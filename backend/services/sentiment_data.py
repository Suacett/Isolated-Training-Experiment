"""
Sentiment data collection from Alpha Vantage News API
Ported from legacy/LSTM_AI_Stock_Predictor/TrainingData/featuresPy/sentiment.py

This module fetches and aggregates news sentiment data for stocks.
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import asyncio
import os

logger = logging.getLogger(__name__)

# Import key pool for multi-key Alpha Vantage support
try:
    from services.alpha_vantage_pool import key_pool
except ImportError:
    key_pool = None


class SentimentDataCollector:
    """Collects sentiment data from Alpha Vantage News API"""

    def __init__(self, api_key: Optional[str] = None):
        # Allow override with explicit key, otherwise use key pool if available
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12  # Alpha Vantage free tier: 5 calls/minute
        self.use_key_pool = key_pool is not None and key_pool.get_key_count() > 0

    def _get_api_key(self) -> Optional[str]:
        """Get API key from key pool (with rotation) or fallback to settings"""
        # If explicit key was provided, use it
        if self.api_key:
            return self.api_key

        # Try key pool first (supports multiple keys with automatic rotation)
        if self.use_key_pool:
            return key_pool.get_key()

        # Fallback to settings object for backwards compatibility
        try:
            from utils.config_loader import settings
            return settings.get("ALPHA_VANTAGE_KEY")
        except Exception as e:
            logger.error(f"Failed to load Alpha Vantage API key: {e}")
            return None

    async def fetch_daily_sentiment(
        self,
        ticker: str,
        date: datetime
    ) -> Optional[dict]:
        """
        Fetch sentiment for a specific date.

        Returns:
            dict with 'sentiment' and 'num_articles' or None if no data
        """
        api_key = self._get_api_key()
        if not api_key:
            logger.warning("Alpha Vantage API key not configured")
            return None

        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ticker,
            'apikey': api_key,
            'time_from': date.strftime('%Y%m%dT0000'),
            'time_to': date.strftime('%Y%m%dT2359'),
            'limit': 1000
        }

        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(self.base_url, params=params, timeout=30)
            )

            if response.status_code != 200:
                logger.error(f"Alpha Vantage API error: {response.status_code}")
                return None

            data = response.json()

            # Check for API limit or error
            if "Note" in data or "Information" in data:
                logger.warning(f"Alpha Vantage rate limit or info: {data}")
                return None

            if "Error Message" in data:
                logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return None

            feed = data.get("feed", [])
            if not feed:
                return {"sentiment": 0.0, "num_articles": 0}

            # Extract sentiment scores for this ticker
            sentiments = []
            for article in feed:
                for ticker_sentiment in article.get("ticker_sentiment", []):
                    if ticker_sentiment.get("ticker", "").upper() == ticker.upper():
                        score = ticker_sentiment.get("ticker_sentiment_score")
                        if score:
                            try:
                                sentiments.append(float(score))
                            except ValueError:
                                continue

            if not sentiments:
                return {"sentiment": 0.0, "num_articles": len(feed)}

            # Average sentiment score
            avg_sentiment = sum(sentiments) / len(sentiments)

            return {
                "sentiment": avg_sentiment,
                "num_articles": len(feed)
            }

        except Exception as e:
            logger.error(f"Failed to fetch sentiment for {ticker} on {date}: {e}")
            return None

    async def fetch_sentiment_range(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch sentiment data for a date range.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with columns: [date, sentiment, num_articles]
        """
        logger.info(f"Fetching sentiment for {ticker} from {start_date} to {end_date}")

        rows = []
        current_date = start_date
        no_news_streak = 0
        max_no_news_days = 30  # Stop if no news for 30 consecutive days

        while current_date <= end_date:
            result = await self.fetch_daily_sentiment(ticker, current_date)

            if result is None:
                # API error or rate limit - stop
                logger.warning(f"Stopping sentiment fetch due to API issue")
                break

            if result['num_articles'] == 0:
                no_news_streak += 1
                if no_news_streak >= max_no_news_days:
                    logger.info(f"No news for {max_no_news_days} days, stopping sentiment fetch")
                    break
            else:
                no_news_streak = 0

            rows.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'sentiment': result['sentiment'],
                'num_articles': result['num_articles']
            })

            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)

            current_date += timedelta(days=1)

        if not rows:
            logger.warning(f"No sentiment data found for {ticker}")
            return pd.DataFrame(columns=['date', 'sentiment', 'num_articles'])

        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])

        logger.info(f"Found sentiment data for {len(df)} days for {ticker}")
        return df

    async def get_recent_sentiment(
        self,
        ticker: str,
        days: int = 90
    ) -> pd.DataFrame:
        """
        Get sentiment data for recent N days.

        Args:
            ticker: Stock ticker symbol
            days: Number of days to fetch (default: 90)

        Returns:
            DataFrame with columns: [date, sentiment, num_articles]
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        return await self.fetch_sentiment_range(ticker, start_date, end_date)

    async def update_sentiment_cache(
        self,
        ticker: str,
        existing_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Update sentiment cache with only missing dates.

        Args:
            ticker: Stock ticker symbol
            existing_df: Existing sentiment DataFrame (if any)

        Returns:
            Updated DataFrame with new sentiment data appended
        """
        if existing_df is not None and not existing_df.empty:
            last_date = pd.to_datetime(existing_df['date']).max()
            start_date = last_date + timedelta(days=1)
        else:
            # Default to last 90 days if no existing data
            start_date = datetime.now() - timedelta(days=90)

        end_date = datetime.now()

        if start_date > end_date:
            logger.info(f"Sentiment data for {ticker} is up to date")
            return existing_df if existing_df is not None else pd.DataFrame()

        # Fetch new data
        new_data = await self.fetch_sentiment_range(ticker, start_date, end_date)

        if existing_df is not None and not existing_df.empty:
            # Concatenate and deduplicate
            combined = pd.concat([existing_df, new_data], ignore_index=True)
            combined = combined.drop_duplicates(subset=['date'], keep='last')
            combined = combined.sort_values('date').reset_index(drop=True)
            return combined
        else:
            return new_data

    async def fetch_bulk_sentiment(
        self,
        tickers: list,
        days: int = 7
    ) -> dict:
        """
        Fetch sentiment for multiple tickers efficiently.

        With key pool (5 keys), rotational usage avoids rate limits, but execution remains sequential.

        Args:
            tickers: List of ticker symbols
            days: Number of days to fetch sentiment for (default: 7)

        Returns:
            Dict mapping ticker -> {sentiment_score, num_articles, last_updated}
        """
        results = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        for ticker in tickers:
            try:
                result = await self.fetch_sentiment_range(ticker, start_date, end_date)

                if not result.empty:
                    # Get most recent sentiment
                    latest = result.iloc[-1]
                    results[ticker] = {
                        "sentiment": float(latest['sentiment']),
                        "num_articles": int(latest['num_articles']),
                        "last_updated": str(latest['date'])
                    }
                else:
                    # No data available
                    results[ticker] = {
                        "sentiment": 0.0,
                        "num_articles": 0,
                        "last_updated": None
                    }

            except Exception as e:
                logger.warning(f"Failed to fetch sentiment for {ticker}: {e}")
                results[ticker] = {
                    "sentiment": None,
                    "num_articles": 0,
                    "last_updated": None,
                    "error": str(e)
                }

        return results


# Global instance
sentiment_collector = SentimentDataCollector()
