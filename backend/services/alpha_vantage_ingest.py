"""
Alpha Vantage Data Client

Provides stock data fetching as a backup to Alpaca.
Note: Alpha Vantage has stricter rate limits (5 calls/min on free tier).
"""

import time
import requests
from datetime import datetime
from typing import List, Optional
from sqlalchemy.dialects.postgresql import insert
from services.db import StockPrice, AsyncSessionLocal
from utils.config_loader import settings


class AlphaVantageClient:
    """
    Alpha Vantage data client for fetching historical stock data.

    Note: Free tier limits:
    - 5 API calls per minute
    - 500 API calls per day
    """

    BASE_URL = "https://www.alphavantage.co/query"
    RATE_LIMIT_DELAY = 15  # Seconds between calls (conservative for free tier)

    def __init__(self):
        self.api_key = settings.ALPHA_VANTAGE_KEY

        if not self.api_key:
            raise ValueError("Alpha Vantage API key not found in settings")

    async def fetch_daily_data(self, ticker: str) -> None:
        """
        Fetches historical daily data for a single ticker and saves to DB.

        Args:
            ticker: Stock symbol (e.g., "AAPL", "MSFT")
        """
        print(f"[Alpha Vantage] Fetching data for {ticker}...")

        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "apikey": self.api_key,
            "apikey": self.api_key,
            "outputsize": "compact"  # Get compact history (100 days) for free tier
        }

        try:
            # Rate limit before API call
            time.sleep(self.RATE_LIMIT_DELAY)

            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            # Check for API errors
            if "Error Message" in data:
                print(f"[Alpha Vantage] API error for {ticker}: {data['Error Message']}")
                return

            if "Note" in data:
                print(f"[Alpha Vantage] Rate limit hit: {data['Note']}")
                return

            # Parse time series data
            time_series = data.get("Time Series (Daily)", {})
            if not time_series:
                print(f"[Alpha Vantage] No data found for {ticker}")
                return

            await self._save_time_series(ticker, time_series)

        except requests.RequestException as e:
            print(f"[Alpha Vantage] Request error for {ticker}: {e}")
        except Exception as e:
            print(f"[Alpha Vantage] Exception fetching {ticker}: {e}")

    async def fetch_data(self, ticker: str) -> None:
        """Alias for fetch_daily_data for compatibility."""
        await self.fetch_daily_data(ticker)

    async def fetch_all_data(self, tickers: List[str]) -> None:
        """
        Fetches historical data for multiple tickers with rate limiting.

        Args:
            tickers: List of stock symbols
        """
        print(f"[Alpha Vantage] Starting fetch for {len(tickers)} tickers...")
        print(f"[Alpha Vantage] Rate limit: {self.RATE_LIMIT_DELAY}s between calls")

        for ticker in tickers:
            await self.fetch_daily_data(ticker)

        print(f"[Alpha Vantage] Completed fetch for {len(tickers)} tickers")

    async def fetch_eps_data(self, ticker: str) -> Optional[dict]:
        """
        Fetches earnings (EPS) data for a ticker.
        Returns quarterly and annual EPS data.

        Args:
            ticker: Stock symbol

        Returns:
            Dict with 'annual' and 'quarterly' EPS data, or None on error
        """
        print(f"[Alpha Vantage] Fetching EPS data for {ticker}...")

        params = {
            "function": "EARNINGS",
            "symbol": ticker,
            "apikey": self.api_key
        }

        try:
            # Rate limit before API call
            time.sleep(self.RATE_LIMIT_DELAY)

            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if "Error Message" in data:
                print(f"[Alpha Vantage] API error for {ticker} EPS: {data['Error Message']}")
                return None

            if "Note" in data:
                print(f"[Alpha Vantage] Rate limit hit: {data['Note']}")
                return None

            annual = data.get("annualEarnings", [])
            quarterly = data.get("quarterlyEarnings", [])

            return {
                "annual": annual,
                "quarterly": quarterly
            }

        except requests.RequestException as e:
            print(f"[Alpha Vantage] Request error for {ticker} EPS: {e}")
            return None
        except Exception as e:
            print(f"[Alpha Vantage] Exception fetching EPS for {ticker}: {e}")
            return None

    async def _save_time_series(self, ticker: str, time_series: dict) -> None:
        """
        Saves time series data to the database.

        Args:
            ticker: Stock symbol
            time_series: Dict of date -> OHLCV data
        """
        stock_prices = []

        for date_str, values in time_series.items():
            try:
                timestamp = datetime.strptime(date_str, "%Y-%m-%d")
                stock_prices.append({
                    "ticker": ticker,
                    "timestamp": timestamp,
                    "open": float(values["1. open"]),
                    "high": float(values["2. high"]),
                    "low": float(values["3. low"]),
                    "close": float(values["4. close"]),
                    "volume": float(values["5. volume"])
                })
            except (ValueError, KeyError) as e:
                print(f"[Alpha Vantage] Error parsing data for {ticker} on {date_str}: {e}")
                continue

        if not stock_prices:
            print(f"[Alpha Vantage] No valid records to save for {ticker}")
            return

        # Upsert data
        async with AsyncSessionLocal() as session:
            stmt = insert(StockPrice).values(stock_prices)
            stmt = stmt.on_conflict_do_update(
                index_elements=[StockPrice.ticker, StockPrice.timestamp],
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume
                }
            )
            await session.execute(stmt)
            await session.commit()
            print(f"[Alpha Vantage] Saved {len(stock_prices)} records for {ticker}")
