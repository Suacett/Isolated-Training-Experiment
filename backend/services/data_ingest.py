"""
Data Ingestion Service - Yahoo Finance Direct API

Uses direct Yahoo Finance API calls instead of yfinance library
to avoid curl_cffi connection issues in Docker containers.
"""

import asyncio
from datetime import datetime, timedelta
import logging
import pandas as pd
import httpx
from sqlalchemy.dialects.postgresql import insert
from services.db import StockPrice, AsyncSessionLocal

logger = logging.getLogger(__name__)


class YahooFinanceClient:
    """
    Yahoo Finance data client using direct API calls.
    No API key required - completely free.
    
    Uses httpx for async HTTP requests instead of yfinance library
    to avoid curl_cffi connectivity issues.
    """
    
    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    def __init__(self):
        logger.info("🔌 Yahoo Finance client initialized (direct API, no yfinance)")

    def verify_ticker(self, ticker: str) -> bool:
        """
        Verify a ticker exists by attempting to fetch minimal data.
        """
        import requests
        try:
            url = f"{self.BASE_URL}/{ticker}"
            params = {"range": "1d", "interval": "1d"}
            response = requests.get(url, params=params, headers=self.HEADERS, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("chart", {}).get("result") is not None
            return False
        except Exception as e:
            logger.warning(f"Ticker verification failed for {ticker}: {e}")
            return False

    def is_crypto(self, ticker: str) -> bool:
        """
        Check if the ticker is a crypto pair.
        Yahoo uses format like BTC-USD, ETH-USD
        """
        return ticker.endswith("-USD") or "USD" in ticker

    def normalize_ticker(self, ticker: str) -> str:
        """
        Convert ticker format to Yahoo Finance format.
        - BTC/USD -> BTC-USD
        - BTCUSD -> BTC-USD
        """
        # Replace / with -
        ticker = ticker.replace("/", "-")
        
        # Handle BTCUSD format
        if ticker.endswith("USD") and not ticker.endswith("-USD"):
            ticker = ticker[:-3] + "-USD"
        
        return ticker.upper()

    async def fetch_data(self, ticker: str, mode: str = "full"):
        """
        Fetches historical data for a ticker using direct Yahoo Finance API.
        
        Args:
            ticker: Stock symbol
            mode: "full" (from 1980) or "daily" (last 5 days)
        """
        original_ticker = ticker
        ticker = self.normalize_ticker(ticker)
        
        logger.info(f"📊 Fetching data for {ticker} from Yahoo Finance (mode={mode})...")
        
        try:
            # Calculate date range
            end_date = datetime.now() + timedelta(days=1)
            
            if mode == "daily":
                # Fetch last 5 days to ensure we catch up on weekends/holidays
                start_date = datetime.now() - timedelta(days=5)
            else:
                # Full history
                start_date = datetime(1980, 1, 1)
            
            logger.info(f"📅 Fetching data from {start_date.date()} to {end_date.date()}")
            
            # Yahoo Finance expects Unix timestamps
            period1 = int(start_date.timestamp())
            period2 = int(end_date.timestamp())
            
            url = f"{self.BASE_URL}/{ticker}"
            params = {
                "period1": period1,
                "period2": period2,
                "interval": "1d",
                "events": "history"
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params, headers=self.HEADERS)
                
                if response.status_code != 200:
                    logger.error(f"Yahoo Finance API returned {response.status_code} for {ticker}")
                    raise ValueError(f"API error: {response.status_code}")
                
                data = response.json()
            
            # Parse the response
            result = data.get("chart", {}).get("result")
            if not result or len(result) == 0:
                logger.warning(f"No data found for {ticker} on Yahoo Finance")
                raise ValueError(f"No data found for {ticker}")
            
            result = result[0]
            timestamps = result.get("timestamp", [])
            quote = result.get("indicators", {}).get("quote", [{}])[0]
            
            if not timestamps:
                raise ValueError(f"No timestamps in response for {ticker}")
            
            opens = quote.get("open", [])
            highs = quote.get("high", [])
            lows = quote.get("low", [])
            closes = quote.get("close", [])
            volumes = quote.get("volume", [])
            
            stock_prices = []
            for i, ts in enumerate(timestamps):
                # Skip if any value is None
                if opens[i] is None or closes[i] is None:
                    continue
                    
                stock_prices.append({
                    "ticker": original_ticker,  # Use original ticker for DB consistency
                    "timestamp": datetime.fromtimestamp(ts),
                    "open": float(opens[i]),
                    "high": float(highs[i]) if highs[i] else float(opens[i]),
                    "low": float(lows[i]) if lows[i] else float(closes[i]),
                    "close": float(closes[i]),
                    "volume": float(volumes[i]) if volumes[i] else 0.0
                })
            
            if not stock_prices:
                logger.warning(f"No valid data parsed for {ticker}")
                raise ValueError(f"No valid data parsed for {ticker}")
            
            # Upsert data in batches to avoid PostgreSQL's 32767 parameter limit
            # Each row has 6 columns, so 1000 rows = 6000 parameters (well under limit)
            BATCH_SIZE = 1000
            async with AsyncSessionLocal() as session:
                for batch_start in range(0, len(stock_prices), BATCH_SIZE):
                    batch = stock_prices[batch_start:batch_start + BATCH_SIZE]
                    stmt = insert(StockPrice).values(batch)
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
                logger.info(f"✅ Saved {len(stock_prices)} records for {original_ticker} (Yahoo Finance) in {(len(stock_prices) + BATCH_SIZE - 1) // BATCH_SIZE} batches")
                
            return len(stock_prices)
        
        except Exception as e:
            logger.error(f"❌ Error fetching data for {ticker} from Yahoo Finance: {e}")
            raise e

    async def fetch_all_data(self, tickers: list[str], mode: str = "full"):
        """
        Fetches historical data for a list of tickers concurrently.
        
        Args:
            tickers: List of stock symbols
            mode: "full" or "daily"
        """
        logger.info(f"📊 Batch fetching {len(tickers)} tickers from Yahoo Finance (mode={mode})...")
        
        results = {"success": [], "failed": []}
        
        # Limit concurrency to avoid rate limiting or connection issues
        sem = asyncio.Semaphore(10)
        
        async def fetch_safe(ticker):
            async with sem:
                try:
                    count = await self.fetch_data(ticker, mode=mode)
                    return {"ticker": ticker, "records": count, "status": "success"}
                except Exception as e:
                    logger.error(f"Failed to fetch {ticker}: {e}")
                    return {"ticker": ticker, "error": str(e), "status": "failed"}

        # Run all fetches concurrently
        fetch_results = await asyncio.gather(*[fetch_safe(t) for t in tickers])
        
        for res in fetch_results:
            if res["status"] == "success":
                results["success"].append({"ticker": res["ticker"], "records": res["records"]})
            else:
                results["failed"].append({"ticker": res["ticker"], "error": res["error"]})
        
        logger.info(f"📊 Batch complete: {len(results['success'])} success, {len(results['failed'])} failed")
        return results


# Keep AlphaVantageClient for EPS data only (intrinsic value calculation)
from utils.config_loader import settings


class AlphaVantageClient:
    """
    Alpha Vantage client - used for EPS data and News Sentiment.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.ALPHA_VANTAGE_KEY
        self.base_url = "https://www.alphavantage.co/query"

    async def fetch_eps_data(self, ticker: str) -> pd.Series:
        """
        Fetches quarterly EPS data for intrinsic value calculation.
        Returns a pandas Series indexed by date.
        """
        if not self.api_key:
            raise ValueError("Alpha Vantage API Key not found")

        params = {
            "function": "EARNINGS",
            "symbol": ticker,
            "apikey": self.api_key
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(self.base_url, params=params)
            data = response.json()
            
        quarterly_earnings = data.get("quarterlyEarnings")
        if not quarterly_earnings:
            return pd.Series(dtype=float)
            
        dates = []
        eps_values = []
        
        for item in quarterly_earnings:
            try:
                dates.append(datetime.strptime(item["fiscalDateEnding"], "%Y-%m-%d"))
                eps_values.append(float(item["reportedEPS"]))
            except (ValueError, KeyError):
                continue
            
        # Create Series, sort by date
        series = pd.Series(data=eps_values, index=dates).sort_index()
        return series

    async def fetch_news_sentiment(self, ticker: str, limit: int = 500) -> pd.DataFrame:
        """
        Fetches news sentiment data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            limit: Max number of articles (default 500)
            
        Returns:
            DataFrame with columns: date, sentiment, num_articles
        """
        if not self.api_key:
            logger.warning("Alpha Vantage API Key not found - skipping sentiment")
            return pd.DataFrame()

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "limit": min(limit, 1000),  # API max is 1000
            "apikey": self.api_key
        }
        
        logger.info(f"📰 Fetching news sentiment for {ticker} from Alpha Vantage...")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(self.base_url, params=params)
                data = response.json()
            
            # Check for errors
            if "Error Message" in data or "Note" in data:
                import re
                msg = data.get('Error Message') or data.get('Note')
                # Mask API key (e.g., "API key as XXXXXX")
                masked_msg = re.sub(r'API key as [A-Z0-9]+', 'API key as [MASKED]', str(msg))
                logger.warning(f"Alpha Vantage API error for {ticker}: {masked_msg}")
                return pd.DataFrame()
            
            feed = data.get("feed", [])
            if not feed:
                logger.info(f"No news sentiment data found for {ticker}")
                return pd.DataFrame()
            
            # Parse articles
            records = []
            for article in feed:
                try:
                    # Parse timestamp (format: 20231215T120000)
                    time_str = article.get("time_published", "")
                    if len(time_str) >= 8:
                        date = datetime.strptime(time_str[:8], "%Y%m%d").date()
                    else:
                        continue
                    
                    # Find sentiment for this specific ticker
                    ticker_sentiments = article.get("ticker_sentiment", [])
                    for ts in ticker_sentiments:
                        if ts.get("ticker", "").upper() == ticker.upper():
                            sentiment_score = float(ts.get("ticker_sentiment_score", 0))
                            records.append({
                                "date": date,
                                "sentiment": sentiment_score
                            })
                            break
                except Exception:
                    continue
            
            if not records:
                return pd.DataFrame()
            
            # Aggregate by date (mean sentiment, count articles)
            df = pd.DataFrame(records)
            daily = df.groupby("date").agg(
                sentiment=("sentiment", "mean"),
                num_articles=("sentiment", "count")
            ).reset_index()
            
            logger.info(f"✅ Fetched {len(daily)} days of sentiment for {ticker}")
            return daily
            
        except Exception as e:
            logger.error(f"Failed to fetch sentiment for {ticker}: {e}")
            return pd.DataFrame()


async def ingest_hybrid_data(ticker: str, mode: str = "full") -> dict:
    """
    Hybrid data ingestion: Yahoo Finance OHLCV + Alpha Vantage Sentiment.
    
    Args:
        ticker: Stock ticker symbol
        mode: "full" or "daily"
        
    Returns:
        Dictionary with ingestion results
    """
    from services.db import SentimentData, save_sentiment_data
    from services.reconciliation import ReconciliationService
    
    results = {
        "ticker": ticker,
        "price_records": 0,
        "sentiment_records": 0,
        "source": "hybrid"
    }
    
    # Step 1: Fetch Yahoo Finance OHLCV
    logger.info(f"🔄 Starting hybrid ingestion for {ticker} (mode={mode})")
    
    try:
        yf_client = YahooFinanceClient()
        price_count = await yf_client.fetch_data(ticker, mode=mode)
        results["price_records"] = price_count
        
        # Trigger reconciliation after new prices are saved
        await ReconciliationService.reconcile_predictions(ticker)
        
    except Exception as e:
        logger.error(f"Failed to fetch price data for {ticker}: {e}")
        raise
    
    # Step 2: Fetch Alpha Vantage Sentiment (rate limited)
    try:
        av_client = AlphaVantageClient()
        sentiment_df = await av_client.fetch_news_sentiment(ticker)
        
        if not sentiment_df.empty:
            # Save sentiment data to database
            for _, row in sentiment_df.iterrows():
                await save_sentiment_data(
                    ticker=ticker,
                    date=datetime.combine(row["date"], datetime.min.time()),
                    sentiment=row["sentiment"],
                    num_articles=int(row["num_articles"])
                )
            results["sentiment_records"] = len(sentiment_df)
            
    except Exception as e:
        logger.warning(f"Failed to fetch sentiment for {ticker}: {e}")
        # Don't fail the whole ingestion if sentiment fails
    
    logger.info(f"✅ Hybrid ingestion complete for {ticker}: {results['price_records']} prices, {results['sentiment_records']} sentiment days")
    
    return results


def sync_ingest_hybrid_data(ticker: str) -> dict:
    """
    Synchronous version of hybrid data ingestion for use in background tasks.
    Uses requests library instead of httpx to avoid event loop issues.
    """
    import requests
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    import os
    
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:password@timescaledb:5432/stock_db")
    # Convert async URL to sync
    SYNC_DATABASE_URL = DATABASE_URL.replace("+asyncpg", "")
    
    results = {
        "ticker": ticker,
        "price_records": 0,
        "sentiment_records": 0,
        "source": "hybrid"
    }
    
    logger.info(f"🔄 Starting hybrid ingestion for {ticker} (sync)")
    
    # Step 1: Fetch Yahoo Finance OHLCV
    try:
        yf = YahooFinanceClient()
        normalized_ticker = yf.normalize_ticker(ticker)
        
        end_date = datetime.now() + timedelta(days=1)
        start_date = datetime(1980, 1, 1)  # Fetch all available history
        
        url = f"{yf.BASE_URL}/{normalized_ticker}"
        params = {
            "period1": int(start_date.timestamp()),
            "period2": int(end_date.timestamp()),
            "interval": "1d",
            "events": "history"
        }
        
        response = requests.get(url, params=params, headers=yf.HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        result = data.get("chart", {}).get("result")
        if not result:
            raise ValueError(f"No data found for {ticker}")
        
        result = result[0]
        timestamps = result.get("timestamp", [])
        quote = result.get("indicators", {}).get("quote", [{}])[0]
        
        stock_prices = []
        for i, ts in enumerate(timestamps):
            opens = quote.get("open", [])
            highs = quote.get("high", [])
            lows = quote.get("low", [])
            closes = quote.get("close", [])
            volumes = quote.get("volume", [])
            
            if i >= len(opens) or opens[i] is None or closes[i] is None:
                continue
                
            stock_prices.append({
                "ticker": ticker,
                "timestamp": datetime.fromtimestamp(ts),
                "open": float(opens[i]),
                "high": float(highs[i]) if highs[i] else float(opens[i]),
                "low": float(lows[i]) if lows[i] else float(closes[i]),
                "close": float(closes[i]),
                "volume": float(volumes[i]) if volumes[i] else 0.0
            })
        
        # Save to database (sync) in batches to avoid PostgreSQL's 32767 parameter limit
        BATCH_SIZE = 1000
        engine = create_engine(SYNC_DATABASE_URL)
        Session = sessionmaker(bind=engine)
        with Session() as session:
            for batch_start in range(0, len(stock_prices), BATCH_SIZE):
                batch = stock_prices[batch_start:batch_start + BATCH_SIZE]
                stmt = pg_insert(StockPrice).values(batch)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["ticker", "timestamp"],
                    set_={
                        "open": stmt.excluded.open,
                        "high": stmt.excluded.high,
                        "low": stmt.excluded.low,
                        "close": stmt.excluded.close,
                        "volume": stmt.excluded.volume
                    }
                )
                session.execute(stmt)
            session.commit()
        
        results["price_records"] = len(stock_prices)
        logger.info(f"✅ Saved {len(stock_prices)} price records for {ticker} in {(len(stock_prices) + BATCH_SIZE - 1) // BATCH_SIZE} batches")
        
    except Exception as e:
        logger.error(f"Failed to fetch price data for {ticker}: {e}")
        raise
    
    # Step 2: Fetch Alpha Vantage Sentiment
    try:
        av_client = AlphaVantageClient()
        if not av_client.api_key:
            logger.warning("Alpha Vantage API Key not found - skipping sentiment")
        else:
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "limit": 500,
                "apikey": av_client.api_key
            }
            
            logger.info(f"📰 Fetching news sentiment for {ticker}...")
            response = requests.get(av_client.base_url, params=params, timeout=30)
            data = response.json()
            
            if "Error Message" not in data and "Note" not in data:
                feed = data.get("feed", [])
                
                records = []
                for article in feed:
                    try:
                        time_str = article.get("time_published", "")
                        if len(time_str) >= 8:
                            date = datetime.strptime(time_str[:8], "%Y%m%d").date()
                        else:
                            continue
                        
                        ticker_sentiments = article.get("ticker_sentiment", [])
                        for ts in ticker_sentiments:
                            if ts.get("ticker", "").upper() == ticker.upper():
                                sentiment_score = float(ts.get("ticker_sentiment_score", 0))
                                records.append({
                                    "date": date,
                                    "sentiment": sentiment_score
                                })
                                break
                    except Exception:
                        continue
                
                if records:
                    from services.db import SentimentData
                    
                    df = pd.DataFrame(records)
                    daily = df.groupby("date").agg(
                        sentiment=("sentiment", "mean"),
                        num_articles=("sentiment", "count")
                    ).reset_index()
                    
                    sentiment_records = []
                    for _, row in daily.iterrows():
                        sentiment_records.append({
                            "ticker": ticker,
                            "date": datetime.combine(row["date"], datetime.min.time()),
                            "sentiment": row["sentiment"],
                            "num_articles": int(row["num_articles"])
                        })
                    
                    engine = create_engine(SYNC_DATABASE_URL)
                    Session = sessionmaker(bind=engine)
                    with Session() as session:
                        for rec in sentiment_records:
                            sentiment = SentimentData(
                                ticker=rec["ticker"],
                                date=rec["date"],
                                sentiment=rec["sentiment"],
                                num_articles=rec["num_articles"]
                            )
                            session.add(sentiment)
                        session.commit()
                    
                    results["sentiment_records"] = len(sentiment_records)
                    logger.info(f"✅ Saved {len(sentiment_records)} sentiment records for {ticker}")
                    
    except Exception as e:
        logger.warning(f"Failed to fetch sentiment for {ticker}: {e}")
    
    logger.info(f"✅ Hybrid ingestion complete for {ticker}: {results['price_records']} prices, {results['sentiment_records']} sentiment days")
    return results



async def ingest_bulk_history(data_dir: str = None) -> dict:
    """
    Bulk ingest historical data from CSV files into database.
    
    Reads all CSV files from the training_raw directory and
    efficiently inserts them into the database using bulk upsert.
    
    Args:
        data_dir: Path to directory containing CSV files.
                  Defaults to backend/data/training_raw/
    
    Returns:
        Dictionary with success count and failed tickers
    """
    from pathlib import Path
    
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "training_raw"
    else:
        data_dir = Path(data_dir)
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return {"success": 0, "failed": [], "error": "Directory not found"}
    
    csv_files = list(data_dir.glob("*.csv"))
    logger.info(f"📂 Found {len(csv_files)} CSV files to ingest")
    
    results = {"success": 0, "failed": []}
    
    for csv_file in csv_files:
        ticker = csv_file.stem  # filename without extension
        
        try:
            # Read CSV
            df = pd.read_csv(csv_file, parse_dates=['date'])
            
            if df.empty:
                results["failed"].append(ticker)
                continue
            
            # Prepare data for insertion
            stock_prices = []
            for _, row in df.iterrows():
                stock_prices.append({
                    "ticker": ticker,
                    "timestamp": row['date'],
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row['volume']) if pd.notna(row['volume']) else 0.0
                })
            
            # Bulk upsert in batches to avoid PostgreSQL's 32767 parameter limit
            BATCH_SIZE = 1000
            async with AsyncSessionLocal() as session:
                for batch_start in range(0, len(stock_prices), BATCH_SIZE):
                    batch = stock_prices[batch_start:batch_start + BATCH_SIZE]
                    stmt = insert(StockPrice).values(batch)
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
            
            results["success"] += 1
            
            if results["success"] % 50 == 0:
                logger.info(f"  Progress: {results['success']}/{len(csv_files)}")
                
        except Exception as e:
            logger.warning(f"Failed to ingest {ticker}: {e}")
            results["failed"].append(ticker)
            continue
    
    logger.info(f"✅ Bulk ingest complete: {results['success']} success, {len(results['failed'])} failed")
    return results

