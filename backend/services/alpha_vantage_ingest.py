"""
Alpha Vantage Data Client

Provides stock data fetching as a backup to Alpaca.
Note: Alpha Vantage has stricter rate limits (5 calls/min on free tier).

Supports multiple API keys with automatic rotation when rate limits are hit.
Set ALPHA_VANTAGE_KEYS as comma-separated keys, or ALPHA_VANTAGE_KEY for single key.
"""

import os
import time
import requests
from datetime import datetime
from typing import List, Optional
from sqlalchemy.dialects.postgresql import insert
from services.db import StockPrice, AsyncSessionLocal
from utils.config_loader import settings
import logging

logger = logging.getLogger(__name__)


class AlphaVantageClient:
    """
    Alpha Vantage data client for fetching historical stock data.

    Supports multiple API keys with automatic rotation.
    
    Note: Free tier limits per key:
    - 5 API calls per minute
    - 500 API calls per day
    """

    BASE_URL = "https://www.alphavantage.co/query"
    RATE_LIMIT_DELAY = 12  # Seconds between calls (conservative for free tier)

    def __init__(self):
        # Support multiple API keys (comma-separated)
        keys_str = os.getenv("ALPHA_VANTAGE_KEYS", "") or settings.ALPHA_VANTAGE_KEY or ""
        
        # Parse comma-separated keys
        self.api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        
        # Also add single key if separate
        if settings.ALPHA_VANTAGE_KEY and settings.ALPHA_VANTAGE_KEY not in self.api_keys:
            self.api_keys.append(settings.ALPHA_VANTAGE_KEY)
        
        if not self.api_keys:
            raise ValueError("No Alpha Vantage API keys found. Set ALPHA_VANTAGE_KEY or ALPHA_VANTAGE_KEYS")
        
        self.current_key_index = 0
        self.key_call_counts = {key: 0 for key in self.api_keys}
        self.key_rate_limited = {key: False for key in self.api_keys}
        
        logger.info(f"[Alpha Vantage] Initialized with {len(self.api_keys)} API key(s)")
    
    @property
    def api_key(self) -> str:
        """Get the current active API key."""
        return self.api_keys[self.current_key_index]
    
    def rotate_key(self) -> bool:
        """
        Rotate to the next API key.
        Returns True if successfully rotated, False if all keys are rate limited.
        """
        # Find a non-rate-limited key
        for _ in range(len(self.api_keys)):
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            if not self.key_rate_limited[self.api_key]:
                logger.info(f"[Alpha Vantage] Rotated to key {self.current_key_index + 1}/{len(self.api_keys)}")
                return True
        
        # All keys are rate limited - reset and try again with delay
        logger.warning("[Alpha Vantage] All API keys rate limited. Resetting...")
        for key in self.api_keys:
            self.key_rate_limited[key] = False
        time.sleep(60)  # Wait a minute before retrying
        return True
    
    def mark_rate_limited(self):
        """Mark the current key as rate limited and rotate."""
        self.key_rate_limited[self.api_key] = True
        logger.warning(f"[Alpha Vantage] Key {self.current_key_index + 1} rate limited, rotating...")
        self.rotate_key()

    async def fetch_daily_data(self, ticker: str) -> None:
        """
        Fetches historical daily data for a single ticker and saves to DB.

        Args:
            ticker: Stock symbol (e.g., "AAPL", "MSFT")
        """
        logger.info(f"[Alpha Vantage] Fetching data for {ticker} with key {self.current_key_index + 1}...")

        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
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
                logger.warning(f"[Alpha Vantage] API error for {ticker}: {data['Error Message']}")
                return

            if "Note" in data:
                logger.warning(f"[Alpha Vantage] Rate limit hit: {data['Note']}")
                self.mark_rate_limited()
                # Retry with new key
                return await self.fetch_daily_data(ticker)

            # Parse time series data
            time_series = data.get("Time Series (Daily)", {})
            if not time_series:
                logger.warning(f"[Alpha Vantage] No data found for {ticker}")
                return

            await self._save_time_series(ticker, time_series)

        except requests.RequestException as e:
            logger.error(f"[Alpha Vantage] Request error for {ticker}: {e}")
        except Exception as e:
            logger.error(f"[Alpha Vantage] Exception fetching {ticker}: {e}")

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

        logger.info(f"[Alpha Vantage] Completed fetch for {len(tickers)} tickers")

    async def fetch_eps_data(self, ticker: str) -> Optional[dict]:
        """
        Fetches earnings (EPS) data for a ticker.
        Returns quarterly and annual EPS data.

        Args:
            ticker: Stock symbol

        Returns:
            Dict with 'annual' and 'quarterly' EPS data, or None on error
        """
        logger.info(f"[Alpha Vantage] Fetching EPS data for {ticker} with key {self.current_key_index + 1}...")

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
                logger.warning(f"[Alpha Vantage] API error for {ticker} EPS: {data['Error Message']}")
                return None

            if "Note" in data:
                logger.warning(f"[Alpha Vantage] Rate limit hit: {data['Note']}")
                self.mark_rate_limited()
                # Retry with new key
                return await self.fetch_eps_data(ticker)

            annual = data.get("annualEarnings", [])
            quarterly = data.get("quarterlyEarnings", [])

            return {
                "annual": annual,
                "quarterly": quarterly
            }

        except requests.RequestException as e:
            logger.error(f"[Alpha Vantage] Request error for {ticker} EPS: {e}")
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
