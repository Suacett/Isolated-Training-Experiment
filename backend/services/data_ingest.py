import asyncio
import os
from datetime import datetime
from sqlalchemy.dialects.postgresql import insert
from services.db import StockPrice, AsyncSessionLocal
from utils.config_loader import settings
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import httpx
import pandas as pd

import logging

logger = logging.getLogger(__name__)

class AlpacaDataClient:
    def __init__(self, api_key: str = None, secret_key: str = None):
        self.api_key = api_key or settings.ALPACA_API_KEY
        self.secret_key = secret_key or settings.ALPACA_SECRET_KEY
        self.base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        
        logger.info(f"🔌 Connecting to Alpaca via: {self.base_url}")
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca credentials not found in settings")

        self.stock_client = StockHistoricalDataClient(self.api_key, self.secret_key, url_override=self.base_url)
        self.crypto_client = CryptoHistoricalDataClient(self.api_key, self.secret_key, url_override=self.base_url)

    def verify_credentials(self):
        """
        Simple check to verify credentials.
        """
        try:
            # Try to fetch 1 bar of AAPL to verify keys
            request_params = StockBarsRequest(
                symbol_or_symbols=["AAPL"],
                timeframe=TimeFrame.Day,
                start=datetime(2023, 1, 1),
                limit=1
            )
            self.stock_client.get_stock_bars(request_params)
            return True
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg:
                logger.error("❌ API AUTHENTICATION FAILED during verification.")
            raise e

    def is_crypto(self, ticker: str) -> bool:
        """
        Check if the ticker is a crypto pair.
        """
        return "USD" in ticker or "/" in ticker

    async def fetch_data(self, ticker: str):
        """
        Fetches historical data for a ticker (Stock or Crypto) and saves it to the DB.
        """
        logger.info(f"Fetching data for {ticker}...")
        
        is_crypto_asset = self.is_crypto(ticker)
        start_date = datetime(2015, 1, 1) # Fetch from 2015
        
        try:
            bars = []
            if is_crypto_asset:
                request_params = CryptoBarsRequest(
                    symbol_or_symbols=[ticker],
                    timeframe=TimeFrame.Day,
                    start=start_date
                )
                # get_crypto_bars returns a BarSet, we need to access the list by symbol
                response = self.crypto_client.get_crypto_bars(request_params)
                bars = response[ticker]
            else:
                request_params = StockBarsRequest(
                    symbol_or_symbols=[ticker],
                    timeframe=TimeFrame.Day,
                    start=start_date
                )
                response = self.stock_client.get_stock_bars(request_params)
                bars = response[ticker]

            if not bars:
                logger.warning(f"No data found for {ticker}")
                return

            stock_prices = []
            for bar in bars:
                stock_prices.append({
                    "ticker": ticker,
                    "timestamp": bar.timestamp.replace(tzinfo=None), # Alpaca returns datetime with timezone, make naive
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume)
                })

            if not stock_prices:
                logger.warning(f"No data parsed for {ticker}")
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
                logger.info(f"Successfully saved {len(stock_prices)} records for {ticker}")

        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg:
                msg = "❌ API AUTHENTICATION FAILED. Please check your API Keys."
                logger.error(msg)
                raise Exception(msg)
            else:
                logger.error(f"Exception occurred while fetching/saving data for {ticker}: {e}")
                raise e

    async def fetch_all_data(self, tickers: list[str]):
        """
        Fetches historical data for a list of tickers (Stock or Crypto) and saves it to the DB.
        Optimized to use multi-ticker endpoints where possible.
        """
        logger.info(f"Starting batch fetch for {len(tickers)} tickers...")
        
        stocks = []
        cryptos = []
        
        for ticker in tickers:
            if self.is_crypto(ticker):
                cryptos.append(ticker)
            else:
                stocks.append(ticker)
                
        start_date = datetime(2015, 1, 1)
        
        # 1. Process Stocks (Batch)
        if stocks:
            logger.info(f"Fetching {len(stocks)} stocks...")
            try:
                request_params = StockBarsRequest(
                    symbol_or_symbols=stocks,
                    timeframe=TimeFrame.Day,
                    start=start_date
                )
                response = self.stock_client.get_stock_bars(request_params)
                
                for ticker in stocks:
                    if ticker in response:
                        bars = response[ticker]
                        await self._save_bars(ticker, bars)
                    else:
                        logger.warning(f"No data found for {ticker}")
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "403" in error_msg:
                    msg = "❌ API AUTHENTICATION FAILED. Please check your API Keys."
                    logger.error(msg)
                    raise Exception(msg)
                else:
                    logger.error(f"Error fetching stocks batch: {e}")
                    raise e
                
        # 2. Process Crypto (Batch if possible, but let's try batch)
        if cryptos:
            logger.info(f"Fetching {len(cryptos)} crypto pairs...")
            try:
                request_params = CryptoBarsRequest(
                    symbol_or_symbols=cryptos,
                    timeframe=TimeFrame.Day,
                    start=start_date
                )
                response = self.crypto_client.get_crypto_bars(request_params)
                
                for ticker in cryptos:
                    if ticker in response:
                        bars = response[ticker]
                        await self._save_bars(ticker, bars)
                    else:
                        logger.warning(f"No data found for {ticker}")
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "403" in error_msg:
                    msg = "❌ API AUTHENTICATION FAILED. Please check your API Keys."
                    logger.error(msg)
                    raise Exception(msg)
                else:
                    logger.error(f"Error fetching crypto batch: {e}")
                    raise e

    async def _save_bars(self, ticker: str, bars):
        if not bars:
            return

        stock_prices = []
        for bar in bars:
            stock_prices.append({
                "ticker": ticker,
                "timestamp": bar.timestamp.replace(tzinfo=None),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume)
            })

        if not stock_prices:
            return

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
            logger.info(f"Saved {len(stock_prices)} records for {ticker}")


class AlphaVantageClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.ALPHA_VANTAGE_KEY
        self.base_url = "https://www.alphavantage.co/query"
        
        if not self.api_key:
            # Don't raise immediately on init if we want to allow instantiation for checking
            pass

    async def fetch_data(self, ticker: str):
        """
        Fetches daily time series (TIME_SERIES_DAILY) and saves to DB.
        """
        if not self.api_key:
             raise ValueError("Alpha Vantage API Key not found")

        logger.info(f"Fetching data for {ticker} from Alpha Vantage...")
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "apikey": self.api_key,
            "outputsize": "compact",
            "datatype": "json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(self.base_url, params=params)
            data = response.json()
            
        if "Error Message" in data:
             raise Exception(f"Alpha Vantage Error: {data['Error Message']}")
        if "Note" in data:
             # Rate limit hit
             raise Exception(f"Alpha Vantage Rate Limit: {data['Note']}")
        if "Information" in data:
             raise Exception(f"Alpha Vantage Information: {data['Information']}")
             
        time_series = data.get("Time Series (Daily)")
        if not time_series:
            logger.warning(f"No data found for {ticker} in Alpha Vantage. Response keys: {list(data.keys())}")
            return

        stock_prices = []
        for date_str, values in time_series.items():
            # Alpha Vantage returns strings
            stock_prices.append({
                "ticker": ticker,
                "timestamp": datetime.strptime(date_str, "%Y-%m-%d"),
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "volume": float(values["5. volume"])
            })
            
        if not stock_prices:
             return

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
            logger.info(f"Successfully saved {len(stock_prices)} records for {ticker} (Alpha Vantage)")

    async def fetch_eps_data(self, ticker: str) -> pd.Series:
        """
        Fetches quarterly EPS data.
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
            dates.append(datetime.strptime(item["fiscalDateEnding"], "%Y-%m-%d"))
            eps_values.append(float(item["reportedEPS"]))
            
        # Create Series, sort by date
        series = pd.Series(data=eps_values, index=dates).sort_index()
        return series
