import asyncio
from datetime import datetime
from sqlalchemy.dialects.postgresql import insert
from services.db import StockPrice, AsyncSessionLocal
from utils.config_loader import Config
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

class AlpacaDataClient:
    def __init__(self):
        self.api_key = Config.ALPACA_API_KEY
        self.secret_key = Config.ALPACA_SECRET_KEY
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca credentials not found in Config")

        self.stock_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        self.crypto_client = CryptoHistoricalDataClient(self.api_key, self.secret_key)

    def is_crypto(self, ticker: str) -> bool:
        """
        Check if the ticker is a crypto pair.
        """
        return "USD" in ticker or "/" in ticker

    async def fetch_data(self, ticker: str):
        """
        Fetches historical data for a ticker (Stock or Crypto) and saves it to the DB.
        """
        print(f"Fetching data for {ticker}...")
        
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
                print(f"No data found for {ticker}")
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
                print(f"No data parsed for {ticker}")
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
                print(f"Successfully saved {len(stock_prices)} records for {ticker}")

        except Exception as e:
            print(f"Exception occurred while fetching/saving data for {ticker}: {e}")

    async def fetch_all_data(self, tickers: list[str]):
        """
        Fetches historical data for a list of tickers (Stock or Crypto) and saves it to the DB.
        Optimized to use multi-ticker endpoints where possible.
        """
        print(f"Starting batch fetch for {len(tickers)} tickers...")
        
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
            print(f"Fetching {len(stocks)} stocks...")
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
                        print(f"No data found for {ticker}")
            except Exception as e:
                print(f"Error fetching stocks batch: {e}")
                
        # 2. Process Crypto (Batch if possible, but let's try batch)
        if cryptos:
            print(f"Fetching {len(cryptos)} crypto pairs...")
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
                        print(f"No data found for {ticker}")
            except Exception as e:
                print(f"Error fetching crypto batch: {e}")

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
            print(f"Saved {len(stock_prices)} records for {ticker}")
