import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Float, DateTime, select, desc, Integer, delete
from sqlalchemy.dialects.postgresql import insert

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:password@timescaledb:5432/stock_db")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

class StockPrice(Base):
    __tablename__ = "stock_prices"
    __table_args__ = {'extend_existing': True}
    
    ticker = Column(String, primary_key=True)
    timestamp = Column(DateTime, primary_key=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

class Watchlist(Base):
    __tablename__ = "watchlist"
    __table_args__ = {'extend_existing': True}
    ticker = Column(String, primary_key=True)

class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String)
    timestamp = Column(DateTime)
    predicted_price = Column(Float)
    actual_price = Column(Float, nullable=True)

async def get_latest_close(ticker: str):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(StockPrice.close)
            .where(StockPrice.ticker == ticker)
            .order_by(desc(StockPrice.timestamp))
            .limit(1)
        )
        return result.scalar_one_or_none()

async def get_unique_tickers():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(StockPrice.ticker).distinct())
        return result.scalars().all()

async def get_historical_data(ticker: str, limit: int = 60):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(StockPrice)
            .where(StockPrice.ticker == ticker)
            .order_by(desc(StockPrice.timestamp))
            .limit(limit)
        )
        data = result.scalars().all()
        # Return chronological
        return data[::-1]

async def add_watchlist_item(ticker: str):
    async with AsyncSessionLocal() as session:
        stmt = insert(Watchlist).values(ticker=ticker).on_conflict_do_nothing()
        await session.execute(stmt)
        await session.commit()

async def remove_watchlist_item(ticker: str):
    async with AsyncSessionLocal() as session:
        await session.execute(delete(Watchlist).where(Watchlist.ticker == ticker))
        await session.commit()

async def get_watchlist():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Watchlist.ticker))
        return result.scalars().all()

async def save_prediction(ticker: str, price: float, timestamp):
    async with AsyncSessionLocal() as session:
        # Simple insert for now, could be upsert if we want one prediction per day per ticker
        session.add(Prediction(ticker=ticker, predicted_price=price, timestamp=timestamp))
        await session.commit()
