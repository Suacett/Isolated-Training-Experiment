import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Float, DateTime, select, desc, Integer, delete, Boolean
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime

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
    prediction_date = Column(DateTime)  # When prediction was made
    target_date = Column(DateTime)      # Date being predicted for
    horizon = Column(String)            # '1d', '1w', '1m', '6m'
    predicted_value = Column(Float)     # Predicted log return or price
    actual_value = Column(Float, nullable=True)  # Actual log return or price
    current_price = Column(Float)       # Price at time of prediction
    is_correct = Column(Boolean, nullable=True)  # Whether prediction was correct
    confidence = Column(Float, nullable=True)    # Uncertainty/confidence score

class InsiderTrade(Base):
    __tablename__ = "insider_trades"
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String)
    date = Column(DateTime)
    shares = Column(Float)
    amount = Column(Float)
    buy_flag = Column(Integer)  # 1 = buy, 0 = sell, -1 = no activity

class SentimentData(Base):
    __tablename__ = "sentiment_data"
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String)
    date = Column(DateTime)
    sentiment = Column(Float)
    num_articles = Column(Integer)

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

async def save_prediction(
    ticker: str,
    prediction_date: datetime,
    target_date: datetime,
    horizon: str,
    predicted_value: float,
    current_price: float,
    confidence: float = None
):
    """Save a prediction to the database"""
    async with AsyncSessionLocal() as session:
        prediction = Prediction(
            ticker=ticker,
            prediction_date=prediction_date,
            target_date=target_date,
            horizon=horizon,
            predicted_value=predicted_value,
            current_price=current_price,
            confidence=confidence
        )
        session.add(prediction)
        await session.commit()

async def update_prediction_actual(prediction_id: int, actual_value: float, is_correct: bool):
    """Update a prediction with actual outcome"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Prediction).where(Prediction.id == prediction_id)
        )
        prediction = result.scalar_one_or_none()
        if prediction:
            prediction.actual_value = actual_value
            prediction.is_correct = is_correct
            await session.commit()

async def get_predictions_for_ticker(ticker: str, limit: int = 50):
    """Get recent predictions for a ticker"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Prediction)
            .where(Prediction.ticker == ticker)
            .order_by(desc(Prediction.prediction_date))
            .limit(limit)
        )
        return result.scalars().all()

async def get_pending_predictions():
    """Get predictions that need to be validated (target_date has passed but no actual_value)"""
    async with AsyncSessionLocal() as session:
        now = datetime.now()
        result = await session.execute(
            select(Prediction)
            .where(Prediction.target_date <= now)
            .where(Prediction.actual_value.is_(None))
        )
        return result.scalars().all()

async def save_insider_trade(ticker: str, date: datetime, shares: float, amount: float, buy_flag: int):
    """Save insider trading data"""
    async with AsyncSessionLocal() as session:
        trade = InsiderTrade(
            ticker=ticker,
            date=date,
            shares=shares,
            amount=amount,
            buy_flag=buy_flag
        )
        session.add(trade)
        await session.commit()

async def get_insider_trades(ticker: str, start_date: datetime = None):
    """Get insider trades for a ticker"""
    async with AsyncSessionLocal() as session:
        query = select(InsiderTrade).where(InsiderTrade.ticker == ticker)
        if start_date:
            query = query.where(InsiderTrade.date >= start_date)
        query = query.order_by(InsiderTrade.date)
        result = await session.execute(query)
        return result.scalars().all()

async def save_sentiment_data(ticker: str, date: datetime, sentiment: float, num_articles: int):
    """Save sentiment data"""
    async with AsyncSessionLocal() as session:
        sentiment_entry = SentimentData(
            ticker=ticker,
            date=date,
            sentiment=sentiment,
            num_articles=num_articles
        )
        session.add(sentiment_entry)
        await session.commit()

async def get_sentiment_data(ticker: str, start_date: datetime = None):
    """Get sentiment data for a ticker"""
    async with AsyncSessionLocal() as session:
        query = select(SentimentData).where(SentimentData.ticker == ticker)
        if start_date:
            query = query.where(SentimentData.date >= start_date)
        query = query.order_by(SentimentData.date)
        result = await session.execute(query)
        return result.scalars().all()
