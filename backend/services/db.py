import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Float, DateTime, select, desc, Integer, delete, Boolean
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime
from config.settings import DATABASE_URL, PAPER_SESSION_ID
from config.constants import INITIAL_PORTFOLIO_CASH, DEFAULT_REBALANCE_DAYS

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
    is_favorite = Column(Boolean, default=False)  # Favorites get Alpha Vantage calls
    added_at = Column(DateTime, default=datetime.now)

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

class CachedIntrinsicValue(Base):
    __tablename__ = "cached_intrinsic_values"
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, unique=True)
    value = Column(Float)
    last_updated = Column(DateTime)
    eps = Column(Float)
    growth_rate = Column(Float)
    bond_yield = Column(Float)

class PaperPortfolio(Base):
    __tablename__ = "paper_portfolio"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, unique=True, nullable=False)
    cash_balance = Column(Float, default=INITIAL_PORTFOLIO_CASH)
    equity_value = Column(Float, default=0.0)
    total_value = Column(Float, default=INITIAL_PORTFOLIO_CASH)
    days_since_rebalance = Column(Integer, default=DEFAULT_REBALANCE_DAYS)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class PaperPortfolioHistory(Base):
    __tablename__ = "paper_portfolio_history"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    total_value = Column(Float, nullable=False)
    equity_value = Column(Float, nullable=False)
    cash_balance = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.now)

class PaperHolding(Base):
    __tablename__ = "paper_holdings"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)
    ticker = Column(String, nullable=False)
    entry_price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    stop_loss_level = Column(Float, nullable=False)
    highest_price = Column(Float, nullable=False)
    entry_date = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class PaperTrade(Base):
    __tablename__ = "paper_trades"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)
    trade_date = Column(DateTime, nullable=False)
    action = Column(String, nullable=False)
    ticker = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    reason = Column(String, nullable=False)
    profit_loss = Column(Float, nullable=True)

async def get_latest_close(ticker: str):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(StockPrice.close)
            .where(StockPrice.ticker == ticker)
            .order_by(desc(StockPrice.timestamp))
            .limit(1)
        )
        return result.scalar_one_or_none()

async def get_latest_date(ticker: str):
    """Get the most recent data date for a ticker"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(StockPrice.timestamp)
            .where(StockPrice.ticker == ticker)
            .order_by(desc(StockPrice.timestamp))
            .limit(1)
        )
        return result.scalar_one_or_none()

async def get_cached_intrinsic_value(ticker: str):
    """Get cached intrinsic value for a ticker"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(CachedIntrinsicValue).where(CachedIntrinsicValue.ticker == ticker)
        )
        return result.scalar_one_or_none()

async def get_all_cached_intrinsic_values():
    """Get all cached intrinsic values in a single query for dashboard efficiency"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(CachedIntrinsicValue))
        rows = result.scalars().all()
        # Return as dict for O(1) lookup
        return {row.ticker: row.value for row in rows}

async def save_cached_intrinsic_value(ticker: str, value: float, eps: float, growth_rate: float, bond_yield: float):
    """Save or update cached intrinsic value"""
    async with AsyncSessionLocal() as session:
        stmt = insert(CachedIntrinsicValue).values(
            ticker=ticker,
            value=value,
            last_updated=datetime.now(),
            eps=eps,
            growth_rate=growth_rate,
            bond_yield=bond_yield
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[CachedIntrinsicValue.ticker],
            set_={
                "value": stmt.excluded.value,
                "last_updated": stmt.excluded.last_updated,
                "eps": stmt.excluded.eps,
                "growth_rate": stmt.excluded.growth_rate,
                "bond_yield": stmt.excluded.bond_yield
            }
        )
        await session.execute(stmt)
        await session.commit()

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

async def get_historical_data_range(ticker: str, start_date: datetime, end_date: datetime):
    """Get historical data for a ticker within a date range"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(StockPrice)
            .where(StockPrice.ticker == ticker)
            .where(StockPrice.timestamp >= start_date)
            .where(StockPrice.timestamp <= end_date)
            .order_by(StockPrice.timestamp)
        )
        return result.scalars().all()

async def add_watchlist_item(ticker: str, is_favorite: bool = False):
    async with AsyncSessionLocal() as session:
        stmt = insert(Watchlist).values(
            ticker=ticker,
            is_favorite=is_favorite,
            added_at=datetime.now()
        ).on_conflict_do_nothing()
        await session.execute(stmt)
        await session.commit()

async def remove_watchlist_item(ticker: str):
    async with AsyncSessionLocal() as session:
        await session.execute(delete(Watchlist).where(Watchlist.ticker == ticker))
        await session.commit()

async def get_watchlist():
    """Get all watchlist tickers (both favorites and non-favorites)"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Watchlist.ticker))
        return result.scalars().all()

async def get_favorites():
    """Get only favorite tickers (these get Alpha Vantage calls)"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Watchlist.ticker).where(Watchlist.is_favorite == True)
        )
        return result.scalars().all()

async def set_favorite(ticker: str, is_favorite: bool):
    """Toggle favorite status for a ticker"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Watchlist).where(Watchlist.ticker == ticker)
        )
        item = result.scalar_one_or_none()
        if item:
            item.is_favorite = is_favorite
            await session.commit()
            return True
        return False

async def get_watchlist_with_favorites():
    """Get all watchlist items with their favorite status"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Watchlist))
        items = result.scalars().all()
        return [{"ticker": item.ticker, "is_favorite": item.is_favorite} for item in items]

async def get_watchlist_item(ticker: str):
    """Get a single watchlist item by ticker"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Watchlist).where(Watchlist.ticker == ticker)
        )
        return result.scalar_one_or_none()

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

async def get_predictions_for_ticker(ticker: str, limit: int = 50, horizon: str = None):
    """Get recent predictions for a ticker, optionally filtered by horizon"""
    async with AsyncSessionLocal() as session:
        query = select(Prediction).where(Prediction.ticker == ticker)

        # Filter by horizon if specified
        if horizon:
            query = query.where(Prediction.horizon == horizon)

        query = query.order_by(desc(Prediction.prediction_date)).limit(limit)

        result = await session.execute(query)
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


async def delete_stock_data(ticker: str):
    """
    Delete all data associated with a ticker:
    - Stock Prices
    - Predictions
    - Insider Trades
    - Sentiment Data
    """
    async with AsyncSessionLocal() as session:
        # Delete from all tables
        await session.execute(delete(StockPrice).where(StockPrice.ticker == ticker))
        await session.execute(delete(Prediction).where(Prediction.ticker == ticker))
        await session.execute(delete(InsiderTrade).where(InsiderTrade.ticker == ticker))
        await session.execute(delete(SentimentData).where(SentimentData.ticker == ticker))
        await session.commit()

async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

