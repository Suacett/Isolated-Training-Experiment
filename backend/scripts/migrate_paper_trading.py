#!/usr/bin/env python3
"""
Migration script to create paper trading tables.

This script creates the following tables:
1. paper_portfolio - Portfolio-level state (cash, equity, total value)
2. paper_holdings - Individual positions with ATR stop tracking
3. paper_trades - Trade log with action/reason

Usage:
    docker exec proxmox_stock_backend python -m scripts.migrate_paper_trading
"""

import sys
import asyncio
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Float, DateTime, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:password@timescaledb:5432/stock_db")
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


# =============================================================================
# PAPER TRADING MODELS
# =============================================================================

class PaperPortfolio(Base):
    """
    Tracks portfolio-level state for a paper trading session.
    """
    __tablename__ = "paper_portfolio"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, unique=True, nullable=False)  # e.g., "v9_golden_2025"
    cash_balance = Column(Float, default=10000.0)
    equity_value = Column(Float, default=0.0)
    total_value = Column(Float, default=10000.0)
    days_since_rebalance = Column(Integer, default=5)  # Start at 5 to trigger rebalance on first run
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class PaperHolding(Base):
    """
    Tracks individual positions in the paper portfolio.
    """
    __tablename__ = "paper_holdings"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)  # Links to PaperPortfolio
    ticker = Column(String, nullable=False)
    entry_price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    stop_loss_level = Column(Float, nullable=False)  # ATR trailing stop
    highest_price = Column(Float, nullable=False)    # For trailing stop calculation
    entry_date = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class PaperTrade(Base):
    """
    Logs all paper trades with action and reason.
    """
    __tablename__ = "paper_trades"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)
    trade_date = Column(DateTime, nullable=False)
    action = Column(String, nullable=False)  # 'BUY' or 'SELL'
    ticker = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    reason = Column(String, nullable=False)  # 'RANK_ENTRY', 'ATR_EXIT', 'REBALANCE_EXIT'
    profit_loss = Column(Float, nullable=True)  # P/L for sells, NULL for buys


# =============================================================================
# MIGRATION LOGIC
# =============================================================================

async def migrate_paper_trading_tables():
    """Create paper trading tables if they don't exist."""
    
    logger.info("=" * 60)
    logger.info("📊 PAPER TRADING TABLE MIGRATION")
    logger.info("=" * 60)
    
    async with AsyncSessionLocal() as session:
        # Check if tables already exist
        tables_to_create = ['paper_portfolio', 'paper_holdings', 'paper_trades']
        existing_tables = []
        
        for table_name in tables_to_create:
            try:
                result = await session.execute(
                    text(f"SELECT 1 FROM information_schema.tables WHERE table_name = '{table_name}'")
                )
                if result.scalar():
                    existing_tables.append(table_name)
            except Exception:
                pass
        
        if existing_tables:
            logger.info(f"Existing tables found: {existing_tables}")
            logger.info("These will be preserved. No data will be lost.")
        
        # Create tables using SQLAlchemy metadata
        logger.info("Creating paper trading tables...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("✅ Tables created/verified successfully!")
        
        # Create indexes for better query performance
        logger.info("Creating indexes...")
        index_statements = [
            "CREATE INDEX IF NOT EXISTS idx_paper_portfolio_session ON paper_portfolio(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_paper_holdings_session ON paper_holdings(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_paper_holdings_ticker ON paper_holdings(ticker)",
            "CREATE INDEX IF NOT EXISTS idx_paper_trades_session ON paper_trades(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_paper_trades_date ON paper_trades(trade_date)",
        ]
        
        for stmt in index_statements:
            try:
                await session.execute(text(stmt))
            except Exception as e:
                # Index may already exist
                pass
        
        await session.commit()
        logger.info("✅ Indexes created/verified!")
        
        # Verify schema
        logger.info("\n📋 Paper Trading Schema:")
        for table_name in tables_to_create:
            result = await session.execute(
                text(f"""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = '{table_name}'
                    ORDER BY ordinal_position
                """)
            )
            columns = result.fetchall()
            logger.info(f"\n  {table_name}:")
            for col_name, col_type in columns:
                logger.info(f"    - {col_name}: {col_type}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ MIGRATION COMPLETE")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("  1. Initialize session: python -m scripts.paper_trader_v9 --init")
        logger.info("  2. Run daily cycle:    python -m scripts.paper_trader_v9 --run-day")
        logger.info("  3. Check status:       python -m scripts.paper_trader_v9 --status")


if __name__ == "__main__":
    asyncio.run(migrate_paper_trading_tables())
