#!/usr/bin/env python3
"""
V9 Paper Trading Engine - "Golden Configuration" Strategy

Simulates the V9 Transformer Ranking strategy with a virtual $10,000 portfolio.
This is a PAPER TRADING system - NO REAL MONEY is involved.

Golden Configuration:
- ATR Multiplier: 3.0x (matches winning backtest Sharpe 1.10)
- Correlation Filter: < 0.60
- Top K: 10 stocks
- Rebalance: Every 5 trading days

Execution Flow (Linear):
1. Valuation & Stops: Fetch prices, check ATR stops, sell if triggered
2. Rebalance: If day >= 5, run V9 inference, apply correlation filter, rebalance
3. Persist: Save new state to database

Usage:
    docker exec proxmox_stock_backend python -m scripts.paper_trader_v9 --init
    docker exec proxmox_stock_backend python -m scripts.paper_trader_v9 --run-day
    docker exec proxmox_stock_backend python -m scripts.paper_trader_v9 --status
"""

import sys
import os
import logging
import pickle
import argparse
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import torch

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from sqlalchemy import text, select, delete, update
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, make_transient
from sqlalchemy import Column, String, Float, DateTime, Integer
from sqlalchemy.dialects.postgresql import insert
from backend.config.constants import DEFAULT_DATABASE_URL

def get_db_url_sync():
    """Convert async DB URL to sync for psycopg2."""
    return DEFAULT_DATABASE_URL.replace("+asyncpg", "")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("paper_trader_v9.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION - V9 "GOLDEN CONFIGURATION"
# =============================================================================

CONFIG = {
    "SESSION_ID": "v9_golden_2025",         # Unique identifier for this paper trading session
    "INITIAL_CAPITAL": 10_000.0,            # Starting cash ($10,000)
    "TOP_K": 10,                            # Number of stocks to hold
    "REBALANCE_DAYS": 5,                    # Rebalance every 5 trading days
    "MAX_CORRELATION": 0.60,                # Correlation filter threshold
    "ATR_MULTIPLIER": 3.0,                  # ATR stop multiplier (winning config)
    "ATR_PERIOD": 14,                       # ATR lookback period
    "WINDOW_SIZE": 60,                      # V9 model input window
    "HISTORY_DAYS": 30,                     # Days of history to fetch for ATR calculation
    "SLIPPAGE": 0.001,                      # 0.1% transaction cost per trade
}

# Paths
MODEL_PATH = backend_path / "models" / "transformer_v9_best.pth"
SCALER_PATH = backend_path / "models" / "scaler_v9.pkl"

# Database
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable must be set")



# =============================================================================
# DATABASE MODELS
# =============================================================================

from services.db import PaperPortfolio, PaperHolding, PaperTrade, AsyncSessionLocal


# =============================================================================
# DATABASE UTILITIES
# =============================================================================

def get_db_url_sync():
    """Get synchronous database URL for raw SQL queries."""
    url = os.getenv("DATABASE_URL", "")
    return url.replace("postgresql+asyncpg://", "postgresql://")


# =============================================================================
# DATABASE-BASED PRICE FETCHING (Primary - works inside Docker)
# =============================================================================

def fetch_current_prices_db(tickers: List[str]) -> Dict[str, float]:
    """
    Fetch current (most recent) prices from the database.
    
    This is the PRIMARY method - uses the existing stock_prices table.
    Works reliably inside Docker containers without external network access.
    
    Returns:
        Dict mapping ticker -> latest close price
    """
    import psycopg2
    
    prices = {}
    
    if not tickers:
        return prices
    
    try:
        with psycopg2.connect(get_db_url_sync()) as conn:
            with conn.cursor() as cur:
                # Get the most recent price for each ticker
                placeholders = ','.join(['%s'] * len(tickers))
                cur.execute(f"""
                    SELECT DISTINCT ON (ticker) ticker, close
                    FROM stock_prices
                    WHERE ticker IN ({placeholders})
                    ORDER BY ticker, timestamp DESC
                """, tickers)
                
                for row in cur.fetchall():
                    ticker, close_price = row
                    if close_price is not None:
                        prices[ticker] = float(close_price)
        
    except Exception as e:
        logger.error(f"Error fetching prices from DB: {e}")
    
    return prices


def load_stock_data_sync(ticker: str) -> Optional[pd.DataFrame]:
    """Load historical data for a ticker from the database."""
    import psycopg2
    
    try:
        with psycopg2.connect(get_db_url_sync()) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT timestamp, open, high, low, close, volume 
                    FROM stock_prices WHERE ticker = %s ORDER BY timestamp
                """, (ticker,))
                records = cur.fetchall()
        
        if records:
            df = pd.DataFrame(records, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df['ticker'] = ticker
            return df
    except Exception as e:
        logger.error(f"Error loading stock data for {ticker}: {e}")
        return None
        
    return None


def fetch_history_for_atr_db(tickers: List[str], days: int = 30) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for ATR calculation from the database.
    
    This is the PRIMARY method - uses the existing stock_prices table.
    
    Args:
        tickers: List of ticker symbols
        days: Number of days of history to fetch
        
    Returns:
        Dict mapping ticker -> DataFrame with OHLCV data
    """
    history = {}
    
    if not tickers:
        return history
    
    for ticker in tickers:
        df = load_stock_data_sync(ticker)
        if df is not None and len(df) >= days:
            # Get the last N days
            history[ticker] = df.tail(days).copy()
        elif df is not None:
            # Use whatever we have
            history[ticker] = df.copy()
    
    return history


# Alias for backward compatibility
def fetch_current_prices_yf(tickers: List[str]) -> Dict[str, float]:
    """Fallback to DB prices (yfinance doesn't work in Docker)."""
    return fetch_current_prices_db(tickers)


def fetch_history_for_atr(tickers: List[str], days: int = 30) -> Dict[str, pd.DataFrame]:
    """Fallback to DB history (yfinance doesn't work in Docker)."""
    return fetch_history_for_atr_db(tickers, days)


# =============================================================================
# ATR CALCULATION
# =============================================================================

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate current ATR value.
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR period (default 14)
        
    Returns:
        Current ATR value
    """
    if len(df) < period + 1:
        return 0.0
    
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    return float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else 0.0


def calculate_stop_loss(highest_price: float, atr: float, multiplier: float = 3.0) -> float:
    """
    Calculate ATR trailing stop loss level.
    
    Stop = Highest_Price - (ATR * Multiplier)
    """
    stop_distance = atr * multiplier
    stop_price = highest_price - stop_distance
    
    # Floor at 3% below highest (prevent too-tight stops in low-vol stocks)
    min_stop = highest_price * 0.97
    return min(stop_price, min_stop)


# =============================================================================
# V9 MODEL INFERENCE
# =============================================================================

def load_v9_model():
    """Load the V9 Transformer model and scaler."""
    from services.transformer_model import TransformerRankModel
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    model = TransformerRankModel.load(str(MODEL_PATH), device=device)
    model.eval()
    logger.info(f"✅ Loaded V9 model from {MODEL_PATH}")
    
    # Load scaler
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
    
    with open(SCALER_PATH, 'rb') as f:
        scaler_data = pickle.load(f)
    scaler = scaler_data['scaler']
    logger.info(f"✅ Loaded scaler from {SCALER_PATH}")
    
    return model, scaler, device


def get_universe_tickers() -> List[str]:
    """Fetch all tickers that meet the criteria for the trading universe."""
    import psycopg2
    
    try:
        with psycopg2.connect(get_db_url_sync()) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ticker, COUNT(*) as count 
                    FROM stock_prices 
                    WHERE ticker NOT IN ('^VIX', 'VIX', 'SPY', 'QQQ', 'DIA', 'IWM')
                    GROUP BY ticker 
                    HAVING COUNT(*) >= 200 
                    ORDER BY count DESC
                """)
                results = cur.fetchall()
                return [row[0] for row in results]
    except Exception as e:
        logger.error(f"Error fetching universe tickers: {e}")
        return []


def run_v9_inference(model, scaler, device) -> Dict[str, float]:
    """
    Run V9 model inference on all stocks in the universe.
    
    Returns:
        Dict mapping ticker -> predicted rank score (0.0 to 1.0)
    """
    from services.feature_engineering_v9 import (
        compute_v9_features,
        process_spy_data,
        process_vix_data,
        V9_FEATURE_NAMES,
    )
    
    logger.info("Running V9 inference on stock universe...")
    
    # Get tickers
    tickers = get_universe_tickers()
    logger.info(f"Found {len(tickers)} tradable stocks")
    
    # Load SPY and VIX for macro context
    spy_df = load_stock_data_sync('SPY')
    vix_df = load_stock_data_sync('^VIX')
    if vix_df is None:
        vix_df = load_stock_data_sync('VIX')
    
    spy_data = process_spy_data(spy_df) if spy_df is not None else None
    vix_data = process_vix_data(vix_df) if vix_df is not None else None
    
    # Process each stock
    rankings = {}
    windows = []
    valid_tickers = []
    
    for i, ticker in enumerate(tickers):
        if (i + 1) % 100 == 0:
            logger.info(f"Processing {i+1}/{len(tickers)}...")
        
        try:
            df = load_stock_data_sync(ticker)
            if df is None or len(df) < CONFIG["WINDOW_SIZE"] + 20:
                continue
            
            # Compute features
            df_feat = compute_v9_features(df, spy_data, vix_data)
            df_feat = df_feat.dropna(subset=V9_FEATURE_NAMES)
            
            if len(df_feat) < CONFIG["WINDOW_SIZE"]:
                continue
            
            # Get last window
            feats = df_feat[V9_FEATURE_NAMES].values[-CONFIG["WINDOW_SIZE"]:]
            feats = np.nan_to_num(feats, nan=0.0, posinf=1e6, neginf=-1e6)
            feats = scaler.transform(feats)
            feats = np.clip(feats, -10, 10).astype(np.float32)
            
            windows.append(feats)
            valid_tickers.append(ticker)
            
        except Exception as e:
            logger.debug(f"Error processing {ticker}: {e}")
            continue
    
    if not windows:
        logger.warning("No valid stocks for inference!")
        return {}
    
    # Batch inference
    X = torch.from_numpy(np.stack(windows)).float().to(device)
    
    model.eval()
    with torch.no_grad():
        preds = model(X).cpu().numpy().flatten()
    
    rankings = {ticker: float(pred) for ticker, pred in zip(valid_tickers, preds)}
    logger.info(f"Generated rankings for {len(rankings)} stocks")
    
    return rankings


def apply_correlation_filter(
    rankings: Dict[str, float],
    max_corr: float = 0.60,
    top_k: int = 10,
    lookback: int = 60
) -> List[str]:
    """
    Apply correlation filter to select diversified Top K stocks.
    
    Args:
        rankings: Dict of ticker -> rank score
        max_corr: Maximum pairwise correlation allowed
        top_k: Target number of holdings
        lookback: Days for correlation calculation
        
    Returns:
        List of selected tickers
    """
    logger.info(f"Applying correlation filter (max_corr={max_corr})...")
    
    # Sort by rank (descending)
    sorted_candidates = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
    
    # Build returns matrix
    returns_dict = {}
    for ticker, rank in sorted_candidates[:top_k * 3]:
        df = load_stock_data_sync(ticker)
        if df is None:
            continue
        
        recent = df.tail(lookback)
        if len(recent) >= lookback // 2:
            ret = recent['close'].pct_change().dropna()
            if len(ret) > 10:
                returns_dict[ticker] = ret.values
    
    if not returns_dict:
        return [t for t, _ in sorted_candidates[:top_k]]
    
    # Create correlation matrix
    min_len = min(len(v) for v in returns_dict.values())
    aligned_returns = {k: v[-min_len:] for k, v in returns_dict.items()}
    returns_df = pd.DataFrame(aligned_returns)
    corr_matrix = returns_df.corr()
    
    # Greedy selection
    selected = []
    for ticker, rank in sorted_candidates:
        if ticker not in corr_matrix.columns:
            continue
        
        if selected:
            # Build list of valid correlations to avoid empty max() error
            pair_corrs = [
                corr_matrix.loc[ticker, s]
                for s in selected
                if s in corr_matrix.columns
            ]
            if pair_corrs:
                max_pair_corr = max(pair_corrs)
                if max_pair_corr > max_corr:
                    continue
            else:
                # No overlap in columns, proceed cautiously
                pass
        
        selected.append(ticker)
        if len(selected) >= top_k:
            break
    
    # Fill if needed
    if len(selected) < top_k:
        for ticker, _ in sorted_candidates:
            if ticker not in selected:
                selected.append(ticker)
                if len(selected) >= top_k:
                    break
    
    logger.info(f"Selected {len(selected)} stocks after correlation filter")
    return selected


# =============================================================================
# PORTFOLIO OPERATIONS
# =============================================================================

async def initialize_session():
    """Initialize a new paper trading session with $10,000."""
    async with AsyncSessionLocal() as session:
        # Check if session exists
        result = await session.execute(
            select(PaperPortfolio).where(PaperPortfolio.session_id == CONFIG["SESSION_ID"])
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            logger.warning(f"Session '{CONFIG['SESSION_ID']}' already exists!")
            logger.info(f"  Cash: ${existing.cash_balance:,.2f}")
            logger.info(f"  Equity: ${existing.equity_value:,.2f}")
            logger.info(f"  Total: ${existing.total_value:,.2f}")
            logger.info("\nTo reset, delete the session from the database first.")
            return
        
        # Create new portfolio
        portfolio = PaperPortfolio(
            session_id=CONFIG["SESSION_ID"],
            cash_balance=CONFIG["INITIAL_CAPITAL"],
            equity_value=0.0,
            total_value=CONFIG["INITIAL_CAPITAL"],
            days_since_rebalance=CONFIG["REBALANCE_DAYS"],  # Trigger rebalance on first run
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        session.add(portfolio)
        await session.commit()
        
        logger.info("=" * 60)
        logger.info("📊 V9 PAPER TRADING SESSION INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"Session ID:      {CONFIG['SESSION_ID']}")
        logger.info(f"Initial Capital: ${CONFIG['INITIAL_CAPITAL']:,.2f}")
        logger.info(f"Strategy:        Top {CONFIG['TOP_K']} by V9 Rank")
        logger.info(f"Rebalance:       Every {CONFIG['REBALANCE_DAYS']} days")
        logger.info(f"ATR Multiplier:  {CONFIG['ATR_MULTIPLIER']}x")
        logger.info(f"Correlation:     < {CONFIG['MAX_CORRELATION']}")
        logger.info("=" * 60)


async def get_portfolio_status():
    """Display current portfolio status."""
    async with AsyncSessionLocal() as session:
        # Get portfolio
        result = await session.execute(
            select(PaperPortfolio).where(PaperPortfolio.session_id == CONFIG["SESSION_ID"])
        )
        portfolio = result.scalar_one_or_none()
        
        if not portfolio:
            logger.error(f"Session '{CONFIG['SESSION_ID']}' not found. Run --init first.")
            return
        
        # Get holdings
        result = await session.execute(
            select(PaperHolding).where(PaperHolding.session_id == CONFIG["SESSION_ID"])
        )
        holdings = result.scalars().all()
        
        # Get recent trades
        result = await session.execute(
            select(PaperTrade)
            .where(PaperTrade.session_id == CONFIG["SESSION_ID"])
            .order_by(PaperTrade.trade_date.desc())
            .limit(10)
        )
        recent_trades = result.scalars().all()
        
        # Calculate P/L
        initial = CONFIG["INITIAL_CAPITAL"]
        total = portfolio.total_value
        pnl = total - initial
        pnl_pct = (pnl / initial) * 100
        
        logger.info("=" * 60)
        logger.info("📊 V9 PAPER TRADING STATUS")
        logger.info("=" * 60)
        logger.info(f"Session:         {CONFIG['SESSION_ID']}")
        logger.info(f"Started:         {portfolio.created_at.strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"Last Updated:    {portfolio.updated_at.strftime('%Y-%m-%d %H:%M')}")
        logger.info("")
        logger.info(f"Portfolio Value: ${total:,.2f} ({pnl_pct:+.2f}%)")
        logger.info(f"Cash Balance:    ${portfolio.cash_balance:,.2f}")
        logger.info(f"Equity Value:    ${portfolio.equity_value:,.2f}")
        logger.info(f"P/L:             ${pnl:+,.2f}")
        logger.info(f"Days to Rebal:   {CONFIG['REBALANCE_DAYS'] - portfolio.days_since_rebalance}")
        
        if holdings:
            logger.info("")
            logger.info(f"📈 HOLDINGS ({len(holdings)} positions):")
            for h in holdings:
                h_pnl = (h.current_price - h.entry_price) / h.entry_price * 100
                h_value = h.quantity * h.current_price
                stop_pct = (h.stop_loss_level / h.highest_price - 1) * 100
                logger.info(f"  {h.ticker:6s} | {h.quantity:6.2f} @ ${h.entry_price:7.2f} | "
                           f"Now ${h.current_price:7.2f} ({h_pnl:+5.1f}%) | "
                           f"Stop ${h.stop_loss_level:7.2f}")
        
        if recent_trades:
            logger.info("")
            logger.info("📋 RECENT TRADES:")
            for t in recent_trades[:5]:
                pnl_str = f"${t.profit_loss:+.2f}" if t.profit_loss else ""
                logger.info(f"  {t.trade_date.strftime('%Y-%m-%d')} | {t.action:4s} | "
                           f"{t.ticker:6s} @ ${t.price:.2f} | {t.reason} {pnl_str}")
        
        logger.info("=" * 60)


async def run_daily_cycle():
    """
    Execute one daily cycle of the paper trading engine.
    
    LINEAR FLOW:
    1. Valuation & Stops: Fetch prices, check ATR stops, SELL if triggered
    2. Rebalance: If day >= 5, run V9, apply correlation filter, rebalance
    3. Persist: Save new state
    """
    logger.info("=" * 60)
    logger.info(f"📊 V9 PAPER TRADING - DAILY CYCLE")
    logger.info(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 60)
    
    async with AsyncSessionLocal() as session:
        # Load portfolio
        result = await session.execute(
            select(PaperPortfolio).where(PaperPortfolio.session_id == CONFIG["SESSION_ID"])
        )
        portfolio = result.scalar_one_or_none()
        
        if not portfolio:
            logger.error(f"Session '{CONFIG['SESSION_ID']}' not found. Run --init first.")
            return
        
        # Load holdings
        result = await session.execute(
            select(PaperHolding).where(PaperHolding.session_id == CONFIG["SESSION_ID"])
        )
        holdings = {h.ticker: h for h in result.scalars().all()}
        
        logger.info(f"Starting state: ${portfolio.total_value:,.2f} ({len(holdings)} positions)")
        
        trades_to_log = []  # Collect trades for persistence
        
        # =====================================================================
        # STEP A: VALUATION & ATR STOP CHECKS
        # =====================================================================
        logger.info("\n📍 STEP A: Valuation & Stop Checks")
        
        if holdings:
            # Fetch current prices
            tickers = list(holdings.keys())
            current_prices = fetch_current_prices_yf(tickers)
            logger.info(f"Fetched prices for {len(current_prices)}/{len(tickers)} holdings")
            
            # Fetch history for ATR calculation
            history_data = fetch_history_for_atr(tickers, days=CONFIG["HISTORY_DAYS"])
            
            # Check each holding
            stops_triggered = []
            
            for ticker, holding in holdings.items():
                if ticker not in current_prices:
                    logger.warning(f"No price data for {ticker}, skipping")
                    continue
                
                current_price = current_prices[ticker]
                
                # Update highest price tracking
                new_highest = max(holding.highest_price, current_price)
                holding.highest_price = new_highest
                holding.current_price = current_price
                
                # Calculate ATR and stop loss
                if ticker in history_data:
                    atr = calculate_atr(history_data[ticker], CONFIG["ATR_PERIOD"])
                else:
                    atr = current_price * 0.04  # Fallback: 4% of price
                
                stop_loss = calculate_stop_loss(new_highest, atr, CONFIG["ATR_MULTIPLIER"])
                holding.stop_loss_level = stop_loss
                
                # Check if stop triggered
                if current_price <= stop_loss:
                    stops_triggered.append((ticker, holding, current_price))
                    logger.info(f"  🛑 ATR STOP: {ticker} @ ${current_price:.2f} <= ${stop_loss:.2f}")
            
            # Execute stop sells
            for ticker, holding, price in stops_triggered:
                proceeds = holding.quantity * price * (1 - CONFIG["SLIPPAGE"])
                pnl = (price - holding.entry_price) * holding.quantity
                
                portfolio.cash_balance += proceeds
                
                trades_to_log.append(PaperTrade(
                    session_id=CONFIG["SESSION_ID"],
                    trade_date=datetime.now(),
                    action="SELL",
                    ticker=ticker,
                    price=price,
                    quantity=holding.quantity,
                    reason="ATR_EXIT",
                    profit_loss=pnl
                ))
                
                # Remove from holdings dict (will delete from DB later)
                del holdings[ticker]
                logger.info(f"  Sold {ticker}: ${proceeds:.2f} proceeds, P/L ${pnl:+.2f}")
            
            logger.info(f"Stop checks complete. {len(stops_triggered)} stops triggered.")
        else:
            logger.info("No holdings to check.")
        
        # =====================================================================
        # STEP B: REBALANCE (Every 5 Days)
        # =====================================================================
        portfolio.days_since_rebalance += 1
        
        if portfolio.days_since_rebalance >= CONFIG["REBALANCE_DAYS"]:
            logger.info(f"\n📍 STEP B: Rebalancing (Day {portfolio.days_since_rebalance})")
            
            # Load V9 model
            model, scaler, device = load_v9_model()
            
            # Run inference
            rankings = run_v9_inference(model, scaler, device)
            
            if rankings:
                # Apply correlation filter
                new_top_k = apply_correlation_filter(
                    rankings,
                    max_corr=CONFIG["MAX_CORRELATION"],
                    top_k=CONFIG["TOP_K"]
                )
                
                logger.info(f"New Top {CONFIG['TOP_K']}: {new_top_k}")
                
                # Determine sells (not in new top K)
                current_tickers = set(holdings.keys())
                new_tickers = set(new_top_k)
                
                to_sell = current_tickers - new_tickers
                to_buy = new_tickers - current_tickers
                
                logger.info(f"Rebalance: Selling {len(to_sell)}, Buying {len(to_buy)}")
                
                # Execute sells first (free up cash)
                for ticker in to_sell:
                    if ticker not in holdings:
                        continue
                    
                    holding = holdings[ticker]
                    price = holding.current_price
                    proceeds = holding.quantity * price * (1 - CONFIG["SLIPPAGE"])
                    pnl = (price - holding.entry_price) * holding.quantity
                    
                    portfolio.cash_balance += proceeds
                    
                    trades_to_log.append(PaperTrade(
                        session_id=CONFIG["SESSION_ID"],
                        trade_date=datetime.now(),
                        action="SELL",
                        ticker=ticker,
                        price=price,
                        quantity=holding.quantity,
                        reason="REBALANCE_EXIT",
                        profit_loss=pnl
                    ))
                    
                    del holdings[ticker]
                    logger.info(f"  Sold {ticker} (rebalance): ${proceeds:.2f}")
                
                # Calculate cash available for buys
                if to_buy:
                    # Get prices for new buys
                    new_prices = fetch_current_prices_yf(list(to_buy))
                    
                    # Fetch history for ATR on new positions
                    new_history = fetch_history_for_atr(list(to_buy), days=CONFIG["HISTORY_DAYS"])
                    
                    # Equal weight allocation
                    valid_buys = [t for t in to_buy if t in new_prices]
                    if valid_buys:
                        cash_per_stock = portfolio.cash_balance / len(valid_buys)
                        
                        for ticker in valid_buys:
                            price = new_prices[ticker]
                            quantity = (cash_per_stock * (1 - CONFIG["SLIPPAGE"])) / price
                            cost = quantity * price
                            
                            # Calculate initial stop loss
                            if ticker in new_history:
                                atr = calculate_atr(new_history[ticker], CONFIG["ATR_PERIOD"])
                            else:
                                atr = price * 0.04  # Fallback
                            
                            stop_loss = calculate_stop_loss(price, atr, CONFIG["ATR_MULTIPLIER"])
                            
                            portfolio.cash_balance -= cost
                            
                            # Create new holding
                            holdings[ticker] = PaperHolding(
                                session_id=CONFIG["SESSION_ID"],
                                ticker=ticker,
                                entry_price=price,
                                quantity=quantity,
                                current_price=price,
                                stop_loss_level=stop_loss,
                                highest_price=price,
                                entry_date=datetime.now(),
                                updated_at=datetime.now()
                            )
                            
                            trades_to_log.append(PaperTrade(
                                session_id=CONFIG["SESSION_ID"],
                                trade_date=datetime.now(),
                                action="BUY",
                                ticker=ticker,
                                price=price,
                                quantity=quantity,
                                reason="RANK_ENTRY",
                                profit_loss=None
                            ))
                            
                            logger.info(f"  Bought {ticker}: {quantity:.2f} @ ${price:.2f}")
                
                portfolio.days_since_rebalance = 0
                logger.info("Rebalance complete.")
        else:
            logger.info(f"\n📍 STEP B: Skipping rebalance (Day {portfolio.days_since_rebalance}/{CONFIG['REBALANCE_DAYS']})")
        
        # =====================================================================
        # STEP C: PERSIST
        # =====================================================================
        logger.info("\n📍 STEP C: Persisting state")
        
        # Calculate equity value
        equity_value = sum(h.quantity * h.current_price for h in holdings.values())
        portfolio.equity_value = equity_value
        portfolio.total_value = portfolio.cash_balance + equity_value
        portfolio.updated_at = datetime.now()
        
        # Clear old holdings and save new ones
        await session.execute(
            delete(PaperHolding).where(PaperHolding.session_id == CONFIG["SESSION_ID"])
        )
        
        for holding in holdings.values():
            make_transient(holding)
            holding.id = None
            session.add(holding)
        
        # Log trades
        for trade in trades_to_log:
            session.add(trade)
        
        await session.commit()
        
        # =====================================================================
        # DAILY SUMMARY
        # =====================================================================
        initial = CONFIG["INITIAL_CAPITAL"]
        pnl = portfolio.total_value - initial
        pnl_pct = (pnl / initial) * 100
        
        logger.info("")
        logger.info("═" * 60)
        logger.info(f"📊 V9 PAPER TRADING SUMMARY - {datetime.now().strftime('%Y-%m-%d')}")
        logger.info("═" * 60)
        logger.info(f"Portfolio Value: ${portfolio.total_value:,.2f} ({pnl_pct:+.2f}%)")
        logger.info(f"Cash Balance:    ${portfolio.cash_balance:,.2f}")
        logger.info(f"Equity Value:    ${portfolio.equity_value:,.2f}")
        logger.info(f"Holdings:        {len(holdings)} stocks")
        logger.info(f"Trades Today:    {len(trades_to_log)}")
        logger.info("═" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="V9 Paper Trading Engine")
    parser.add_argument("--init", action="store_true", help="Initialize new trading session")
    parser.add_argument("--run-day", action="store_true", help="Run one daily trading cycle")
    parser.add_argument("--status", action="store_true", help="Show current portfolio status")
    parser.add_argument("--session", type=str, default=None, help="Custom session ID")
    args = parser.parse_args()
    
    if args.session:
        CONFIG["SESSION_ID"] = args.session
    
    if args.init:
        asyncio.run(initialize_session())
    elif args.run_day:
        asyncio.run(run_daily_cycle())
    elif args.status:
        asyncio.run(get_portfolio_status())
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python -m scripts.paper_trader_v9 --init      # Start new session")
        print("  python -m scripts.paper_trader_v9 --run-day   # Run daily cycle")
        print("  python -m scripts.paper_trader_v9 --status    # Check status")


if __name__ == "__main__":
    main()
