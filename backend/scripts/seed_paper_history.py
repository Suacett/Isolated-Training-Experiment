#!/usr/bin/env python3
"""
Seed Paper History - V9 Hedge Fund Simulator "Time Machine"

This script simulates 1-5 years of V9 trading history to populate the database
with realistic paper trading data for the dashboard.

Features:
- Loads actual V9 Transformer model for real inference
- Runs daily V9 rankings with proper feature engineering
- Applies "Golden Config" filters: Top 10 stocks + Correlation < 0.60
- Generates descriptive "Why" reasons for each trade
- Populates: paper_portfolio_history, paper_trades, paper_holdings, paper_portfolio

Usage:
    # 1 year backtest (default, ~252 trading days)
    docker exec proxmox_stock_backend python -m scripts.seed_paper_history

    # 5 year backtest (~1260 trading days)
    docker exec proxmox_stock_backend python -m scripts.seed_paper_history --years=5

    # Custom session ID for comparing multiple backtests
    docker exec proxmox_stock_backend python -m scripts.seed_paper_history --years=3 --session-id=v9_3year_test

    # Force re-seed (overwrites existing data)
    docker exec proxmox_stock_backend python -m scripts.seed_paper_history --force

    # Or locally:
    # export DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/stock_db
    python backend/scripts/seed_paper_history.py --years=2
"""

import asyncio
import argparse
import logging
import sys
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np
import torch
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

# Setup backend path
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# Import models and services
from services.db import Base, PaperPortfolio, PaperHolding, PaperTrade, PaperPortfolioHistory, init_db
from services.feature_engineering_v9 import (
    compute_v9_features,
    process_spy_data,
    process_vix_data,
    V9_FEATURE_NAMES,
    N_FEATURES,
)
from services.transformer_model import TransformerRankModel, get_device
from sqlalchemy import select, delete

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(BACKEND_DIR / "seed_paper_history.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "START_CAPITAL": 10000.0,
    "SESSION_ID": "v9_golden_2025",
    "SIMULATION_DAYS": 1825,  # 5 years = 365 days/year * 5 (Calendar days for timedelta)
    "TOP_K": 10,
    "REBALANCE_DAYS": 5,  # Weekly rebalance
    "WINDOW_SIZE": 60,     # V9 model lookback window
    "CORRELATION_THRESHOLD": 0.75,
    "VIX_THRESHOLD": 45.0,
}

# S&P 500 Selection (Diversified across sectors)
TICKERS = [
    # Tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "AVGO", "ADBE",
    "CRM", "NFLX", "INTC", "QCOM", "TXN", "PYPL", "INTU", "ORCL", "CSCO", "IBM",
    # Finance
    "JPM", "BAC", "V", "MA", "GS", "MS", "WFC", "BLK", "SCHW", "AXP",
    # Consumer
    "WMT", "COST", "TGT", "HD", "LOW", "MCD", "SBUX", "NKE", "DIS", "PG",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "DHR", "BMY",
    # Energy/Industrial
    "XOM", "CVX", "COP", "NEE", "SO", "CAT", "DE", "UNP", "HON", "RTX",
]


def fetch_sp500_tickers() -> list:
    """
    Fetch full S&P 500 ticker list from Wikipedia.
    Returns ~500 tickers for realistic backtesting.
    """
    import requests
    from bs4 import BeautifulSoup
    
    logger.info("Fetching S&P 500 ticker list from Wikipedia...")
    
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        table = soup.find('table', {'class': 'wikitable', 'id': 'constituents'})
        if not table:
            table = soup.find('table', {'class': 'wikitable'})
        
        tickers = []
        for row in table.find_all('tr')[1:]:  # Skip header
            cells = row.find_all('td')
            if cells:
                ticker = cells[0].text.strip()
                # Clean ticker (remove special chars, convert BRK.B -> BRK-B)
                ticker = ticker.replace('.', '-')
                tickers.append(ticker)
        
        logger.info(f"Fetched {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        logger.warning(f"Failed to fetch S&P 500 list: {e}. Using default 60 tickers.")
        return TICKERS


# Model paths
MODEL_PATH = BACKEND_DIR / "models" / "transformer_v9_best.pth"
SCALER_PATH = BACKEND_DIR / "models" / "scaler_v9.pkl"

# Database Connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:password@localhost:5432/stock_db")


# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

# Singleton engine and sessionmaker to avoid leaks
_engine = create_async_engine(DATABASE_URL, echo=False)
_AsyncSessionLocal = sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)

async def get_session():
    return _AsyncSessionLocal()


# =============================================================================
# DATA LOADING
# =============================================================================

def fetch_yfinance_data(tickers: list, start_date: datetime, end_date: datetime) -> dict:
    """
    Fetch historical data from Yahoo Finance using BULK download.

    Downloads ALL tickers in a single yfinance request for maximum speed.
    """
    import time
    start_time = time.time()

    logger.info(f"🚀 BULK DOWNLOADING {len(tickers)} tickers from {start_date.date()} to {end_date.date()}...")
    logger.info("Using vectorized download (1 request for all tickers)...")

    hist_data = {}

    try:
        # BULK DOWNLOAD - Single yfinance call for ALL tickers
        raw_data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            group_by='ticker',  # Critical for multi-ticker parsing
            progress=True,
            auto_adjust=True,
            threads=True  # Enable parallel processing
        )

        # Parse into per-ticker DataFrames
        for ticker in tickers:
            try:
                # Extract ticker's DataFrame
                if len(tickers) > 1:
                    df = raw_data[ticker].copy()
                else:
                    df = raw_data.copy()

                # Clean and filter
                df = df.dropna(how='all')  # Remove completely empty rows

                if len(df) < 60:  # Need at least WINDOW_SIZE days
                    logger.debug(f"Skipping {ticker}: Only {len(df)} days (need ≥60)")
                    continue

                # Standardize column names
                df.columns = [c.lower().replace(' ', '_') for c in df.columns]
                if 'adj_close' not in df.columns and 'close' in df.columns:
                    df['adj_close'] = df['close']

                df['date'] = df.index
                df = df.reset_index(drop=True)
                df['ticker'] = ticker
                hist_data[ticker] = df

            except Exception as e:
                logger.debug(f"Failed to parse {ticker}: {e}")
                continue

    except Exception as e:
        logger.error(f"Bulk download failed: {e}")
        logger.error("Falling back to per-ticker download...")

        # Fallback: Individual downloads (old method)
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                if not df.empty and len(df) >= 60:
                    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
                    if 'adj_close' not in df.columns and 'close' in df.columns:
                        df['adj_close'] = df['close']
                    df['date'] = df.index
                    df = df.reset_index(drop=True)
                    df['ticker'] = ticker
                    hist_data[ticker] = df
            except Exception as e2:
                logger.debug(f"Failed {ticker}: {e2}")
                continue

    total_time = time.time() - start_time
    logger.info(f"✅ Successfully fetched {len(hist_data)}/{len(tickers)} tickers in {total_time:.1f}s ({len(tickers)/total_time:.1f} tickers/sec)")
    return hist_data


def fetch_macro_data(start_date: datetime, end_date: datetime):
    """Fetch SPY and VIX data for macro features."""
    logger.info("Fetching SPY and VIX data...")
    
    spy_df = None
    vix_df = None
    
    try:
        spy = yf.download("SPY", start=start_date, end=end_date, progress=False, multi_level_index=False)
        if not spy.empty:
            spy = spy.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
            spy['date'] = spy.index
            spy = spy.reset_index(drop=True)
            spy_df = spy
    except Exception as e:
        logger.warning(f"Failed to fetch SPY: {e}")
    
    try:
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False, multi_level_index=False)
        if not vix.empty:
            vix = vix.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
            vix['date'] = vix.index
            vix = vix.reset_index(drop=True)
            vix_df = vix
    except Exception as e:
        logger.warning(f"Failed to fetch VIX: {e}")
    
    return spy_df, vix_df


# =============================================================================
# V9 MODEL INFERENCE
# =============================================================================

def precompute_features(stock_data: dict, spy_data, vix_data, scaler):
    """Pre-compute V9 features for all stocks."""
    logger.info("Pre-computing V9 features for all stocks...")
    
    processed = {}
    
    for i, (ticker, df) in enumerate(stock_data.items()):
        if (i + 1) % 20 == 0:
            logger.info(f"  Processing features {i+1}/{len(stock_data)}...")
        
        try:
            # Compute features
            df_feat = compute_v9_features(df, spy_data, vix_data)
            df_feat = df_feat.dropna(subset=V9_FEATURE_NAMES)
            
            if len(df_feat) < CONFIG["WINDOW_SIZE"]:
                continue
            
            # Scale features
            feats = df_feat[V9_FEATURE_NAMES].values
            feats = np.nan_to_num(feats, nan=0.0, posinf=1e6, neginf=-1e6)
            feats = scaler.transform(feats)
            feats = np.clip(feats, -10, 10).astype(np.float32)
            
            # Map dates to indices for O(1) lookup
            date_map = {pd.Timestamp(ts): idx for idx, ts in enumerate(df_feat['date'])}
            
            processed[ticker] = {
                'features': feats,
                'date_map': date_map,
                'prices': df_feat[['date', 'close']].copy(),
            }
        except Exception as e:
            logger.debug(f"Error processing {ticker}: {e}")
            continue
    
    logger.info(f"Features computed for {len(processed)} stocks")
    return processed


def get_window_for_date(processed_data: dict, ticker: str, date: pd.Timestamp, window_size: int = 60):
    """Get feature window for a specific date."""
    if ticker not in processed_data:
        return None
    
    p_data = processed_data[ticker]
    date_map = p_data['date_map']
    
    if date not in date_map:
        return None
    
    idx = date_map[date]
    start_idx = idx - window_size + 1
    
    if start_idx < 0:
        return None
    
    features = p_data['features'][start_idx:idx + 1]
    
    if len(features) != window_size:
        return None
    
    return features


def run_daily_inference(model, processed_data: dict, date: pd.Timestamp, device):
    """Run V9 model inference on all stocks for a single date."""
    windows = []
    tickers = []
    
    for ticker in processed_data.keys():
        window = get_window_for_date(processed_data, ticker, date, CONFIG["WINDOW_SIZE"])
        if window is not None:
            windows.append(window)
            tickers.append(ticker)
    
    if not windows:
        return {}
    
    # Batch inference
    X = torch.from_numpy(np.stack(windows)).float().to(device)
    
    model.eval()
    with torch.no_grad():
        preds = model(X).cpu().numpy().flatten()
    
    rankings = {ticker: float(pred) for ticker, pred in zip(tickers, preds)}
    return rankings


def get_price_for_date(processed_data: dict, ticker: str, date: pd.Timestamp):
    """Get closing price for a specific date."""
    if ticker not in processed_data:
        return None
    
    prices_df = processed_data[ticker]['prices']
    row = prices_df[prices_df['date'] == date]
    
    if len(row) == 0:
        return None
    
    return float(row['close'].values[0])


def compute_correlation_matrix(processed_data: dict, tickers: list, date: pd.Timestamp, lookback: int = 20):
    """Compute correlation matrix for a set of tickers."""
    returns_data = {}
    
    for ticker in tickers:
        if ticker not in processed_data:
            continue
        
        prices_df = processed_data[ticker]['prices']
        date_idx = prices_df[prices_df['date'] <= date].index
        
        if len(date_idx) < lookback:
            continue
        
        recent = prices_df.loc[date_idx[-lookback:]]
        returns = recent['close'].pct_change().dropna().values
        
        if len(returns) >= lookback - 1:
            returns_data[ticker] = returns[-lookback+1:]
    
    if len(returns_data) < 2:
        return {}
    
    # Build correlation dict (average correlation with others)
    tickers_list = list(returns_data.keys())
    n = len(tickers_list)
    corr_avg = {}
    
    for i, t1 in enumerate(tickers_list):
        correlations = []
        for j, t2 in enumerate(tickers_list):
            if i != j:
                try:
                    # Align lengths if needed (should be aligned by lookback)
                    r1 = returns_data[t1]
                    r2 = returns_data[t2]
                    min_len = min(len(r1), len(r2))
                    corr = np.corrcoef(r1[-min_len:], r2[-min_len:])[0, 1]
                    
                    if not np.isnan(corr):
                        correlations.append(corr)
                except (IndexError, ValueError, np.linalg.LinAlgError) as e:
                    logger.debug(f"Correlation error between {t1} and {t2}: {e}")
                    pass
        corr_avg[t1] = np.mean(correlations) if correlations else 0.5
    
    return corr_avg


# =============================================================================
# MAIN SIMULATION
# =============================================================================

async def seed_history(force=False):
    """Main function to seed paper trading history."""
    logger.info("=" * 70)
    logger.info("V9 HEDGE FUND SIMULATOR - TIME MACHINE")
    logger.info("=" * 70)
    logger.info(f"Session ID: {CONFIG['SESSION_ID']}")
    logger.info(f"Starting Capital: ${CONFIG['START_CAPITAL']:,.2f}")
    logger.info(f"Simulation Days: {CONFIG['SIMULATION_DAYS']}")
    logger.info(f"Top-K Holdings: {CONFIG['TOP_K']}")
    logger.info(f"Correlation Threshold: {CONFIG['CORRELATION_THRESHOLD']}")

    session = await get_session()
    device = get_device()
    logger.info(f"Using device: {device}")

    try:
        # Check if data already exists (skip if --force not specified)
        
        existing = await session.execute(
            select(PaperPortfolioHistory).where(
                PaperPortfolioHistory.session_id == CONFIG['SESSION_ID']
            ).limit(1)
        )
        if existing.scalar() and not force:
            logger.info("✅ Session data already exists! Use --force to re-seed.")
            logger.info("   Example: python -m scripts.seed_paper_history --force")
            return
        
        # 1. Cleanup existing data
        logger.info(f"Cleaning up session {CONFIG['SESSION_ID']}...")
        await session.execute(delete(PaperPortfolioHistory).where(PaperPortfolioHistory.session_id == CONFIG['SESSION_ID']))
        await session.execute(delete(PaperTrade).where(PaperTrade.session_id == CONFIG['SESSION_ID']))
        await session.execute(delete(PaperHolding).where(PaperHolding.session_id == CONFIG['SESSION_ID']))
        await session.execute(delete(PaperPortfolio).where(PaperPortfolio.session_id == CONFIG['SESSION_ID']))
        await session.commit()
        
        # 2. Load V9 Model
        if not MODEL_PATH.exists():
            logger.warning(f"V9 model not found at {MODEL_PATH}, using mock scores")
            model = None
            scaler = None
        else:
            logger.info(f"Loading V9 model from {MODEL_PATH}...")
            model = TransformerRankModel.load(str(MODEL_PATH), device=device)
            model.eval()
            
            logger.info(f"Loading scaler from {SCALER_PATH}...")
            with open(SCALER_PATH, 'rb') as f:
                scaler_data = pickle.load(f)
            scaler = scaler_data['scaler']
        
        # 3. Fetch Data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=CONFIG["SIMULATION_DAYS"] + 100)  # Extra for warmup
        
        stock_data = fetch_yfinance_data(TICKERS, start_date, end_date)
        spy_df, vix_df = fetch_macro_data(start_date, end_date)
        
        # Process macro data - CRITICAL for V9's rel_strength feature
        spy_data = process_spy_data(spy_df) if spy_df is not None else None
        vix_data = process_vix_data(vix_df) if vix_df is not None else None
        
        # Validate macro data (V9 needs SPY for relative strength)
        if spy_data is None:
            logger.error("⚠️ CRITICAL: SPY data not fetched! V9 will be 'blind' without relative strength features.")
            logger.error("Check internet connectivity (VPN off?) and try again.")
        else:
            logger.info(f"✅ SPY data loaded: {len(spy_data['spy_ret_5d'])} days")
        
        if vix_data is None:
            logger.warning("⚠️ VIX data not fetched. Using default VIX=20 for all days.")
        else:
            logger.info(f"✅ VIX data loaded: {len(vix_data)} days")
        
        # 4. Pre-compute features (if model available)
        if model is not None and scaler is not None:
            processed_data = precompute_features(stock_data, spy_data, vix_data, scaler)
        else:
            processed_data = None
        
        # 5. Get trading calendar
        if not stock_data:
            logger.error("No stock data fetched! Check internet connectivity.")
            logger.error("Try running locally: python backend/scripts/seed_paper_history.py")
            await session.rollback()
            return
        
        ref_ticker = 'AAPL' if 'AAPL' in stock_data else list(stock_data.keys())[0]
        all_dates = sorted(stock_data[ref_ticker]['date'].tolist())
        
        # Filter to simulation period
        sim_start = datetime.now() - timedelta(days=CONFIG["SIMULATION_DAYS"])
        trading_days = [d for d in all_dates if d >= pd.Timestamp(sim_start)]
        
        logger.info(f"Simulating {len(trading_days)} trading days...")
        
        # 6. Initialize portfolio state
        current_cash = CONFIG["START_CAPITAL"]
        holdings = {}  # ticker -> {qty, entry_price}
        portfolio_value = CONFIG["START_CAPITAL"]
        days_since_rebalance = CONFIG["REBALANCE_DAYS"]  # Force rebalance on first day
        
        # 7. Simulation Loop
        import time
        rankings = {}
        for i, current_date in enumerate(trading_days):
            day_start_time = time.time()  # Track per-day speed

            # Progress logging
            if i % 50 == 0:
                pct = (i / len(trading_days)) * 100
                logger.info(f"Day {i+1}/{len(trading_days)} ({pct:.0f}%) - {current_date.strftime('%Y-%m-%d')}")

            # Get current prices
            current_prices = {}
            for ticker in TICKERS:
                if ticker in stock_data:
                    df = stock_data[ticker]
                    row = df[df['date'] == current_date]
                    if len(row) > 0:
                        current_prices[ticker] = float(row['close'].values[0])
            
            # Update equity value
            equity_value = 0.0
            for ticker, pos in holdings.items():
                if ticker in current_prices:
                    equity_value += pos['qty'] * current_prices[ticker]
            
            total_value = current_cash + equity_value
            
            # VIX Check (Optional - skip trading in panic)
            vix_level = 20.0
            if vix_data is not None and current_date in vix_data.index:
                val = vix_data.loc[current_date]
                # Scale if it looks like a decimal (e.g. 0.20 instead of 20.0)
                vix_level = val * 100 if val < 2.0 else val
                # Sanity clamp
                vix_level = max(5.0, min(100.0, vix_level))
            
            # REBALANCE (every 5 days)
            if days_since_rebalance >= CONFIG["REBALANCE_DAYS"] and vix_level < CONFIG["VIX_THRESHOLD"]:
                days_since_rebalance = 0
                
                # Get V9 Rankings
                if processed_data is not None:
                    rankings = run_daily_inference(model, processed_data, current_date, device)
                else:
                    # Fallback: Mock scores
                    import random
                    rankings = {t: random.uniform(0.5, 0.99) for t in current_prices.keys()}
                
                if len(rankings) < CONFIG["TOP_K"]:
                    days_since_rebalance = CONFIG["REBALANCE_DAYS"]
                    continue
                
                # Sort by rank (descending)
                sorted_stocks = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
                
                # Apply correlation filter
                top_candidates = [t for t, _ in sorted_stocks[:CONFIG["TOP_K"] * 2]]
                correlations = compute_correlation_matrix(processed_data or {}, top_candidates, current_date)
                
                filtered = []
                # 1. First pass: High rank + Low correlation
                for t in top_candidates:
                    if t not in current_prices: continue
                    
                    # Check correlation against already selected
                    is_correlated = False
                    current_corr = 0.5
                    
                    for existing_item in filtered:
                        existing_ticker = existing_item['ticker']
                        pair = tuple(sorted((t, existing_ticker)))
                        if pair in correlations:
                            corr_val = correlations[pair]
                            if corr_val > CONFIG["CORRELATION_THRESHOLD"]:
                                is_correlated = True
                                break
                    
                    if not is_correlated:
                        filtered.append({
                            'ticker': t, 
                            'rank': rankings.get(t, 0), 
                            'corr': 0.0, # Placeholder/Avg
                            'price': current_prices[t]
                        })
                        if len(filtered) >= CONFIG["TOP_K"]:
                            break
                            
                # 2. Fallback: Fill remaining spots with highest ranked (ignoring correlation) if needed
                if len(filtered) < CONFIG["TOP_K"]:
                    for ticker, rank in sorted_stocks:
                        # Skip if already in filtered
                        if any(f['ticker'] == ticker for f in filtered):
                            continue
                        if ticker not in current_prices:
                            continue
                            
                        filtered.append({
                            'ticker': ticker, 
                            'rank': rank, 
                            'corr': 1.0, # High correlation penalty
                            'price': current_prices[ticker]
                        })
                        if len(filtered) >= CONFIG["TOP_K"]:
                            break
                
                # RECONSTRUCT top_picks with reasons
                top_picks = []
                for item in filtered:
                    ticker = item['ticker']
                    rank = item['rank']
                    corr = item.get('corr', 0.5)
                    top_picks.append({
                        "ticker": ticker,
                        "rank": rank,
                        "reason": f"Rank #{list(rankings.keys()).index(ticker)+1} ({rank:.2f}), Corr: {corr:.2f}",
                        "price": item['price'],
                        "corr": corr
                    })
                
                top_tickers = [p['ticker'] for p in top_picks]
                
                # SELL: Stocks not in top picks
                for ticker in list(holdings.keys()):
                    if ticker not in top_tickers and ticker in current_prices:
                        qty = holdings[ticker]['qty']
                        price = current_prices[ticker]
                        revenue = qty * price
                        entry = holdings[ticker]['entry_price']
                        pnl = revenue - (qty * entry)
                        
                        # Generate "Why" reason for sell
                        reason = f"Rank Dropped (>Top {CONFIG['TOP_K']}), PnL: ${pnl:+.0f}"
                        
                        trade = PaperTrade(
                            session_id=CONFIG['SESSION_ID'],
                            trade_date=current_date,
                            action="SELL",
                            ticker=ticker,
                            price=price,
                            quantity=qty,
                            reason=reason,
                            profit_loss=pnl
                        )
                        session.add(trade)
                        current_cash += revenue
                        del holdings[ticker]
                
                # BUY: New top picks
                target_allocation = total_value / CONFIG["TOP_K"]
                for pick in top_picks:
                    ticker = pick['ticker']
                    if ticker not in holdings:
                        price = pick['price']
                        if current_cash >= target_allocation * 0.9:
                            qty = int(target_allocation / price)
                            if qty > 0:
                                cost = qty * price
                                current_cash -= cost
                                holdings[ticker] = {'qty': qty, 'entry_price': price}
                                
                                # Generate "Why" reason for buy (Educational!)
                                rank_position = top_picks.index(pick) + 1
                                reason = f"Rank #{rank_position} ({pick['rank']:.2f}), Low Corr ({pick['corr']:.2f})"
                                
                                trade = PaperTrade(
                                    session_id=CONFIG['SESSION_ID'],
                                    trade_date=current_date,
                                    action="BUY",
                                    ticker=ticker,
                                    price=price,
                                    quantity=qty,
                                    reason=reason,
                                    profit_loss=0
                                )
                                session.add(trade)
            else:
                days_since_rebalance += 1
            
            # Save daily history
            hist_entry = PaperPortfolioHistory(
                session_id=CONFIG['SESSION_ID'],
                date=current_date,
                total_value=total_value,
                equity_value=equity_value,
                cash_balance=current_cash
            )
            session.add(hist_entry)

            # Speed Metrics
            day_elapsed = time.time() - day_start_time
            if days_since_rebalance == 1:  # Log rebalance days (most compute-intensive)
                n_ranked = len(rankings) if 'rankings' in locals() else 0
                logger.info(f"  ⚡ Day {current_date.strftime('%Y-%m-%d')}: Ranked {n_ranked} stocks in {day_elapsed:.3f}s")
        
        # 8. Final State
        # Recalculate final values
        final_equity = 0.0
        for ticker, pos in holdings.items():
            if ticker in current_prices:
                final_equity += pos['qty'] * current_prices[ticker]
        final_value = current_cash + final_equity
        
        logger.info(f"\n{'='*70}")
        logger.info(f"SIMULATION COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Final Value: ${final_value:,.2f}")
        logger.info(f"Total Return: {((final_value / CONFIG['START_CAPITAL']) - 1) * 100:+.2f}%")
        logger.info(f"Current Holdings: {len(holdings)}")
        
        # Save final portfolio state
        # Delete any existing portfolio first (in case cleanup was incomplete)
        await session.execute(delete(PaperPortfolio).where(PaperPortfolio.session_id == CONFIG['SESSION_ID']))
        await session.execute(delete(PaperHolding).where(PaperHolding.session_id == CONFIG['SESSION_ID']))
        
        portfolio = PaperPortfolio(
            session_id=CONFIG['SESSION_ID'],
            cash_balance=current_cash,
            equity_value=final_equity,
            total_value=final_value,
            days_since_rebalance=days_since_rebalance
        )
        session.add(portfolio)
        
        # Save current holdings
        for ticker, pos in holdings.items():
            if ticker in current_prices:
                holding = PaperHolding(
                    session_id=CONFIG['SESSION_ID'],
                    ticker=ticker,
                    quantity=pos['qty'],
                    entry_price=pos['entry_price'],
                    current_price=current_prices[ticker],
                    stop_loss_level=current_prices[ticker] * 0.93,  # 7% stop loss
                    highest_price=current_prices[ticker]
                )
                session.add(holding)
        
        await session.commit()
        logger.info("✅ Seed Completed Successfully!")
        
    except Exception as e:
        logger.error(f"Seed Error: {e}")
        import traceback
        traceback.print_exc()
        await session.rollback()
    finally:
        await session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed V9 paper trading history")
    parser.add_argument("--years", type=float, default=5,
                        help="Years of history to backtest (e.g. 0.2, 1, 5, default: 5)")
    parser.add_argument("--session-id", type=str, default="v9_golden_2025",
                        help="Session ID for this backtest (default: v9_golden_2025)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top stocks to pick (default: 10)")
    parser.add_argument("--corr-threshold", type=float, default=0.6, help="Correlation threshold (default: 0.6)")
    parser.add_argument("--rebalance-days", type=int, default=5, help="Rebalance every N days (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-seed even if session data exists")
    parser.add_argument("--sp500", action="store_true",
                        help="Use full S&P 500 (~500 stocks) instead of curated 60")
    args = parser.parse_args()

    # Update CONFIG based on arguments
    CONFIG["SIMULATION_DAYS"] = int(args.years * 252)  # 252 trading days per year
    CONFIG["SESSION_ID"] = args.session_id
    CONFIG["TOP_K"] = args.top_k
    CONFIG["CORRELATION_THRESHOLD"] = args.corr_threshold
    CONFIG["REBALANCE_DAYS"] = args.rebalance_days
    CONFIG["SEED"] = args.seed
    
    # Apply seeds immediately
    import random
    import numpy as np
    import torch
    random.seed(CONFIG["SEED"])
    np.random.seed(CONFIG["SEED"])
    torch.manual_seed(CONFIG["SEED"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["SEED"])
    
    # Use full S&P 500 if requested
    if args.sp500:
        TICKERS = fetch_sp500_tickers()
        logger.info(f"🚀 FULL S&P 500 MODE: {len(TICKERS)} stocks")
    else:
        logger.info(f"Standard mode: {len(TICKERS)} curated stocks")

    logger.info(f"Configuration: {args.years} year(s) = {CONFIG['SIMULATION_DAYS']} trading days")
    logger.info(f"Session ID: {CONFIG['SESSION_ID']}")

    asyncio.run(seed_history(force=args.force))
