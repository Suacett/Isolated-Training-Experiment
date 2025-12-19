#!/usr/bin/env python3
"""
V9 Backtest - Transformer Ranking Portfolio (Optimized)

This script backtests the V9 ranking model by:
1. Pre-computing features for ALL stocks (Optimization: O(N) instead of O(N^2))
2. Running inference on all stocks daily using pre-computed features
3. Selecting top K stocks by predicted rank
4. Rebalancing weekly with equal weights
5. Comparing vs SPY buy & hold

Metrics:
- Annualized Return (Time-Weighted)
- Sharpe Ratio
- Max Drawdown
- Turnover Rate

Usage:
    docker exec proxmox_stock_backend python -m scripts.backtest_v9_portfolio
    docker exec proxmox_stock_backend python -m scripts.backtest_v9_portfolio --top-k 20
"""

import sys
import os
import logging
import pickle
import argparse
import pandas as pd
import numpy as np
import psycopg2
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import torch

backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from services.feature_engineering_v9 import (
    compute_v9_features,
    process_spy_data,
    process_vix_data,
    V9_FEATURE_NAMES,
    N_FEATURES,
)
from services.transformer_model import TransformerRankModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest_v9.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "WINDOW_SIZE": 60,           # Lookback window for model
    "TOP_K": 10,                 # Number of stocks to hold
    "REBALANCE_DAYS": 5,         # Rebalance weekly (5 trading days)
    "SLIPPAGE": 0.001,           # 0.1% transaction cost per trade
    "INITIAL_CAPITAL": 100000,   # Starting capital ($100k)
    "MIN_HISTORY": 60,           # Minimum days to have valid features
    "BACKTEST_START_DATE": "2023-01-01",  # Only backtest from this date forward
}

MODEL_PATH = backend_path / "models" / "transformer_v9.pth"
MODEL_PATH_BEST = backend_path / "models" / "transformer_v9_best.pth"
SCALER_PATH = backend_path / "models" / "scaler_v9.pkl"


# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def get_db_url():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL environment variable must be set")
    return url.replace("postgresql+asyncpg://", "postgresql://")


def get_all_tickers():
    """Get all tickers with sufficient history, excluding non-tradable indices."""
    try:
        with psycopg2.connect(get_db_url()) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ticker, COUNT(*) as count 
                    FROM stock_prices 
                    WHERE ticker NOT IN ('^VIX', 'VIX', 'SPY', 'QQQ', 'DIA', 'IWM')
                    GROUP BY ticker 
                    HAVING COUNT(*) >= 500 
                    ORDER BY count DESC
                """)
                results = cur.fetchall()
        return [row[0] for row in results]
    except Exception as e:
        logger.error(f"Error fetching tickers: {e}")
        return []


def load_stock_data(ticker: str) -> pd.DataFrame:
    """Load OHLCV data for a single stock."""
    try:
        with psycopg2.connect(get_db_url()) as conn:
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


def load_all_stocks_data(tickers: list):
    """Load data for all tickers."""
    all_data = {}
    for i, ticker in enumerate(tickers):
        if (i + 1) % 100 == 0:
            logger.info(f"Loading stock {i+1}/{len(tickers)}...")
        df = load_stock_data(ticker)
        if df is not None and len(df) > CONFIG["MIN_HISTORY"]:
            all_data[ticker] = df
    return all_data


# =============================================================================
# OPTIMIZED FEATURE ENGINEERING
# =============================================================================

def precompute_model_data(stock_data, spy_data, vix_data, scaler):
    """
    Pre-compute features for all stocks to avoid O(N^2) re-calculation.
    
    Returns:
        processed_data: Dict mapping ticker -> {
            'features': np.array (T, F),
            'date_map': dict(timestamp -> index)
        }
    """
    logger.info("Pre-computing features for all stocks (Optimized)...")
    processed = {}
    
    for i, (ticker, df) in enumerate(stock_data.items()):
        if (i + 1) % 50 == 0:
            logger.info(f"Processing features {i+1}/{len(stock_data)}...")
            
        try:
            # Features
            df_feat = compute_v9_features(df, spy_data, vix_data)
            df_feat = df_feat.dropna(subset=V9_FEATURE_NAMES)
            
            if len(df_feat) < CONFIG["WINDOW_SIZE"]:
                continue
                
            # Scale
            feats = df_feat[V9_FEATURE_NAMES].values
            feats = np.nan_to_num(feats, nan=0.0, posinf=1e6, neginf=-1e6)
            feats = scaler.transform(feats)
            feats = np.clip(feats, -10, 10).astype(np.float32)
            
            # Map Dates -> Index for O(1) lookup
            # Use pandas Timestamp objects as keys
            date_map = {ts: idx for idx, ts in enumerate(df_feat['date'])}
            
            processed[ticker] = {
                'features': feats,
                'date_map': date_map
            }
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            continue
            
    return processed


def get_inference_window_fast(processed_stock, date, window_size=60):
    """
    Retrieve pre-computed window for a specific date (O(1)).
    """
    date_map = processed_stock['date_map']
    if date not in date_map:
        return None
        
    idx = date_map[date]
    
    # We need a window ending at 'idx' (inclusive) with length 'window_size'
    # Start index = idx - window_size + 1
    # If starting index is negative, we don't have enough history
    start_idx = idx - window_size + 1
    
    if start_idx < 0:
        return None
        
    # Python slicing [start:end] excludes end, so end_idx = idx + 1
    end_idx = idx + 1
    
    features = processed_stock['features'][start_idx : end_idx]
    
    # Double check length
    if len(features) != window_size:
        return None
        
    return features


# =============================================================================
# MODEL INFERENCE
# =============================================================================

def run_daily_inference(
    model,
    processed_data: dict,  # Now uses pre-computed data
    date: pd.Timestamp,
    device,
):
    """
    Run inference on all stocks for a single date using pre-computed features.
    
    Returns:
        rankings: Dict mapping ticker -> predicted_rank
    """
    windows = []
    tickers = []
    
    for ticker, p_data in processed_data.items():
        window = get_inference_window_fast(
            p_data, date, CONFIG["WINDOW_SIZE"]
        )
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
    
    rankings = {ticker: pred for ticker, pred in zip(tickers, preds)}
    return rankings


# =============================================================================
# PORTFOLIO SIMULATION
# =============================================================================

def get_next_day_return(df: pd.DataFrame, date: pd.Timestamp) -> float:
    """Get the 1-day return after the given date."""
    # Optimization: Filter roughly first
    try:
        # Assuming df is sorted by date
        # Find index of date
        mask = df['date'] == date
        if not mask.any():
            return 0.0
            
        current_idx = df.index[mask][0]
        # Next row
        if current_idx + 1 >= len(df):
            return 0.0
            
        current_close = df.at[current_idx, 'close']
        future_close = df.at[current_idx + 1, 'close']
        
        return (future_close - current_close) / current_close
    except (KeyError, IndexError, ValueError, Exception) as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        # Fallback to slow method if index not aligned
        logger.debug(f"Index error in next day return calculation, falling back: {e}")
        future = df[df['date'] > date].head(1)
        if len(future) == 0:
            return 0.0
        
        current = df[df['date'] == date]
        if len(current) == 0:
            return 0.0
        
        current_close = current['close'].values[0]
        future_close = future['close'].values[0]
        
        return (future_close - current_close) / current_close


def run_backtest(
    model,
    stock_data: dict,
    processed_data: dict, # Pre-computed features
    vix_data,  # VIX data for volatility filter
    device,
    start_date=None,
    end_date=None,
    top_k: int = 10,
    rebalance_days: int = 5,
    slippage: float = 0.001,
    vix_threshold: float = 30.0,  # VIX level to trigger defensive mode
    stop_loss: float = -0.07,     # -7% stop loss
    take_profit: float = 0.15,    # +15% take profit
):
    """
    Run the full backtest simulation with VIX volatility filter and SL/TP.
    
    Risk Management:
    - VIX > vix_threshold: portfolio goes to 100% cash
    - Stop Loss: sell if position drops stop_loss% from entry
    - Take Profit: sell if position gains take_profit% from entry
    """
    # Get all unique trading dates from keys of stock_data
    if not stock_data:
        logger.error("No stock data available for backtest")
        raise RuntimeError("Empty stock_data dict")
    
    sample_ticker = 'AAPL' if 'AAPL' in stock_data else list(stock_data.keys())[0]
    all_dates = sorted(stock_data[sample_ticker]['date'].tolist())
    
    # Apply start date filter
    backtest_start = pd.Timestamp(CONFIG["BACKTEST_START_DATE"])
    if start_date:
        backtest_start = max(backtest_start, pd.Timestamp(start_date))
    
    all_dates = [d for d in all_dates if d >= backtest_start]
    
    if end_date:
        all_dates = [d for d in all_dates if d <= end_date]
    
    if not all_dates:
        logger.error("No trading dates available after filtering")
        return None
    
    total_days = len(all_dates)
    logger.info(f"Backtesting from {all_dates[0]} to {all_dates[-1]}")
    logger.info(f"Total trading days: {total_days}")
    logger.info(f"Top-K: {top_k}, Rebalance: Every {rebalance_days} days")
    logger.info(f"VIX Volatility Filter: Go to cash when VIX > {vix_threshold}")
    logger.info(f"Risk Management: Stop Loss {stop_loss*100:.1f}%, Take Profit {take_profit*100:.1f}%")
    
    # Initialize portfolio
    portfolio_value = CONFIG["INITIAL_CAPITAL"]
    current_holdings = {}  # ticker -> weight
    entry_prices = {}  # ticker -> entry price for SL/TP tracking
    
    # Tracking
    portfolio_history = []
    trade_log = []
    turnover_events = []
    vix_veto_days = 0  # Count days in defensive mode
    risk_exits = {'stop_loss': 0, 'take_profit': 0}  # Count risk-based exits
    days_since_rebalance = rebalance_days  # Force rebalance on first day
    
    for day_idx, date in enumerate(all_dates):
        # Progress logging every 50 days
        if day_idx % 50 == 0:
            pct = (day_idx / total_days) * 100
            logger.info(f"Processing day {day_idx+1}/{total_days} ({pct:.0f}%) - {date.strftime('%Y-%m-%d')}...")
        
        # =====================================================================
        # VIX VOLATILITY FILTER: Check if we should be defensive
        # =====================================================================
        vix_level = 20.0  # Default to normal if no data
        if vix_data is not None and date in vix_data.index:
            vix_level = vix_data.loc[date] * 100  # Convert from normalized (0.20 -> 20)
        
        # VIX VETO: If VIX > threshold, go to 100% cash
        if vix_level > vix_threshold:
            vix_veto_days += 1
            
            # Liquidate all holdings
            if current_holdings:
                trade_log.append({
                    'date': date,
                    'action': 'vix_veto',
                    'vix_level': vix_level,
                    'holdings_liquidated': list(current_holdings.keys()),
                })
                current_holdings = {}
            
            # Skip trading, stay in cash
            portfolio_history.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'n_holdings': 0,
                'vix_veto': True,
            })
            days_since_rebalance += 1
            continue  # Skip to next day
        
        # =====================================================================
        # STOP LOSS / TAKE PROFIT: Check daily before any rebalance
        # =====================================================================
        stocks_to_exit = []
        for ticker in list(current_holdings.keys()):
            if ticker not in entry_prices or ticker not in stock_data:
                continue
            
            # Get current price for this date
            ticker_df = stock_data[ticker]
            current_row = ticker_df[ticker_df['date'] == date]
            if len(current_row) == 0:
                continue
            
            current_price = current_row['close'].values[0]
            entry_price = entry_prices[ticker]
            
            # Calculate return since entry
            return_since_entry = (current_price - entry_price) / entry_price
            
            # Check stop loss
            if return_since_entry <= stop_loss:
                stocks_to_exit.append((ticker, 'stop_loss', return_since_entry))
                risk_exits['stop_loss'] += 1
            # Check take profit
            elif return_since_entry >= take_profit:
                stocks_to_exit.append((ticker, 'take_profit', return_since_entry))
                risk_exits['take_profit'] += 1
        
        # Execute SL/TP exits
        for ticker, reason, ret in stocks_to_exit:
            # Remove from holdings
            if ticker in current_holdings:
                del current_holdings[ticker]
            if ticker in entry_prices:
                del entry_prices[ticker]
            
            trade_log.append({
                'date': date,
                'action': reason,
                'ticker': ticker,
                'return_pct': ret * 100,
            })
        
        # Check if we need to rebalance
        if days_since_rebalance >= rebalance_days:
            # Get rankings
            rankings = run_daily_inference(
                model, processed_data, date, device
            )
            
            if len(rankings) >= top_k:
                # Sort by predicted rank (descending - higher is better)
                sorted_stocks = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
                new_top_k = [t for t, _ in sorted_stocks[:top_k]]
                
                # Calculate turnover
                old_holdings = set(current_holdings.keys())
                new_holdings = set(new_top_k)
                trades = len(old_holdings.symmetric_difference(new_holdings))
                turnover = trades / (2 * top_k) if top_k > 0 else 0
                turnover_events.append(turnover)
                
                # Apply slippage for trades
                if current_holdings:
                    stocks_sold = old_holdings - new_holdings
                    stocks_bought = new_holdings - old_holdings
                    total_trades = len(stocks_sold) + len(stocks_bought)
                    slippage_cost = slippage * (total_trades / top_k)
                    portfolio_value *= (1 - slippage_cost)
                
                # Update holdings (equal weight)
                current_holdings = {ticker: 1.0 / top_k for ticker in new_top_k}
                
                # Record entry prices for SL/TP tracking
                for ticker in new_top_k:
                    if ticker not in entry_prices and ticker in stock_data:
                        ticker_df = stock_data[ticker]
                        entry_row = ticker_df[ticker_df['date'] == date]
                        if len(entry_row) > 0:
                            entry_prices[ticker] = entry_row['close'].values[0]
                
                trade_log.append({
                    'date': date,
                    'action': 'rebalance',
                    'holdings': list(current_holdings.keys()),
                    'top_predictions': sorted_stocks[:top_k],
                    'turnover': turnover,
                })
                
                days_since_rebalance = 0
            else:
                days_since_rebalance += 1
        else:
            days_since_rebalance += 1
        
        # Calculate daily return based on current holdings
        if current_holdings:
            daily_return = 0.0
            for ticker, weight in current_holdings.items():
                if ticker in stock_data:
                    ret = get_next_day_return(stock_data[ticker], date)
                    daily_return += weight * ret
            
            portfolio_value *= (1 + daily_return)
        
        portfolio_history.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'n_holdings': len(current_holdings),
        })
    
    logger.info(f"VIX Volatility Filter: Spent {vix_veto_days} days in cash (VIX > {vix_threshold})")
    logger.info(f"Risk Exits: {risk_exits['stop_loss']} Stop Loss, {risk_exits['take_profit']} Take Profit")
    
    return {
        'portfolio_history': pd.DataFrame(portfolio_history),
        'trade_log': trade_log,
        'turnover_events': turnover_events,
        'vix_veto_days': vix_veto_days,
        'risk_exits': risk_exits,
    }


def calculate_spy_benchmark(stock_data: dict, portfolio_history: pd.DataFrame):
    """Calculate SPY buy & hold benchmark for the same period."""
    if 'SPY' not in stock_data:
        logger.warning("SPY not in stock data, cannot calculate benchmark")
        return None
    
    spy_df = stock_data['SPY'].copy()
    start_date = portfolio_history['date'].min()
    end_date = portfolio_history['date'].max()
    
    spy_df = spy_df[(spy_df['date'] >= start_date) & (spy_df['date'] <= end_date)]
    
    if len(spy_df) == 0:
        return None
    
    initial_price = spy_df['close'].iloc[0]
    spy_df['spy_value'] = CONFIG["INITIAL_CAPITAL"] * (spy_df['close'] / initial_price)
    
    return spy_df[['date', 'spy_value']].rename(columns={'spy_value': 'benchmark_value'})


def calculate_metrics(portfolio_df: pd.DataFrame, benchmark_df: pd.DataFrame = None):
    """Calculate performance metrics."""
    portfolio_df = portfolio_df.copy()
    portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
    
    # Remove NaN
    returns = portfolio_df['returns'].dropna()
    
    # Annualized Return (Time-Weighted)
    start_date = portfolio_df['date'].iloc[0]
    end_date = portfolio_df['date'].iloc[-1]
    calendar_days = (end_date - start_date).days
    if calendar_days < 1:
        calendar_days = 1
        
    total_return = (portfolio_df['portfolio_value'].iloc[-1] / portfolio_df['portfolio_value'].iloc[0]) - 1
    annualized_return = (1 + total_return) ** (365.25 / calendar_days) - 1
    
    # Sharpe Ratio (assuming 0% risk-free rate)
    daily_sharpe = returns.mean() / (returns.std() + 1e-10)
    annualized_sharpe = daily_sharpe * np.sqrt(252)
    
    # Max Drawdown
    cummax = portfolio_df['portfolio_value'].cummax()
    drawdown = (portfolio_df['portfolio_value'] - cummax) / cummax
    max_drawdown = drawdown.min()
    
    # Final values
    final_value = portfolio_df['portfolio_value'].iloc[-1]
    
    metrics = {
        'total_return': total_return * 100,
        'annualized_return': annualized_return * 100,
        'sharpe_ratio': annualized_sharpe,
        'max_drawdown': max_drawdown * 100,
        'final_value': final_value,
        'total_days': calendar_days,
    }
    
    # Benchmark comparison
    if benchmark_df is not None and len(benchmark_df) > 0:
        merged = portfolio_df.merge(benchmark_df, on='date', how='inner')
        if len(merged) > 1:
            bench_total = (merged['benchmark_value'].iloc[-1] / merged['benchmark_value'].iloc[0]) - 1
            
            b_days = (merged['date'].iloc[-1] - merged['date'].iloc[0]).days
            if b_days < 1: b_days = 1
            bench_ann = (1 + bench_total) ** (365.25 / b_days) - 1
            
            bench_returns = merged['benchmark_value'].pct_change().dropna()
            bench_sharpe = (bench_returns.mean() / (bench_returns.std() + 1e-10)) * np.sqrt(252)
            
            bench_cummax = merged['benchmark_value'].cummax()
            bench_dd = (merged['benchmark_value'] - bench_cummax) / bench_cummax
            
            metrics['benchmark_total_return'] = bench_total * 100
            metrics['benchmark_annualized_return'] = bench_ann * 100
            metrics['benchmark_sharpe'] = bench_sharpe
            metrics['benchmark_max_drawdown'] = bench_dd.min() * 100
            metrics['benchmark_final_value'] = merged['benchmark_value'].iloc[-1]
            metrics['alpha'] = metrics['annualized_return'] - metrics['benchmark_annualized_return']
    
    return metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="V9 Backtest - Transformer Ranking")
    parser.add_argument("--top-k", type=int, default=10, help="Number of stocks to hold")
    parser.add_argument("--rebalance", type=int, default=5, help="Rebalance every N days")
    parser.add_argument("--slippage", type=float, default=0.001, help="Transaction cost per trade")
    parser.add_argument("--best", action="store_true", help="Use best correlation model")
    parser.add_argument("--start-date", type=str, default="2023-01-01", help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date for backtest (YYYY-MM-DD)")
    args = parser.parse_args()
    
    CONFIG["TOP_K"] = args.top_k
    CONFIG["REBALANCE_DAYS"] = args.rebalance
    CONFIG["SLIPPAGE"] = args.slippage
    CONFIG["BACKTEST_START_DATE"] = args.start_date
    if args.end_date:
        CONFIG["BACKTEST_END_DATE"] = args.end_date
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    model_path = MODEL_PATH_BEST if args.best else MODEL_PATH
    if not model_path.exists():
        model_path = MODEL_PATH_BEST if MODEL_PATH_BEST.exists() else MODEL_PATH
    
    logger.info(f"Loading model from {model_path}...")
    model = TransformerRankModel.load(str(model_path), device=device)
    model.eval()
    
    # Load scaler
    logger.info(f"Loading scaler from {SCALER_PATH}...")
    with open(SCALER_PATH, 'rb') as f:
        scaler_data = pickle.load(f)
    scaler = scaler_data['scaler']
    
    # Load all stock data
    logger.info("Loading stock data...")
    tickers = get_all_tickers()
    stock_data = load_all_stocks_data(tickers)
    logger.info(f"Loaded {len(stock_data)} stocks")
    
    # Load macro data
    if 'SPY' in stock_data:
        spy_df = stock_data['SPY']
    else:
        # Try loading SPY specifically if not in tickers
        spy_df = load_stock_data('SPY') 
        # Add to stock_data for benchmark calculation later
        if spy_df is not None:
            stock_data['SPY'] = spy_df
            
    # Try loading VIX
    vix_df = load_stock_data('^VIX')
    if vix_df is None:
        vix_df = load_stock_data('VIX')
    
    spy_data = process_spy_data(spy_df) if spy_df is not None else None
    vix_data = process_vix_data(vix_df) if vix_df is not None else None
    
    # Ensure all data has numeric indices for optimized fetching where possible
    for df in stock_data.values():
        df.reset_index(drop=True, inplace=True)

    # PRE-COMPUTE FEATURES
    processed_data = precompute_model_data(stock_data, spy_data, vix_data, scaler)
    logger.info(f"Optimized features computed for {len(processed_data)} stocks")
    
    # Run backtest
    logger.info("\n" + "="*70)
    logger.info("RUNNING V9 BACKTEST (OPTIMIZED)")
    logger.info("="*70)
    
    results = run_backtest(
        model=model,
        stock_data=stock_data,
        processed_data=processed_data,
        vix_data=vix_data,  # VIX data for volatility filter
        device=device,
        top_k=args.top_k,
        rebalance_days=args.rebalance,
        slippage=args.slippage,
        vix_threshold=30.0,  # Go to cash when VIX > 30
    )
    
    if results is None:
        logger.error("Backtest failed")
        return
    
    portfolio_history = results['portfolio_history']
    
    # Calculate benchmark
    benchmark_df = calculate_spy_benchmark(stock_data, portfolio_history)
    
    # Calculate metrics
    metrics = calculate_metrics(portfolio_history, benchmark_df)
    
    # Calculate turnover
    avg_turnover = np.mean(results['turnover_events']) * 100 if results['turnover_events'] else 0
    
    # Report
    logger.info("\n" + "="*70)
    logger.info("📊 V9 BACKTEST RESULTS")
    logger.info("="*70)
    logger.info(f"Strategy: Top-{args.top_k} by Predicted Rank, Rebalance every {args.rebalance} days")
    logger.info(f"Transaction Cost: {args.slippage*100:.2f}%")
    logger.info("")
    
    logger.info("--- V9 STRATEGY ---")
    logger.info(f"  Total Return:      {metrics['total_return']:+.2f}%")
    logger.info(f"  Annualized Return: {metrics['annualized_return']:+.2f}%")
    logger.info(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:.3f}")
    logger.info(f"  Max Drawdown:      {metrics['max_drawdown']:.2f}%")
    logger.info(f"  Final Value:       ${metrics['final_value']:,.2f}")
    logger.info(f"  Avg Turnover:      {avg_turnover:.1f}%")
    
    if 'benchmark_total_return' in metrics:
        logger.info("")
        logger.info("--- SPY BUY & HOLD ---")
        logger.info(f"  Total Return:      {metrics['benchmark_total_return']:+.2f}%")
        logger.info(f"  Annualized Return: {metrics['benchmark_annualized_return']:+.2f}%")
        logger.info(f"  Sharpe Ratio:      {metrics['benchmark_sharpe']:.3f}")
        logger.info(f"  Max Drawdown:      {metrics['benchmark_max_drawdown']:.2f}%")
        logger.info(f"  Final Value:       ${metrics['benchmark_final_value']:,.2f}")
        logger.info("")
        logger.info(f">>> ALPHA (V9 - SPY): {metrics['alpha']:+.2f}% annualized")
    
    logger.info("="*70)
    
    # Save results
    results_path = backend_path / "backtest_v9_results.csv"
    portfolio_history.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to {results_path}")
    
    # Show sample trades
    if results['trade_log']:
        logger.info("\n📋 Sample Trades (First 5 events):")
        for trade in results['trade_log'][:5]:
            if trade.get('action') == 'rebalance':
                holdings_str = str(trade.get('holdings', []))[:50] + "..."
                turnover = trade.get('turnover', 0) * 100
                logger.info(f"  {trade['date'].strftime('%Y-%m-%d')}: REBALANCE - {holdings_str} (turnover: {turnover:.1f}%)")
            else:
                action = trade.get('action', 'unknown').upper()
                ticker = trade.get('ticker', 'N/A')
                
                # Use available fields instead of non-existent 'reason'
                details = ""
                if action in ['STOP_LOSS', 'TAKE_PROFIT']:
                    ret = trade.get('return_pct', 0.0)
                    details = f"Return: {ret:+.2f}%"
                else:
                    details = f"Price: {trade.get('price', 'N/A')}"
                    
                logger.info(f"  {trade['date'].strftime('%Y-%m-%d')}: {action} - {ticker} ({details})")


if __name__ == "__main__":
    main()
