#!/usr/bin/env python3
"""
V9 Backtest - Transformer Relative Strength Ranking Portfolio

This script backtests the V9 ranking model by:
1. Running inference on all stocks daily
2. Selecting top K stocks by predicted rank
3. Rebalancing weekly with equal weights
4. Comparing vs SPY buy & hold

Metrics:
- Annualized Return
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
    url = os.getenv("DATABASE_URL", "")
    return url.replace("postgresql+asyncpg://", "postgresql://")


def get_all_tickers():
    """Get all tickers with sufficient history."""
    import psycopg2
    conn = psycopg2.connect(get_db_url())
    cur = conn.cursor()
    cur.execute("""
        SELECT ticker, COUNT(*) as count 
        FROM stock_prices GROUP BY ticker 
        HAVING COUNT(*) >= 500 ORDER BY count DESC
    """)
    results = cur.fetchall()
    cur.close()
    conn.close()
    return [row[0] for row in results]


def load_stock_data(ticker: str) -> pd.DataFrame:
    """Load OHLCV data for a single stock."""
    import psycopg2
    conn = psycopg2.connect(get_db_url())
    cur = conn.cursor()
    cur.execute("""
        SELECT timestamp, open, high, low, close, volume 
        FROM stock_prices WHERE ticker = %s ORDER BY timestamp
    """, (ticker,))
    records = cur.fetchall()
    cur.close()
    conn.close()
    
    if records:
        df = pd.DataFrame(records, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df['ticker'] = ticker
        return df
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
# MODEL INFERENCE
# =============================================================================

def prepare_inference_window(
    df: pd.DataFrame,
    date: pd.Timestamp,
    scaler,
    spy_data,
    vix_data,
    window_size: int = 60,
):
    """
    Prepare a single inference window for a stock on a specific date.
    
    Returns:
        window: numpy array of shape (window_size, n_features) or None if not available
    """
    # Filter to dates up to and including target date
    df = df[df['date'] <= date].copy()
    
    if len(df) < window_size + 30:  # Need enough history
        return None
    
    try:
        # Compute features
        df = compute_v9_features(df, spy_data, vix_data)
        df = df.dropna(subset=V9_FEATURE_NAMES)
        
        if len(df) < window_size:
            return None
        
        # Get last window_size rows
        features = df[V9_FEATURE_NAMES].tail(window_size).values
        
        # Scale
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        features = scaler.transform(features)
        features = np.clip(features, -10, 10).astype(np.float32)
        
        return features
    except Exception as e:
        return None


def run_daily_inference(
    model,
    stock_data: dict,
    date: pd.Timestamp,
    scaler,
    spy_data,
    vix_data,
    device,
):
    """
    Run inference on all stocks for a single date.
    
    Returns:
        rankings: Dict mapping ticker -> predicted_rank
    """
    windows = []
    tickers = []
    
    for ticker, df in stock_data.items():
        window = prepare_inference_window(
            df, date, scaler, spy_data, vix_data, CONFIG["WINDOW_SIZE"]
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
    scaler,
    spy_data,
    vix_data,
    device,
    start_date=None,
    end_date=None,
    top_k: int = 10,
    rebalance_days: int = 5,
    slippage: float = 0.001,
):
    """
    Run the full backtest simulation.
    
    Returns:
        results: Dict with portfolio values, metrics, trade log
    """
    # Get all unique trading dates
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df['date'].tolist())
    all_dates = sorted([d for d in all_dates if pd.notna(d)])
    
    # Apply start date filter FIRST (before warmup skip)
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
    
    # Initialize portfolio
    portfolio_value = CONFIG["INITIAL_CAPITAL"]
    current_holdings = {}  # ticker -> weight
    
    # Tracking
    portfolio_history = []
    trade_log = []
    turnover_events = []
    days_since_rebalance = rebalance_days  # Force rebalance on first day
    
    for day_idx, date in enumerate(all_dates):
        # Progress logging every 20 days
        if day_idx % 20 == 0:
            pct = (day_idx / total_days) * 100
            logger.info(f"Processing day {day_idx+1}/{total_days} ({pct:.0f}%) - {date.strftime('%Y-%m-%d')}...")
        
        # Check if we need to rebalance
        if days_since_rebalance >= rebalance_days:
            # Get rankings for today
            rankings = run_daily_inference(
                model, stock_data, date, scaler, spy_data, vix_data, device
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
    
    return {
        'portfolio_history': pd.DataFrame(portfolio_history),
        'trade_log': trade_log,
        'turnover_events': turnover_events,
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
    
    # Annualized Return
    total_days = len(portfolio_df)
    total_return = (portfolio_df['portfolio_value'].iloc[-1] / portfolio_df['portfolio_value'].iloc[0]) - 1
    annualized_return = (1 + total_return) ** (252 / total_days) - 1
    
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
        'total_days': total_days,
    }
    
    # Benchmark comparison
    if benchmark_df is not None and len(benchmark_df) > 0:
        merged = portfolio_df.merge(benchmark_df, on='date', how='inner')
        if len(merged) > 1:
            bench_total = (merged['benchmark_value'].iloc[-1] / merged['benchmark_value'].iloc[0]) - 1
            bench_ann = (1 + bench_total) ** (252 / len(merged)) - 1
            
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
    args = parser.parse_args()
    
    CONFIG["TOP_K"] = args.top_k
    CONFIG["REBALANCE_DAYS"] = args.rebalance
    CONFIG["SLIPPAGE"] = args.slippage
    
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
    spy_df = stock_data.get('SPY')
    vix_df = load_stock_data('^VIX')
    if vix_df is None:
        vix_df = load_stock_data('VIX')
    
    spy_data = process_spy_data(spy_df) if spy_df is not None else None
    vix_data = process_vix_data(vix_df) if vix_df is not None else None
    
    # Run backtest
    logger.info("\n" + "="*70)
    logger.info("RUNNING V9 BACKTEST")
    logger.info("="*70)
    
    results = run_backtest(
        model=model,
        stock_data=stock_data,
        scaler=scaler,
        spy_data=spy_data,
        vix_data=vix_data,
        device=device,
        top_k=args.top_k,
        rebalance_days=args.rebalance,
        slippage=args.slippage,
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
        logger.info("\n📋 Sample Trades (First 5 Rebalances):")
        for trade in results['trade_log'][:5]:
            logger.info(f"  {trade['date'].strftime('%Y-%m-%d')}: {trade['holdings'][:5]}... (turnover: {trade['turnover']*100:.1f}%)")


if __name__ == "__main__":
    main()
