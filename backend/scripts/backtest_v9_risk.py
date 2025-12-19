#!/usr/bin/env python3
"""
V9 Backtest with Risk Management - Transformer Ranking Portfolio

This script backtests the V9 ranking model with adaptive risk management:
1. ATR-based trailing stops (replaces broken fixed SL/TP)
2. Drawdown-based exposure scaling (portfolio-level protection)
3. Enhanced regime filter (SPY trend + VIX)

Metrics:
- Annualized Return (Time-Weighted)
- Sharpe Ratio
- Max Drawdown
- Turnover Rate

Usage:
    docker exec proxmox_stock_backend python -m scripts.backtest_v9_risk --top-k 10
    docker exec proxmox_stock_backend python -m scripts.backtest_v9_risk --top-k 10 --atr-mult 2.5 --no-risk
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
from services.risk_management import (
    calculate_atr,
    get_atr_trailing_stop,
    get_drawdown_exposure,
    get_regime_exposure,
    calculate_final_exposure,
    PositionTracker,
    select_with_correlation_filter,
    calculate_volatility_weights,
    calculate_hybrid_weights,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest_v9_risk.log"),
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

    # Risk Management Parameters
    "ATR_MULTIPLIER": 2.5,       # ATR trailing stop multiplier
    "ATR_PERIOD": 14,            # ATR lookback period
    "ENABLE_ATR_STOPS": True,    # Enable ATR trailing stops
    "ENABLE_DRAWDOWN": False,    # Enable drawdown-based scaling (DISABLED BY DEFAULT)
    "ENABLE_REGIME": False,      # Enable regime filter (DISABLED BY DEFAULT)

    # Diversification Parameters
    "ENABLE_CORRELATION_FILTER": True,  # Enable correlation filtering
    "MAX_CORRELATION": 0.70,            # Maximum pairwise correlation
    "ENABLE_VOL_WEIGHTING": True,       # Enable volatility-adjusted weights
    "VOL_LOOKBACK": 20,                 # Volatility lookback period
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
    """Get all tickers with sufficient history, excluding non-tradable indices."""
    import psycopg2
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
        logger.error(f"Failed to get tickers: {e}")
        return []


def load_stock_data(ticker: str) -> pd.DataFrame:
    """Load OHLCV data for a single stock."""
    import psycopg2
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
        logger.error(f"Failed to load data for {ticker}: {e}")
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
    start_idx = idx - window_size + 1

    if start_idx < 0:
        return None

    end_idx = idx + 1
    features = processed_stock['features'][start_idx : end_idx]

    if len(features) != window_size:
        return None

    return features


# =============================================================================
# MODEL INFERENCE
# =============================================================================

def run_daily_inference(
    model,
    processed_data: dict,
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
# PORTFOLIO SIMULATION WITH RISK MANAGEMENT
# =============================================================================

def get_next_day_return(df: pd.DataFrame, date: pd.Timestamp) -> float:
    """Get the 1-day return after the given date."""
    try:
        mask = df['date'] == date
        if not mask.any():
            return 0.0

        current_idx = df.index[mask][0]
        if current_idx + 1 >= len(df):
            return 0.0

        current_close = df.at[current_idx, 'close']
        future_close = df.at[current_idx + 1, 'close']

        return (future_close - current_close) / current_close
    except Exception:
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
    processed_data: dict,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    device,
    start_date=None,
    end_date=None,
    top_k: int = 10,
    rebalance_days: int = 5,
    slippage: float = 0.001,
    atr_multiplier: float = 2.5,
    enable_atr_stops: bool = True,
    enable_drawdown: bool = True,
    enable_regime: bool = True,
    enable_correlation_filter: bool = True,
    max_correlation: float = 0.70,
    enable_vol_weighting: bool = True,
    vol_lookback: int = 20,
):
    """
    Run the full backtest simulation with adaptive risk management.

    Risk Management:
    1. ATR-based trailing stops (volatility-adjusted)
    2. Drawdown-based exposure scaling (portfolio protection)
    3. Market regime filter (SPY trend + VIX)

    Diversification:
    4. Correlation filtering (max pairwise correlation)
    5. Volatility-adjusted position sizing (inverse vol weighting)
    """
    # Get all unique trading dates
    if not stock_data:
        logger.error("No stock data available for backtest")
        return None

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
    logger.info(f"Risk Management: ATR Stops={enable_atr_stops}, Drawdown Scaling={enable_drawdown}, Regime Filter={enable_regime}")
    if enable_atr_stops:
        logger.info(f"  ATR Multiplier: {atr_multiplier}x (adaptive stops)")
    logger.info(f"Diversification: Correlation Filter={enable_correlation_filter}, Vol Weighting={enable_vol_weighting}")
    if enable_correlation_filter:
        logger.info(f"  Max Correlation: {max_correlation}")
    if enable_vol_weighting:
        logger.info(f"  Vol Lookback: {vol_lookback} days")

    # Initialize portfolio
    portfolio_value = CONFIG["INITIAL_CAPITAL"]
    current_holdings = {}  # ticker -> weight
    position_tracker = PositionTracker()

    # Tracking
    portfolio_history = []
    portfolio_values = [portfolio_value]  # For drawdown calculation
    trade_log = []
    turnover_events = []
    risk_exits = {'atr_stop': 0, 'drawdown_scale': 0, 'regime_filter': 0}
    days_since_rebalance = rebalance_days  # Force rebalance on first day

    for day_idx, date in enumerate(all_dates):
        # Progress logging every 50 days
        if day_idx % 50 == 0:
            pct = (day_idx / total_days) * 100
            logger.info(f"Processing day {day_idx+1}/{total_days} ({pct:.0f}%) - {date.strftime('%Y-%m-%d')}...")

        # =====================================================================
        # RISK MANAGEMENT: Calculate exposure multipliers
        # =====================================================================
        exposure_mult, exposure_details = calculate_final_exposure(
            portfolio_values=portfolio_values,
            spy_df=spy_df,
            vix_df=vix_df,
            date=date,
            enable_drawdown=enable_drawdown,
            enable_regime=enable_regime
        )

        # Log regime changes
        if exposure_mult < 1.0:
            regime_label = exposure_details.get('regime_label', 'unknown')
            if day_idx % 10 == 0:  # Log every 10 days to avoid spam
                logger.info(f"  Risk adjustment: Exposure={exposure_mult:.2f} (Regime: {regime_label})")

        # =====================================================================
        # ATR TRAILING STOPS: Check each position
        # =====================================================================
        stocks_to_exit = []

        if enable_atr_stops:
            for ticker in list(current_holdings.keys()):
                if ticker not in stock_data:
                    continue

                # Get current price
                ticker_df = stock_data[ticker]
                current_row = ticker_df[ticker_df['date'] == date]
                if len(current_row) == 0:
                    continue

                current_price = current_row['close'].values[0]

                # Update highest price
                position_tracker.update_highest(ticker, current_price)
                highest_price = position_tracker.get_highest(ticker)

                if highest_price is None:
                    continue

                # Calculate ATR trailing stop
                stop_price = get_atr_trailing_stop(
                    stock_data=stock_data,
                    ticker=ticker,
                    date=date,
                    highest_price=highest_price,
                    multiplier=atr_multiplier
                )

                # Check if stop triggered
                if current_price < stop_price:
                    return_since_entry = (current_price - position_tracker.get_entry_price(ticker)) / position_tracker.get_entry_price(ticker)
                    stocks_to_exit.append((ticker, 'atr_stop', return_since_entry))
                    risk_exits['atr_stop'] += 1

        # Execute ATR stop exits
        for ticker, reason, ret in stocks_to_exit:
            if ticker in current_holdings:
                del current_holdings[ticker]
            position_tracker.remove_position(ticker)

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
                # DIVERSIFICATION: Apply correlation filter
                if enable_correlation_filter:
                    new_top_k = select_with_correlation_filter(
                        stock_data=stock_data,
                        rankings=rankings,
                        date=date,
                        top_k=top_k,
                        max_corr=max_correlation,
                        lookback=60
                    )
                else:
                    # Simple top-K selection without correlation filter
                    sorted_stocks = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
                    new_top_k = [t for t, _ in sorted_stocks[:top_k]]

                sorted_stocks = sorted([(t, rankings[t]) for t in new_top_k], key=lambda x: x[1], reverse=True)

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

                # DIVERSIFICATION: Calculate position weights
                if enable_vol_weighting:
                    # Volatility-adjusted weighting
                    position_weights = calculate_volatility_weights(
                        stock_data=stock_data,
                        holdings=new_top_k,
                        date=date,
                        lookback=vol_lookback
                    )
                else:
                    # Equal weighting
                    position_weights = {t: 1.0 / len(new_top_k) for t in new_top_k}

                # Apply exposure multiplier to all positions
                current_holdings = {ticker: weight * exposure_mult for ticker, weight in position_weights.items()}

                # Track new positions
                for ticker in new_top_k:
                    if ticker in stock_data:
                        ticker_df = stock_data[ticker]
                        entry_row = ticker_df[ticker_df['date'] == date]
                        if len(entry_row) > 0:
                            entry_price = entry_row['close'].values[0]
                            if ticker not in old_holdings:
                                # New position
                                position_tracker.add_position(ticker, entry_price, date)

                # Remove exited positions from tracker
                for ticker in old_holdings - new_holdings:
                    position_tracker.remove_position(ticker)

                trade_log.append({
                    'date': date,
                    'action': 'rebalance',
                    'holdings': list(current_holdings.keys()),
                    'top_predictions': sorted_stocks[:top_k],
                    'turnover': turnover,
                    'exposure_mult': exposure_mult,
                })

                days_since_rebalance = 0
            else:
                days_since_rebalance += 1
        else:
            # Not rebalancing, but still apply exposure scaling to existing holdings
            if current_holdings and (enable_drawdown or enable_regime):
                base_weight = 1.0 / len(current_holdings)
                current_holdings = {ticker: base_weight * exposure_mult for ticker in current_holdings.keys()}

            days_since_rebalance += 1

        # Calculate daily return based on current holdings
        if current_holdings:
            daily_return = 0.0
            for ticker, weight in current_holdings.items():
                if ticker in stock_data:
                    ret = get_next_day_return(stock_data[ticker], date)
                    daily_return += weight * ret

            portfolio_value *= (1 + daily_return)

        # Track portfolio value for drawdown calculation
        portfolio_values.append(portfolio_value)

        portfolio_history.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'n_holdings': len(current_holdings),
            'exposure_mult': exposure_mult,
            'regime': exposure_details.get('regime_label', 'unknown'),
        })

    logger.info(f"Risk Exits: {risk_exits['atr_stop']} ATR Stops")

    return {
        'portfolio_history': pd.DataFrame(portfolio_history),
        'trade_log': trade_log,
        'turnover_events': turnover_events,
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
    parser = argparse.ArgumentParser(description="V9 Backtest with Risk Management & Diversification")
    parser.add_argument("--top-k", type=int, default=10, help="Number of stocks to hold")
    parser.add_argument("--rebalance", type=int, default=5, help="Rebalance every N days")
    parser.add_argument("--slippage", type=float, default=0.001, help="Transaction cost per trade")
    parser.add_argument("--best", action="store_true", help="Use best correlation model")
    parser.add_argument("--atr-mult", type=float, default=2.5, help="ATR trailing stop multiplier")
    parser.add_argument("--no-risk", action="store_true", help="Disable all risk management (for comparison)")
    parser.add_argument("--no-corr", action="store_true", help="Disable correlation filter")
    parser.add_argument("--no-vol-weight", action="store_true", help="Disable volatility weighting")
    parser.add_argument("--max-corr", type=float, default=0.70, help="Maximum pairwise correlation")
    args = parser.parse_args()

    CONFIG["TOP_K"] = args.top_k
    CONFIG["REBALANCE_DAYS"] = args.rebalance
    CONFIG["SLIPPAGE"] = args.slippage
    CONFIG["ATR_MULTIPLIER"] = args.atr_mult
    CONFIG["MAX_CORRELATION"] = args.max_corr

    # Risk management flags
    if args.no_risk:
        CONFIG["ENABLE_ATR_STOPS"] = False
        CONFIG["ENABLE_DRAWDOWN"] = False
        CONFIG["ENABLE_REGIME"] = False
        logger.info("⚠️  ALL RISK MANAGEMENT DISABLED (--no-risk)")

    # Diversification flags
    if args.no_corr:
        CONFIG["ENABLE_CORRELATION_FILTER"] = False
    if args.no_vol_weight:
        CONFIG["ENABLE_VOL_WEIGHTING"] = False

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
        spy_df = load_stock_data('SPY')
        if spy_df is not None:
            stock_data['SPY'] = spy_df

    # Try loading VIX
    vix_df = load_stock_data('^VIX')
    if vix_df is None:
        vix_df = load_stock_data('VIX')

    spy_data = process_spy_data(spy_df) if spy_df is not None else None
    vix_data = process_vix_data(vix_df) if vix_df is not None else None

    # Ensure all data has numeric indices
    for df in stock_data.values():
        df.reset_index(drop=True, inplace=True)

    # PRE-COMPUTE FEATURES
    processed_data = precompute_model_data(stock_data, spy_data, vix_data, scaler)
    logger.info(f"Optimized features computed for {len(processed_data)} stocks")

    # Run backtest
    logger.info("\n" + "="*70)
    logger.info("RUNNING V9 BACKTEST WITH RISK MANAGEMENT & DIVERSIFICATION")
    logger.info("="*70)

    results = run_backtest(
        model=model,
        stock_data=stock_data,
        processed_data=processed_data,
        spy_df=spy_df,
        vix_df=vix_df,
        device=device,
        top_k=args.top_k,
        rebalance_days=args.rebalance,
        slippage=args.slippage,
        atr_multiplier=args.atr_mult,
        enable_atr_stops=CONFIG["ENABLE_ATR_STOPS"],
        enable_drawdown=CONFIG["ENABLE_DRAWDOWN"],
        enable_regime=CONFIG["ENABLE_REGIME"],
        enable_correlation_filter=CONFIG["ENABLE_CORRELATION_FILTER"],
        max_correlation=CONFIG["MAX_CORRELATION"],
        enable_vol_weighting=CONFIG["ENABLE_VOL_WEIGHTING"],
        vol_lookback=CONFIG["VOL_LOOKBACK"],
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
    logger.info("📊 V9 BACKTEST RESULTS (WITH RISK MANAGEMENT)")
    logger.info("="*70)
    logger.info(f"Strategy: Top-{args.top_k} by Predicted Rank, Rebalance every {args.rebalance} days")
    logger.info(f"Transaction Cost: {args.slippage*100:.2f}%")
    logger.info(f"ATR Multiplier: {args.atr_mult}x")
    logger.info("")

    logger.info("--- V9 RISK-MANAGED STRATEGY ---")
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
    results_path = backend_path / "backtest_v9_risk_results.csv"
    portfolio_history.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to {results_path}")

    # Show sample trades
    if results['trade_log']:
        logger.info("\n📋 Sample Trades (First 5 Rebalances):")
        for trade in [t for t in results['trade_log'] if t['action'] == 'rebalance'][:5]:
            exposure = trade.get('exposure_mult', 1.0)
            logger.info(f"  {trade['date'].strftime('%Y-%m-%d')}: {trade['holdings'][:5]}... (turnover: {trade['turnover']*100:.1f}%, exposure: {exposure*100:.0f}%)")


if __name__ == "__main__":
    main()
