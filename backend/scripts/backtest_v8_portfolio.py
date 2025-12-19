#!/usr/bin/env python3
"""
V8 Portfolio Backtest - Realistic Trading Simulation with Risk Management

Simulates a portfolio using V8 classification model predictions.

RISK MANAGEMENT FEATURES:
1. SMA 200 Trend Filter: Only buy when Price > 200-day SMA (uptrend)
2. RSI 14 Overbought Filter: Only buy when RSI < 70 (not overbought)
3. Trailing Stop Loss: Exit if price drops 5% from highest since entry
4. Time Exit: Hold for max 21 days

Strategy:
- "Top 5 Picks": Every day, rank stocks by buy probability
- Only enter trades in UPTREND (Price > SMA 200)
- Protect profits with 10% trailing stop
- Compare to SPY Buy & Hold baseline

Usage:
    docker exec proxmox_stock_backend python -m scripts.backtest_v8_portfolio
    docker exec proxmox_stock_backend python -m scripts.backtest_v8_portfolio --threshold 0.65
    docker exec proxmox_stock_backend python -m scripts.backtest_v8_portfolio --no-trend-filter
"""

import sys
import os
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import torch
import pickle

backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from services.lstm_model_v8_class import LSTMModelV8Class
from services.feature_engineering import process_stock_data, get_model_input_features_v7

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Config
CONFIG = {
    "WINDOW_SIZE": 60,
    "HOLD_DAYS": 21,  # 1 month horizon
    "INITIAL_CAPITAL": 10000,
    "MAX_POSITIONS": 5,
    "TRAIN_CUTOFF": 0.80,
    "TOP_STOCKS": 50,
    "SMA_PERIOD": 200,        # Trend filter
    "RSI_PERIOD": 14,         # RSI for overbought filter
    "RSI_OVERBOUGHT": 70,     # Reject if RSI > 70
    "TRAILING_STOP": 0.05,    # 5% trailing stop (tightened from 10%)
}

MODEL_PATH = backend_path / "models" / "lstm_model_v8_class.pth"
SCALER_PATH = backend_path / "models" / "scaler_v8_class.pkl"


@dataclass
class Position:
    """Represents an open position."""
    ticker: str
    entry_date: datetime
    entry_price: float
    shares: int
    entry_probability: float
    high_since_entry: float = 0.0  # For trailing stop
    
    def __post_init__(self):
        self.high_since_entry = self.entry_price
    
    def days_held(self, current_date: datetime) -> int:
        return (current_date - self.entry_date).days
    
    def update_high(self, current_price: float):
        if current_price > self.high_since_entry:
            self.high_since_entry = current_price
    
    def trailing_stop_price(self) -> float:
        return self.high_since_entry * (1 - CONFIG["TRAILING_STOP"])


@dataclass
class Trade:
    """Completed trade record."""
    ticker: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    hold_days: int
    exit_reason: str


class Portfolio:
    """Portfolio manager for backtesting."""
    
    def __init__(self, initial_capital: float, max_positions: int):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_positions = max_positions
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
    def available_slots(self) -> int:
        return self.max_positions - len(self.positions)
    
    def position_size(self) -> float:
        return self.initial_capital / self.max_positions
    
    def total_equity(self, current_prices: Dict[str, float]) -> float:
        positions_value = sum(
            pos.shares * current_prices.get(pos.ticker, pos.entry_price)
            for pos in self.positions.values()
        )
        return self.cash + positions_value
    
    def open_position(self, ticker: str, date: datetime, price: float, probability: float):
        if ticker in self.positions:
            return False
        if self.available_slots() <= 0:
            return False
        
        allocation = self.position_size()
        shares = int(allocation / price)
        if shares <= 0:
            return False
        
        cost = shares * price
        if cost > self.cash:
            return False
        
        self.cash -= cost
        self.positions[ticker] = Position(
            ticker=ticker,
            entry_date=date,
            entry_price=price,
            shares=shares,
            entry_probability=probability
        )
        return True
    
    def close_position(self, ticker: str, date: datetime, price: float, reason: str):
        if ticker not in self.positions:
            return None
        
        pos = self.positions.pop(ticker)
        proceeds = pos.shares * price
        self.cash += proceeds
        
        pnl = proceeds - (pos.shares * pos.entry_price)
        pnl_pct = (price / pos.entry_price - 1) * 100
        
        trade = Trade(
            ticker=ticker,
            entry_date=pos.entry_date,
            exit_date=date,
            entry_price=pos.entry_price,
            exit_price=price,
            shares=pos.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            hold_days=pos.days_held(date),
            exit_reason=reason
        )
        self.trades.append(trade)
        return trade
    
    def record_equity(self, date: datetime, current_prices: Dict[str, float]):
        equity = self.total_equity(current_prices)
        self.equity_curve.append((date, equity))


def get_db_url():
    url = os.getenv("DATABASE_URL", "")
    return url.replace("postgresql+asyncpg://", "postgresql://")


def get_top_tickers(n: int = 50) -> List[str]:
    """Get top N most liquid tickers."""
    import psycopg2
    try:
        with psycopg2.connect(get_db_url()) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ticker, COUNT(*) as count 
                    FROM stock_prices 
                    GROUP BY ticker 
                    HAVING COUNT(*) >= 500 
                    ORDER BY count DESC 
                    LIMIT %s
                """, (n,))
                results = [row[0] for row in cur.fetchall()]
                return results
    except Exception as e:
        logger.error(f"Error fetching top tickers: {e}")
        return []


def load_stock_data(ticker: str) -> Optional[pd.DataFrame]:
    """Load full OHLCV history for a ticker."""
    import psycopg2
    try:
        with psycopg2.connect(get_db_url()) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT timestamp, open, high, low, close, volume 
                    FROM stock_prices 
                    WHERE ticker = %s 
                    ORDER BY timestamp
                """, (ticker,))
                records = cur.fetchall()
                
                if records:
                    df = pd.DataFrame(records, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                    df['date'] = pd.to_datetime(df['date'])
                    return df
    except Exception as e:
        logger.error(f"Error loading stock data for {ticker}: {e}")
    return None


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def prepare_stock_features(df: pd.DataFrame, feature_cols: list, scaler: StandardScaler) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Prepare features for the entire stock history.
    Also calculates SMA 200 and RSI 14 for filters.
    """
    try:
        processed = process_stock_data(df.copy(), create_targets=True)
    except Exception as e:
        logger.error(f"Error preparing features for stock: {e}", exc_info=True)
        return None, None
    
    # Calculate SMA 200 for trend filter
    processed['SMA_200'] = processed['close'].rolling(window=CONFIG["SMA_PERIOD"], min_periods=1).mean()
    
    # Calculate RSI 14 for overbought filter
    processed['RSI_14'] = calculate_rsi(processed['close'], CONFIG["RSI_PERIOD"])
    
    for col in feature_cols:
        if col not in processed.columns:
            processed[col] = 0.0
    
    features = processed[feature_cols].values
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    features_scaled = scaler.transform(features)
    features_scaled = np.clip(features_scaled, -10.0, 10.0)
    
    return features_scaled.astype(np.float32), processed


def run_backtest(
    threshold: float = 0.75,
    top_k: int = 5,
    horizon: str = "1m",
    use_trend_filter: bool = True,
    use_trailing_stop: bool = True
):
    """
    Run portfolio backtest with risk management.
    """
    logger.info("=" * 70)
    logger.info("🎯 V8 PORTFOLIO BACKTEST (with Risk Management)")
    logger.info("=" * 70)
    logger.info(f"Threshold: {threshold}")
    logger.info(f"Top K picks: {top_k}")
    logger.info(f"Horizon: {horizon}")
    logger.info(f"Hold days: {CONFIG['HOLD_DAYS']}")
    logger.info(f"Trend Filter (SMA 200): {'ON' if use_trend_filter else 'OFF'}")
    logger.info(f"Trailing Stop (10%): {'ON' if use_trailing_stop else 'OFF'}")
    
    # Load model
    if not MODEL_PATH.exists():
        logger.error(f"Model not found: {MODEL_PATH}")
        return
    
    model = LSTMModelV8Class.load(str(MODEL_PATH))
    model.eval()
    device = model.device
    logger.info(f"Model loaded on {device}")
    
    # Load scaler
    if not SCALER_PATH.exists():
        logger.error(f"Scaler not found at {SCALER_PATH}")
        return

    with open(SCALER_PATH, 'rb') as f:
        scaler_data = pickle.load(f)
        scaler = scaler_data['scaler']
        feature_cols = scaler_data['feature_names']
    
    # Horizon index
    horizon_map = {"1d": 0, "1w": 1, "1m": 2, "6m": 3}
    horizon_idx = horizon_map.get(horizon, 2)
    
    # Get tickers
    tickers = get_top_tickers(CONFIG["TOP_STOCKS"])
    logger.info(f"Using {len(tickers)} stocks")
    
    # Prepare all stock data
    logger.info("\n📊 Preparing stock data (with SMA 200)...")
    stock_data = {}
    for ticker in tickers:
        df = load_stock_data(ticker)
        if df is None or len(df) < CONFIG["WINDOW_SIZE"] + CONFIG["SMA_PERIOD"]:
            continue
        
        features, processed = prepare_stock_features(df, feature_cols, scaler)
        if features is None:
            continue
        
        # Get validation period (last 20%)
        val_start = int(len(processed) * CONFIG["TRAIN_CUTOFF"])
        
        stock_data[ticker] = {
            "features": features,
            "processed": processed,
            "val_start": val_start
        }
    
    logger.info(f"Prepared {len(stock_data)} stocks")
    
    if len(stock_data) < 5:
        logger.error("Not enough stocks with valid data")
        return
    
    # Get common date range for validation
    all_val_dates = []
    for ticker, data in stock_data.items():
        val_dates = data["processed"]["date"].iloc[data["val_start"]:].tolist()
        all_val_dates.extend(val_dates)
    
    if not all_val_dates:
        logger.error("No validation dates found")
        return
    
    min_date = min(all_val_dates)
    max_date = max(all_val_dates)
    logger.info(f"Validation period: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    
    # Build date index
    date_range = pd.date_range(min_date, max_date, freq='D')
    trading_days = []
    
    for date in date_range:
        count = sum(1 for data in stock_data.values() 
                   if date in data["processed"]["date"].values)
        if count >= 10:
            trading_days.append(date)
    
    logger.info(f"Trading days in validation: {len(trading_days)}")
    
    # Initialize portfolio
    portfolio = Portfolio(CONFIG["INITIAL_CAPITAL"], CONFIG["MAX_POSITIONS"])
    
    # Counters for analysis
    signals_passed_trend = 0
    signals_blocked_trend = 0
    signals_blocked_rsi = 0
    stops_triggered = 0
    
    # Load SPY for comparison
    spy_data = load_stock_data("SPY")
    if spy_data is not None:
        spy_data = spy_data.set_index('date')
    
    # Run simulation
    logger.info("\n🚀 Running simulation...")
    
    for i, date in enumerate(trading_days):
        if i < CONFIG["WINDOW_SIZE"]:
            continue
        
        if (i + 1) % 100 == 0:
            logger.info(f"  Day {i+1}/{len(trading_days)}...")
        
        current_prices = {}
        daily_signals = []
        
        # Get predictions and prices for all stocks
        for ticker, data in stock_data.items():
            proc = data["processed"]
            features = data["features"]
            
            # Find index for current date
            date_matches = proc[proc["date"] == date]
            if len(date_matches) == 0:
                continue
            
            idx = date_matches.index[0]
            if idx < CONFIG["WINDOW_SIZE"]:
                continue
            
            # Get current price, SMA, and RSI
            price = proc.loc[idx, "close"]
            sma_200 = proc.loc[idx, "SMA_200"]
            rsi_14 = proc.loc[idx, "RSI_14"] if "RSI_14" in proc.columns else 50
            current_prices[ticker] = price
            
            # Update trailing stop high for existing positions
            if ticker in portfolio.positions:
                portfolio.positions[ticker].update_high(price)
            
            # Get window for prediction
            window = features[idx - CONFIG["WINDOW_SIZE"]:idx]
            if len(window) < CONFIG["WINDOW_SIZE"]:
                continue
            
            # Run inference
            with torch.no_grad():
                x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
            
            prob = probs[horizon_idx]
            
            if prob >= threshold:
                # FILTER 1: Only buy in uptrend (Price > SMA 200)
                if use_trend_filter and price <= sma_200:
                    signals_blocked_trend += 1
                    continue
                
                # FILTER 2: Only buy if not overbought (RSI < 70)
                if rsi_14 >= CONFIG["RSI_OVERBOUGHT"]:
                    signals_blocked_rsi += 1
                    continue
                
                # Passed all filters
                signals_passed_trend += 1
                daily_signals.append({
                    "ticker": ticker,
                    "probability": prob,
                    "price": price,
                    "rsi": rsi_14
                })
        
        # Sort by probability
        daily_signals.sort(key=lambda x: x["probability"], reverse=True)
        
        # === EXIT LOGIC ===
        tickers_to_close = []
        
        for ticker, pos in portfolio.positions.items():
            current_price = current_prices.get(ticker, pos.entry_price)
            
            # Exit 1: Trailing Stop Loss
            if use_trailing_stop:
                stop_price = pos.trailing_stop_price()
                if current_price < stop_price:
                    tickers_to_close.append((ticker, "stop_loss"))
                    stops_triggered += 1
                    continue
            
            # Exit 2: Time-based exit
            if pos.days_held(date) >= CONFIG["HOLD_DAYS"]:
                tickers_to_close.append((ticker, "time_exit"))
        
        for ticker, reason in tickers_to_close:
            price = current_prices.get(ticker, portfolio.positions[ticker].entry_price)
            portfolio.close_position(ticker, date, price, reason)
        
        # Open new positions (top K picks that passed trend filter)
        for signal in daily_signals[:top_k]:
            if portfolio.available_slots() > 0:
                if signal["ticker"] not in portfolio.positions:
                    portfolio.open_position(
                        signal["ticker"],
                        date,
                        signal["price"],
                        signal["probability"]
                    )
        
        # Record equity
        portfolio.record_equity(date, current_prices)
    
    # Close all remaining positions
    final_date = trading_days[-1]
    for ticker in list(portfolio.positions.keys()):
        price = current_prices.get(ticker, portfolio.positions[ticker].entry_price)
        portfolio.close_position(ticker, final_date, price, "end_of_backtest")
    
    # === RESULTS ===
    logger.info("\n" + "=" * 70)
    logger.info("📊 BACKTEST RESULTS")
    logger.info("=" * 70)
    
    if not portfolio.equity_curve:
        logger.error("No equity data recorded")
        return
    
    # V8 Strategy results
    initial = CONFIG["INITIAL_CAPITAL"]
    final = portfolio.total_equity(current_prices)
    total_return = (final / initial - 1) * 100
    
    # Calculate drawdown
    equity_values = [e[1] for e in portfolio.equity_curve]
    peak = equity_values[0]
    max_drawdown = 0
    for val in equity_values:
        if val > peak:
            peak = val
        drawdown = (peak - val) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # Trade statistics
    num_trades = len(portfolio.trades)
    if num_trades > 0:
        wins = sum(1 for t in portfolio.trades if t.pnl > 0)
        win_rate = wins / num_trades * 100
        avg_pnl = np.mean([t.pnl_pct for t in portfolio.trades])
        avg_win = np.mean([t.pnl_pct for t in portfolio.trades if t.pnl > 0]) if wins > 0 else 0
        avg_loss = np.mean([t.pnl_pct for t in portfolio.trades if t.pnl <= 0]) if (num_trades - wins) > 0 else 0
    else:
        win_rate = avg_pnl = avg_win = avg_loss = 0
    
    # Exit reason breakdown
    exit_reasons = {}
    for t in portfolio.trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
    
    # SPY comparison
    if spy_data is not None:
        try:
            spy_start = spy_data.loc[trading_days[CONFIG["WINDOW_SIZE"]], "close"]
            spy_end = spy_data.loc[final_date, "close"]
            spy_return = (spy_end / spy_start - 1) * 100
        except:
            spy_return = 0
    else:
        spy_return = 0
    
    # Print results
    print("\n" + "=" * 60)
    print("V8 CLASSIFICATION STRATEGY (with Risk Management)")
    print("=" * 60)
    print(f"Initial Capital:     ${initial:,.2f}")
    print(f"Final Equity:        ${final:,.2f}")
    print(f"Total Return:        {total_return:+.2f}%")
    print(f"Max Drawdown:        {max_drawdown:.2f}%")
    print(f"Number of Trades:    {num_trades}")
    print(f"Win Rate:            {win_rate:.1f}%")
    print(f"Avg Trade P/L:       {avg_pnl:+.2f}%")
    print(f"Avg Win:             {avg_win:+.2f}%")
    print(f"Avg Loss:            {avg_loss:+.2f}%")
    
    print("\n" + "-" * 60)
    print("RISK MANAGEMENT STATS")
    print("-" * 60)
    print(f"Trend Filter (SMA 200):        {'ON' if use_trend_filter else 'OFF'}")
    print(f"RSI Overbought Filter (<70):   ON")
    print(f"Trailing Stop (5%):            {'ON' if use_trailing_stop else 'OFF'}")
    print(f"  Signals Passed All Filters:  {signals_passed_trend}")
    print(f"  Blocked by Downtrend:        {signals_blocked_trend}")
    print(f"  Blocked by Overbought RSI:   {signals_blocked_rsi}")
    print(f"Trailing Stop:")
    print(f"  Stops Triggered:   {stops_triggered}")
    print(f"Exit Reasons:        {exit_reasons}")
    
    print("\n" + "-" * 60)
    print("BENCHMARK COMPARISON")
    print("-" * 60)
    print(f"SPY Buy & Hold:      {spy_return:+.2f}%")
    print(f"Alpha (vs SPY):      {total_return - spy_return:+.2f}%")
    
    print("\n" + "=" * 60)
    
    # Show top trades
    if portfolio.trades:
        print("\nTOP 5 WINNING TRADES:")
        sorted_trades = sorted(portfolio.trades, key=lambda x: x.pnl_pct, reverse=True)
        for t in sorted_trades[:5]:
            print(f"  {t.ticker}: {t.pnl_pct:+.1f}% over {t.hold_days}d ({t.exit_reason})")
        
        print("\nTOP 5 LOSING TRADES:")
        for t in sorted_trades[-5:]:
            print(f"  {t.ticker}: {t.pnl_pct:+.1f}% over {t.hold_days}d ({t.exit_reason})")
    
    return portfolio


def main():
    parser = argparse.ArgumentParser(description="V8 Portfolio Backtest with Risk Management")
    parser.add_argument("--threshold", type=float, default=0.75, help="Buy probability threshold")
    parser.add_argument("--top-k", type=int, default=5, help="Top K picks per day")
    parser.add_argument("--horizon", type=str, default="1m", choices=["1d", "1w", "1m", "6m"])
    parser.add_argument("--no-trend-filter", action="store_true", help="Disable SMA 200 trend filter")
    parser.add_argument("--no-trailing-stop", action="store_true", help="Disable trailing stop loss")
    args = parser.parse_args()
    
    run_backtest(
        threshold=args.threshold,
        top_k=args.top_k,
        horizon=args.horizon,
        use_trend_filter=not args.no_trend_filter,
        use_trailing_stop=not args.no_trailing_stop
    )


if __name__ == "__main__":
    main()
