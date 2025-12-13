#!/usr/bin/env python3
"""
Multi-Portfolio Seed Script - V9 Parameter Comparison

Creates 7 different portfolio simulations to compare V9 behavior under:
1. Different parameter configurations (aggressive, conservative, etc.)
2. Different historical periods (2008 crisis, 2022 bear market)

Usage:
    python -m scripts.seed_multi_portfolio --all           # Run all 7 portfolios
    python -m scripts.seed_multi_portfolio --config golden # Run specific config
    python -m scripts.seed_multi_portfolio --list          # List available configs
"""

import asyncio
import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
import pickle

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sqlalchemy import select, delete

# Setup path
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from services.db import (
    Base, PaperPortfolio, PaperHolding, PaperTrade, 
    PaperPortfolioHistory, init_db, AsyncSessionLocal
)
from services.feature_engineering_v9 import (
    compute_v9_features, process_spy_data, process_vix_data, V9_FEATURE_NAMES
)
from services.transformer_model import TransformerRankModel, get_device

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(BACKEND_DIR / "seed_multi_portfolio.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Model paths
MODEL_PATH = BACKEND_DIR / "models" / "transformer_v9_best.pth"
SCALER_PATH = BACKEND_DIR / "models" / "scaler_v9.pkl"

# =============================================================================
# PORTFOLIO CONFIGURATIONS
# =============================================================================

PORTFOLIO_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Current production config (baseline) - Recent 5 years
    "golden_2020": {
        "display_name": "🏆 Golden Config (Recent)",
        "description": "Current production settings on recent data (2020-2024).",
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
        "vix_threshold": 30.0,
        "top_k": 10,
        "correlation_threshold": 0.60,
        "rebalance_days": 5,
        "start_capital": 10000.0,
    },
    
    # Aggressive - Higher risk tolerance
    "aggressive": {
        "display_name": "🔥 Aggressive",
        "description": "Higher VIX tolerance (40), more holdings (15), looser correlation (0.80).",
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
        "vix_threshold": 40.0,  # Stay invested during moderate panic
        "top_k": 15,            # More diversified
        "correlation_threshold": 0.80,  # Allow more correlated holdings
        "rebalance_days": 5,
        "start_capital": 10000.0,
    },
    
    # Conservative - Lower risk tolerance
    "conservative": {
        "display_name": "🛡️ Conservative",
        "description": "Exit earlier (VIX>25), fewer holdings (5), strict diversification (0.40).",
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
        "vix_threshold": 25.0,  # Exit at first sign of fear
        "top_k": 5,             # Concentrated bets
        "correlation_threshold": 0.40,  # Very strict diversification
        "rebalance_days": 5,
        "start_capital": 10000.0,
    },
    
    # No VIX Filter - Full exposure to crashes
    "no_vix": {
        "display_name": "⚠️ No VIX Filter",
        "description": "VIX filter disabled. Full exposure to market crashes.",
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
        "vix_threshold": 999.0,  # Effectively disabled
        "top_k": 10,
        "correlation_threshold": 0.60,
        "rebalance_days": 5,
        "start_capital": 10000.0,
    },
    
    # Daily Rebalancing - More frequent trading
    "daily": {
        "display_name": "⚡ Daily Rebalance",
        "description": "Rebalance every day instead of weekly. Higher turnover.",
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
        "vix_threshold": 30.0,
        "top_k": 10,
        "correlation_threshold": 0.60,
        "rebalance_days": 1,  # Daily
        "start_capital": 10000.0,
    },
    
    # COVID CRASH - March 2020 in the MIDDLE (2018-2023)
    "covid_2020": {
        "display_name": "🦠 COVID Crash (2018-2023)",
        "description": "5 years with March 2020 crash in the MIDDLE. Tests V9 panic response.",
        "start_date": "2018-01-01",
        "end_date": "2023-12-31",
        "vix_threshold": 30.0,
        "top_k": 10,
        "correlation_threshold": 0.60,
        "rebalance_days": 5,
        "start_capital": 10000.0,
    },
    
    # 2008 Financial Crisis (centered) - 2006-2011
    "crisis_2008": {
        "display_name": "💥 2008 Crisis (2006-2011)",
        "description": "5 years with 2008-2009 crash in MIDDLE. Worst since Great Depression.",
        "start_date": "2006-01-01",
        "end_date": "2011-12-31",
        "vix_threshold": 30.0,
        "top_k": 10,
        "correlation_threshold": 0.60,
        "rebalance_days": 5,
        "start_capital": 10000.0,
    },
    
    # 2022 Bear Market (centered) - 2020-2025
    "bear_2022": {
        "display_name": "🐻 2022 Bear (2020-2025)",
        "description": "5 years with 2022 tech bear market in MIDDLE. COVID recovery → bear → recovery.",
        "start_date": "2020-01-01",
        "end_date": "2025-12-12",
        "vix_threshold": 30.0,
        "top_k": 10,
        "correlation_threshold": 0.60,
        "rebalance_days": 5,
        "start_capital": 10000.0,
    },
    
    # 20-YEAR ULTIMATE TEST - Covers 2008, 2020, 2022
    "ultimate_20yr": {
        "display_name": "📊 20-Year Ultimate (2005-2024)",
        "description": "20 years covering 2008 crisis, 2020 COVID crash, AND 2022 bear market.",
        "start_date": "2005-01-01",
        "end_date": "2024-12-31",
        "vix_threshold": 30.0,
        "top_k": 10,
        "correlation_threshold": 0.60,
        "rebalance_days": 5,
        "start_capital": 10000.0,
    },
}

# Stock universe (same as main seed script)
TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "INTC", "CSCO", "ORCL",
    # Finance
    "JPM", "BAC", "GS", "MS", "WFC", "V", "MA", "AXP", "BLK", "SCHW",
    # Consumer
    "WMT", "COST", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "DIS", "PG",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "DHR", "BMY",
    # Industrial
    "CAT", "BA", "GE", "HON", "UPS", "RTX", "LMT", "MMM", "DE", "UNP",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "PSX", "VLO", "MPC", "HAL",
]


def fetch_yfinance_data(tickers: List[str], start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
    """Bulk fetch data from yfinance."""
    logger.info(f"Fetching {len(tickers)} tickers from {start.date()} to {end.date()}...")
    
    try:
        data = yf.download(
            tickers,
            start=start,
            end=end,
            progress=False,
            group_by='ticker',
            auto_adjust=True,
            multi_level_index=False
        )
        
        result = {}
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    df = data.copy()
                else:
                    df = data[ticker].copy()
                
                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]
                df = df.rename(columns={'adj close': 'close'})
                df = df.dropna(subset=['close'])
                
                if len(df) > 60:
                    result[ticker] = df
            except Exception as e:
                logger.warning(f"Could not process {ticker}: {e}")
        
        logger.info(f"Successfully fetched {len(result)}/{len(tickers)} tickers")
        return result
        
    except Exception as e:
        logger.error(f"Bulk download failed: {e}")
        return {}


async def run_portfolio_simulation(
    config_name: str,
    config: Dict[str, Any],
    model: TransformerRankModel,
    scaler: Any,
    device: torch.device
) -> Dict[str, Any]:
    """Run a single portfolio simulation with given config."""
    
    session_id = f"v9_{config_name}"
    logger.info(f"\n{'='*70}")
    logger.info(f"STARTING: {config['display_name']}")
    logger.info(f"Session: {session_id}")
    logger.info(f"Period: {config['start_date']} to {config['end_date']}")
    logger.info(f"VIX: {config['vix_threshold']}, Top-K: {config['top_k']}, Corr: {config['correlation_threshold']}")
    logger.info(f"{'='*70}")
    
    start_date = datetime.strptime(config['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(config['end_date'], '%Y-%m-%d')
    
    # Fetch data
    stock_data = fetch_yfinance_data(TICKERS, start_date - timedelta(days=100), end_date)
    
    if len(stock_data) < 10:
        logger.error(f"Insufficient data for {config_name}")
        return {"error": "Insufficient data"}
    
    # Fetch SPY and VIX
    spy_df = yf.download("SPY", start=start_date - timedelta(days=100), end=end_date, progress=False, auto_adjust=True, multi_level_index=False)
    vix_df = yf.download("^VIX", start=start_date - timedelta(days=100), end=end_date, progress=False, auto_adjust=True, multi_level_index=False)
    
    spy_df = spy_df.reset_index()
    spy_df.columns = [c.lower() for c in spy_df.columns]
    vix_df = vix_df.reset_index()
    vix_df.columns = [c.lower() for c in vix_df.columns]
    
    # Process SPY/VIX
    # Process SPY/VIX
    if spy_df.empty:
        logger.error("⚠️ CRITICAL: SPY data fetch failed! Simulation will be invalid.")
        spy_data = None
    else:
        spy_data = process_spy_data(spy_df)
        
    if vix_df.empty:
        logger.warning("⚠️ VIX data fetch failed! Using default VIX=20.")
        vix_data = None
    else:
        vix_data = process_vix_data(vix_df)
    
    # Get trading days from SPY
    trading_days = spy_df['date'].tolist()
    trading_days = [d for d in trading_days if start_date <= d <= end_date]
    
    async with AsyncSessionLocal() as session:
        try:
            # Cleanup existing data for this session
            await session.execute(delete(PaperPortfolioHistory).where(PaperPortfolioHistory.session_id == session_id))
            await session.execute(delete(PaperTrade).where(PaperTrade.session_id == session_id))
            await session.execute(delete(PaperHolding).where(PaperHolding.session_id == session_id))
            await session.execute(delete(PaperPortfolio).where(PaperPortfolio.session_id == session_id))
            await session.commit()
            
            # Initialize portfolio state
            current_cash = config['start_capital']
            holdings = {}
            days_since_rebalance = config['rebalance_days']  # Ready to rebalance immediately
            
            total_trades = 0
            max_value = config['start_capital']
            min_value = config['start_capital']
            
            # Pre-compute features for all stocks
            processed_data = {}
            for ticker, df in stock_data.items():
                try:
                    features_df = compute_v9_features(df, spy_data, vix_data)
                    features_df = features_df.dropna(subset=V9_FEATURE_NAMES)
                    if len(features_df) >= 60:
                        processed_data[ticker] = {
                            'features': features_df[V9_FEATURE_NAMES].values,
                            'dates': features_df['date'].tolist(),
                            'closes': features_df['close'].tolist()
                        }
                except Exception as e:
                    pass
            
            logger.info(f"Processed {len(processed_data)} stocks for simulation")
            
            # Main simulation loop
            for i, current_date in enumerate(trading_days):
                if i % 50 == 0:
                    logger.info(f"Day {i}/{len(trading_days)} - {current_date.strftime('%Y-%m-%d')}")
                
                # Get current prices
                current_prices = {}
                for ticker, data in processed_data.items():
                    try:
                        idx = data['dates'].index(current_date)
                        current_prices[ticker] = data['closes'][idx]
                    except (ValueError, IndexError):
                        pass
                
                # Calculate portfolio value
                equity_value = sum(
                    holdings[t]['qty'] * current_prices.get(t, holdings[t]['entry_price'])
                    for t in holdings
                )
                total_value = current_cash + equity_value
                max_value = max(max_value, total_value)
                min_value = min(min_value, total_value)
                
                # Get VIX level for this day
                vix_level = 20.0
                if vix_data is not None:
                    try:
                        vix_level = vix_data.get(current_date, 0.20) * 100
                    except:
                        pass
                
                # Check VIX threshold
                if vix_level > config['vix_threshold']:
                    # CRISIS MODE - Go to cash
                    if holdings:
                        logger.info(f"  🛑 VIX={vix_level:.1f} > {config['vix_threshold']} - Going to cash")
                        for ticker in list(holdings.keys()):
                            if ticker in current_prices:
                                qty = holdings[ticker]['qty']
                                price = current_prices[ticker]
                                current_cash += qty * price
                                
                                trade = PaperTrade(
                                    session_id=session_id,
                                    trade_date=current_date,
                                    action="SELL",
                                    ticker=ticker,
                                    price=price,
                                    quantity=qty,
                                    reason=f"VIX Crisis ({vix_level:.0f})",
                                    profit_loss=(price - holdings[ticker]['entry_price']) * qty
                                )
                                session.add(trade)
                                total_trades += 1
                                del holdings[ticker]
                    
                    days_since_rebalance = 0
                
                # Normal rebalancing (if VIX is OK)
                elif days_since_rebalance >= config['rebalance_days']:
                    days_since_rebalance = 0
                    
                    # Run V9 inference
                    rankings = {}
                    for ticker, data in processed_data.items():
                        try:
                            dates = data['dates']
                            if current_date in dates:
                                idx = dates.index(current_date)
                                if idx >= 59:
                                    window = data['features'][idx-59:idx+1]
                                    if len(window) == 60:
                                        if scaler:
                                            window = scaler.transform(window)
                                        tensor = torch.from_numpy(window).float().unsqueeze(0).to(device)
                                        with torch.no_grad():
                                            score = model(tensor).cpu().item()
                                        rankings[ticker] = score
                        except:
                            pass
                    
                    if len(rankings) >= config['top_k']:
                        # Get top K by rank
                        sorted_tickers = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
                        top_tickers = [t for t, _ in sorted_tickers[:config['top_k']]]
                        
                        # Sell tickers no longer in top K
                        for ticker in list(holdings.keys()):
                            if ticker not in top_tickers and ticker in current_prices:
                                qty = holdings[ticker]['qty']
                                price = current_prices[ticker]
                                pnl = (price - holdings[ticker]['entry_price']) * qty
                                current_cash += qty * price
                                
                                trade = PaperTrade(
                                    session_id=session_id,
                                    trade_date=current_date,
                                    action="SELL",
                                    ticker=ticker,
                                    price=price,
                                    quantity=qty,
                                    reason=f"Rank dropped",
                                    profit_loss=pnl
                                )
                                session.add(trade)
                                total_trades += 1
                                del holdings[ticker]
                        
                        # Buy new top tickers
                        target_allocation = total_value / config['top_k']
                        for ticker in top_tickers:
                            if ticker not in holdings and ticker in current_prices:
                                price = current_prices[ticker]
                                if current_cash >= target_allocation * 0.9:
                                    qty = int(target_allocation / price)
                                    if qty > 0:
                                        cost = qty * price
                                        current_cash -= cost
                                        holdings[ticker] = {'qty': qty, 'entry_price': price}
                                        
                                        rank_pos = top_tickers.index(ticker) + 1
                                        trade = PaperTrade(
                                            session_id=session_id,
                                            trade_date=current_date,
                                            action="BUY",
                                            ticker=ticker,
                                            price=price,
                                            quantity=qty,
                                            reason=f"Rank #{rank_pos}",
                                            profit_loss=0
                                        )
                                        session.add(trade)
                                        total_trades += 1
                else:
                    days_since_rebalance += 1
                
                # Save daily history
                hist_entry = PaperPortfolioHistory(
                    session_id=session_id,
                    date=current_date,
                    total_value=total_value,
                    equity_value=equity_value,
                    cash_balance=current_cash
                )
                session.add(hist_entry)
            
            # Final state
            final_equity = sum(
                holdings[t]['qty'] * current_prices.get(t, holdings[t]['entry_price'])
                for t in holdings
            )
            final_value = current_cash + final_equity
            
            # Calculate metrics
            total_return = ((final_value / config['start_capital']) - 1) * 100
            max_drawdown = ((max_value - min_value) / max_value) * 100
            
            logger.info(f"\n{'='*70}")
            logger.info(f"COMPLETED: {config['display_name']}")
            logger.info(f"Final Value: ${final_value:,.2f}")
            logger.info(f"Total Return: {total_return:+.2f}%")
            logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
            logger.info(f"Total Trades: {total_trades}")
            logger.info(f"{'='*70}")
            
            # Save final portfolio
            await session.execute(delete(PaperPortfolio).where(PaperPortfolio.session_id == session_id))
            await session.execute(delete(PaperHolding).where(PaperHolding.session_id == session_id))
            
            portfolio = PaperPortfolio(
                session_id=session_id,
                cash_balance=current_cash,
                equity_value=final_equity,
                total_value=final_value,
                days_since_rebalance=days_since_rebalance
            )
            session.add(portfolio)
            
            for ticker, pos in holdings.items():
                if ticker in current_prices:
                    holding = PaperHolding(
                        session_id=session_id,
                        ticker=ticker,
                        quantity=pos['qty'],
                        entry_price=pos['entry_price'],
                        current_price=current_prices[ticker],
                        stop_loss_level=current_prices[ticker] * 0.93,
                        highest_price=current_prices[ticker]
                    )
                    session.add(holding)
            
            await session.commit()
            
            return {
                "session_id": session_id,
                "config_name": config_name,
                "display_name": config['display_name'],
                "description": config['description'],
                "final_value": final_value,
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "total_trades": total_trades,
                "start_date": config['start_date'],
                "end_date": config['end_date'],
            }
            
        except Exception as e:
            logger.error(f"Simulation error for {config_name}: {e}")
            import traceback
            traceback.print_exc()
            await session.rollback()
            return {"error": str(e)}


async def main():
    parser = argparse.ArgumentParser(description="Multi-Portfolio V9 Seed Script")
    parser.add_argument("--all", action="store_true", help="Run all portfolio configs")
    parser.add_argument("--config", type=str, help="Run specific config (e.g., 'golden_2020')")
    parser.add_argument("--list", action="store_true", help="List available configs")
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Portfolio Configurations:")
        print("-" * 60)
        for name, cfg in PORTFOLIO_CONFIGS.items():
            print(f"  {name:20} - {cfg['display_name']}")
        print()
        return
    
    # Initialize DB
    await init_db()
    
    # Load model
    device = get_device()
    logger.info(f"Using device: {device}")
    
    if MODEL_PATH.exists():
        model = TransformerRankModel.load(str(MODEL_PATH), device=device)
        model.eval()
        with open(SCALER_PATH, 'rb') as f:
            scaler_data = pickle.load(f)
        scaler = scaler_data['scaler']
    else:
        logger.error(f"Model not found at {MODEL_PATH}")
        return
    
    # Determine which configs to run
    if args.all:
        configs_to_run = list(PORTFOLIO_CONFIGS.keys())
    elif args.config:
        if args.config not in PORTFOLIO_CONFIGS:
            logger.error(f"Unknown config: {args.config}")
            return
        configs_to_run = [args.config]
    else:
        # Default: run golden config only
        configs_to_run = ["golden_2020"]
    
    # Run simulations
    results = []
    for config_name in configs_to_run:
        config = PORTFOLIO_CONFIGS[config_name]
        result = await run_portfolio_simulation(config_name, config, model, scaler, device)
        results.append(result)
    
    # Print summary
    print("\n" + "="*80)
    print("PORTFOLIO COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Config':<20} {'Return':>12} {'Drawdown':>12} {'Trades':>10}")
    print("-"*80)
    for r in results:
        if "error" not in r:
            print(f"{r['config_name']:<20} {r['total_return']:>+11.2f}% {r['max_drawdown']:>11.2f}% {r['total_trades']:>10}")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
