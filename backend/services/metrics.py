"""
Financial Metrics Service

Provides reusable functions for calculating portfolio performance metrics.
Extracted from routers/paper.py for testability and reuse (Phase 2.1).
"""

import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Constants
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0  # Assuming 0 for simplicity


def calculate_sharpe_ratio(portfolio_values: List[float], risk_free_rate: float = RISK_FREE_RATE) -> float:
    """
    Calculate the annualized Sharpe Ratio from a list of portfolio values.
    
    Sharpe = (mean_return - risk_free_rate) / std_return * sqrt(252)
    
    Args:
        portfolio_values: List of daily portfolio values (chronological order)
        risk_free_rate: Daily risk-free rate (default 0)
    
    Returns:
        Annualized Sharpe Ratio (float). Returns 0.0 if insufficient data.
    """
    if len(portfolio_values) < 6:
        return 0.0
    
    # Calculate daily returns
    values = np.array(portfolio_values)
    returns = np.diff(values) / values[:-1]
    
    if len(returns) < 2:
        return 0.0
    
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    
    if std_ret <= 0:
        return 0.0
    
    # Annualized Sharpe
    sharpe = ((mean_ret - risk_free_rate) / std_ret) * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return float(sharpe)


def calculate_alpha(
    portfolio_start_value: float,
    portfolio_end_value: float,
    benchmark_start_price: float,
    benchmark_end_price: float
) -> float:
    """
    Calculate Alpha (excess return vs benchmark).
    
    Alpha = Portfolio_Return - Benchmark_Return (in percentage points)
    
    Args:
        portfolio_start_value: Portfolio value at start of period
        portfolio_end_value: Portfolio value at end of period
        benchmark_start_price: Benchmark (e.g., SPY) price at start
        benchmark_end_price: Benchmark price at end
    
    Returns:
        Alpha in percentage points (e.g., 5.0 means +5% excess return)
    """
    if portfolio_start_value <= 0 or benchmark_start_price <= 0:
        return 0.0
    
    portfolio_return = (portfolio_end_value / portfolio_start_value) - 1
    benchmark_return = (benchmark_end_price / benchmark_start_price) - 1
    
    # Alpha in percentage points
    alpha = (portfolio_return - benchmark_return) * 100
    
    if np.isnan(alpha) or np.isinf(alpha):
        return 0.0
        
    return float(alpha)


def calculate_daily_pnl(current_value: float, previous_value: float, initial_capital: float) -> Tuple[float, float]:
    """
    Calculate daily profit/loss and percentage.
    
    Args:
        current_value: Current portfolio value
        previous_value: Previous day's portfolio value
        initial_capital: Starting capital for percentage calculation
    
    Returns:
        Tuple of (daily_pnl_dollars, daily_pnl_percentage)
    """
    if initial_capital <= 0:
        return 0.0, 0.0
    
    pnl = current_value - previous_value
    pnl_pct = (pnl / initial_capital) * 100
    
    return float(pnl), float(pnl_pct)


def calculate_total_return(current_value: float, initial_capital: float) -> Tuple[float, float]:
    """
    Calculate total return in dollars and percentage.
    
    Args:
        current_value: Current portfolio value
        initial_capital: Starting capital
    
    Returns:
        Tuple of (total_pnl_dollars, total_pnl_percentage)
    """
    if initial_capital <= 0:
        return 0.0, 0.0
    
    pnl = current_value - initial_capital
    pnl_pct = (pnl / initial_capital) * 100
    
    return float(pnl), float(pnl_pct)
