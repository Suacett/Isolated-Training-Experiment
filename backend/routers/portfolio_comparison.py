"""
Portfolio Comparison Router
===========================

Provides API endpoints for comparing multiple V9 portfolio strategies
(Monte Carlo variations + parameter variations).

Endpoints:
- GET /portfolio/comparison - Full comparison data for all 10 portfolios
- GET /portfolio/comparison/{session_id} - Single portfolio details
- GET /portfolio/comparison/metrics - Summary metrics table
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.db import (
    AsyncSessionLocal,
    PaperPortfolio,
    PaperPortfolioHistory,
    PAPER_SESSION_ID,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/portfolio", tags=["Portfolio Comparison"])


# Dependency
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


# Session IDs for all 10 portfolios
PORTFOLIO_SESSIONS = {
    "v9_mc_seed1": {"name": "Monte Carlo Seed 1", "category": "monte_carlo"},
    "v9_mc_seed2": {"name": "Monte Carlo Seed 2", "category": "monte_carlo"},
    "v9_mc_seed3": {"name": "Monte Carlo Seed 3", "category": "monte_carlo"},
    "v9_mc_seed4": {"name": "Monte Carlo Seed 4", "category": "monte_carlo"},
    "v9_mc_seed5": {"name": "Monte Carlo Seed 5", "category": "monte_carlo"},
    "v9_top5_tight": {"name": "Conservative (Top-5, Tight)", "category": "parameter"},
    "v9_top15_loose": {"name": "Aggressive (Top-15, Loose)", "category": "parameter"},
    "v9_weekly_rebal": {"name": "Weekly Rebalance", "category": "parameter"},
    "v9_monthly_rebal": {"name": "Monthly Rebalance", "category": "parameter"},
    "v9_golden_2025": {"name": "Baseline (5-day Rebal)", "category": "baseline"},
}


def calculate_max_drawdown(equity_values: List[float]) -> float:
    """Calculate maximum drawdown as percentage."""
    if not equity_values or len(equity_values) < 2:
        return 0.0

    equity_array = np.array(equity_values, dtype=float)
    
    # Protect against zero or negative values
    if np.any(equity_array <= 0):
        return 0.0
        
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - running_max) / running_max
    return float(np.min(drawdown) * 100)  # Convert to percentage


def calculate_sharpe_ratio(equity_values: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio (252 trading days/year)."""
    if not equity_values or len(equity_values) < 2:
        return 0.0

    equity_array = np.array(equity_values, dtype=float)
    
    # Filter out entries where previous equity is zero to avoid division by zero
    # and compute returns only for valid transitions.
    valid_mask = equity_array[:-1] > 0
    if not np.any(valid_mask):
        return 0.0
        
    valid_previous = equity_array[:-1][valid_mask]
    valid_current = equity_array[1:][valid_mask]
    
    returns = (valid_current - valid_previous) / valid_previous

    if len(returns) < 2:
        return 0.0

    # Annualize mean and volatility
    mean_return = np.mean(returns) * 252
    # Use sample standard deviation (ddof=1)
    std_return = np.std(returns, ddof=1) * np.sqrt(252)

    if std_return == 0:
        return 0.0

    sharpe = (mean_return - risk_free_rate) / std_return
    return float(sharpe)


# Helper function to fetch data using a dedicated session
async def fetch_session_data(session_id: str) -> Optional[Dict[str, Any]]:
    """Helper to fetch data with its own DB session to avoid concurrency issues."""
    async with AsyncSessionLocal() as db:
        return await get_portfolio_data(session_id, db)


async def get_portfolio_data(session_id: str, db: AsyncSession) -> Optional[Dict[str, Any]]:
    """Fetch portfolio data for a single session."""
    try:
        # Get portfolio summary
        portfolio_result = await db.execute(
            select(PaperPortfolio).where(PaperPortfolio.session_id == session_id)
        )
        portfolio = portfolio_result.scalar_one_or_none()

        if not portfolio:
            return None

        # Get equity curve history
        history_result = await db.execute(
            select(PaperPortfolioHistory)
            .where(PaperPortfolioHistory.session_id == session_id)
            .order_by(PaperPortfolioHistory.date)
        )
        history = history_result.scalars().all()

        if not history:
            return None

        # Extract equity values
        equity_values = [h.total_value for h in history]
        dates = [h.date.isoformat() for h in history]

        # Get actual initial value from history or fall back to DB default
        initial_value = history[0].total_value if history else 10000.0
        final_value = portfolio.total_value
        
        total_return = ((final_value - initial_value) / initial_value) * 100 if initial_value > 0 else 0.0
        max_drawdown = calculate_max_drawdown(equity_values)
        sharpe_ratio = calculate_sharpe_ratio(equity_values)

        return {
            "session_id": session_id,
            "name": PORTFOLIO_SESSIONS.get(session_id, {}).get("name", session_id),
            "category": PORTFOLIO_SESSIONS.get(session_id, {}).get("category", "unknown"),
            "initial_value": initial_value,
            "final_value": final_value,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "days_simulated": len(history),
            "equity_curve": [
                {"date": dates[i], "value": equity_values[i]}
                for i in range(len(history))
            ]
        }

    except Exception as e:
        logger.error(f"Error fetching portfolio data for {session_id}: {e}")
        return None


@router.get("/comparison")
async def get_portfolio_comparison(db: AsyncSession = Depends(get_db)):
    """
    Get comparison data for all 10 portfolio strategies.

    Returns:
        - Portfolio metrics (return, drawdown, sharpe ratio)
        - Equity curve for charting
        - Sorted by total return (descending)
    """
    try:
        portfolios = []

        # Fetch data for all sessions in parallel using independent sessions
        portfolio_tasks = [
            fetch_session_data(session_id) 
            for session_id in PORTFOLIO_SESSIONS.keys()
        ]
        portfolio_results = await asyncio.gather(*portfolio_tasks)
        portfolios = [p for p in portfolio_results if p is not None]

        if not portfolios:
            raise HTTPException(status_code=404, detail="No portfolio data found. Run generate_portfolio_ensemble.py first.")

        # Sort by total return (descending)
        portfolios.sort(key=lambda p: p["total_return"], reverse=True)

        # Calculate statistics for context
        returns = [p["total_return"] for p in portfolios]
        drawdowns = [p["max_drawdown"] for p in portfolios]
        sharpes = [p["sharpe_ratio"] for p in portfolios]

        return {
            "portfolios": portfolios,
            "statistics": {
                "best_return": max(returns),
                "worst_return": min(returns),
                "avg_return": np.mean(returns),
                "std_return": np.std(returns),
                "best_sharpe": max(sharpes),
                "worst_drawdown": min(drawdowns),  # Most negative
                "avg_drawdown": np.mean(drawdowns),
            },
            "count": len(portfolios),
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in get_portfolio_comparison: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comparison/{session_id}")
async def get_single_portfolio(session_id: str, db: AsyncSession = Depends(get_db)):
    """
    Get detailed data for a single portfolio.

    Args:
        session_id: Portfolio session ID (e.g., v9_golden_2025)

    Returns:
        Portfolio metrics and full equity curve
    """
    portfolio_data = await get_portfolio_data(session_id, db)

    if not portfolio_data:
        raise HTTPException(
            status_code=404,
            detail=f"Portfolio {session_id} not found"
        )

    return portfolio_data


@router.get("/comparison/metrics")
async def get_portfolio_metrics(db: AsyncSession = Depends(get_db)):
    """
    Get summary metrics table for comparison.

    Returns:
        Simplified metrics for table display (strategy, return, drawdown, sharpe, final value)
    """
    try:
        # Fetch all portfolios in parallel using independent sessions
        portfolio_tasks = [
            fetch_session_data(session_id)
            for session_id in PORTFOLIO_SESSIONS.keys()
        ]
        portfolio_results = await asyncio.gather(*portfolio_tasks)
        portfolio_data_list = [p for p in portfolio_results if p is not None]
        
        # Sort by return before formatting
        portfolio_data_list.sort(key=lambda p: p["total_return"], reverse=True)
        
        # Format metrics after sorting
        metrics = [
            {
                "session_id": p["session_id"],
                "strategy": p["name"],
                "category": p["category"],
                "return": f"{p['total_return']:.2f}%",
                "max_drawdown": f"{p['max_drawdown']:.2f}%",
                "sharpe_ratio": f"{p['sharpe_ratio']:.2f}",
                "final_value": f"${p['final_value']:,.0f}",
                "gain_loss": f"${p['final_value'] - p['initial_value']:,.0f}",
            }
            for p in portfolio_data_list
        ]

        return {
            "metrics": metrics,
            "count": len(metrics)
        }

    except Exception as e:
        logger.error(f"Error in get_portfolio_metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comparison/status")
async def get_comparison_status(db: AsyncSession = Depends(get_db)):
    """
    Check if all 10 portfolios have been generated.

    Returns:
        Status of each portfolio (exists, ready, missing)
    """
    try:
        # Check all portfolios using a single query for efficiency
        session_ids = list(PORTFOLIO_SESSIONS.keys())
        result = await db.execute(
            select(PaperPortfolio).where(PaperPortfolio.session_id.in_(session_ids))
        )
        portfolios_map = {p.session_id: p for p in result.scalars().all()}
        
        status = {}
        for session_id, info in PORTFOLIO_SESSIONS.items():
            portfolio = portfolios_map.get(session_id)
            status[session_id] = {
                "name": info["name"],
                "exists": portfolio is not None,
                "ready": portfolio is not None and portfolio.total_value > 0
            }

        total = len(status)
        ready = sum(1 for s in status.values() if s["ready"])

        return {
            "portfolios": status,
            "total": total,
            "ready": ready,
            "missing": total - ready,
            "percent_complete": (ready / total * 100) if total > 0 else 0
        }

    except Exception as e:
        logger.error(f"Error in get_comparison_status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
