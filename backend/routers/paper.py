from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

from services.db import (
    AsyncSessionLocal, 
    PaperPortfolio, 
    PaperHolding, 
    PaperTrade, 
    PAPER_SESSION_ID
)

router = APIRouter(prefix="/paper", tags=["Paper Trading"])

# Dependency
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

# Pydantic Models
class HoldingResponse(BaseModel):
    ticker: str
    quantity: float
    entry_price: float
    current_price: float
    stop_loss_level: float
    highest_price: float
    profit_pct: float
    value: float

class TradeResponse(BaseModel):
    date: datetime
    action: str
    ticker: str
    price: float
    quantity: float
    reason: str
    profit_loss: Optional[float] = None

class PortfolioStatusResponse(BaseModel):
    session_id: str
    total_value: float
    cash_balance: float
    equity_value: float
    pnl: float
    pnl_pct: float
    days_since_rebalance: int
    holdings: List[HoldingResponse]
    recent_trades: List[TradeResponse]

@router.get("/status", response_model=PortfolioStatusResponse)
async def get_paper_status(db: AsyncSession = Depends(get_db)):
    """Get the current status of the paper trading portfolio."""
    
    # Get Portfolio
    result = await db.execute(
        select(PaperPortfolio).where(PaperPortfolio.session_id == PAPER_SESSION_ID)
    )
    portfolio = result.scalar_one_or_none()
    
    if not portfolio:
        # If no portfolio exists, return default/empty state
        return PortfolioStatusResponse(
            session_id=PAPER_SESSION_ID,
            total_value=10000.0,
            cash_balance=10000.0,
            equity_value=0.0,
            pnl=0.0,
            pnl_pct=0.0,
            days_since_rebalance=0,
            holdings=[],
            recent_trades=[]
        )
    
    # Get Holdings
    result = await db.execute(
        select(PaperHolding).where(PaperHolding.session_id == PAPER_SESSION_ID)
    )
    db_holdings = result.scalars().all()
    
    holdings_list = []
    for h in db_holdings:
        profit_pct = ((h.current_price - h.entry_price) / h.entry_price) * 100
        value = h.quantity * h.current_price
        
        holdings_list.append(HoldingResponse(
            ticker=h.ticker,
            quantity=h.quantity,
            entry_price=h.entry_price,
            current_price=h.current_price,
            stop_loss_level=h.stop_loss_level,
            highest_price=h.highest_price,
            profit_pct=profit_pct,
            value=value
        ))
        
    # Get Recent Trades (up to 200 most recent trades)
    result = await db.execute(
        select(PaperTrade)
        .where(PaperTrade.session_id == PAPER_SESSION_ID)
        .order_by(PaperTrade.trade_date.desc())
        .limit(200)
    )
    db_trades = result.scalars().all()
    
    trades_list = []
    for t in db_trades:
        trades_list.append(TradeResponse(
            date=t.trade_date,
            action=t.action,
            ticker=t.ticker,
            price=t.price,
            quantity=t.quantity,
            reason=t.reason,
            profit_loss=t.profit_loss
        ))
        
    # Calculate Total P/L
    initial_capital = 10000.0 # Could store this in DB too, but hardcoded for now per script
    pnl = portfolio.total_value - initial_capital
    pnl_pct = (pnl / initial_capital) * 100
    
    return PortfolioStatusResponse(
        session_id=portfolio.session_id,
        total_value=portfolio.total_value,
        cash_balance=portfolio.cash_balance,
        equity_value=portfolio.equity_value,
        pnl=pnl,
        pnl_pct=pnl_pct,
        days_since_rebalance=portfolio.days_since_rebalance,
        holdings=holdings_list,
        recent_trades=trades_list
    )

@router.post("/reset")
async def reset_paper_portfolio(db: AsyncSession = Depends(get_db)):
    """Reset the paper trading portfolio (Delete and Re-init)."""
    from sqlalchemy import delete
    
    # Delete all data for this session
    await db.execute(delete(PaperTrade).where(PaperTrade.session_id == PAPER_SESSION_ID))
    await db.execute(delete(PaperHolding).where(PaperHolding.session_id == PAPER_SESSION_ID))
    await db.execute(delete(PaperPortfolio).where(PaperPortfolio.session_id == PAPER_SESSION_ID))
    
    # Create new portfolio
    new_portfolio = PaperPortfolio(
        session_id=PAPER_SESSION_ID,
        cash_balance=10000.0,
        equity_value=0.0,
        total_value=10000.0,
        days_since_rebalance=5 # Ready to rebalance immediately
    )
    db.add(new_portfolio)
    await db.commit()
    
    return {"message": f"Session {PAPER_SESSION_ID} reset successfully"}

@router.get("/history")
async def get_paper_history(db: AsyncSession = Depends(get_db)):
    """
    Get historical equity curve.
    Returns empty array if no history exists (run seed_paper_history.py to populate).
    """
    from services.db import PaperPortfolioHistory
    
    # Try fetching from DB
    result = await db.execute(
        select(PaperPortfolioHistory)
        .where(PaperPortfolioHistory.session_id == PAPER_SESSION_ID)
        .order_by(PaperPortfolioHistory.date)
    )
    history = result.scalars().all()
    
    if history and len(history) > 10:
        return [{
            "date": h.date.isoformat(),
            "value": h.total_value,
            "equity": h.equity_value,
            "cash": h.cash_balance
        } for h in history]

    # NO DATA - Return empty array instead of random mock data
    # Frontend should display "No historical data. Run seed script."
    return []


@router.get("/trades")
async def get_paper_trades(
    limit: int = 200,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """
    Get trades with pagination support.

    Args:
        limit: Number of trades to return (default 200, max 500)
        offset: Number of trades to skip (default 0)

    Returns:
        trades: List of trades
        total: Total number of trades for session
        limit: Requested limit
        offset: Requested offset
    """
    from sqlalchemy import func

    # Clamp limit to reasonable bounds
    limit = min(limit, 500)
    limit = max(limit, 10)

    # Get total count
    count_result = await db.execute(
        select(func.count(PaperTrade.id))
        .where(PaperTrade.session_id == PAPER_SESSION_ID)
    )
    total_count = count_result.scalar() or 0

    # Get paginated trades (most recent first)
    result = await db.execute(
        select(PaperTrade)
        .where(PaperTrade.session_id == PAPER_SESSION_ID)
        .order_by(PaperTrade.trade_date.desc())
        .limit(limit)
        .offset(offset)
    )
    db_trades = result.scalars().all()

    trades_list = []
    for t in db_trades:
        trades_list.append(TradeResponse(
            date=t.trade_date,
            action=t.action,
            ticker=t.ticker,
            price=t.price,
            quantity=t.quantity,
            reason=t.reason,
            profit_loss=t.profit_loss
        ))

    return {
        "trades": trades_list,
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "remaining": max(0, total_count - (offset + limit))
    }


@router.post("/reset-and-simulate")
async def reset_and_simulate(
    years: int = 5,
    force: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """
    Reset paper trading portfolio and run 5-year backtest simulation.

    This endpoint triggers seed_paper_history.py to:
    1. Clear existing paper trading data for this session
    2. Run V9 model on 5 years of historical data
    3. Generate realistic trade history with portfolio snapshots

    Args:
        years: Number of years to simulate (1-5, default: 5)
        force: Force re-seed even if session exists (default: True)

    Returns:
        Status message with simulation results
    """
    import subprocess
    from pathlib import Path

    try:
        # Build command to run seed script
        backend_dir = Path(__file__).parent.parent
        cmd = [
            "python", "-m", "scripts.seed_paper_history",
            "--years", str(min(5, max(1, years))),  # Clamp between 1-5
            "--force" if force else ""
        ]

        # Remove empty string from command
        cmd = [c for c in cmd if c]

        # Run the script
        result = subprocess.run(
            cmd,
            cwd=str(backend_dir),
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for 5-year simulation
        )

        if result.returncode == 0:
            return {
                "status": "success",
                "message": f"5-year portfolio simulation completed successfully",
                "years_simulated": years,
                "trading_days": years * 252,
                "output": result.stdout[-500:] if result.stdout else ""  # Last 500 chars
            }
        else:
            return {
                "status": "error",
                "message": f"Simulation failed: {result.stderr}",
                "error": result.stderr
            }

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "Simulation timed out after 10 minutes",
            "error": "timeout"
        }

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Reset and simulate failed: {e}", exc_info=True)

        return {
            "status": "error",
            "message": f"Failed to start simulation: {str(e)}",
            "error": str(e)
        }


# =============================================================================
# PORTFOLIO COMPARISON ENDPOINTS
# =============================================================================

# Portfolio configuration metadata (matching seed_multi_portfolio.py)
PORTFOLIO_CONFIGS = {
    "v9_golden_2020": {
        "display_name": "🏆 Golden Config",
        "description": "Current production settings. Balanced risk/reward.",
        "parameters": {"vix": 30, "top_k": 10, "correlation": 0.60, "rebalance": 5},
    },
    "v9_aggressive": {
        "display_name": "🔥 Aggressive",
        "description": "Higher VIX tolerance, more holdings, looser correlation.",
        "parameters": {"vix": 40, "top_k": 15, "correlation": 0.80, "rebalance": 5},
    },
    "v9_conservative": {
        "display_name": "🛡️ Conservative", 
        "description": "Exit earlier, fewer holdings, strict diversification.",
        "parameters": {"vix": 25, "top_k": 5, "correlation": 0.40, "rebalance": 5},
    },
    "v9_no_vix": {
        "display_name": "⚠️ No VIX Filter",
        "description": "VIX filter disabled. Full exposure to crashes.",
        "parameters": {"vix": 999, "top_k": 10, "correlation": 0.60, "rebalance": 5},
    },
    "v9_daily": {
        "display_name": "⚡ Daily Rebalance",
        "description": "Rebalance every day. Higher turnover.",
        "parameters": {"vix": 30, "top_k": 10, "correlation": 0.60, "rebalance": 1},
    },
    "v9_covid_2020": {
        "display_name": "🦠 COVID Crash",
        "description": "2018-2023: March 2020 crash in MIDDLE.",
        "parameters": {"vix": 30, "top_k": 10, "correlation": 0.60, "rebalance": 5, "period": "2018-2023"},
    },
    "v9_crisis_2008": {
        "display_name": "💥 2008 Crisis",
        "description": "2006-2011: Worst financial crisis since Great Depression.",
        "parameters": {"vix": 30, "top_k": 10, "correlation": 0.60, "rebalance": 5, "period": "2006-2011"},
    },
    "v9_bear_2022": {
        "display_name": "🐻 2022 Bear",
        "description": "2020-2025: COVID crash + tech bear market.",
        "parameters": {"vix": 30, "top_k": 10, "correlation": 0.60, "rebalance": 5, "period": "2020-2025"},
    },
    "v9_ultimate_20yr": {
        "display_name": "📊 20-Year Ultimate",
        "description": "2005-2024: Covers 2008, 2020, AND 2022 crashes.",
        "parameters": {"vix": 30, "top_k": 10, "correlation": 0.60, "rebalance": 5, "period": "2005-2024"},
    },
    # Also include the existing golden session
    "v9_golden_2025": {
        "display_name": "🏆 Golden (Current)",
        "description": "Active production portfolio.",
        "parameters": {"vix": 30, "top_k": 10, "correlation": 0.60, "rebalance": 5},
    },
}


@router.get("/portfolios/compare")
async def get_portfolio_comparison(db: AsyncSession = Depends(get_db)):
    """
    Get all V9 portfolio sessions with comparison metrics.
    Returns performance data for each portfolio configuration.
    """
    from sqlalchemy import func
    from services.db import PaperPortfolioHistory
    
    results = []
    
    for session_id, config in PORTFOLIO_CONFIGS.items():
        try:
            # Get portfolio summary
            portfolio_result = await db.execute(
                select(PaperPortfolio).where(PaperPortfolio.session_id == session_id)
            )
            portfolio = portfolio_result.scalar_one_or_none()
            
            if not portfolio:
                continue
            
            # Get trade count
            trade_count_result = await db.execute(
                select(func.count(PaperTrade.id)).where(PaperTrade.session_id == session_id)
            )
            trade_count = trade_count_result.scalar() or 0
            
            # Get history for metrics
            history_result = await db.execute(
                select(PaperPortfolioHistory)
                .where(PaperPortfolioHistory.session_id == session_id)
                .order_by(PaperPortfolioHistory.date)
            )
            history = history_result.scalars().all()
            
            # Calculate metrics
            start_value = history[0].total_value if history else 10000.0
            final_value = portfolio.total_value
            total_return = ((final_value / start_value) - 1) * 100
            
            # Calculate max drawdown
            max_drawdown = 0.0
            peak = start_value
            for h in history:
                if h.total_value > peak:
                    peak = h.total_value
                drawdown = ((peak - h.total_value) / peak) * 100
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            # Get date range
            start_date = history[0].date.isoformat() if history else None
            end_date = history[-1].date.isoformat() if history else None
            
            results.append({
                "session_id": session_id,
                "display_name": config["display_name"],
                "description": config["description"],
                "parameters": config["parameters"],
                "start_value": start_value,
                "final_value": final_value,
                "total_return": round(total_return, 2),
                "max_drawdown": round(max_drawdown, 2),
                "trade_count": trade_count,
                "start_date": start_date,
                "end_date": end_date,
                "holdings_count": len([]),  # Could add actual count
            })
            
        except Exception as e:
            # Portfolio doesn't exist yet
            pass
    
    # Sort by total return (best first)
    results.sort(key=lambda x: x["total_return"], reverse=True)
    
    return {
        "portfolios": results,
        "count": len(results),
        "available_configs": list(PORTFOLIO_CONFIGS.keys())
    }


@router.get("/portfolios/{session_id}/history")
async def get_portfolio_session_history(session_id: str, db: AsyncSession = Depends(get_db)):
    """Get historical equity curve for a specific portfolio session."""
    from services.db import PaperPortfolioHistory
    
    if session_id not in PORTFOLIO_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
    
    result = await db.execute(
        select(PaperPortfolioHistory)
        .where(PaperPortfolioHistory.session_id == session_id)
        .order_by(PaperPortfolioHistory.date)
    )
    history = result.scalars().all()
    
    return [{
        "date": h.date.isoformat(),
        "value": h.total_value,
        "equity": h.equity_value,
        "cash": h.cash_balance
    } for h in history]
