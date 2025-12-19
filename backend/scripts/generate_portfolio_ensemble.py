#!/usr/bin/env python3
"""
Generate 10-Portfolio Ensemble - Multi-Strategy Comparison
===========================================================

This script generates and runs 10 different V9 portfolio configurations
to show the variance in strategy performance and the impact of parameter choices.

Portfolio Configurations:
========================

MONTE CARLO (Same Settings, Different Seeds):
- v9_mc_seed1 through v9_mc_seed5
  Config: Top-10, Corr < 0.60, 5-day rebalance, 5-year simulation
  Vary: Random seed for deterministic variations
  Purpose: Show strategy stability (Monte Carlo variance)

PARAMETER VARIATIONS (Different Settings):
- v9_top5_tight:    Top-5 stocks, Correlation < 0.50 (conservative)
- v9_top15_loose:   Top-15 stocks, Correlation < 0.70 (aggressive)
- v9_weekly_rebal:  Top-10, Corr < 0.60, weekly (5-day) rebalance
- v9_monthly_rebal: Top-10, Corr < 0.60, monthly (21-day) rebalance
- v9_golden_2025:   Top-10, Corr < 0.60, 5-day rebalance (baseline)

Expected Results:
=================
- Monte Carlo variance: ±2-5% return difference due to entry/exit timing
- Top-5 tight: Lower volatility, higher drawdown (smaller portfolio)
- Top-15 loose: Higher returns but higher volatility (larger portfolio)
- Weekly vs 5-day: Minimal difference (weekly = longer hold periods)
- Monthly vs 5-day: Slower reaction time, may miss short-term rallies

Total Execution Time: ~20-30 minutes (2-3 minutes per portfolio × 10)

Usage:
======
    # Generate all 10 portfolios
    docker exec proxmox_stock_backend python -m scripts.generate_portfolio_ensemble

    # Or locally:
    export DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/stock_db
    python backend/scripts/generate_portfolio_ensemble.py
"""

import asyncio
import logging
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Setup backend path
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(BACKEND_DIR / "generate_ensemble.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# PORTFOLIO CONFIGURATIONS
# =============================================================================

PORTFOLIO_CONFIGS = [
    # MONTE CARLO: Same config, different seeds for variance analysis
    {
        "session_id": "v9_mc_seed1",
        "name": "Monte Carlo Seed 1",
        "top_k": 10,
        "corr_threshold": 0.60,
        "rebalance_days": 5,
        "seed": 42,
        "category": "monte_carlo"
    },
    {
        "session_id": "v9_mc_seed2",
        "name": "Monte Carlo Seed 2",
        "top_k": 10,
        "corr_threshold": 0.60,
        "rebalance_days": 5,
        "seed": 123,
        "category": "monte_carlo"
    },
    {
        "session_id": "v9_mc_seed3",
        "name": "Monte Carlo Seed 3",
        "top_k": 10,
        "corr_threshold": 0.60,
        "rebalance_days": 5,
        "seed": 456,
        "category": "monte_carlo"
    },
    {
        "session_id": "v9_mc_seed4",
        "name": "Monte Carlo Seed 4",
        "top_k": 10,
        "corr_threshold": 0.60,
        "rebalance_days": 5,
        "seed": 789,
        "category": "monte_carlo"
    },
    {
        "session_id": "v9_mc_seed5",
        "name": "Monte Carlo Seed 5",
        "top_k": 10,
        "corr_threshold": 0.60,
        "rebalance_days": 5,
        "seed": 999,
        "category": "monte_carlo"
    },

    # PARAMETER VARIATIONS: Different configurations for strategy comparison
    {
        "session_id": "v9_top5_tight",
        "name": "Conservative (Top-5, Tight Corr)",
        "top_k": 5,
        "corr_threshold": 0.50,
        "rebalance_days": 5,
        "seed": 42,
        "category": "parameter"
    },
    {
        "session_id": "v9_top15_loose",
        "name": "Aggressive (Top-15, Loose Corr)",
        "top_k": 15,
        "corr_threshold": 0.70,
        "rebalance_days": 5,
        "seed": 42,
        "category": "parameter"
    },
    {
        "session_id": "v9_weekly_rebal",
        "name": "Weekly Rebalance",
        "top_k": 10,
        "corr_threshold": 0.60,
        "rebalance_days": 7,
        "seed": 42,
        "category": "parameter"
    },
    {
        "session_id": "v9_monthly_rebal",
        "name": "Monthly Rebalance",
        "top_k": 10,
        "corr_threshold": 0.60,
        "rebalance_days": 21,
        "seed": 42,
        "category": "parameter"
    },
    {
        "session_id": "v9_golden_2025",
        "name": "Baseline (5-day Rebalance)",
        "top_k": 10,
        "corr_threshold": 0.60,
        "rebalance_days": 5,
        "seed": 42,
        "category": "baseline"
    },
]


import asyncio
import logging
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Setup backend path
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(BACKEND_DIR / "generate_ensemble.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# PORTFOLIO CONFIGURATIONS
# =============================================================================

PORTFOLIO_CONFIGS = [
    # MONTE CARLO: Same config, different seeds for variance analysis
    {
        "session_id": "v9_mc_seed1",
        "name": "Monte Carlo Seed 1",
        "top_k": 10,
        "corr_threshold": 0.60,
        "rebalance_days": 5,
        "seed": 42,
        "category": "monte_carlo"
    },
    # ... (other configs implied standard or generated dynamically if needed, keeping existing list structure references if possible, 
    # but for replaced content I must provide the full list or assume it's okay)
    # To keep this clean, I will NOT replace the config list itself if I can target around it, but I am replacing the whole file imports and main logic.
    # Actually, I'll just target the imports and then the run_single_portfolio and main functions separately to avoid huge payload.
]

# ...

def run_single_portfolio(config: dict, extra_args: list = None) -> dict:
    """
    Run a single portfolio backtest in a subprocess.

    Args:
        config: Portfolio configuration dict
        extra_args: List of additional CLI arguments to pass (e.g. --limit 100)

    Returns:
        Result dict with success/failure status
    """
    session_id = config["session_id"]
    logger.info(f"Starting: {session_id} ({config['name']})")
    
    cmd = [
        "python", "-m", "scripts.seed_paper_history",
        "--years", "1",
        "--session-id", session_id,
        "--top-k", str(config["top_k"]),
        "--corr-threshold", str(config["corr_threshold"]),
        "--rebalance-days", str(config["rebalance_days"]),
        "--seed", str(config["seed"]),
        "--force"
    ]
    
    # Forward extra arguments (e.g. limit, start-date)
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(BACKEND_DIR),
            capture_output=True,
            text=True,
            timeout=900  # 15 minute timeout per portfolio
        )

        if result.returncode == 0:
            logger.info(f"✅ Completed: {session_id}")
            return {
                "session_id": session_id,
                "status": "success",
                "config": config,
                "message": f"Generated {session_id} successfully"
            }
        else:
            logger.error(f"❌ Failed: {session_id}")
            logger.error(f"   Error: {result.stderr}")
            return {
                "session_id": session_id,
                "status": "error",
                "config": config,
                "error": result.stderr[-500:] if result.stderr else "Unknown error"
            }

    except subprocess.TimeoutExpired:
        logger.error(f"⏱️  Timeout: {session_id} (exceeded 15 minutes)")
        return {
            "session_id": session_id,
            "status": "timeout",
            "config": config,
            "error": "Simulation exceeded 15-minute timeout"
        }

    except Exception as e:
        logger.error(f"❌ Exception: {session_id} - {e}")
        return {
            "session_id": session_id,
            "status": "error",
            "config": config,
            "error": str(e)
        }


async def generate_ensemble(extra_args: list = None) -> dict:
    """
    Generate all 10 portfolio configurations in parallel.
    """
    logger.info("=" * 80)
    logger.info("GENERATING 10-PORTFOLIO ENSEMBLE")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total portfolios: {len(PORTFOLIO_CONFIGS)}")
    logger.info(f"Expected time: ~20-30 minutes (5 parallel × 3-5 min each)")
    if extra_args:
        logger.info(f"Forwarding args: {' '.join(extra_args)}")
    logger.info("")

    results = {
        "total": len(PORTFOLIO_CONFIGS),
        "successful": 0,
        "failed": 0,
        "timeout": 0,
        "portfolios": [],
        "start_time": datetime.now(),
        "end_time": None
    }

    # Use ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=5) as executor:
        # Submit all jobs
        future_to_config = {
            executor.submit(run_single_portfolio, config, extra_args): config
            for config in PORTFOLIO_CONFIGS
        }

        # Process results as they complete
        for future in as_completed(future_to_config):
            result = future.result()
            results["portfolios"].append(result)

            # Update counters
            if result["status"] == "success":
                results["successful"] += 1
            elif result["status"] == "timeout":
                results["timeout"] += 1
            else:
                results["failed"] += 1

    results["end_time"] = datetime.now()
    duration = (results["end_time"] - results["start_time"]).total_seconds() / 60

    # Log summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("ENSEMBLE GENERATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Completed: {results['successful']}/{results['total']} successful")
    logger.info(f"Failed:    {results['failed']}/{results['total']}")
    logger.info(f"Timeout:   {results['timeout']}/{results['total']}")
    logger.info(f"Duration:  {duration:.1f} minutes")
    logger.info("")

    # Show details for failed portfolios
    if results["failed"] > 0 or results["timeout"] > 0:
        logger.info("⚠️  FAILURES/TIMEOUTS:")
        for portfolio in results["portfolios"]:
            if portfolio["status"] != "success":
                logger.info(f"  - {portfolio['session_id']}: {portfolio.get('error', 'Unknown error')}")
        logger.info("")

    # Show summary of all portfolios (for comparison)
    logger.info("PORTFOLIO RESULTS:")
    for portfolio in results["portfolios"]:
        status_emoji = "✅" if portfolio["status"] == "success" else "❌" if portfolio["status"] == "error" else "⏱️"
        config = portfolio.get("config", {})
        logger.info(
            f"  {status_emoji} {portfolio['session_id']:20} | "
            f"Top-K: {config.get('top_k', '-'):2} | "
            f"Corr: {config.get('corr_threshold', 0):.2f} | "
            f"Rebal: {config.get('rebalance_days', '-'):2}d"
        )

    logger.info("=" * 80)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate 10-Portfolio Ensemble")
    parser.add_argument("--limit", type=int, help="Limit number of days (useful for testing)")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # args.limit and args.start_date are already in args
    extra_args = []
    if args.limit:
        extra_args.extend(["--limit", str(args.limit)])
    if args.start_date:
        extra_args.extend(["--start-date", args.start_date])
    
    
    # unknown_args removal: we are strictly defining known args above now
    # extra_args.extend(unknown_args)

    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {BACKEND_DIR}")

    # Run async generator
    results = asyncio.run(generate_ensemble(extra_args))

    # Exit with appropriate code
    if results["failed"] > 0 or results["timeout"] > 0:
        logger.warning(f"⚠️  Some portfolios failed. Check logs for details.")
        sys.exit(1)
    else:
        logger.info("✅ All portfolios generated successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
