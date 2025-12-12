#!/usr/bin/env python3
"""
Training Data Fetcher for V7 Model

Fetches comprehensive market data:
- Major indices (SPY, QQQ, DIA, IWM)
- Sector ETFs (XLK, XLF, etc.)
- Macro indicators (VIX proxy, bond yields)
- Top 30 liquid stocks

Data is fetched from 2007 onwards to capture:
- 2008 Financial Crisis (crash patterns)
- 2020 COVID crash (V-shape recovery)
- 2022 Bear Market (inflation regime)

Usage:
    docker exec proxmox_stock_backend python -m scripts.fetch_training_data
    docker exec proxmox_stock_backend python -m scripts.fetch_training_data --validate
"""

import sys
import os
import asyncio
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta

backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fetch_training_data.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# === TRAINING UNIVERSE ===
# A curated list of high-quality data sources for v7 training

TRAINING_UNIVERSE = [
    # --- INDICES (The Baseline) ---
    "SPY",   # S&P 500 ETF - THE market baseline
    "QQQ",   # Nasdaq 100 - Tech heavy
    "IWM",   # Russell 2000 - Small caps
    "DIA",   # Dow Jones Industrial
    
    # --- SECTOR ETFS (Flow of Funds) ---
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Healthcare
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLI",   # Industrials
    "XLB",   # Materials
    "XLU",   # Utilities
    "XLRE",  # Real Estate
    "XLC",   # Communications
    "SMH",   # Semiconductors (lead indicator for tech)
    "KRE",   # Regional Banks (risk indicator)
    "XBI",   # Biotech (risk-on indicator)
    
    # --- MACRO INDICATORS ---
    "GLD",   # Gold - Defensive/Inflation hedge
    "TLT",   # 20+ Year Treasury Bonds
    "UUP",   # US Dollar Index
    "USO",   # Oil (WTI Crude)
    "HYG",   # High Yield Corporate Bonds (risk indicator)
    
    # --- TOP 30 LIQUID STOCKS (Diverse Industries) ---
    # Tech / Growth
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "CRM", "NFLX",
    # Finance
    "JPM", "BAC", "V", "MA", "GS",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE",
    # Retail / Staples
    "WMT", "COST", "PG", "KO", "PEP",
    # Industrial / Energy
    "XOM", "CVX", "CAT", "BA", "UPS",
    # Defense
    "LMT", "RTX"
]


async def fetch_ticker_data(ticker: str, mode: str = "full") -> dict:
    """Fetch data for a single ticker."""
    from services.data_ingest import YahooFinanceClient
    
    client = YahooFinanceClient()
    try:
        records = await client.fetch_data(ticker, mode=mode)
        return {"ticker": ticker, "records": records, "status": "success"}
    except Exception as e:
        logger.warning(f"Failed to fetch {ticker}: {e}")
        return {"ticker": ticker, "error": str(e), "status": "failed"}


async def fetch_all_data(tickers: list, mode: str = "full", concurrency: int = 5):
    """Fetch data for all tickers with rate limiting."""
    import asyncio
    
    sem = asyncio.Semaphore(concurrency)
    
    async def fetch_with_semaphore(ticker):
        async with sem:
            result = await fetch_ticker_data(ticker, mode)
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)
            return result
    
    logger.info(f"📊 Fetching {len(tickers)} tickers (mode={mode})...")
    results = await asyncio.gather(*[fetch_with_semaphore(t) for t in tickers])
    
    success = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    
    logger.info(f"✅ Success: {len(success)}, ❌ Failed: {len(failed)}")
    if failed:
        logger.warning(f"Failed tickers: {[r['ticker'] for r in failed]}")
    
    return {"success": success, "failed": failed}


def validate_data():
    """Validate data quality in the database."""
    import psycopg2
    
    DATABASE_URL = os.getenv("DATABASE_URL", "")
    db_url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    logger.info("="*60)
    logger.info("📋 DATA QUALITY VALIDATION")
    logger.info("="*60)
    
    # Check each ticker in training universe
    issues = []
    
    for ticker in TRAINING_UNIVERSE:
        cur.execute("""
            SELECT 
                COUNT(*) as total_records,
                MIN(timestamp) as first_date,
                MAX(timestamp) as last_date,
                COUNT(*) FILTER (WHERE volume IS NULL OR volume = 0) as zero_volume_days,
                COUNT(*) FILTER (WHERE close IS NULL) as null_close_days
            FROM stock_prices 
            WHERE ticker = %s
        """, (ticker,))
        
        result = cur.fetchone()
        total, first_date, last_date, zero_vol, null_close = result
        
        if total == 0:
            issues.append(f"{ticker}: NO DATA")
            logger.warning(f"❌ {ticker}: No data found")
        elif total < 500:
            issues.append(f"{ticker}: Only {total} records (need 500+)")
            logger.warning(f"⚠️ {ticker}: Only {total} records (need 500+ for LSTM)")
        else:
            # Check if we have recession data (2008)
            cur.execute("""
                SELECT COUNT(*) FROM stock_prices 
                WHERE ticker = %s AND timestamp < '2010-01-01'
            """, (ticker,))
            pre_2010 = cur.fetchone()[0]
            
            recession_status = "✅" if pre_2010 > 0 else "⚠️ Missing 2008 data"
            
            logger.info(
                f"✅ {ticker}: {total:,} records | "
                f"{first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')} | "
                f"{recession_status}"
            )
    
    cur.close()
    conn.close()
    
    logger.info("="*60)
    if issues:
        logger.warning(f"⚠️ {len(issues)} tickers have issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("✅ All tickers validated successfully!")
    logger.info("="*60)
    
    return len(issues) == 0


def data_quality_cleanup():
    """Clean up data quality issues."""
    import psycopg2
    import pandas as pd
    
    DATABASE_URL = os.getenv("DATABASE_URL", "")
    db_url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    logger.info("🧹 Running data quality cleanup...")
    
    # 1. Forward fill missing closes (within reason)
    # This handles single-day gaps from holidays
    fixed_count = 0
    
    for ticker in TRAINING_UNIVERSE:
        cur.execute("""
            SELECT timestamp, close FROM stock_prices 
            WHERE ticker = %s 
            ORDER BY timestamp
        """, (ticker,))
        
        records = cur.fetchall()
        if len(records) < 2:
            continue
            
        for i in range(1, len(records)):
            if records[i][1] is None:
                prev_close = records[i-1][1]
                cur.execute("""
                    UPDATE stock_prices 
                    SET close = %s 
                    WHERE ticker = %s AND timestamp = %s AND close IS NULL
                """, (prev_close, ticker, records[i][0]))
                fixed_count += 1
    
    conn.commit()
    cur.close()
    conn.close()
    
    logger.info(f"✅ Fixed {fixed_count} null close prices via forward fill")


async def main():
    parser = argparse.ArgumentParser(description="Fetch training data for v7 model")
    parser.add_argument("--validate", action="store_true", help="Validate data quality")
    parser.add_argument("--cleanup", action="store_true", help="Run data cleanup")
    parser.add_argument("--daily", action="store_true", help="Fetch only recent data (last 5 days)")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent fetches")
    args = parser.parse_args()
    
    if args.validate:
        validate_data()
        return
    
    if args.cleanup:
        data_quality_cleanup()
        return
    
    mode = "daily" if args.daily else "full"
    
    logger.info("="*60)
    logger.info("🚀 V7 TRAINING DATA FETCHER")
    logger.info(f"Fetching {len(TRAINING_UNIVERSE)} tickers")
    logger.info(f"Mode: {mode}")
    logger.info("="*60)
    
    # Fetch all data
    results = await fetch_all_data(TRAINING_UNIVERSE, mode=mode, concurrency=args.concurrency)
    
    total_records = sum(r["records"] for r in results["success"])
    logger.info(f"📊 Total records fetched: {total_records:,}")
    
    # Run validation
    logger.info("")
    validate_data()
    
    logger.info("")
    logger.info("="*60)
    logger.info("🎉 Data fetch complete!")
    logger.info("Next step: Run training with:")
    logger.info("  docker exec proxmox_stock_backend python -m scripts.train_model_v7")
    logger.info("="*60)


if __name__ == "__main__":
    asyncio.run(main())
