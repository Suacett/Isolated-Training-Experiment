#!/usr/bin/env python3
"""
Training Data Fetcher - FULL S&P 500 + Extended Universe

This script fetches the ENTIRE S&P 500 (500+ stocks) plus additional sector ETFs
and international indices. This provides the massive dataset needed for V8's
non-overlapping window approach (which throws away 98% of data).

Data is fetched from 2007 onwards to capture:
- 2008 Financial Crisis (crash patterns)
- 2010 Flash Crash
- 2020 COVID crash (V-shape recovery)
- 2022 Bear Market (inflation regime)

Usage:
    docker exec proxmox_stock_backend python -m scripts.fetch_sp500_full
    docker exec proxmox_stock_backend python -m scripts.fetch_sp500_full --validate
    docker exec proxmox_stock_backend python -m scripts.fetch_sp500_full --daily
"""

import sys
import os
import asyncio
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fetch_sp500_full.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def get_sp500_tickers() -> list:
    """
    Dynamically fetch the current S&P 500 constituents from Wikipedia.
    This ensures we always have the latest list.
    """
    logger.info("📊 Fetching current S&P 500 constituents from Wikipedia...")
    
    try:
        # Read S&P 500 list from Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]  # First table is the S&P 500 list
        
        # Get ticker symbols
        tickers = sp500_table['Symbol'].tolist()
        
        # Clean up tickers (some have dots that need to be replaced with dashes for Yahoo)
        tickers = [t.replace('.', '-') for t in tickers]
        
        logger.info(f"✅ Found {len(tickers)} S&P 500 stocks")
        return tickers
    except Exception as e:
        logger.warning(f"Failed to fetch S&P 500 from Wikipedia: {e}")
        logger.warning("Falling back to hardcoded list...")
        return get_sp500_fallback()


def get_sp500_fallback() -> list:
    """Fallback list of S&P 500 stocks (top 200 by market cap)."""
    return [
        # Top 50 by market cap
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "V", "UNH",
        "JNJ", "JPM", "XOM", "WMT", "PG", "MA", "HD", "CVX", "LLY", "ABBV",
        "MRK", "KO", "PEP", "AVGO", "COST", "PFE", "TMO", "CSCO", "MCD", "ACN",
        "ABT", "DHR", "ADBE", "DIS", "WFC", "VZ", "CMCSA", "NKE", "TXN", "NEE",
        "PM", "CRM", "RTX", "BMY", "INTC", "LIN", "ORCL", "COP", "AMD", "QCOM",
        # 51-100
        "T", "UPS", "HON", "IBM", "AMGN", "LOW", "GE", "CAT", "ELV", "SPGI",
        "BA", "MS", "GS", "BLK", "INTU", "DE", "SBUX", "ISRG", "ADP", "BKNG",
        "CVS", "GILD", "MDLZ", "PLD", "AMT", "REGN", "TJX", "AXP", "ADI", "CI",
        "SYK", "ZTS", "VRTX", "LRCX", "MO", "CB", "TMUS", "NOW", "CME", "EOG",
        "SCHW", "NOC", "SNPS", "PGR", "DUK", "SO", "CDNS", "SLB", "MMC", "BSX",
        # 101-150
        "ITW", "BDX", "MU", "APD", "FDX", "ETN", "CSX", "CL", "ATVI", "ICE",
        "HUM", "MCK", "GD", "TGT", "EMR", "FISV", "PNC", "NSC", "WM", "NXPI",
        "MAR", "ROP", "ECL", "USB", "AON", "FCX", "MCO", "GM", "HCA", "PSX",
        "KMB", "APH", "AZO", "SRE", "CTSH", "VLO", "F", "CNC", "MPC", "ADM",
        "CTAS", "PCAR", "OXY", "OKE", "PSA", "KLAC", "A", "TEL", "CMG", "MCHP",
        # 151-200
        "D", "AEP", "AFL", "SHW", "TRV", "PH", "DHI", "NEM", "ALL", "GIS",
        "DXCM", "KMI", "EW", "DD", "TT", "HES", "MSCI", "IQV", "PAYX", "YUM",
        "WELL", "PRU", "HAL", "STZ", "WMB", "LEN", "MTD", "PPG", "EXC", "IDXX",
        "STT", "EA", "FTNT", "AMP", "ALB", "ODFL", "KEYS", "BK", "FRC", "WST",
        "DLR", "ROK", "RMD", "FIS", "CPRT", "KDP", "HPQ", "GWW", "BIIB", "CHD"
    ]


# === EXTENDED UNIVERSE (Beyond S&P 500) ===
EXTENDED_UNIVERSE = [
    # Major Indices / ETFs
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO",
    
    # Sector ETFs
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC",
    "SMH",  # Semiconductors
    "KRE",  # Regional Banks
    "XBI",  # Biotech
    "VNQ",  # Real Estate
    "GDX",  # Gold Miners
    
    # International
    "EFA",  # EAFE (Europe, Australasia, Far East)
    "EEM",  # Emerging Markets
    "VEU",  # Non-US Developed
    "FXI",  # China
    "EWJ",  # Japan
    "EWG",  # Germany
    "EWZ",  # Brazil
    
    # Commodities & Alternatives
    "GLD",  # Gold
    "SLV",  # Silver
    "USO",  # Oil
    "UNG",  # Natural Gas
    
    # Fixed Income
    "TLT",  # 20+ Year Treasury
    "IEF",  # 7-10 Year Treasury
    "HYG",  # High Yield
    "LQD",  # Investment Grade Corporate
    
    # Currency
    "UUP",  # US Dollar Index
    
    # Volatility
    "VIXY", # VIX Short-Term Futures
    
    # Popular non-S&P stocks
    "PLTR", "COIN", "RBLX", "SHOP", "SQ", "SNAP", "SOFI", "RIVN", "LCID",
    "ARKK", "MARA", "RIOT", "SPCE", "NIO", "BABA"
]


async def fetch_ticker_data(ticker: str, mode: str = "full") -> dict:
    """Fetch data for a single ticker."""
    from services.data_ingest import YahooFinanceClient
    
    client = YahooFinanceClient()
    try:
        records = await client.fetch_data(ticker, mode=mode)
        return {"ticker": ticker, "records": records, "status": "success"}
    except Exception as e:
        return {"ticker": ticker, "error": str(e), "status": "failed"}


async def fetch_all_data(tickers: list, mode: str = "full", concurrency: int = 10):
    """Fetch data for all tickers with rate limiting."""
    
    sem = asyncio.Semaphore(concurrency)
    
    async def fetch_with_semaphore(ticker):
        async with sem:
            result = await fetch_ticker_data(ticker, mode)
            await asyncio.sleep(0.3)  # Small delay to avoid rate limiting
            return result
    
    logger.info(f"📊 Fetching {len(tickers)} tickers (mode={mode})...")
    
    # Process in batches to show progress
    batch_size = 50
    all_results = []
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        logger.info(f"  Processing batch {i//batch_size + 1}/{(len(tickers) + batch_size - 1)//batch_size} ({len(batch)} tickers)...")
        
        results = await asyncio.gather(*[fetch_with_semaphore(t) for t in batch])
        all_results.extend(results)
        
        success = sum(1 for r in results if r["status"] == "success")
        logger.info(f"    ✅ {success}/{len(batch)} success")
    
    success = [r for r in all_results if r["status"] == "success"]
    failed = [r for r in all_results if r["status"] == "failed"]
    
    return {"success": success, "failed": failed}


def get_database_stats():
    """Get current database statistics."""
    import psycopg2
    
    DATABASE_URL = os.getenv("DATABASE_URL", "")
    db_url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    # Total unique tickers
    cur.execute("SELECT COUNT(DISTINCT ticker) FROM stock_prices")
    total_tickers = cur.fetchone()[0]
    
    # Total records
    cur.execute("SELECT COUNT(*) FROM stock_prices")
    total_records = cur.fetchone()[0]
    
    # Tickers with 500+ records (usable for training)
    cur.execute("""
        SELECT COUNT(*) FROM (
            SELECT ticker FROM stock_prices 
            GROUP BY ticker HAVING COUNT(*) >= 500
        ) t
    """)
    usable_tickers = cur.fetchone()[0]
    
    # Date range
    cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM stock_prices")
    min_date, max_date = cur.fetchone()
    
    cur.close()
    conn.close()
    
    return {
        "total_tickers": total_tickers,
        "total_records": total_records,
        "usable_tickers": usable_tickers,
        "min_date": min_date,
        "max_date": max_date
    }


def validate_data():
    """Validate data quality in the database."""
    import psycopg2
    
    DATABASE_URL = os.getenv("DATABASE_URL", "")
    db_url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    stats = get_database_stats()
    
    logger.info("=" * 60)
    logger.info("📋 DATABASE STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total unique tickers: {stats['total_tickers']:,}")
    logger.info(f"Tickers with 500+ records (usable): {stats['usable_tickers']:,}")
    logger.info(f"Total records: {stats['total_records']:,}")
    if stats['min_date'] and stats['max_date']:
        logger.info(f"Date range: {stats['min_date'].strftime('%Y-%m-%d')} to {stats['max_date'].strftime('%Y-%m-%d')}")
    
    # V8 Non-Overlapping Window estimate
    # With stride=126, each 4000-day stock gives ~30 samples
    # Usable = 500+ records = ~3-4 years = ~1000 records = ~7 samples
    # With 500 stocks = 500 * 7 = 3500 samples
    estimated_v8_samples = stats['usable_tickers'] * 7
    logger.info(f"\n🎯 Estimated V8 training samples (non-overlapping): ~{estimated_v8_samples:,}")
    
    # Check for stocks with long history (2008)
    cur.execute("""
        SELECT COUNT(*) FROM (
            SELECT ticker FROM stock_prices 
            WHERE timestamp < '2010-01-01'
            GROUP BY ticker HAVING COUNT(*) >= 100
        ) t
    """)
    stocks_with_2008 = cur.fetchone()[0]
    logger.info(f"Stocks with 2008 crisis data: {stocks_with_2008:,}")
    
    cur.close()
    conn.close()
    
    logger.info("=" * 60)
    return stats


async def main():
    parser = argparse.ArgumentParser(description="Fetch FULL S&P 500 training data")
    parser.add_argument("--validate", action="store_true", help="Validate data quality")
    parser.add_argument("--daily", action="store_true", help="Fetch only recent data (last 5 days)")
    parser.add_argument("--extended", action="store_true", help="Include extended universe (ETFs, international)")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent fetches")
    parser.add_argument("--skip-sp500", action="store_true", help="Skip S&P 500 (only fetch extended)")
    args = parser.parse_args()
    
    if args.validate:
        validate_data()
        return
    
    mode = "daily" if args.daily else "full"
    
    # Build ticker list
    all_tickers = []
    
    if not args.skip_sp500:
        sp500 = get_sp500_tickers()
        all_tickers.extend(sp500)
    
    if args.extended or args.skip_sp500:
        all_tickers.extend(EXTENDED_UNIVERSE)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tickers = []
    for t in all_tickers:
        if t not in seen:
            seen.add(t)
            unique_tickers.append(t)
    
    logger.info("=" * 60)
    logger.info("🚀 FULL S&P 500 + EXTENDED DATA FETCHER")
    logger.info(f"Fetching {len(unique_tickers)} tickers")
    logger.info(f"Mode: {mode}")
    logger.info(f"Extended universe: {args.extended}")
    logger.info("=" * 60)
    
    # Show current stats
    try:
        stats = get_database_stats()
        logger.info(f"Current DB: {stats['usable_tickers']} usable tickers, {stats['total_records']:,} records")
    except:
        pass
    
    logger.info("")
    
    # Fetch all data
    results = await fetch_all_data(unique_tickers, mode=mode, concurrency=args.concurrency)
    
    total_records = sum(r["records"] for r in results["success"])
    logger.info(f"\n📊 Fetch Results:")
    logger.info(f"   Success: {len(results['success'])}")
    logger.info(f"   Failed: {len(results['failed'])}")
    logger.info(f"   Total new records: {total_records:,}")
    
    if results["failed"]:
        logger.warning(f"   Failed tickers: {[r['ticker'] for r in results['failed'][:10]]}...")
    
    # Show updated stats
    logger.info("\n")
    validate_data()
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 Data fetch complete!")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Run V7 lazy audit:")
    logger.info("     docker exec proxmox_stock_backend python -m scripts.audit_v7")
    logger.info("")
    logger.info("  2. Train V8 classification model:")
    logger.info("     docker exec proxmox_stock_backend python -m scripts.train_model_v8_class")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
