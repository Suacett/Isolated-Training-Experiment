#!/usr/bin/env python3
"""
Bulk Ingest Script

Reads CSV files from data/training_raw/ and inserts into database.

Usage:
    docker exec proxmox_stock_backend python -m scripts.ingest_bulk
"""

import asyncio
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.dialects.postgresql import insert
from services.db import StockPrice, AsyncSessionLocal

DATA_DIR = Path(__file__).parent.parent / "data" / "training_raw"
REQUIRED_COLUMNS = {'date', 'open', 'high', 'low', 'close', 'volume'}


async def ingest_file(csv_path: Path) -> tuple[str, int, str | None]:
    """
    Ingest a single CSV file into the database.
    
    Returns: (ticker, records_count, error_message)
    """
    ticker = csv_path.stem  # filename without extension
    
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Normalize column names
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Validate columns
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            return (ticker, 0, f"Missing columns: {missing}")
        
        # Parse dates
        df['date'] = pd.to_datetime(df['date'])
        
        # Drop rows with NaN in critical columns
        df = df.dropna(subset=['date', 'close'])
        
        if df.empty:
            return (ticker, 0, "No valid data rows")
        
        # Prepare data for insertion
        stock_prices = []
        for _, row in df.iterrows():
            stock_prices.append({
                "ticker": ticker,
                "timestamp": row['date'].to_pydatetime(),
                "open": float(row['open']) if pd.notna(row['open']) else 0.0,
                "high": float(row['high']) if pd.notna(row['high']) else 0.0,
                "low": float(row['low']) if pd.notna(row['low']) else 0.0,
                "close": float(row['close']),
                "volume": float(row['volume']) if pd.notna(row['volume']) else 0.0
            })
        
        # Bulk upsert in batches
        BATCH_SIZE = 1000
        async with AsyncSessionLocal() as session:
            for i in range(0, len(stock_prices), BATCH_SIZE):
                batch = stock_prices[i:i + BATCH_SIZE]
                stmt = insert(StockPrice).values(batch)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[StockPrice.ticker, StockPrice.timestamp],
                    set_={
                        "open": stmt.excluded.open,
                        "high": stmt.excluded.high,
                        "low": stmt.excluded.low,
                        "close": stmt.excluded.close,
                        "volume": stmt.excluded.volume
                    }
                )
                await session.execute(stmt)
            await session.commit()
        
        return (ticker, len(stock_prices), None)
        
    except Exception as e:
        return (ticker, 0, str(e))


async def ingest_batch(files: list[Path]) -> list[tuple[str, int, str | None]]:
    """Ingest a batch of files in parallel."""
    tasks = [ingest_file(f) for f in files]
    return await asyncio.gather(*tasks)


async def main():
    """Main entry point."""
    print("=" * 60)
    print("📥 Bulk Database Ingest")
    print("=" * 60)
    
    # Find all CSV files
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found in {DATA_DIR}")
        return 1
    
    print(f"📂 Found {len(csv_files)} CSV files to ingest")
    
    # Process in batches of 5
    BATCH_SIZE = 5
    total_success = 0
    total_records = 0
    failed = []
    
    for i in range(0, len(csv_files), BATCH_SIZE):
        batch = csv_files[i:i + BATCH_SIZE]
        results = await ingest_batch(batch)
        
        for ticker, count, error in results:
            if error:
                failed.append((ticker, error))
            else:
                total_success += 1
                total_records += count
        
        # Progress
        processed = min(i + BATCH_SIZE, len(csv_files))
        print(f"  Progress: {processed}/{len(csv_files)} files ({total_success} success, {len(failed)} failed)")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"✅ Ingested {total_success} tickers ({total_records:,} total records)")
    
    if failed:
        print(f"⚠️ Failed: {len(failed)} tickers")
        for ticker, error in failed[:5]:
            print(f"   - {ticker}: {error}")
        if len(failed) > 5:
            print(f"   ... and {len(failed) - 5} more")
    
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
