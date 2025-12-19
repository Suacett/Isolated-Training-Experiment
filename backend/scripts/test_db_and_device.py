import asyncio
import sys
import os
from datetime import datetime

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.db import AsyncSessionLocal, bulk_upsert_stock_prices
from services.transformer_model import TransformerRankModel
from services.device_utils import get_device

async def test_bulk_upsert():
    print("\n--- Testing Bulk Upsert ---")
    async with AsyncSessionLocal() as session:
        # Create some dummy data
        dummy_data = [
            {
                "ticker": "TEST_TICKER",
                "timestamp": datetime(2025, 1, 1),
                "open": 100.0,
                "high": 110.0,
                "low": 90.0,
                "close": 105.0,
                "volume": 1000.0
            },
            {
                "ticker": "TEST_TICKER",
                "timestamp": datetime(2025, 1, 2),
                "open": 105.0,
                "high": 115.0,
                "low": 100.0,
                "close": 110.0,
                "volume": 1100.0
            }
        ]
        
        print(f"Upserting {len(dummy_data)} records...")
        await bulk_upsert_stock_prices(session, dummy_data)
        print("✅ Bulk upsert completed.")

async def test_device_detection():
    print("\n--- Testing Device Detection ---")
    device = get_device()
    print(f"Detected device: {device}")
    
    # Initialize model to check if it uses the device
    model = TransformerRankModel(input_dim=12, d_model=64)
    print(f"Model device: {model.device}")
    
    if str(device) == str(model.device):
        print("✅ Device detection matches model initialization.")
    else:
        print("❌ Device detection mismatch!")

async def main():
    try:
        await test_device_detection()
        # We might not be able to connect to the DB if the container is down and we are running locally
        # but let's try.
        await test_bulk_upsert()
    except Exception as e:
        print(f"\nCaught expected or unexpected error: {e}")
        print("Note: DB test might fail if PostgreSQL is not reachable locally.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
