import asyncio
import logging
import sys
import os

# Add current directory to sys.path to allow imports
sys.path.append(os.getcwd())

# Set DB URL for local debugging BEFORE importing services.db
os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres:password@localhost:5432/stock_predictor"

from services.data_ingest import AlphaVantageClient
from services.db import get_latest_close

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

async def main():
    ticker = "AMD"
    logger.info(f"Testing Alpha Vantage ingestion for {ticker}...")
    
    try:
        client = AlphaVantageClient()
        await client.fetch_data(ticker)
        logger.info("Fetch data call completed.")
        
        # Verify
        price = await get_latest_close(ticker)
        if price:
            logger.info(f"✅ SUCCESS: Found latest close for {ticker}: {price}")
        else:
            logger.error(f"❌ FAILURE: No data found for {ticker} in DB after ingestion.")
            
    except Exception as e:
        logger.error(f"❌ EXCEPTION: {e}")

if __name__ == "__main__":
    asyncio.run(main())
