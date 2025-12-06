import asyncio
import sys
import os
import csv

# Add backend to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from services.db import init_db
from services.data_ingest import AlphaVantageClient

async def seed_data():
    print("Initializing database...")
    await init_db()
    print("Database initialized.")

    # Path to stock list
    stock_list_path = os.path.join(os.path.dirname(__file__), '../../legacy/LSTM_AI_Stock_Predictor/TrainingData/stockList.csv')
    
    if not os.path.exists(stock_list_path):
        print(f"Error: Stock list not found at {stock_list_path}")
        return

    tickers = []
    with open(stock_list_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                tickers.append(row[0])
    
    # Pick first 3 tickers
    target_tickers = tickers[:3]
    print(f"Seeding data for: {target_tickers}")

    client = AlphaVantageClient()

    for ticker in target_tickers:
        await client.fetch_daily_data(ticker)

    print("Seeding complete.")

if __name__ == "__main__":
    # Load environment variables if not already loaded
    from dotenv import load_dotenv
    load_dotenv()
    
    asyncio.run(seed_data())
