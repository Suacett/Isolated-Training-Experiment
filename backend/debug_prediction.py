import asyncio
import logging
import sys
import os
import torch
import pandas as pd
import numpy as np

# Add current directory to sys.path to allow imports
sys.path.append(os.getcwd())

# Set DB URL for local debugging BEFORE importing services.db
os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres:password@localhost:5432/stock_predictor"

from services.db import get_historical_data
from services.feature_engineering import process_stock_data, get_model_input_features
from services.lstm_model import LSTMModel
from services.scaler import FeatureScaler

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

async def main():
    ticker = "AMD"
    logger.info(f"Testing Prediction Pipeline for {ticker}...")
    
    # 1. Load Model and Scaler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    scaler_path = "models/scaler.pkl"
    model_path = "models/lstm_model.pth"
    
    scaler = FeatureScaler(scaler_path)
    if not scaler.load():
        logger.error("Failed to load scaler")
        return

    model = LSTMModel(input_dim=37, window_size=60, device=device)
    try:
        model.load(model_path)
        logger.info("Model loaded")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # 2. Get Data
    historical_data = await get_historical_data(ticker, limit=100)
    logger.info(f"Fetched {len(historical_data)} records")
    
    if len(historical_data) < 60:
        logger.error("Not enough data")
        return

    # 3. Process Features
    df = pd.DataFrame([{
        "date": d.timestamp,
        "open": d.open,
        "high": d.high,
        "low": d.low,
        "close": d.close,
        "volume": d.volume
    } for d in historical_data])
    
    processed_df = process_stock_data(df, create_targets=False)
    feature_cols = get_model_input_features()
    
    logger.info(f"Processed DF shape: {processed_df.shape}")
    
    if len(processed_df) >= 60:
        # Get latest sequence of 60 steps
        seq_data = processed_df.iloc[-60:][feature_cols].values
        
        # Scale
        scaled_seq = scaler.transform(seq_data)
        
        # Input shape should be (Batch, SeqLen, Features) -> (1, 60, 37)
        # The model handles the permute for Conv1D internally.
        input_tensor = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0)
        
        logger.info(f"Input Tensor Shape: {input_tensor.shape}")
        
        try:
            prediction = model.predict(input_tensor, device)
            logger.info(f"✅ SUCCESS: Prediction: {prediction}")
        except Exception as e:
            logger.error(f"❌ PREDICTION FAILED: {e}")
    else:
        logger.error("Not enough processed data")

if __name__ == "__main__":
    asyncio.run(main())
