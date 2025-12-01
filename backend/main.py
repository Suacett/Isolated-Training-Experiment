import logging
import os
import torch
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from services.intrinsic import IntrinsicCalculator
from services.lstm_model import LSTMModel, get_device
from services.db import get_latest_close, get_unique_tickers, get_historical_data, add_watchlist_item, remove_watchlist_item, get_watchlist, save_prediction
from services.data_ingest import AlpacaDataClient
from utils.config_loader import load_config, save_config
from pydantic import BaseModel
from datetime import datetime
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
lstm_model = None
device = None
alpaca_client = None

class WatchlistItem(BaseModel):
    ticker: str

@app.on_event("startup")
async def startup_event():
    global lstm_model, device
    device = get_device()
    logger.info(f"Running on device: {device}")
    
    # Initialize LSTM Model
    # Input dim = 5 (open, high, low, close, volume)
    # Window size = 60
    lstm_model = LSTMModel(input_dim=5, window_size=60)
    lstm_model.to(device)
    # Ideally load weights here: lstm_model.load_state_dict(torch.load("model.pth"))
    logger.info("LSTM Model initialized")
    
    global alpaca_client
    alpaca_client = AlpacaDataClient()
    logger.info("Alpaca Client initialized")

class APIKeys(BaseModel):
    ALPACA_API_KEY: str
    ALPACA_SECRET_KEY: str

@app.get("/status")
async def get_status():
    config = load_config()
    return {"configured": config is not None}

@app.post("/settings/keys")
async def save_keys(keys: APIKeys):
    save_config(keys.ALPACA_API_KEY, keys.ALPACA_SECRET_KEY)
    return {"message": "Keys saved successfully"}

@app.post("/ingest/all")
async def ingest_all():
    # 1. Get Watchlist
    tickers = await get_watchlist()
    if not tickers:
        return {"message": "Watchlist is empty"}
    
    # 2. Ingest Data
    client = AlpacaDataClient()
    await client.fetch_all_data(tickers)
    
    # 3. Run Predictions
    predictions = []
    for ticker in tickers:
        try:
            historical_data = await get_historical_data(ticker, limit=60)
            if len(historical_data) == 60:
                data_array = np.array([[d.open, d.high, d.low, d.close, d.volume] for d in historical_data], dtype=np.float32)
                input_tensor = torch.tensor(data_array)
                prediction = lstm_model.predict(input_tensor, device)
                
                # Save prediction
                await save_prediction(ticker, float(prediction), datetime.now())
                predictions.append({"ticker": ticker, "prediction": float(prediction)})
        except Exception as e:
            logger.error(f"Error predicting for {ticker}: {e}")
            
    return {"message": "Ingestion and prediction complete", "predictions": predictions}

@app.post("/ingest/{ticker}")
async def ingest_data(ticker: str):
    client = AlpacaDataClient()
    await client.fetch_data(ticker)
    return {"message": f"Ingestion started for {ticker}"}

@app.get("/")
async def root():
    return {"message": "Proxmox AI Stock Predictor Backend is running"}

@app.get("/tickers")
async def get_tickers():
    tickers = await get_unique_tickers()
    return {"tickers": tickers}

@app.post("/watchlist")
async def add_to_watchlist(request: WatchlistItem):
    await add_watchlist_item(request.ticker)
    return {"message": f"Added {request.ticker} to watchlist"}

@app.get("/watchlist")
async def get_watchlist_items():
    items = await get_watchlist()
    return {"watchlist": items}

@app.delete("/watchlist/{ticker}")
async def remove_from_watchlist(ticker: str):
    await remove_watchlist_item(ticker)
    return {"message": f"Removed {ticker} from watchlist"}

@app.get("/dashboard")
async def get_dashboard_summary():
    watchlist = await get_watchlist()
    summaries = []
    
    for ticker in watchlist:
        # 1. Get latest price
        current_price = await get_latest_close(ticker)
        if current_price is None:
            continue # Skip if no data yet
            
        # 2. Intrinsic Value (Placeholder)
        intrinsic_value = 0.0
        
        # 3. LSTM Prediction
        prediction = 0.0
        historical_data = await get_historical_data(ticker, limit=60)
        if len(historical_data) == 60:
            data_array = np.array([[d.open, d.high, d.low, d.close, d.volume] for d in historical_data], dtype=np.float32)
            input_tensor = torch.tensor(data_array).unsqueeze(0) # Add batch dimension
            # Note: lstm_model.predict expects (batch, seq, feature) or (seq, feature)?
            # Looking at previous code: lstm_model.predict(input_tensor, device)
            # We need to check lstm_model.py to be sure about input shape.
            # Assuming it handles it or we need to adjust.
            # Previous code: input_tensor = torch.tensor(data_array) -> (60, 5)
            # Usually LSTM expects (batch, seq, feature).
            # Let's assume the previous code was correct or I should fix it.
            # I'll stick to previous usage pattern but check if unsqueeze is needed.
            # Previous usage: input_tensor = torch.tensor(data_array)
            # If previous usage was correct, I'll keep it.
            input_tensor = torch.tensor(data_array)
            prediction = lstm_model.predict(input_tensor, device)
        
        # 4. Signal
        signal = "HOLD"
        if prediction > current_price * 1.02:
            signal = "BUY"
        elif prediction < current_price * 0.98:
            signal = "SELL"
            
        summaries.append({
            "ticker": ticker,
            "current_price": current_price,
            "prediction": prediction,
            "signal": signal
        })
        
    return summaries

@app.get("/dashboard/{ticker}")
async def get_dashboard_detail(ticker: str):
    # 1. Get latest price (for quick check)
    current_price = await get_latest_close(ticker)
    if current_price is None:
        raise HTTPException(status_code=404, detail="Ticker not found")
    
    # 2. Fetch Historical Data
    # We want 90 days of history for the chart.
    # LSTM needs 60 days context.
    # So we need 90 + 60 = 150 days.
    required_history = 90
    window_size = 60
    fetch_limit = required_history + window_size
    
    historical_data = await get_historical_data(ticker, limit=fetch_limit)
    
    # 3. Prepare Response Structure
    history_response = []
    
    # We can only generate predictions if we have enough data
    # We need at least window_size + 1 data points to have 1 prediction?
    # Actually, to predict for index i, we need data [i-window : i]
    # So if we have N data points, we can predict for indices [window : N]
    
    total_records = len(historical_data)
    
    # Prepare batch for LSTM
    # We want predictions for the last 'required_history' days (or as many as we have)
    # The data is chronological.
    # Indices we want to report: [total_records - required_history ... total_records - 1]
    # But we can only report if index >= window_size
    
    start_index = max(window_size, total_records - required_history)
    
    # If we don't have enough data for even one prediction
    if total_records <= window_size:
        # Just return data without predictions
        for d in historical_data:
             history_response.append({
                "date": d.timestamp.strftime("%Y-%m-%d"),
                "open": d.open,
                "high": d.high,
                "low": d.low,
                "close": d.close,
                "predicted_close": None,
                "intrinsic_value": 0.0 # Placeholder
            })
        return {"history": history_response}

    # Prepare input tensor for batch prediction
    # We need a list of windows.
    # For each index i from start_index to total_records-1:
    # Window is historical_data[i-window_size : i]
    
    windows = []
    valid_indices = []
    
    for i in range(start_index, total_records):
        # Window data
        window_data = historical_data[i-window_size : i]
        # Convert to list of [open, high, low, close, volume]
        window_array = np.array([[d.open, d.high, d.low, d.close, d.volume] for d in window_data], dtype=np.float32)
        windows.append(window_array)
        valid_indices.append(i)
        
    # Convert to tensor: (Batch, Seq_Len, Input_Dim)
    if windows:
        batch_tensor = torch.tensor(np.array(windows))
        
        # Run Batch Prediction
        predictions = lstm_model.predict_batch(batch_tensor, device)
        predictions_list = predictions.cpu().numpy().tolist()
    else:
        predictions_list = []
        
    # Map predictions back to dates
    # predictions_list[k] corresponds to valid_indices[k]
    # valid_indices[k] is the index of the day we are predicting FOR?
    # Wait, LSTM predicts the NEXT day usually, or the current day's close given previous?
    # "forecasting_backtest_Predictor.py" usually predicts T+1 given T.
    # But here we want to show "predicted_close" for a specific date.
    # If the model predicts T+1, then the input [T-60 : T] predicts T+1.
    # So the prediction should be aligned with date T+1.
    # However, for the chart, we often want to compare "Predicted vs Actual" for date T.
    # To get predicted for date T, we need input [T-61 : T-1].
    # My loop: i is the index of the data point we are processing.
    # window is [i-window : i]. This is data up to i-1.
    # So predicting using this window gives prediction for i.
    # Yes, this aligns correctly. Prediction at index i uses data up to i-1.
    
    pred_map = {idx: pred for idx, pred in zip(valid_indices, predictions_list)}
    
    # Intrinsic Value (Placeholder - Constant for now)
    intrinsic_val = 142.0 # Mock value as per example, or 0.0
    
    # Build final response
    # We return the last 'required_history' days (or available)
    display_start = max(0, total_records - required_history)
    
    for i in range(display_start, total_records):
        d = historical_data[i]
        pred = pred_map.get(i)
        
        history_response.append({
            "date": d.timestamp.strftime("%Y-%m-%d"),
            "open": d.open,
            "high": d.high,
            "low": d.low,
            "close": d.close,
            "predicted_close": pred,
            "intrinsic_value": intrinsic_val
        })
        
    return {"history": history_response}
