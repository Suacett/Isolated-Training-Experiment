import logging
import sys
import re
from pathlib import Path
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from services.lstm_model import LSTMModel, get_device
from services.db import get_latest_close, get_unique_tickers, get_historical_data, add_watchlist_item, remove_watchlist_item, get_watchlist, save_prediction
from services.data_ingest import AlpacaDataClient
from utils.config_loader import settings, save_config, clear_config, is_alpaca_configured, is_alpha_vantage_configured
from state import state
from routers import ingestion

# Configure logging
LOG_FILE = Path("backend.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Include Routers
app.include_router(ingestion.router)

# Model weights path
MODEL_WEIGHTS_PATH = Path(__file__).parent / "models" / "lstm_model.pth"

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WatchlistItem(BaseModel):
    ticker: str

class APIKeys(BaseModel):
    ALPACA_API_KEY: Optional[str] = None
    ALPACA_SECRET_KEY: Optional[str] = None
    ALPHA_VANTAGE_KEY: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    # Dynamic Device Detection
    state.device = get_device()
    logger.info(f"Initializing Local LSTM Model on device: {state.device}")

    # Create models directory if it doesn't exist
    MODEL_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Try to load existing model weights
    if MODEL_WEIGHTS_PATH.exists():
        try:
            state.lstm_model = LSTMModel.load(str(MODEL_WEIGHTS_PATH), state.device)
            logger.info(f"✅ Successfully loaded trained model from {MODEL_WEIGHTS_PATH}")
        except Exception as e:
            logger.error(f"❌ Failed to load model weights: {e}. Creating new model.")
            state.lstm_model = LSTMModel(input_dim=5, window_size=60, device=state.device)
    else:
        # Initialize new LSTM Model
        state.lstm_model = LSTMModel(input_dim=5, window_size=60, device=state.device)
        logger.info("Initialized new LSTM model (no trained weights found)")

    if state.lstm_model:
        logger.info(f"✅ AI Model Active: LSTM (Input: {state.lstm_model.input_dim}, Hidden: {state.lstm_model.hidden_dim})")

    # Log API Key Status
    if is_alpaca_configured():
        logger.info("✅ Alpaca API Keys: Loaded")
    else:
        logger.warning("⚠️ Alpaca API Keys: Missing")

    if is_alpha_vantage_configured():
        logger.info("✅ Alpha Vantage Key: Loaded")
    else:
        logger.warning("⚠️ Alpha Vantage Key: Missing")

@app.get("/status")
async def get_status():
    has_alpaca = is_alpaca_configured()
    has_alpha_vantage = is_alpha_vantage_configured()
    return {
        "configured": has_alpaca or has_alpha_vantage,
        "alpaca_configured": has_alpaca,
        "alpha_vantage_configured": has_alpha_vantage
    }

@app.post("/settings/keys")
async def save_keys(keys: APIKeys):
    try:
        # Verify Alpaca keys if provided
        if keys.ALPACA_API_KEY and keys.ALPACA_SECRET_KEY:
            try:
                test_client = AlpacaDataClient(api_key=keys.ALPACA_API_KEY, secret_key=keys.ALPACA_SECRET_KEY)
                test_client.verify_credentials()
            except Exception as e:
                logger.error(f"Alpaca keys validation failed: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid API Keys: {str(e)}")

        save_config(
            alpaca_api_key=keys.ALPACA_API_KEY,
            alpaca_secret_key=keys.ALPACA_SECRET_KEY,
            alpha_vantage_key=keys.ALPHA_VANTAGE_KEY
        )
        
        return {"message": "Keys saved successfully", "status": "valid"}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Failed to save keys: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save keys: {str(e)}")

@app.delete("/settings/keys")
async def reset_keys():
    try:
        clear_config()
        return {"message": "Keys reset successfully"}
    except Exception as e:
        logger.error(f"Failed to reset keys: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset keys: {str(e)}")

def mask_log_line(line: str) -> str:
    # Mask potential API keys
    line = re.sub(r'(PK[A-Z0-9]{10,})', 'PK***********', line)
    return line

@app.get("/logs")
async def get_logs():
    """Return the last 100 lines of logs."""
    if not LOG_FILE.exists():
        return {"logs": []}
    
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
            masked_lines = [mask_log_line(line) for line in lines[-100:]]
            return {"logs": masked_lines}
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        return {"logs": [f"Error reading logs: {e}"]}

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
        current_price = await get_latest_close(ticker)
        if current_price is None:
            continue 
            
        intrinsic_value = 0.0
        prediction = 0.0
        
        # Use global state model
        if state.lstm_model:
            try:
                historical_data = await get_historical_data(ticker, limit=60)
                if len(historical_data) == 60:
                    data_array = np.array([[d.open, d.high, d.low, d.close, d.volume] for d in historical_data], dtype=np.float32)
                    input_tensor = torch.tensor(data_array)
                    prediction = state.lstm_model.predict(input_tensor, state.device)
            except Exception as e:
                logger.error(f"Prediction error for {ticker}: {e}")
        
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
    current_price = await get_latest_close(ticker)
    if current_price is None:
        raise HTTPException(status_code=404, detail="Ticker not found")
    
    required_history = 90
    window_size = 60
    fetch_limit = required_history + window_size
    
    historical_data = await get_historical_data(ticker, limit=fetch_limit)
    history_response = []
    
    total_records = len(historical_data)
    start_index = max(window_size, total_records - required_history)
    
    if total_records <= window_size or not state.lstm_model:
        for d in historical_data:
             history_response.append({
                "date": d.timestamp.strftime("%Y-%m-%d"),
                "open": d.open,
                "high": d.high,
                "low": d.low,
                "close": d.close,
                "predicted_close": None,
                "intrinsic_value": 0.0 
            })
        return {"history": history_response}

    windows = []
    valid_indices = []
    
    for i in range(start_index, total_records):
        window_data = historical_data[i-window_size : i]
        window_array = np.array([[d.open, d.high, d.low, d.close, d.volume] for d in window_data], dtype=np.float32)
        windows.append(window_array)
        valid_indices.append(i)
        
    if windows:
        batch_tensor = torch.tensor(np.array(windows))
        predictions = state.lstm_model.predict_batch(batch_tensor, state.device)
        predictions_list = predictions.cpu().numpy().tolist()
    else:
        predictions_list = []
        
    pred_map = {idx: pred for idx, pred in zip(valid_indices, predictions_list)}
    intrinsic_val = 142.0 
    
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

@app.post("/model/save")
async def save_model():
    if state.lstm_model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    try:
        state.lstm_model.save(str(MODEL_WEIGHTS_PATH))
        return {"message": f"Model saved to {MODEL_WEIGHTS_PATH}"}
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/status")
async def get_model_status():
    if state.lstm_model is None:
        return {"initialized": False}

    return {
        "initialized": True,
        "device": str(state.device),
        "input_dim": state.lstm_model.input_dim,
        "hidden_dim": state.lstm_model.hidden_dim,
        "output_dim": state.lstm_model.output_dim,
        "window_size": state.lstm_model.window_size,
        "weights_path": str(MODEL_WEIGHTS_PATH),
        "weights_exist": MODEL_WEIGHTS_PATH.exists()
    }
