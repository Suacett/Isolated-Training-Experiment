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
import pandas as pd

from services.lstm_model import LSTMModel, get_device
from services.db import get_latest_close, get_unique_tickers, get_historical_data, add_watchlist_item, remove_watchlist_item, get_watchlist, save_prediction
from services.data_ingest import AlpacaDataClient, AlphaVantageClient
from services.intrinsic import IntrinsicCalculator
from services.scaler import FeatureScaler
from services.feature_engineering import process_stock_data, get_model_input_features
from utils.config_loader import settings, save_config, clear_config, is_alpaca_configured, is_alpha_vantage_configured
from state import state
from routers import ingestion, predictions

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
app.include_router(predictions.router)

# Model and scaler paths
MODEL_WEIGHTS_PATH = Path(__file__).parent / "models" / "lstm_model.pth"
SCALER_PATH = Path(__file__).parent / "models" / "scaler.pkl"

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

    # Load Scaler
    state.scaler = FeatureScaler(str(SCALER_PATH))
    if SCALER_PATH.exists():
        if state.scaler.load():
            logger.info(f"✅ Successfully loaded feature scaler from {SCALER_PATH}")
        else:
            logger.warning("⚠️ Scaler file exists but failed to load")
    else:
        logger.warning("⚠️ Scaler not found - predictions will not work without scaler")

    # Try to load existing model weights (37 features)
    if MODEL_WEIGHTS_PATH.exists():
        try:
            state.lstm_model = LSTMModel(input_dim=37, window_size=60, device=state.device)
            state.lstm_model.load(str(MODEL_WEIGHTS_PATH))
            logger.info(f"✅ Successfully loaded trained model from {MODEL_WEIGHTS_PATH}")
        except Exception as e:
            logger.error(f"❌ Failed to load model weights: {e}. Creating new model.")
            state.lstm_model = LSTMModel(input_dim=37, window_size=60, device=state.device)
    else:
        # Initialize new LSTM Model with 37 features
        state.lstm_model = LSTMModel(input_dim=37, window_size=60, device=state.device)
        logger.info("⚠️ Initialized new LSTM model (no trained weights found)")

    if state.lstm_model:
        logger.info(f"✅ AI Model Active: LSTM (Input: {state.lstm_model.input_dim} features, Hidden: {state.lstm_model.hidden_dim})")

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
        alpaca_valid = False
        if keys.ALPACA_API_KEY and keys.ALPACA_SECRET_KEY:
            try:
                test_client = AlpacaDataClient(api_key=keys.ALPACA_API_KEY, secret_key=keys.ALPACA_SECRET_KEY)
                test_client.verify_credentials()
                alpaca_valid = True
            except Exception as e:
                logger.error(f"Alpaca keys validation failed: {e}")
                # Only raise if Alpha Vantage is also missing/invalid
                if not keys.ALPHA_VANTAGE_KEY:
                     raise HTTPException(status_code=400, detail=f"Invalid API Keys: {str(e)}")
                # If we have AV key, just warn
                logger.warning("Alpaca keys invalid, but proceeding because Alpha Vantage key is present.")

        save_config(
            alpaca_api_key=keys.ALPACA_API_KEY,
            alpaca_secret_key=keys.ALPACA_SECRET_KEY,
            alpha_vantage_key=keys.ALPHA_VANTAGE_KEY
        )
        
        status = "valid" if alpaca_valid else "partial"
        return {"message": "Keys saved successfully", "status": status}
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
        
        # Calculate Intrinsic Value
        try:
            av_client = AlphaVantageClient()
            eps_data = await av_client.fetch_eps_data(ticker)
            calculator = IntrinsicCalculator()
            eps_ttm = calculator.calculate_eps_ttm(eps_data)
            growth = calculator.estimate_growth_rate(eps_data)
            intrinsic_value = calculator.calculate_graham(eps_ttm, growth)
        except Exception as e:
            logger.warning(f"Intrinsic value calc failed for {ticker}: {e}")

        # Use global state model
        if state.lstm_model:
            try:
                historical_data = await get_historical_data(ticker, limit=100) # Need more data for features
                if len(historical_data) >= 60:
                    # Convert to DataFrame
                    df = pd.DataFrame([{
                        "date": d.timestamp,
                        "open": d.open,
                        "high": d.high,
                        "low": d.low,
                        "close": d.close,
                        "volume": d.volume
                    } for d in historical_data])
                    
                    # Process features
                    processed_df = process_stock_data(df, create_targets=False)
                    feature_cols = get_model_input_features()
                    
                    if len(processed_df) > 0:
                        # Get latest features
                        latest_features = processed_df.iloc[-1][feature_cols].values.reshape(1, -1)
                        
                        # Scale
                        if state.scaler:
                            scaled_features = state.scaler.transform(latest_features)
                            
                            # Reshape for LSTM [batch, features, seq_len] -> actually model expects [batch, seq_len, features] or similar?
                            # Wait, the error said: expected input[1, 5, 60] to have 37 channels, but got 5 channels instead
                            # The model likely expects [batch, seq_len, features] or [batch, features, seq_len] depending on implementation.
                            # Let's check LSTMModel.predict.
                            # Assuming it takes [batch, seq_len, features] based on standard PyTorch LSTM, 
                            # BUT the error "expected input[1, 5, 60] to have 37 channels" suggests Conv1d or similar?
                            # Let's look at the error again: "Given groups=1, weight of size [32, 37, 3], expected input[1, 5, 60] to have 37 channels, but got 5 channels instead"
                            # This implies the input is [1, 5, 60] (Batch, Channels, SeqLen) and it wants 37 channels.
                            # So we need to provide 37 features.
                            
                            # We need a sequence of 60 steps.
                            # So we need the last 60 rows of processed_df.
                            
                            if len(processed_df) >= 60:
                                seq_data = processed_df.iloc[-60:][feature_cols].values # (60, 37)
                                scaled_seq = state.scaler.transform(seq_data) # (60, 37)
                                
                                # Transpose to [1, 37, 60] if model expects [channels, seq_len]
                                # The error "expected input[1, 5, 60]" implies it got 5 channels (OHLCV) and 60 steps.
                                # So we need to pass [1, 37, 60].
                                
                                # CORRECTION: The model forward method permutes (Batch, SeqLen, Features) -> (Batch, Features, SeqLen).
                                # So we should pass (Batch, SeqLen, Features) -> (1, 60, 37).
                                input_tensor = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0)
                                prediction = state.lstm_model.predict(input_tensor, state.device)
            except Exception as e:
                logger.error(f"Prediction error for {ticker}: {e}")
        
        signal = "HOLD"
        # Combine Intrinsic Value and AI Prediction for Signal
        # Logic: If Prediction > Current + 2% AND Price < Intrinsic -> STRONG BUY
        #        If Prediction > Current + 2% -> BUY
        #        If Price < 0.5 * Intrinsic -> VALUE BUY
        
        is_bullish_prediction = prediction > current_price * 1.02
        is_undervalued = intrinsic_value > 0 and current_price < intrinsic_value
        is_deep_value = intrinsic_value > 0 and current_price < 0.5 * intrinsic_value
        
        if is_bullish_prediction and is_undervalued:
            signal = "STRONG BUY"
        elif is_bullish_prediction:
            signal = "BUY"
        elif is_deep_value:
            signal = "VALUE BUY"
        elif prediction < current_price * 0.98:
            signal = "SELL"
            
        summaries.append({
            "ticker": ticker,
            "current_price": current_price,
            "prediction": prediction,
            "intrinsic_value": intrinsic_value,
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
    
    # Calculate Intrinsic Value
    intrinsic_val = 0.0
    try:
        av_client = AlphaVantageClient()
        eps_data = await av_client.fetch_eps_data(ticker)
        calculator = IntrinsicCalculator()
        eps_ttm = calculator.calculate_eps_ttm(eps_data)
        growth = calculator.estimate_growth_rate(eps_data)
        intrinsic_val = calculator.calculate_graham(eps_ttm, growth)
    except Exception as e:
        logger.warning(f"Intrinsic value calc failed for {ticker}: {e}")

    if total_records <= window_size or not state.lstm_model:
        for d in historical_data:
             history_response.append({
                "date": d.timestamp.strftime("%Y-%m-%d"),
                "open": d.open,
                "high": d.high,
                "low": d.low,
                "close": d.close,
                "predicted_close": None,
                "intrinsic_value": intrinsic_val 
            })
        return {"history": history_response}

    windows = []
    valid_indices = []
    
    windows = []
    valid_indices = []
    
    # Need sufficient history for feature engineering (e.g. 30 days for volatility)
    # We fetched 'fetch_limit' which is required_history + window_size.
    # Let's convert all historical data to DF first.
    
    full_df = pd.DataFrame([{
        "date": d.timestamp,
        "open": d.open,
        "high": d.high,
        "low": d.low,
        "close": d.close,
        "volume": d.volume
    } for d in historical_data])
    
    if not full_df.empty:
        processed_full_df = process_stock_data(full_df, create_targets=False)
        feature_cols = get_model_input_features()
        
        # We need to align processed_df with original indices.
        # process_stock_data drops NaNs, so indices shift.
        # We'll map by date.
        processed_full_df.set_index("date", inplace=True)
        
        for i in range(start_index, total_records):
            # We need the window ending at i (exclusive of i? No, historical_data[i] is the target usually, 
            # but here we want to predict FOR i? Or is i the current time?
            # The loop logic: window_data = historical_data[i-window_size : i]
            # So we use data up to i-1 to predict i?
            # Let's stick to the existing logic: window is [i-window_size : i]
            
            target_date = historical_data[i-1].timestamp # The last data point in the window
            
            # We need a sequence of 60 steps ending at i-1.
            # In processed_df, we need to find the row corresponding to target_date and take it + 59 previous rows?
            # Actually, simpler: just take the slice from the processed dataframe if possible.
            
            # Re-slice from full processed DF is safer.
            # But processed_df might be shorter due to NaN dropping.
            
            # Let's try to grab the window from the processed DF based on dates.
            window_end_date = historical_data[i-1].timestamp
            
            if window_end_date in processed_full_df.index:
                # Get location of this date
                loc = processed_full_df.index.get_loc(window_end_date)
                
                if isinstance(loc, int):
                    if loc >= window_size - 1:
                        window_seq = processed_full_df.iloc[loc-window_size+1 : loc+1][feature_cols].values
                        if len(window_seq) == window_size:
                            if state.scaler:
                                window_seq = state.scaler.transform(window_seq)
                            
                            # Shape: (60, 37)
                            windows.append(window_seq)
                            valid_indices.append(i)

    if windows:
        # Shape: (Batch, SeqLen, Features)
        batch_tensor = torch.tensor(np.array(windows), dtype=torch.float32)
        predictions = state.lstm_model.predict_batch(batch_tensor, state.device)
        predictions_list = predictions.cpu().numpy().tolist()
    else:
        predictions_list = []
        
    pred_map = {idx: pred for idx, pred in zip(valid_indices, predictions_list)}
    
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
@app.post("/model/train")
async def train_model():
    """
    Stub for model training.
    """
    import asyncio
    # Simulate training delay
    await asyncio.sleep(5)
    
    # In a real implementation, this would trigger the training loop
    if state.lstm_model:
        # Save dummy weights to simulate "training" completion
        state.lstm_model.save(str(MODEL_WEIGHTS_PATH))
        
    return {"message": "Model training started (simulation)", "status": "training"}

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
