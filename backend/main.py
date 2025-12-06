"""
Proxmox AI Stock Predictor - Main FastAPI Application

This module serves as the entry point for the backend API, providing:
- REST endpoints for stock data, predictions, and dashboard views
- AI model loading and inference using LSTM neural networks
- Integration with Yahoo Finance (free) and Alpha Vantage (for EPS data)
- Real-time prediction generation with multi-horizon forecasting

Architecture:
    - FastAPI application with CORS middleware
    - Routers for modular endpoint organization
    - Global state management for model and scaler instances
    - Dynamic GPU/CPU device selection for inference

Author: Proxmox AI Stock Predictor Team
License: MIT
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import logging
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Third-party imports
import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Local application imports - Services
from services.lstm_model import LSTMModel, get_device
from services.db import (
    get_watchlist,
    add_watchlist_item,
    remove_watchlist_item,
    get_latest_close,
    get_unique_tickers,
    get_historical_data,
    get_favorites,
    get_watchlist_with_favorites,
    save_prediction,
    get_all_cached_intrinsic_values,
)
from services.data_ingest import YahooFinanceClient, AlphaVantageClient
from services.intrinsic import IntrinsicCalculator
from services.scaler import FeatureScaler
from services.feature_engineering import process_stock_data, get_model_input_features
from services.model_metadata import get_available_models, get_model_info, MODEL_REGISTRY
from services.model_loader import model_loader

# Local application imports - Configuration and State
from utils.config_loader import (
    settings, save_config, clear_config, 
    is_alpha_vantage_configured, get_playground_enabled, set_playground_enabled
)
from state import state

# Local application imports - Routers
from routers import ingestion, predictions, dashboard, backtest, stocks

# Configure logging
LOG_FILE = Path("backend.log")

class APIKeyFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        # Regex to find apikey=... and replace with apikey=REDACTED
        if "apikey=" in msg:
            record.msg = re.sub(r'apikey=[^&"\s]+', 'apikey=REDACTED', msg)
            record.args = () # Clear args to prevent formatting issues if msg was modified
        return True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
# Apply filter to root logger handlers
for handler in logging.getLogger().handlers:
    handler.addFilter(APIKeyFilter())

# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

app = FastAPI(
    title="Proxmox AI Stock Predictor",
    description="LSTM-based stock prediction API with multi-horizon forecasting",
    version="1.0.0"
)

# Include modular routers for endpoint organization
app.include_router(stocks.router)
app.include_router(ingestion.router)
app.include_router(predictions.router)
app.include_router(dashboard.router)
app.include_router(backtest.router)

# Model and scaler paths
# We will dynamically select the best available model
MODELS_DIR = Path(__file__).parent / "models"
MODEL_VERSIONS = ["v6", "v5", "v4", "v3", "v2"]

# Default to v2 if nothing else found
MODEL_WEIGHTS_PATH = MODELS_DIR / "lstm_model_v2.pth"
SCALER_PATH = MODELS_DIR / "scaler_v2.pkl"
ACTIVE_MODEL_VERSION = "v2"

# Check for newer versions
for version in MODEL_VERSIONS:
    model_path = MODELS_DIR / f"lstm_model_{version}.pth"
    scaler_path = MODELS_DIR / f"scaler_{version}.pkl"
    if model_path.exists() and scaler_path.exists():
        MODEL_WEIGHTS_PATH = model_path
        SCALER_PATH = scaler_path
        ACTIVE_MODEL_VERSION = version
        break

# CORS Configuration - Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# =============================================================================

class WatchlistItem(BaseModel):
    ticker: str

class APIKeys(BaseModel):
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

    # Try to load existing model weights (dimensions read from checkpoint)
    if MODEL_WEIGHTS_PATH.exists():
        print(f"🚀 ACTIVATING MODEL: {MODEL_WEIGHTS_PATH.name} ({ACTIVE_MODEL_VERSION})")
        try:
            # Use class method to load - reads input_dim, window_size from checkpoint
            state.lstm_model = LSTMModel.load(str(MODEL_WEIGHTS_PATH), device=state.device)
            logger.info(f"✅ Successfully loaded trained model from {MODEL_WEIGHTS_PATH}")
            logger.info(f"   Model config: input_dim={state.lstm_model.input_dim}, window_size={state.lstm_model.window_size}")
        except Exception as e:
            logger.error(f"❌ Failed to load model weights: {e}. Creating new model.")
            state.lstm_model = LSTMModel(input_dim=37, window_size=60, device=state.device)
    else:
        # Initialize new LSTM Model with default features
        state.lstm_model = LSTMModel(input_dim=37, window_size=60, device=state.device)
        logger.info("⚠️ Initialized new LSTM model (no trained weights found)")

    if state.lstm_model:
        logger.info(f"✅ AI Model Active: LSTM (Input: {state.lstm_model.input_dim} features, Hidden: {state.lstm_model.hidden_dim})")

    # Log API Key Status - Only Alpha Vantage needed (Yahoo Finance is free)
    logger.info("✅ Yahoo Finance: Always Available (No API Key Required)")
    
    if is_alpha_vantage_configured():
        av_key = settings.ALPHA_VANTAGE_KEY
        if av_key and "placeholder" in av_key.lower():
            logger.critical("❌ CRITICAL: Alpha Vantage Key contains 'placeholder' - Intrinsic Value calculations DISABLED")
            state.alpha_vantage_enabled = False
        else:
            logger.info("✅ Alpha Vantage Key: Loaded (for EPS/Intrinsic Value)")
            state.alpha_vantage_enabled = True
    else:
        logger.warning("⚠️ Alpha Vantage Key: Missing - Intrinsic Value calculations will use defaults")
        state.alpha_vantage_enabled = False

    # Auto-seed watchlist with default assets if empty
    DEFAULT_TICKERS = ["SPY", "BTC-USD", "AAPL", "AMD", "TSLA", "AMZN"]
    try:
        watchlist = await get_watchlist()
        if len(watchlist) == 0:
            logger.info("📋 Watchlist empty - seeding with default assets...")
            for ticker in DEFAULT_TICKERS:
                try:
                    await add_watchlist_item(ticker)
                    logger.info(f"  ✅ Added {ticker} to watchlist")
                except Exception as e:
                    logger.warning(f"  ⚠️ Failed to add {ticker}: {e}")
            
            # Trigger data ingestion for all default tickers
            logger.info("📊 Starting background data ingestion for default tickers...")
            yf_client = YahooFinanceClient()
            for ticker in DEFAULT_TICKERS:
                try:
                    await yf_client.fetch_data(ticker)
                    logger.info(f"  ✅ Ingested data for {ticker}")
                except Exception as e:
                    logger.warning(f"  ⚠️ Failed to ingest {ticker}: {e}")
        else:
            logger.info(f"📋 Watchlist has {len(watchlist)} items: {watchlist}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to check/seed watchlist: {e}")

@app.get("/status")
async def get_status():
    """Get system status including data sources and AI model."""
    has_alpha_vantage = is_alpha_vantage_configured()
    
    return {
        "configured": True,  # Yahoo Finance always available
        "data_source": "yahoo_finance",
        "yahoo_finance_active": True,
        "alpha_vantage_configured": has_alpha_vantage,
        "ai_model_loaded": state.lstm_model is not None,
        "scaler_loaded": state.scaler is not None,
        "device": str(state.device) if state.device else None,
        "model_version": ACTIVE_MODEL_VERSION
    }

@app.get("/status/ai")
async def get_ai_status():
    """Get detailed AI model status and configuration."""
    model_info = None
    if state.lstm_model:
        model_info = {
            "type": "LSTM",
            "input_features": state.lstm_model.input_dim,
            "hidden_dim": state.lstm_model.hidden_dim,
            "num_layers": getattr(state.lstm_model, 'num_layers', 2),
            "window_size": getattr(state.lstm_model, 'window_size', 60),
            "horizons": ["1d", "1w", "1m", "6m"]
        }
    
    return {
        "model_loaded": state.lstm_model is not None,
        "model_version": ACTIVE_MODEL_VERSION,
        "model_info": model_info,
        "scaler_loaded": state.scaler is not None,
        "device": str(state.device) if state.device else "cpu",
        "alpha_vantage_enabled": state.alpha_vantage_enabled,
        "ready_for_predictions": state.lstm_model is not None and state.scaler is not None
    }


@app.get("/debug/model_info")
async def debug_model_info():
    """Debug endpoint: Get current model information."""
    return {
        "model_file": MODEL_WEIGHTS_PATH.name,
        "model_path": str(MODEL_WEIGHTS_PATH),
        "model_exists": MODEL_WEIGHTS_PATH.exists(),
        "scaler_file": SCALER_PATH.name,
        "scaler_exists": SCALER_PATH.exists(),
        "input_features": state.lstm_model.input_dim if state.lstm_model else None,
        "hidden_dim": state.lstm_model.hidden_dim if state.lstm_model else None,
        "trained_date": "2025-12-05",
        "device": str(state.device) if state.device else "cpu"
    }


@app.get("/debug/features/{ticker}")
async def debug_features(ticker: str):
    """Debug endpoint: Get the raw 37-feature input vector for a ticker."""
    from services.db import get_historical_data
    
    try:
        # Get historical data
        history = await get_historical_data(ticker, limit=100)
        if not history or len(history) < 61:
            return {"error": f"Not enough data for {ticker}. Need 61 days, have {len(history) if history else 0}"}
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            "date": h.timestamp,
            "open": h.open,
            "high": h.high,
            "low": h.low,
            "close": h.close,
            "volume": h.volume
        } for h in history])
        
        # Process features
        processed = process_stock_data(df, create_targets=False)
        
        if processed.empty:
            return {"error": "Feature processing failed"}
        
        # Get feature columns
        feature_cols = get_model_input_features()
        
        # Get the last row's features
        last_row = processed.iloc[-1]
        features = {col: float(last_row[col]) if col in processed.columns else None for col in feature_cols}
        
        return {
            "ticker": ticker,
            "date": str(last_row.get("date", "unknown")),
            "feature_count": len(feature_cols),
            "features": features,
            "non_zero_features": sum(1 for v in features.values() if v and abs(v) > 0.0001),
            "feature_columns": feature_cols
        }
        
    except Exception as e:
        logger.error(f"Debug features failed for {ticker}: {e}")
        return {"error": str(e)}

@app.post("/settings/keys")
async def save_keys(keys: APIKeys):
    """Save API keys. Only Alpha Vantage needed (Yahoo Finance is free)."""
    try:
        save_config(alpha_vantage_key=keys.ALPHA_VANTAGE_KEY)
        
        # Update state
        if keys.ALPHA_VANTAGE_KEY:
            state.alpha_vantage_enabled = True
            logger.info("✅ Alpha Vantage key saved")
        
        return {
            "message": "Settings saved successfully",
            "alpha_vantage_saved": bool(keys.ALPHA_VANTAGE_KEY),
            "data_source": "yahoo_finance"
        }
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
    # Get watchlist with favorite status
    watchlist_items = await get_watchlist_with_favorites()
    
    # Get ALL cached intrinsic values in ONE query (efficient batch lookup)
    cached_intrinsic_values = await get_all_cached_intrinsic_values()
    
    summaries = []
    
    for item in watchlist_items:
        ticker = item["ticker"]
        is_favorite = item.get("is_favorite", True)
        
        current_price = await get_latest_close(ticker)
        if current_price is None:
            continue 
            
        intrinsic_value = 0.0
        prediction = 0.0

        # Get Intrinsic Value from cache (favorites only)
        # Uses pre-fetched batch cache - no per-ticker DB queries
        if is_favorite:
            cached_val = cached_intrinsic_values.get(ticker)
            if cached_val and cached_val > 0:
                intrinsic_value = cached_val
            # Note: If not in cache, we don't calculate here to avoid API spam
            # Background job or detail view will populate cache

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
                                log_return = state.lstm_model.predict(input_tensor, state.device)
                                # CRITICAL: Model outputs log returns, not prices!
                                # predicted_price = current_price * exp(log_return)
                                prediction = current_price * np.exp(log_return)
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
            "signal": signal,
            "is_favorite": is_favorite
        })
        
    return summaries

@app.get("/dashboard/{ticker}")
async def get_dashboard_detail(ticker: str):
    current_price = await get_latest_close(ticker)
    if current_price is None:
        raise HTTPException(status_code=404, detail="Ticker not found")
    
    required_history = 10000  # ~40 years of trading days
    window_size = 60
    fetch_limit = required_history + window_size
    
    historical_data = await get_historical_data(ticker, limit=fetch_limit)
    history_response = []
    
    total_records = len(historical_data)
    start_index = max(window_size, total_records - required_history)
    
    # Calculate Intrinsic Value ONLY for favorites (uses cache - won't spam API)
    intrinsic_val = 0.0
    
    # Check if this ticker is a favorite
    favorites = await get_favorites()
    is_favorite = ticker.upper() in [f.upper() for f in favorites]
    
    if state.alpha_vantage_enabled and is_favorite:
        try:
            # Use cached IntrinsicCalculator.calculate() - caches for 30 days
            calculator = IntrinsicCalculator()
            cached_value = await calculator.calculate(ticker, use_live_bond_yield=False)
            if cached_value is not None and cached_value > 0:
                intrinsic_val = cached_value
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
        log_returns = state.lstm_model.predict_batch(batch_tensor, state.device)
        log_returns_list = log_returns.cpu().numpy().tolist()
        
        # Convert log returns to predicted prices: pred_price = current_price * exp(log_return)
        predictions_list = []
        for idx, (valid_idx, log_ret) in enumerate(zip(valid_indices, log_returns_list)):
            # The prediction is for day i, using window ending at i-1
            # The "current" price is the close at i-1
            current_close = float(historical_data[valid_idx - 1].close)
            predicted_price = current_close * np.exp(log_ret)
            predictions_list.append(predicted_price)
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
            "predicted_close": round(pred, 2) if pred is not None else None,
            "intrinsic_value": intrinsic_val
        })

    # Get the last sync date (most recent data point)
    last_sync_date = None
    if historical_data:
        last_sync_date = historical_data[0].timestamp.strftime("%Y-%m-%d") if historical_data[0].timestamp else None

    return {"history": history_response, "last_sync_date": last_sync_date}


@app.get("/forecasts/{ticker}")
async def get_forecasts(ticker: str):
    """
    Get multi-horizon AI forecasts for a ticker.
    Returns predictions for 1d, 1w, 1m, 6m horizons.
    """
    current_price = await get_latest_close(ticker)
    if current_price is None:
        raise HTTPException(status_code=404, detail="Ticker not found")
    
    if not state.lstm_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        historical_data = await get_historical_data(ticker, limit=100)
        if len(historical_data) < 60:
            raise HTTPException(status_code=400, detail="Insufficient data")
            
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
        
        if len(processed_df) < 60:
            raise HTTPException(status_code=400, detail="Insufficient processed data")
            
        # Get latest 60-day sequence
        seq_data = processed_df.iloc[-60:][feature_cols].values
        
        # Scale
        if state.scaler:
            scaled_seq = state.scaler.transform(seq_data)
        else:
            scaled_seq = seq_data
            
        # Create tensor and predict
        input_tensor = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0)
        
        # Use predict_with_uncertainty to get all horizons
        result = state.lstm_model.predict_with_uncertainty(input_tensor, state.device)
        
        # All horizons: [1d, 1w, 1m, 6m]
        all_horizons = result["prediction_all_horizons"][0]  # Shape: (4,)
        confidence = float(result["confidence"][0])
        
        # Define horizons
        horizon_labels = ["1d", "1w", "1m", "6m"]
        horizon_names = {
            "1d": "Tomorrow",
            "1w": "Next Week", 
            "1m": "Next Month",
            "6m": "6 Months"
        }
        
        forecasts = {}
        for i, label in enumerate(horizon_labels):
            # Model outputs are RAW log returns (not scaled)
            # log_return = ln(future_price / current_price)
            log_return = float(all_horizons[i])
            
            # Debug: log the raw values
            logger.debug(f"Forecast {label}: raw={log_return:.4f}")
            
            # Convert log return to price: price * exp(log_return)
            predicted_price = current_price * np.exp(log_return)
            change_pct = (np.exp(log_return) - 1) * 100  # More accurate conversion
            
            forecasts[label] = {
                "label": horizon_names[label],
                "price": round(predicted_price, 2),
                "change_pct": round(change_pct, 2),
                "log_return": round(log_return, 6)
            }
            
        return {
            "ticker": ticker,
            "current_price": current_price,
            "confidence": confidence,
            "forecasts": forecasts
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


# ============================================================
# MODEL PLAYGROUND ENDPOINTS
# ============================================================

@app.get("/models")
async def list_models():
    """
    List all available AI models with metadata and reasoning.
    """
    available = get_available_models()
    
    return {
        "models": [
            {
                "version": m.version,
                "display_name": m.display_name,
                "description": m.description,
                "reasoning": m.reasoning.strip(),
                "training_notes": m.training_notes,
                "is_available": m.is_available
            }
            for m in available
        ],
        "total": len(available)
    }


@app.get("/settings/playground")
async def get_playground_setting():
    """
    Get current Model Playground toggle status.
    """
    enabled = get_playground_enabled()
    return {
        "enabled": enabled,
        "loader_status": model_loader.get_status() if enabled else None
    }


@app.post("/settings/playground")
async def set_playground_setting(enabled: bool):
    """
    Enable or disable Model Playground mode.
    This setting persists across restarts.
    
    WARNING: Enabling loads multiple models which may use 4-8GB VRAM/RAM.
    """
    set_playground_enabled(enabled)
    
    if enabled:
        model_loader.enable()
        # Load all models with VRAM -> RAM fallback
        model_loader.load_all_models()
        status = model_loader.get_status()
        logger.info(f"🎮 Model Playground ENABLED - Loaded {status['loaded_count']} models")
    else:
        model_loader.disable()
        logger.info("🎮 Model Playground DISABLED - All models unloaded")
    
    return {
        "enabled": enabled,
        "message": "Model Playground enabled" if enabled else "Model Playground disabled",
        "loader_status": model_loader.get_status()
    }


@app.get("/playground/compare/{ticker}")
async def playground_compare(ticker: str, accuracy_days: int = 10):
    """
    Get predictions from ALL loaded models for a specific ticker.
    Does NOT call Alpha Vantage API (uses existing DB data).
    """
    if not get_playground_enabled():
        raise HTTPException(status_code=400, detail="Model Playground is disabled. Enable it in Settings first.")

    # Auto-load models if playground is enabled but models aren't loaded yet
    if not model_loader.loaded_models:
        logger.info("Auto-loading models for playground...")
        model_loader.enable()
        model_loader.load_all_models()

        if not model_loader.loaded_models:
            raise HTTPException(status_code=500, detail="Failed to load models. Check server logs.")
        
    # Get historical data (Need enough for window + accuracy backtest)
    # 500 days should be safe for 60 window + 90 days backtest + indicators
    historical_data = await get_historical_data(ticker, limit=500) 
    current_price = await get_latest_close(ticker)
    
    if not historical_data or len(historical_data) < 60:
        raise HTTPException(status_code=404, detail=f"Insufficient data for {ticker}")
    
    # Convert to DataFrame and process features
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
    
    if len(processed_df) == 0:
        raise HTTPException(status_code=400, detail="Feature processing failed")
    
    # Get predictions from each model
    results = {
        "ticker": ticker,
        "current_price": current_price,
        "timestamp": datetime.now().isoformat(),
        "models": {}
    }
    
    for version, loaded in model_loader.loaded_models.items():
        try:
            # Use the model's scaler if available, else fall back to main scaler
            scaler = loaded.scaler or state.scaler
            
            # --- PREDICTION 1: TOMORROW (Using Today's Data T) ---
            latest_features = processed_df.iloc[-1][feature_cols].values.reshape(1, -1)
            if scaler:
                scaled_features = scaler.transform(latest_features)
            else:
                scaled_features = latest_features
                
            # Create window if needed
            window_size = getattr(loaded.model, 'window_size', 60)
            if window_size > 1 and len(processed_df) >= window_size:
                window_features = processed_df.iloc[-window_size:][feature_cols].values
                if scaler:
                    window_features = scaler.transform(window_features)
                features_tensor = torch.tensor(window_features, dtype=torch.float32).unsqueeze(0)
            else:
                features_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0)
                
            prediction_tomorrow = model_loader.get_prediction(version, features_tensor, current_price)

            # --- PREDICTION 2: TODAY (Using Yesterday's Data T-1) ---
            # This simulates "Yesterday's Prediction" for comparison with today's price
            yesterday_features = processed_df.iloc[-2][feature_cols].values.reshape(1, -1)
            if scaler:
                scaled_yesterday = scaler.transform(yesterday_features)
            else:
                scaled_yesterday = yesterday_features

            if window_size > 1 and len(processed_df) >= window_size + 1:
                # Window ending at T-1
                window_yesterday = processed_df.iloc[-(window_size+1):-1][feature_cols].values
                if scaler:
                    window_yesterday = scaler.transform(window_yesterday)
                features_tensor_yesterday = torch.tensor(window_yesterday, dtype=torch.float32).unsqueeze(0)
            else:
                features_tensor_yesterday = torch.tensor(scaled_yesterday, dtype=torch.float32).unsqueeze(0)
                
            # For "Yesterday's Prediction", the "current_price" reference should be Yesterday's Close
            # to calculate the predicted price correctly from log returns.
            yesterday_close = float(processed_df.iloc[-2]["close"])
            prediction_today = model_loader.get_prediction(version, features_tensor_yesterday, yesterday_close)
            
            # Calculate Accuracy (Last N days)
            accuracy = model_loader.calculate_recent_accuracy(version, processed_df, feature_cols, days=accuracy_days)
            
            results["models"][version] = {
                "display_name": loaded.info.display_name,
                "device": loaded.device,
                "predictions": prediction_tomorrow,
                "prediction_today": prediction_today, # The prediction FOR today (made yesterday)
                "accuracy": accuracy,
                "error": None
            }

            if not prediction_tomorrow:
                 results["models"][version]["error"] = "Prediction failed"
                
        except Exception as e:
            logger.error(f"Playground prediction error for {version}: {e}")
            results["models"][version] = {
                "display_name": get_model_info(version).display_name if get_model_info(version) else version,
                "device": "unknown",
                "predictions": None,
                "prediction_today": None,
                "accuracy": None,
                "error": str(e)
            }
    
    return results


@app.get("/playground/compare_all")
async def playground_compare_all(accuracy_days: int = 10):
    """
    Get predictions from ALL loaded models for ALL watchlist tickers.
    Returns comprehensive comparison matrix.
    
    NOTE: Does NOT call Alpha Vantage API.
    """
    if not get_playground_enabled():
        raise HTTPException(status_code=400, detail="Model Playground is disabled. Enable it in Settings first.")

    # Auto-load models if playground is enabled but models aren't loaded yet
    if not model_loader.loaded_models:
        logger.info("Auto-loading models for playground...")
        model_loader.enable()
        model_loader.load_all_models()

        if not model_loader.loaded_models:
            raise HTTPException(status_code=500, detail="Failed to load models. Check server logs.")

    watchlist = await get_watchlist()
    results = {}
    
    for ticker in watchlist:
        try:
            data = await playground_compare(ticker, accuracy_days=accuracy_days)
            results[ticker] = data
        except Exception as e:
            logger.error(f"Compare all error for {ticker}: {e}")
            results[ticker] = {"error": str(e)}
            
    return {"tickers": results}

