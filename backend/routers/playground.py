"""
Model Playground Router - Multi-Model Comparison

Routes for Model Playground feature that enables comparison
across multiple AI model versions.
"""

import logging
import torch
import pandas as pd
from datetime import datetime
from fastapi import APIRouter, HTTPException
from services.db import get_historical_data, get_latest_close, get_watchlist
from services.feature_engineering import process_stock_data, get_model_input_features, get_model_input_features_v7
from services.model_loader import model_loader
from services.model_metadata import get_available_models, get_model_info
from utils.config_loader import get_playground_enabled, set_playground_enabled

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/models")
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


@router.get("/settings/playground")
async def get_playground_setting():
    """
    Get current Model Playground toggle status.
    """
    enabled = get_playground_enabled()
    return {
        "enabled": enabled,
        "loader_status": model_loader.get_status() if enabled else None
    }


@router.post("/settings/playground")
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


@router.get("/playground/compare/{ticker}")
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
    
    from state import state  # Import here to avoid circular imports
    
    for version, loaded in model_loader.loaded_models.items():
        try:
            # Use version-specific feature columns
            if version == "v7":
                feature_cols = get_model_input_features_v7()  # 41 features
            else:
                feature_cols = get_model_input_features()      # 37 features
            
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
                "prediction_today": prediction_today,  # The prediction FOR today (made yesterday)
                "accuracy": accuracy,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Playground compare error for {version} on {ticker}: {e}")
            results["models"][version] = {
                "display_name": get_model_info(version).display_name if get_model_info(version) else version,
                "device": "unknown",
                "predictions": None,
                "prediction_today": None,
                "accuracy": None,
                "error": str(e)
            }
    
    return results


@router.get("/playground/compare_all")
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
