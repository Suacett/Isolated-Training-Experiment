"""
Forecasts Router - Multi-horizon AI Predictions

Routes for getting AI forecasts at various time horizons.
"""

import logging
import re
import torch
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from services.db import get_latest_close, get_historical_data
from services.feature_engineering import process_stock_data, get_model_input_features
from state import state

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/forecasts/{ticker}")
async def get_forecasts(ticker: str):
    """
    Get multi-horizon AI forecasts for a ticker.
    Returns predictions for 1d, 1w, 1m, 6m horizons.
    """
    # Input validation
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")
        
    ticker = ticker.strip().upper()
    if not re.match(r'^[A-Z0-9.-]{1,6}$', ticker):
        logger.warning(f"Invalid ticker format received: {ticker}")
        raise HTTPException(status_code=400, detail="Invalid ticker format")
        
    current_price = await get_latest_close(ticker)
    if current_price is None:
        raise HTTPException(status_code=404, detail="Ticker not found")
    
    if not state.lstm_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    import asyncio
    try:
        try:
            historical_data = await asyncio.wait_for(get_historical_data(ticker, limit=100), timeout=10.0)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Timeout fetching historical data")
            
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
        raise HTTPException(status_code=500, detail="Internal server error")
