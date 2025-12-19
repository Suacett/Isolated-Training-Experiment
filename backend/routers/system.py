"""
System Router - For monitoring application status, AI model info, and logs.
Extracted from main.py (Phase 5.2).
"""

import logging
import re
import pandas as pd
import numpy as np
import torch
from fastapi import APIRouter
from pathlib import Path

from services.db import get_historical_data
from services.feature_engineering import process_stock_data, get_model_input_features
from state import state
from utils.config_loader import is_alpha_vantage_configured

router = APIRouter(
    prefix="/system",
    tags=["system"]
)

logger = logging.getLogger(__name__)
# Use a configurable path or absolute path
LOG_FILE = Path("backend.log").resolve()


@router.get("/status")
async def get_status():
    """Get system status including data sources and AI model."""
    return {
        "configured": True,  # Yahoo Finance always available
        "data_source": "yahoo_finance",
        "yahoo_finance_active": True,
        "alpha_vantage_configured": is_alpha_vantage_configured(),
        "ai_model_loaded": state.lstm_model is not None,
        "scaler_loaded": state.scaler is not None,
        "device": str(state.device) if state.device else None,
        "model_version": getattr(state, "active_model_version", "unknown")
    }


@router.get("/status/ai")
async def get_ai_status():
    """Get detailed AI model status and configuration."""
    model_info = None
    version = getattr(state, "active_model_version", "unknown")
    
    if state.lstm_model:
        if version == "v9":
            model_info = {
                "type": "Transformer V9",
                "input_features": getattr(state.lstm_model, 'feature_dim', 12),
                "hidden_dim": getattr(state.lstm_model, 'd_model', 128),
                "num_layers": getattr(state.lstm_model, 'num_layers', 2),
                "window_size": 60,
                "horizons": ["5d"]
            }
        else:
            model_info = {
                "type": "LSTM",
                "input_features": getattr(state.lstm_model, 'input_dim', None),
                "hidden_dim": getattr(state.lstm_model, 'hidden_dim', None),
                "num_layers": getattr(state.lstm_model, 'num_layers', 2),
                "window_size": getattr(state.lstm_model, 'window_size', 60),
                "horizons": ["1d", "1w", "1m", "6m"]
            }
    
    return {
        "model_loaded": state.lstm_model is not None,
        "model_version": version,
        "model_info": model_info,
        "scaler_loaded": state.scaler is not None,
        "device": str(state.device) if state.device else "cpu",
        "alpha_vantage_enabled": state.alpha_vantage_enabled,
        "ready_for_predictions": state.lstm_model is not None and state.scaler is not None
    }


@router.get("/logs")
async def get_logs():
    """Return the last 100 lines of logs with sensitive info masked."""
    if not LOG_FILE.exists():
        return {"logs": []}
    
    try:
        def mask_log_line(line: str) -> str:
            # Mask potential API keys
            line = re.sub(r'(PK[A-Z0-9]{10,})', 'PK***********', line)
            return line

        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
            masked_lines = [mask_log_line(line) for line in lines[-100:]]
            return {"logs": masked_lines}
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        return {"logs": [f"Error reading logs: {e}"]}


@router.get("/debug/model-info")
async def debug_model_info():
    """Debug endpoint: Get current model information."""
    return {
        "model_version": getattr(state, "active_model_version", "unknown"),
        "ai_model_loaded": state.lstm_model is not None,
        "device": str(state.device) if state.device else "cpu",
        "input_features": getattr(state.lstm_model, 'input_dim', getattr(state.lstm_model, 'feature_dim', None)) if state.lstm_model else None,
        "hidden_dim": getattr(state.lstm_model, 'hidden_dim', getattr(state.lstm_model, 'd_model', None)) if state.lstm_model else None,
    }


@router.get("/debug/features/{ticker}")
async def debug_features(ticker: str):
    """Debug endpoint: Get the raw feature input vector for a ticker."""
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
        
        # Ensure data is sorted by date before taking last row
        if "date" in processed.columns:
            processed = processed.sort_values("date")
        else:
            processed = processed.sort_index()

        # Get feature columns
        feature_cols = get_model_input_features()
        last_row = processed.iloc[-1]
        features = {col: float(last_row[col]) if col in processed.columns else None for col in feature_cols}
        
        return {
            "ticker": ticker,
            "date": str(last_row.get("date", "unknown")),
            "feature_count": len(feature_cols),
            "features": features,
            "feature_columns": feature_cols
        }
    except Exception as e:
        logger.error(f"Debug features failed for {ticker}: {e}")
        return {"error": str(e)}
