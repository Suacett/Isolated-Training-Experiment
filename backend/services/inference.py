"""
AI Inference Service

Provides helper functions for running V9 Transformer model inference.
Extracted from routers/dashboard.py for testability and reuse (Phase 2.2).
"""

import logging
import pandas as pd
import numpy as np
import torch
from typing import Optional, List, Any

logger = logging.getLogger(__name__)

# V9 Model Constants
V9_WINDOW_SIZE = 60


def prepare_stock_dataframe(historical_data: List[Any]) -> pd.DataFrame:
    """
    Convert list of StockPrice ORM objects to a pandas DataFrame.
    
    Args:
        historical_data: List of StockPrice objects with timestamp, open, high, low, close, volume
    
    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    return pd.DataFrame([{
        "date": d.timestamp,
        "open": d.open,
        "high": d.high,
        "low": d.low,
        "close": d.close,
        "volume": d.volume
    } for d in historical_data])


def get_v9_rank(
    ticker_df: pd.DataFrame,
    spy_data: Optional[dict],
    vix_data: Optional[pd.DataFrame],
    model: Any,
    scaler: Any,
    device: Optional[torch.device] = None
) -> Optional[float]:
    """
    Calculate V9 Transformer rank score for a single ticker.
    
    Args:
        ticker_df: DataFrame with ticker's OHLCV data
        spy_data: Processed SPY data dict from process_spy_data()
        vix_data: Processed VIX data from process_vix_data()
        model: V9 TransformerRankModel instance
        scaler: Fitted feature scaler
        device: Torch device (optional, uses CPU if not provided)
    
    Returns:
        Rank score (0-100) or None if insufficient data
    """
    from services.feature_engineering_v9 import (
        compute_v9_features,
        get_v9_feature_names,
    )
    
    if model is None:
        logger.warning("V9 model not loaded, cannot calculate rank")
        return None
    
    try:
        # Compute features
        processed_df = compute_v9_features(ticker_df, spy_data, vix_data)
        
        if len(processed_df) < V9_WINDOW_SIZE:
            logger.debug(f"Insufficient data for V9 rank: {len(processed_df)} < {V9_WINDOW_SIZE}")
            return None
        
        # Get feature names and extract window
        v9_features = get_v9_feature_names()
        window_seq = processed_df.iloc[-V9_WINDOW_SIZE:][v9_features].values
        
        # Scale features
        if scaler is not None:
            window_seq = scaler.transform(window_seq)
        
        # Create tensor and run inference
        input_tensor = torch.tensor(window_seq, dtype=torch.float32).unsqueeze(0)
        
        if device is not None:
            input_tensor = input_tensor.to(device)
        
        rank_score = model.predict(input_tensor).item()
        
        # Convert to 0-100 scale
        return rank_score * 100
        
    except Exception as e:
        logger.error(f"V9 rank calculation failed: {e}")
        return None


def batch_v9_inference(
    ticker_dfs: dict,
    spy_data: Optional[dict],
    vix_data: Optional[pd.DataFrame],
    model: Any,
    scaler: Any,
    device: Optional[torch.device] = None
) -> dict:
    """
    Run batched V9 inference on multiple tickers for efficiency.
    
    Args:
        ticker_dfs: Dict mapping ticker -> DataFrame
        spy_data: Processed SPY data
        vix_data: Processed VIX data
        model: V9 model
        scaler: Feature scaler
        device: Torch device
    
    Returns:
        Dict mapping ticker -> rank score (0-100)
    """
    from services.feature_engineering_v9 import (
        compute_v9_features,
        get_v9_feature_names,
    )
    
    if model is None:
        return {}
    
    results = {}
    valid_windows = []
    valid_tickers = []
    
    try:
        v9_features = get_v9_feature_names()
        
        # Prepare all windows
        for ticker, df in ticker_dfs.items():
            processed_df = compute_v9_features(df, spy_data, vix_data)
            
            if len(processed_df) >= V9_WINDOW_SIZE:
                window = processed_df.iloc[-V9_WINDOW_SIZE:][v9_features].values
                
                if scaler is not None:
                    window = scaler.transform(window)
                
                valid_windows.append(window)
                valid_tickers.append(ticker)
        
        if not valid_windows:
            return {}
        
        # Batch inference
        batch_tensor = torch.tensor(np.stack(valid_windows), dtype=torch.float32)
        
        if device is not None:
            batch_tensor = batch_tensor.to(device)
        
        with torch.no_grad():
            predictions = model(batch_tensor).cpu().numpy().flatten()
        
        # Map back to tickers
        for ticker, pred in zip(valid_tickers, predictions):
            results[ticker] = float(pred) * 100
        
        return results
        
    except Exception as e:
        logger.error(f"Batch V9 inference failed: {e}")
        return {}
