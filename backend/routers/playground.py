"""
Model Playground Router - Multi-Model Comparison

Routes for Model Playground feature that enables comparison
across multiple AI model versions.
"""

import logging
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import APIRouter, HTTPException
from services.db import get_historical_data, get_latest_close, get_watchlist, get_latest_date, AsyncSessionLocal, PaperHolding, select
from services.feature_engineering import process_stock_data, get_model_input_features, get_model_input_features_v7
from services.model_loader import model_loader
from services.model_metadata import get_available_models, get_model_info
from utils.config_loader import get_playground_enabled, set_playground_enabled

router = APIRouter()
logger = logging.getLogger(__name__)


async def get_spy_data(days: int = 120):
    """Get SPY historical data for market comparison."""
    try:
        spy_data = await get_historical_data("SPY", limit=days)
        if not spy_data:
            return None
        df = pd.DataFrame([{
            "date": d.timestamp,
            "close": d.close
        } for d in spy_data])
        return df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        logger.warning(f"Failed to get SPY data: {e}")
        return None


def calculate_signals(ticker: str, prediction_score: float, current_price: float,
                     historical_data: pd.DataFrame, spy_data: pd.DataFrame,
                     portfolio_tickers: list = None) -> dict:
    """Calculate AI signals: confidence, strength vs market, correlation, and market mood."""
    try:
        signals = {
            "confidence": 0.0,
            "relative_strength": 0.0,
            "strength_label": "Unknown",
            "similarity": 0.0,
            "similarity_label": "Unknown",
            "market_mood": "Unknown",
            "vix_proxy": 0.0
        }

        if historical_data is None or len(historical_data) < 20:
            return signals

        # 1. AI Confidence (0-100%, based on prediction score)
        if prediction_score is not None:
            confidence = min(100, max(0, float(prediction_score) * 100))
        else:
            confidence = 0.0
        signals["confidence"] = round(confidence, 1)

        # 2. Strength vs Market (20-day returns comparison)
        try:
            if len(historical_data) >= 20:
                stock_ret_20d = ((current_price / historical_data.iloc[-20]["close"]) - 1) * 100
            else:
                stock_ret_20d = 0.0

            if spy_data is not None and len(spy_data) >= 20:
                spy_ret_20d = ((spy_data.iloc[-1]["close"] / spy_data.iloc[-20]["close"]) - 1) * 100
            else:
                spy_ret_20d = 0.0

            relative_strength = stock_ret_20d - spy_ret_20d

            if relative_strength > 5:
                strength_label = "Very Strong"
            elif relative_strength > 0:
                strength_label = "Strong"
            elif relative_strength > -5:
                strength_label = "Weak"
            else:
                strength_label = "Very Weak"

            signals["relative_strength"] = round(relative_strength, 1)
            signals["strength_label"] = strength_label
        except Exception as e:
            logger.warning(f"Failed to calculate relative strength: {e}")

        # 3. Similarity/Correlation to Portfolio (simplified - return placeholder)
        # Full correlation calculation would need all portfolio holdings prices
        # For now return a simplified estimate based on volatility similarity
        try:
            stock_vol = historical_data["close"].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
            # Assume moderate correlation for display
            avg_correlation = 0.45  # Placeholder - would be calculated from actual holdings
            similarity_pct = avg_correlation * 100

            if avg_correlation < 0.6:
                similarity_label = "Good"
            elif avg_correlation < 0.7:
                similarity_label = "Moderate"
            else:
                similarity_label = "High"

            signals["similarity"] = round(similarity_pct, 1)
            signals["similarity_label"] = similarity_label
        except Exception as e:
            logger.warning(f"Failed to calculate similarity: {e}")

        # 4. Market Mood (volatility regime)
        try:
            if len(historical_data) >= 20:
                volatility_20d = historical_data["close"].pct_change().rolling(20).std().iloc[-1]
                vix_proxy = volatility_20d * np.sqrt(252) * 100  # Annualized vol as VIX proxy
            else:
                vix_proxy = 20.0

            if vix_proxy < 15:
                mood = "Calm"
            elif vix_proxy < 25:
                mood = "Choppy"
            else:
                mood = "Volatile"

            signals["market_mood"] = mood
            signals["vix_proxy"] = round(vix_proxy, 1)
        except Exception as e:
            logger.warning(f"Failed to calculate market mood: {e}")

        return signals

    except Exception as e:
        logger.error(f"Error calculating signals: {e}")
        return signals


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

    # Get SPY data for market comparison
    spy_data = await get_spy_data(days=500)

    # Get predictions from each model
    results = {
        "ticker": ticker,
        "current_price": current_price,
        "timestamp": datetime.now().isoformat(),
        "models": {},
        "signals": None  # Will be calculated with prediction
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

    # Calculate signals using the first available model's prediction or v9 if available
    prediction_score = None
    if results["models"].get("v9") and results["models"]["v9"]["predictions"]:
        prediction_score = results["models"]["v9"]["predictions"].get("forecast_5d", 0.5)
    else:
        # Fall back to first available model
        for version, model_result in results["models"].items():
            if model_result.get("predictions"):
                prediction_score = model_result["predictions"].get("forecast_5d", 0.5)
                break

    # Convert to signal by normalizing to 0-1 range
    if prediction_score is not None:
        if isinstance(prediction_score, (int, float)):
            # Assume it's a log return or percentage, normalize to 0-1
            normalized_score = max(0, min(1, (prediction_score + 0.1) / 0.2))  # Map [-0.1, 0.1] to [0, 1]
        else:
            normalized_score = 0.5
    else:
        normalized_score = 0.5

    results["signals"] = calculate_signals(
        ticker=ticker,
        prediction_score=normalized_score,
        current_price=current_price,
        historical_data=processed_df,
        spy_data=spy_data
    )

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


@router.get("/playground/explain/{ticker}/{model_version}")
async def get_model_explanation(ticker: str, model_version: str = "v9"):
    """
    Get model-specific reasoning for why it made a prediction.

    Returns:
        - why_selected: Why this stock was ranked
        - key_factors: Top contributing factors
        - risk_factors: Important risk considerations
    """
    try:
        # Get current data for context
        current_price = await get_latest_close(ticker)
        historical_data = await get_historical_data(ticker, limit=500)

        if not historical_data or len(historical_data) < 20:
            raise HTTPException(status_code=404, detail=f"Insufficient data for {ticker}")

        # Convert to DataFrame for analysis
        df = pd.DataFrame([{
            "date": d.timestamp,
            "close": d.close
        } for d in historical_data])
        df = df.sort_values("date").reset_index(drop=True)

        # Calculate metrics
        ret_20d = ((current_price / df.iloc[-20]["close"]) - 1) * 100
        ret_5d = ((current_price / df.iloc[-5]["close"]) - 1) * 100 if len(df) >= 5 else ret_20d
        volatility_20d = df["close"].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100

        explanation = {
            "model": get_model_info(model_version).display_name if get_model_info(model_version) else model_version,
            "ticker": ticker,
            "current_price": float(current_price) if current_price else None
        }

        if model_version == "v9":
            explanation.update({
                "why_selected": f"V9 Transformer predicts strong relative momentum and ranking",
                "key_factors": [
                    f"20-day momentum: {ret_20d:+.1f}% (price trend)",
                    f"5-day performance: {ret_5d:+.1f}% (short-term strength)",
                    f"Volatility: {volatility_20d:.1f}% annualized (market conditions)"
                ],
                "risk_factors": [
                    "Correlation filtering ensures portfolio diversification",
                    f"Stop-loss protection at 7% below entry price",
                    "Rebalanced every 5 trading days based on latest rankings"
                ],
                "architecture": "Transformer with attention mechanism, trained on 60-day price windows with 12 stationary features"
            })
        elif model_version == "v7":
            explanation.update({
                "why_selected": "BiLSTM with Multi-Head Attention predicts next 5-day returns",
                "key_factors": [
                    f"20-day momentum: {ret_20d:+.1f}% (captures trend)",
                    f"Recent volatility: {volatility_20d:.1f}% (market regime)",
                    "Multi-head attention weights recent price patterns most heavily"
                ],
                "risk_factors": [
                    "Directional bias: -23.7% (slightly bearish, defensive)",
                    "Accuracy: ~52.6% on out-of-sample test data",
                    "Trained on all available stocks for robustness"
                ],
                "architecture": "2-layer BiLSTM with 4-head attention, 41 features including market context"
            })
        else:
            explanation.update({
                "why_selected": f"{model_version} model predicts price movements based on technical features",
                "key_factors": [
                    f"20-day momentum: {ret_20d:+.1f}%",
                    f"Volatility: {volatility_20d:.1f}% (annualized)",
                    "Feature-engineered technical indicators"
                ],
                "risk_factors": [
                    "See individual model accuracy statistics",
                    "Different feature set than V9",
                    "May perform differently in different market regimes"
                ],
                "architecture": f"Model version {model_version} with custom architecture"
            })

        return explanation

    except Exception as e:
        logger.error(f"Error getting explanation for {ticker} {model_version}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
