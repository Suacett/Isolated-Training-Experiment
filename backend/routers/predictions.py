"""
Prediction API routes for frontend integration.

Provides endpoints to:
- Get prediction history for a ticker
- Get current/future predictions
- Get prediction accuracy statistics
"""

import logging
import torch
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta

from services.db import (
    get_predictions_for_ticker,
    get_pending_predictions,
    get_historical_data_range,
    save_prediction
)
from services.feature_engineering import process_stock_data, get_model_input_features
from state import state

router = APIRouter(prefix="/predictions", tags=["predictions"])
logger = logging.getLogger(__name__)


class PredictionResponse(BaseModel):
    id: int
    ticker: str
    prediction_date: datetime
    target_date: datetime
    horizon: str
    predicted_value: float
    predicted_price: float  # Converted from log return
    actual_value: Optional[float]
    current_price: float
    is_correct: Optional[bool]
    confidence: Optional[float]


class PredictionStatsResponse(BaseModel):
    ticker: str
    total_predictions: int
    validated_predictions: int
    correct_predictions: int
    accuracy: float
    by_horizon: dict


@router.get("/{ticker}", response_model=List[PredictionResponse])
async def get_ticker_predictions(ticker: str, limit: int = 50):
    """
    Get prediction history for a ticker.

    Args:
        ticker: Stock ticker symbol
        limit: Maximum number of predictions to return

    Returns:
        List of predictions, newest first
    """
    try:
        predictions = await get_predictions_for_ticker(ticker, limit)

        return [
            PredictionResponse(
                id=p.id,
                ticker=p.ticker,
                prediction_date=p.prediction_date,
                target_date=p.target_date,
                horizon=p.horizon,
                predicted_value=p.predicted_value,
                # Convert log return to price: Price * exp(log_return)
                predicted_price=p.current_price * np.exp(p.predicted_value),
                actual_value=p.actual_value,
                current_price=p.current_price,
                is_correct=p.is_correct,
                confidence=p.confidence
            )
            for p in predictions
        ]
    except Exception as e:
        logger.error(f"Failed to get predictions for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticker}/stats", response_model=PredictionStatsResponse)
async def get_prediction_stats(ticker: str):
    """
    Get prediction accuracy statistics for a ticker.

    Returns:
        Statistics including overall accuracy and per-horizon breakdown
    """
    try:
        predictions = await get_predictions_for_ticker(ticker, limit=1000)

        if not predictions:
            return PredictionStatsResponse(
                ticker=ticker,
                total_predictions=0,
                validated_predictions=0,
                correct_predictions=0,
                accuracy=0.0,
                by_horizon={}
            )

        total = len(predictions)
        validated = sum(1 for p in predictions if p.actual_value is not None)
        correct = sum(1 for p in predictions if p.is_correct is True)

        accuracy = (correct / validated * 100) if validated > 0 else 0.0

        # Breakdown by horizon
        by_horizon = {}
        for horizon in ['1d', '1w', '1m', '6m']:
            horizon_preds = [p for p in predictions if p.horizon == horizon]
            horizon_validated = sum(1 for p in horizon_preds if p.actual_value is not None)
            horizon_correct = sum(1 for p in horizon_preds if p.is_correct is True)

            by_horizon[horizon] = {
                'total': len(horizon_preds),
                'validated': horizon_validated,
                'correct': horizon_correct,
                'accuracy': (horizon_correct / horizon_validated * 100) if horizon_validated > 0 else 0.0
            }

        return PredictionStatsResponse(
            ticker=ticker,
            total_predictions=total,
            validated_predictions=validated,
            correct_predictions=correct,
            accuracy=accuracy,
            by_horizon=by_horizon
        )

    except Exception as e:
        logger.error(f"Failed to get stats for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pending/validate")
async def get_pending_validation():
    """
    Get predictions that need validation (target date has passed).

    This endpoint is used by background tasks to update predictions with actual outcomes.
    """
    try:
        pending = await get_pending_predictions()

        return [
            {
                'id': p.id,
                'ticker': p.ticker,
                'target_date': p.target_date,
                'predicted_value': p.predicted_value,
                'current_price': p.current_price
            }
            for p in pending
        ]

    except Exception as e:
        logger.error(f"Failed to get pending predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class BackfillDataPoint(BaseModel):
    date: datetime
    current_price: float  # Price at prediction time
    actual_price: float   # Price at target time
    predicted_price: float
    correct_direction: bool
    error_percentage: float  # Percentage difference between predicted and actual
    confidence: Optional[float] = None


@router.post("/{ticker}/backfill", response_model=List[BackfillDataPoint])
async def backfill_predictions(ticker: str, days: int = 90):
    """
    Generate historical predictions for backtesting.

    For each of the last N days, fetches the prior 60 days of data,
    generates a next-day prediction, and compares with actual price.

    Args:
        ticker: Stock ticker symbol
        days: Number of days to backfill (default: 90)

    Returns:
        List of backfill data points with date, actual price, predicted price, and direction correctness
    """
    try:
        # Check if model is loaded
        if state.lstm_model is None:
            raise HTTPException(status_code=503, detail="LSTM model not initialized")

        if not state.scaler:
            raise HTTPException(status_code=503, detail="Scaler not loaded")

        # Get current date and start date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 60)  # Extra 60 days for initial window

        logger.info(f"Backfilling predictions for {ticker} from {start_date} to {end_date}")

        # Fetch all historical data for the period
        all_data = await get_historical_data_range(ticker, start_date, end_date)

        if len(all_data) < 61:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for {ticker}: need at least 61 days, got {len(all_data)}"
            )

        # Convert to DataFrame
        df_all = pd.DataFrame([{
            "date": d.timestamp,
            "open": d.open,
            "high": d.high,
            "low": d.low,
            "close": d.close,
            "volume": d.volume
        } for d in all_data])

        df_all = df_all.sort_values("date").reset_index(drop=True)

        # Process features for entire dataset
        processed_df = process_stock_data(df_all, create_targets=False)
        feature_cols = get_model_input_features()

        window_size = state.lstm_model.window_size if state.lstm_model else 60

        if len(processed_df) < window_size + 1:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient processed data for {ticker}: need at least {window_size + 1} days after feature engineering"
            )

        backfill_results = []

        # Iterate through the last N days (skip first window_size for window)
        for i in range(window_size, min(len(processed_df) - 1, window_size + days)):
            # Get window_size-day window ending at day i
            window_data = processed_df.iloc[i-window_size:i][feature_cols].values

            # Current price (at end of window)
            current_price = float(df_all.iloc[i]["close"])

            # Actual next-day price
            actual_price = float(df_all.iloc[i + 1]["close"])

            # Scale the window
            scaled_window = state.scaler.transform(window_data)

            # Create tensor
            input_tensor = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0)

            # Generate prediction (1-day horizon only)
            # Model outputs LOG RETURNS, not prices!
            prediction_result = state.lstm_model.predict_with_uncertainty(input_tensor, state.device)
            log_return = float(prediction_result["prediction_all_horizons"][0][0])  # 1d horizon
            
            # CRITICAL FIX: Convert log return to predicted price
            # predicted_price = current_price * exp(log_return)
            predicted_price = current_price * np.exp(log_return)

            # Check if direction is correct
            actual_direction = actual_price > current_price
            predicted_direction = predicted_price > current_price
            correct_direction = actual_direction == predicted_direction

            # Calculate error percentage
            error_percentage = abs((predicted_price - actual_price) / actual_price) * 100
            confidence = float(prediction_result["confidence"][0])

            # Store the backfill data point
            prediction_date = df_all.iloc[i]["date"] # Define prediction_date here
            backfill_results.append(BackfillDataPoint(
                date=prediction_date,
                current_price=current_price,
                actual_price=actual_price,
                predicted_price=predicted_price,
                correct_direction=correct_direction,
                error_percentage=error_percentage,
                confidence=confidence
            ))

            # Optionally save to database for later analysis
            # NOTE: We save the log_return (raw model output) to DB for consistency
            prediction_date = df_all.iloc[i]["date"]
            target_date = df_all.iloc[i + 1]["date"]
            confidence = float(prediction_result["confidence"][0])

            await save_prediction(
                ticker=ticker,
                prediction_date=prediction_date,
                target_date=target_date,
                horizon="1d",
                predicted_value=log_return,  # Store raw log return
                current_price=current_price,
                confidence=confidence
            )

        logger.info(f"Backfill complete for {ticker}: {len(backfill_results)} predictions generated")

        return backfill_results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to backfill predictions for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{ticker}/backfill/full", response_model=List[BackfillDataPoint])
async def backfill_predictions_full(ticker: str):
    """
    Generate complete historical predictions for deep backtesting visualization.

    Fetches ALL available historical data (up to 5000 days), generates predictions
    for each day using a sliding 60-day window, and returns the complete dataset
    for visualization.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Complete list of backfill data points with date, actual price, predicted price, and direction correctness
    """
    try:
        # Check if model is loaded
        if state.lstm_model is None:
            raise HTTPException(status_code=503, detail="LSTM model not initialized")

        if not state.scaler:
            raise HTTPException(status_code=503, detail="Scaler not loaded")

        # Fetch ALL historical data (up to 5000 records)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=20000)  # ~55 years of data

        logger.info(f"Deep backfill: Fetching complete history for {ticker} from {start_date} to {end_date}")

        # Fetch all historical data
        all_data = await get_historical_data_range(ticker, start_date, end_date)

        if len(all_data) < 61:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for {ticker}: need at least 61 days, got {len(all_data)}"
            )

        logger.info(f"Deep backfill: Retrieved {len(all_data)} days of data for {ticker}")

        # Convert to DataFrame
        df_all = pd.DataFrame([{
            "date": d.timestamp,
            "open": d.open,
            "high": d.high,
            "low": d.low,
            "close": d.close,
            "volume": d.volume
        } for d in all_data])

        df_all = df_all.sort_values("date").reset_index(drop=True)

        # Process features for entire dataset
        processed_df = process_stock_data(df_all, create_targets=False)
        feature_cols = get_model_input_features()

        window_size = state.lstm_model.window_size if state.lstm_model else 60

        if len(processed_df) < window_size + 1:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient processed data for {ticker}: need at least {window_size + 1} days after feature engineering"
            )

        backfill_results = []
        total_predictions = len(processed_df) - window_size - 1

        logger.info(f"Deep backfill: Generating {total_predictions} predictions for {ticker}")

        # --- VECTORIZED BATCH INFERENCE ---
        
        # 1. Prepare Windows (CPU)
        # Extract numpy arrays for faster slicing
        # CRITICAL: Use processed_df for ALL data (dates, prices, features)
        # processed_df has already aligned indices after dropna()
        feature_data = processed_df[feature_cols].values
        close_prices = processed_df["close"].values  # Use processed_df, NOT df_all!
        dates_series = processed_df["date"].values    # Use processed_df, NOT df_all!
        
        # Create a list of window_size-day windows
        # shape: (num_samples, window_size, num_features)
        # FIX: Use i-window_size+1 : i+1 to include the current day 'i'
        indices = range(window_size, len(processed_df) - 1)
        windows = [feature_data[i-window_size+1:i+1] for i in indices]
        
        if not windows:
            return []
            
        # Convert to numpy array
        windows_array = np.array(windows)
        N, T, F = windows_array.shape
        
        # 2. Batch Scaling
        # Reshape to (N*T, F) for scaler, then back to (N, T, F)
        windows_flat = windows_array.reshape(-1, F)
        scaled_flat = state.scaler.transform(windows_flat)
        scaled_windows = scaled_flat.reshape(N, T, F)
        
        # 3. Create Tensor
        input_tensor = torch.tensor(scaled_windows, dtype=torch.float32)
        
        logger.info(f"Deep backfill: Running batch inference on {input_tensor.shape}...")
        
        # 4. Run Batch Inference (GPU)
        # predict_with_uncertainty handles batches efficiently
        batch_results = state.lstm_model.predict_with_uncertainty(input_tensor, state.device)
        
        # 5. Process Results
        log_returns = batch_results["prediction"] # (N,)
        confidences = batch_results["confidence"] # (N,)
        
        backfill_results = []
        for idx, i in enumerate(indices):
            log_return = float(log_returns[idx])
            current_price = float(close_prices[i])
            actual_price = float(close_prices[i+1])
            
            # Convert log return to price
            predicted_price = current_price * np.exp(log_return)
            
            # Direction
            actual_direction = actual_price > current_price
            predicted_direction = predicted_price > current_price
            correct_direction = actual_direction == predicted_direction
            
            # Error
            error_percentage = abs((predicted_price - actual_price) / actual_price) * 100
            
            # Convert date
            date_val = dates_series[i]
            if isinstance(date_val, np.datetime64):
                date_val = pd.to_datetime(date_val).to_pydatetime()

            backfill_results.append(BackfillDataPoint(
                date=date_val,
                current_price=current_price,
                actual_price=actual_price,
                predicted_price=predicted_price,
                correct_direction=correct_direction,
                error_percentage=error_percentage,
                confidence=float(confidences[idx])
            ))

            # SAVE TO DB
            # We need to save this so it shows up in the graph
            target_date = dates_series[i+1]
            if isinstance(target_date, np.datetime64):
                target_date = pd.to_datetime(target_date).to_pydatetime()

            await save_prediction(
                ticker=ticker,
                prediction_date=date_val,
                target_date=target_date,
                horizon="1d",
                predicted_value=log_return,
                current_price=current_price,
                confidence=float(confidences[idx])
            )

        logger.info(f"Deep backfill complete for {ticker}: {len(backfill_results)} predictions generated")

        # Calculate accuracy
        correct_count = sum(1 for r in backfill_results if r.correct_direction)
        accuracy = (correct_count / len(backfill_results) * 100) if backfill_results else 0
        logger.info(f"Deep backfill accuracy for {ticker}: {accuracy:.2f}% ({correct_count}/{len(backfill_results)})")

        return backfill_results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to perform deep backfill for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
