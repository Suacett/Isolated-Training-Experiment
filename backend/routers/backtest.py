import logging
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
from services.db import get_historical_data, save_prediction
from services.feature_engineering import process_stock_data, get_model_input_features
from state import state

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/backtest/all")
async def backtest_all(days: int = 30):
    """
    Generate historical predictions for all tickers in the watchlist.

    Args:
        days: Number of days to backtest for each ticker (default 30)

    Returns:
        Summary of predictions generated for all tickers
    """
    try:
        from services.db import get_watchlist

        tickers = await get_watchlist()

        if not tickers:
            return {"message": "Watchlist is empty"}

        logger.info(f"Starting backtest for {len(tickers)} tickers")

        results = []
        total_predictions = 0

        for ticker in tickers:
            try:
                result = await backtest_ticker(ticker, days)
                results.append(result)
                total_predictions += result["predictions_generated"]
            except Exception as e:
                logger.error(f"Failed to backtest {ticker}: {e}")
                results.append({
                    "ticker": ticker,
                    "error": str(e)
                })

        return {
            "message": f"Backtest complete for {len(tickers)} tickers",
            "total_predictions": total_predictions,
            "results": results
        }

    except Exception as e:
        logger.error(f"Backtest all failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest/{ticker}")
async def backtest_ticker(ticker: str, days: int = 30):
    """
    Generate historical predictions for backtesting and chart visualization.

    This creates predictions for past dates so the chart can display them.
    For each historical day (going back 'days'), we:
    1. Use data available up to that date
    2. Generate a prediction for the next day
    3. Save it with the actual next day as target_date

    Args:
        ticker: Stock symbol
        days: Number of days to backtest (default 30)

    Returns:
        Number of predictions generated
    """
    try:
        logger.info(f"Starting backtest for {ticker} ({days} days)")

        # Fetch more historical data than we need for backtesting
        historical_data = await get_historical_data(ticker, limit=150)

        if len(historical_data) < 90:
            raise HTTPException(status_code=400, detail=f"Insufficient data for {ticker}")

        # Convert to DataFrame
        df = pd.DataFrame([{
            "date": d.timestamp,
            "open": d.open,
            "high": d.high,
            "low": d.low,
            "close": d.close,
            "volume": d.volume
        } for d in historical_data])

        df = df.sort_values("date").reset_index(drop=True)

        # Process all features at once
        processed_df = process_stock_data(df, create_targets=False)
        feature_cols = get_model_input_features()

        logger.info(f"Processed {len(processed_df)} data points for {ticker} (from {len(df)} raw)")

        # Need at least 61 points (60 for LSTM window + 1 for prediction)
        if len(processed_df) < 61:
            raise HTTPException(status_code=400, detail=f"Insufficient processed data for {ticker}: {len(processed_df)} points (need 61+)")

        predictions_saved = 0
        errors = 0

        # Start from the end and work backwards
        # We need at least 60 days of history for the LSTM
        for i in range(min(days, len(processed_df) - 61)):
            # Index for the "current" day we're pretending to be on
            current_idx = len(processed_df) - 1 - i

            # Get data up to current_idx (simulating we only have data up to this date)
            available_data = processed_df.iloc[:current_idx + 1]

            if len(available_data) < 60:
                continue

            # Get the last 60 days of features
            seq_data = available_data.iloc[-60:][feature_cols].values

            # Get current price and date
            current_price = float(df.iloc[current_idx]["close"])
            current_date = df.iloc[current_idx]["date"]

            # Target date is the next trading day
            if current_idx + 1 < len(df):
                target_date = df.iloc[current_idx + 1]["date"]
            else:
                # No next day available
                continue

            try:
                # Scale
                if not state.scaler:
                    logger.error("Scaler not loaded")
                    break

                scaled_seq = state.scaler.transform(seq_data)

                # Create tensor
                input_tensor = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0)

                # Generate prediction with uncertainty
                prediction_result = state.lstm_model.predict_with_uncertainty(input_tensor, state.device)

                # Extract 1-day prediction
                predicted_value = float(prediction_result["prediction"][0])
                confidence = float(prediction_result["confidence"][0])

                # Save prediction (only 1-day horizon for backtesting)
                await save_prediction(
                    ticker=ticker,
                    prediction_date=current_date,
                    target_date=target_date,
                    horizon="1d",
                    predicted_value=predicted_value,
                    current_price=current_price,
                    confidence=confidence
                )

                predictions_saved += 1

            except Exception as e:
                logger.error(f"Error generating backtest prediction for {ticker} at {current_date}: {e}")
                errors += 1
                continue

        logger.info(f"Backtest complete for {ticker}: {predictions_saved} predictions saved, {errors} errors")

        return {
            "ticker": ticker,
            "predictions_generated": predictions_saved,
            "errors": errors,
            "message": f"Generated {predictions_saved} historical predictions for chart visualization"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
