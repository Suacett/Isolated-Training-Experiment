#!/usr/bin/env python3
"""
Daily Data Synchronization Script

This script fetches the previous day's closing data for all watchlist tickers
and runs predictions. It is designed to be run daily via cron or scheduler.

Usage:
    python -m scripts.daily_sync

Schedule (add to crontab):
    0 6 * * * cd /opt/stock-predictor && docker exec proxmox_stock_backend python -m scripts.daily_sync

The script runs at 6 AM local time by default, which provides buffer time after
US market close (4 PM ET) for data availability.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.db import get_watchlist, get_historical_data, AsyncSessionLocal
from services.data_ingest import YahooFinanceClient
from services.feature_engineering import process_stock_data, get_model_input_features
from state import state
import pandas as pd
import torch
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("daily_sync.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


async def fetch_daily_data(tickers: list[str]) -> dict:
    """
    Fetch the latest daily data for all tickers.
    
    Args:
        tickers: List of stock symbols
        
    Returns:
        Dictionary with success/failure counts
    """
    client = YahooFinanceClient()
    results = {"success": 0, "failed": []}
    
    logger.info(f"Fetching daily data for {len(tickers)} tickers...")
    
    # Use daily mode to only fetch recent data
    fetch_results = await client.fetch_all_data(tickers, mode="daily")
    
    for ticker, result in fetch_results.items():
        if "error" in str(result).lower():
            results["failed"].append(ticker)
        else:
            results["success"] += 1
    
    return results


async def generate_predictions(tickers: list[str]) -> dict:
    """
    Generate predictions for all tickers using the loaded model.
    
    Args:
        tickers: List of stock symbols
        
    Returns:
        Dictionary with predictions and accuracy stats
    """
    from services.db import save_prediction
    
    if state.lstm_model is None:
        logger.warning("LSTM model not loaded. Skipping predictions.")
        return {"error": "Model not loaded"}
    
    if state.scaler is None:
        logger.warning("Scaler not loaded. Skipping predictions.")
        return {"error": "Scaler not loaded"}
    
    results = {"success": 0, "failed": [], "predictions": []}
    
    for ticker in tickers:
        try:
            # Get historical data
            historical_data = await get_historical_data(ticker, limit=100)
            
            if len(historical_data) < 60:
                logger.warning(f"Insufficient data for {ticker}: {len(historical_data)} days")
                results["failed"].append({"ticker": ticker, "error": "Insufficient data"})
                continue
            
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
                logger.warning(f"Insufficient processed data for {ticker}")
                results["failed"].append({"ticker": ticker, "error": "Feature processing failed"})
                continue
            
            # Get current price
            current_price = float(df.iloc[-1]["close"])
            
            # Get latest 60-day sequence
            seq_data = processed_df.iloc[-60:][feature_cols].values
            scaled_seq = state.scaler.transform(seq_data)
            
            # Create tensor and predict
            input_tensor = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0)
            prediction_result = state.lstm_model.predict_with_uncertainty(input_tensor, state.device)
            
            # Extract 1-day prediction
            log_return = float(prediction_result["prediction_all_horizons"][0][0])
            predicted_price = current_price * np.exp(log_return)
            confidence = float(prediction_result["confidence"][0])
            
            # Save prediction to database
            await save_prediction(
                ticker=ticker,
                prediction_date=datetime.now(),
                target_date=datetime.now() + timedelta(days=1),
                horizon="1d",
                predicted_value=log_return,
                current_price=current_price,
                confidence=confidence
            )
            
            results["success"] += 1
            results["predictions"].append({
                "ticker": ticker,
                "current_price": current_price,
                "predicted_price": round(predicted_price, 2),
                "change_pct": round((np.exp(log_return) - 1) * 100, 2),
                "confidence": round(confidence, 2)
            })
            
            logger.info(f"Prediction for {ticker}: ${current_price:.2f} -> ${predicted_price:.2f} ({(np.exp(log_return) - 1) * 100:.2f}%)")
            
        except Exception as e:
            logger.error(f"Prediction failed for {ticker}: {e}")
            results["failed"].append({"ticker": ticker, "error": str(e)})
    
    return results


async def validate_past_predictions() -> dict:
    """
    Validate predictions made in the past against actual prices.
    
    Returns:
        Dictionary with validation statistics
    """
    from services.db import get_pending_predictions, update_prediction_actual
    
    try:
        pending = await get_pending_predictions()
        
        if not pending:
            logger.info("No pending predictions to validate")
            return {"validated": 0, "correct": 0}
        
        logger.info(f"Validating {len(pending)} pending predictions...")
        
        validated = 0
        correct = 0
        
        for pred in pending:
            try:
                # Get actual price at target date
                historical = await get_historical_data(pred.ticker, limit=30)
                
                # Find the price for the target date
                actual_price = None
                for h in historical:
                    if h.timestamp.date() == pred.target_date.date():
                        actual_price = h.close
                        break
                
                if actual_price is None:
                    continue
                
                # Calculate actual log return
                actual_return = np.log(actual_price / pred.current_price)
                
                # Check if direction is correct
                is_correct = (
                    (pred.predicted_value > 0 and actual_return > 0) or
                    (pred.predicted_value < 0 and actual_return < 0)
                )
                
                # Update prediction in database
                await update_prediction_actual(
                    pred.id,
                    actual_value=actual_return,
                    is_correct=is_correct
                )
                
                validated += 1
                if is_correct:
                    correct += 1
                    
            except Exception as e:
                logger.error(f"Validation failed for prediction {pred.id}: {e}")
        
        accuracy = (correct / validated * 100) if validated > 0 else 0
        logger.info(f"Validation complete: {correct}/{validated} correct ({accuracy:.1f}%)")
        
        return {"validated": validated, "correct": correct, "accuracy": accuracy}
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {"error": str(e)}


async def main():
    """
    Main daily sync routine.
    
    1. Fetch latest closing data for all watchlist tickers
    2. Generate predictions for the next trading day
    3. Validate past predictions against actual prices
    """
    logger.info("=" * 60)
    logger.info("DAILY SYNC STARTED")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    try:
        # Get watchlist
        tickers = await get_watchlist()
        
        if not tickers:
            logger.warning("Watchlist is empty. Nothing to sync.")
            return
        
        logger.info(f"Watchlist contains {len(tickers)} tickers: {tickers}")
        
        # Step 1: Fetch daily data
        logger.info("-" * 40)
        logger.info("STEP 1: Fetching Daily Data")
        logger.info("-" * 40)
        
        fetch_results = await fetch_daily_data(tickers)
        logger.info(f"Fetch results: {fetch_results['success']} successful, {len(fetch_results['failed'])} failed")
        
        if fetch_results["failed"]:
            logger.warning(f"Failed tickers: {fetch_results['failed']}")
        
        # Step 2: Generate predictions
        logger.info("-" * 40)
        logger.info("STEP 2: Generating Predictions")
        logger.info("-" * 40)
        
        pred_results = await generate_predictions(tickers)
        
        if "error" not in pred_results:
            logger.info(f"Prediction results: {pred_results['success']} successful, {len(pred_results['failed'])} failed")
        else:
            logger.error(f"Prediction error: {pred_results['error']}")
        
        # Step 3: Validate past predictions
        logger.info("-" * 40)
        logger.info("STEP 3: Validating Past Predictions")
        logger.info("-" * 40)
        
        validation_results = await validate_past_predictions()
        
        if "error" not in validation_results:
            logger.info(f"Validation: {validation_results['correct']}/{validation_results['validated']} correct")
        else:
            logger.error(f"Validation error: {validation_results['error']}")
        
        # Summary
        logger.info("=" * 60)
        logger.info("DAILY SYNC COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Data fetched: {fetch_results['success']}/{len(tickers)}")
        logger.info(f"Predictions made: {pred_results.get('success', 0)}")
        logger.info(f"Predictions validated: {validation_results.get('validated', 0)}")
        
    except Exception as e:
        logger.error(f"Daily sync failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
