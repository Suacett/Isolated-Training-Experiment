"""
Data Ingestion Router - Yahoo Finance Primary

Routes for ingesting stock data.
"""

import logging
import torch
import numpy as np
import asyncio
from datetime import datetime
from fastapi import APIRouter, HTTPException
from services.db import get_watchlist, get_historical_data, save_prediction
from services.data_ingest import YahooFinanceClient
from services.feature_engineering import process_stock_data, get_model_input_features
from services.scaler import FeatureScaler
from state import state
import pandas as pd

router = APIRouter()
logger = logging.getLogger(__name__)


async def process_single_ticker_prediction(ticker: str) -> dict:
    """
    Process prediction for a single ticker.
    Returns dict with ticker and prediction, or error info.
    """
    try:
        historical_data = await get_historical_data(ticker, limit=100)
        if len(historical_data) < 60:
            return {"ticker": ticker, "error": "Insufficient data"}

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
            return {"ticker": ticker, "error": "Insufficient processed data"}

        # Get current price (last close price before processing)
        current_price = float(df.iloc[-1]["close"])

        # Get latest sequence of 60 steps
        seq_data = processed_df.iloc[-60:][feature_cols].values

        # Scale
        if not state.scaler:
            return {"ticker": ticker, "error": "Scaler not loaded"}

        scaled_seq = state.scaler.transform(seq_data)

        # Create tensor (Batch, SeqLen, Features) -> (1, 60, 37)
        input_tensor = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0)

        logger.info(f"Predicting for {ticker} with input shape {input_tensor.shape}")

        # Use the global model with uncertainty estimation
        prediction_result = state.lstm_model.predict_with_uncertainty(input_tensor, state.device)

        # Extract predictions for all horizons
        all_horizons = prediction_result["prediction_all_horizons"][0]  # Shape: (4,) for 4 horizons
        confidence = float(prediction_result["confidence"][0])

        # Get prediction date (now)
        prediction_date = datetime.now()

        # Define horizons and their day offsets
        horizons = [
            ("1d", 1),
            ("1w", 5),
            ("1m", 21),
            ("6m", 126)
        ]

        # Save predictions for all horizons
        # Model outputs log returns - convert to prices
        for i, (horizon_name, days_ahead) in enumerate(horizons):
            target_date = prediction_date + pd.Timedelta(days=days_ahead)
            log_return = float(all_horizons[i])
            predicted_price = current_price * np.exp(log_return)

            await save_prediction(
                ticker=ticker,
                prediction_date=prediction_date,
                target_date=target_date,
                horizon=horizon_name,
                predicted_value=log_return,  # Store raw log return
                current_price=current_price,
                confidence=confidence
            )

        # Return 1-day prediction (converted to price) for immediate use
        log_return_1d = float(all_horizons[0])
        predicted_price_1d = current_price * np.exp(log_return_1d)
        
        return {
            "ticker": ticker,
            "prediction": predicted_price_1d,
            "log_return": log_return_1d,
            "current_price": current_price,
            "confidence": confidence
        }

    except Exception as e:
        logger.error(f"Error predicting for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}


@router.post("/ingest/all")
async def ingest_all(mode: str = "daily"):
    """
    Ingest data for all watched tickers using Yahoo Finance.
    Args:
        mode: "daily" (default) or "full"
    """
    # 1. Get Watchlist
    tickers = await get_watchlist()
    if not tickers:
        return {"message": "Watchlist is empty"}

    logger.info(f"Starting ingestion for {len(tickers)} tickers via Yahoo Finance (mode={mode})")

    # 2. Ingest Data
    # fetch_all_data doesn't support mode yet, but it fetches what's needed.
    # If mode is full, we might want to force full fetch in fetch_all_data?
    # For now, let's assume fetch_all_data gets enough data.
    # Actually, fetch_all_data in YahooFinanceClient might need a mode too.
    # But let's stick to the current flow: ingest then predict.
    
    client = YahooFinanceClient()
    # If mode is full, we should probably use fetch_data(ticker) in loop or update fetch_all_data
    # But fetch_all_data is efficient. Let's use it.
    results = await client.fetch_all_data(tickers, mode=mode)

    # 3. Run Predictions / Backfill
    predictions = []
    backfills = []

    # Check if model is loaded
    if state.lstm_model is None:
        logger.warning("LSTM model not initialized, skipping predictions")
        return {
            "message": "Ingestion complete, but model not initialized",
            "ingestion": results,
            "predictions": []
        }

    # Helper to process single ticker with backfill if needed
    async def process_ticker(ticker):
        pred_res = await process_single_ticker_prediction(ticker)
        
        backfill_res = None
        if mode == "full" and "error" not in pred_res:
             try:
                from routers.predictions import backfill_predictions_full
                bf_data = await backfill_predictions_full(ticker)
                backfill_res = {"ticker": ticker, "count": len(bf_data), "status": "success"}
             except Exception as e:
                logger.error(f"Backfill failed for {ticker}: {e}")
                backfill_res = {"ticker": ticker, "error": str(e)}
        
        return {"prediction": pred_res, "backfill": backfill_res}

    # Run in parallel
    # Use semaphore to prevent OOM with full backfill on many tickers
    sem = asyncio.Semaphore(5) # Limit to 5 concurrent backfills
    
    async def safe_process(ticker):
        async with sem:
            return await process_ticker(ticker)

    all_results = await asyncio.gather(*[safe_process(t) for t in tickers])

    # Organize results
    predictions = [r["prediction"] for r in all_results if "prediction" in r and "error" not in r["prediction"]]
    errors = [r["prediction"] for r in all_results if "prediction" in r and "error" in r["prediction"]]
    backfills = [r["backfill"] for r in all_results if r["backfill"]]

    logger.info(f"Batch processing complete: {len(predictions)} predictions, {len(backfills)} backfills")

    return {
        "message": "Ingestion and prediction complete",
        "source": "yahoo_finance",
        "ingestion": results,
        "predictions": predictions,
        "backfills": backfills,
        "errors": errors if errors else None
    }


@router.post("/ingest/{ticker}")
async def ingest_data(ticker: str, mode: str = "full"):
    """
    Ingest data for a single ticker using Yahoo Finance and run predictions.
    
    Args:
        ticker: Stock symbol
        mode: "full" (default) or "daily"
    """
    logger.info(f"📊 Ingesting data for {ticker} via Yahoo Finance (mode={mode})...")
    
    try:
        # 1. Ingest Data (Hybrid: Yahoo Prices + Alpha Vantage Sentiment)
        # This also handles reconciliation now
        from services.data_ingest import ingest_hybrid_data
        ingest_result = await ingest_hybrid_data(ticker, mode=mode)
        
        # 2. Run prediction if model is loaded
        prediction_result = None
        backfill_result = None
        if state.lstm_model is not None:
            prediction_result = await process_single_ticker_prediction(ticker)
            if "error" in prediction_result:
                logger.warning(f"Prediction failed for {ticker}: {prediction_result['error']}")
            else:
                logger.info(f"✅ Generated prediction for {ticker}: ${prediction_result.get('prediction', 0):.2f}")
            
            # 3. If full sync, also backfill historical predictions
            if mode == "full":
                try:
                    from routers.predictions import backfill_predictions_full
                    backfill_data = await backfill_predictions_full(ticker)
                    backfill_result = {
                        "predictions_generated": len(backfill_data),
                        "status": "complete"
                    }
                    logger.info(f"✅ Full backfill complete for {ticker}: {len(backfill_data)} predictions")
                except Exception as e:
                    logger.warning(f"⚠️ Backfill failed for {ticker}: {e}")
                    backfill_result = {"status": "failed", "error": str(e)}
        
        return {
            "message": f"Ingestion complete for {ticker}",
            "source": "hybrid",
            "ingestion": ingest_result,
            "prediction": prediction_result,
            "backfill": backfill_result
        }
    except Exception as e:
        logger.error(f"❌ Failed to ingest {ticker}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch data for {ticker}: {str(e)}")

