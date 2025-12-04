import logging
import torch
import numpy as np
from datetime import datetime
from fastapi import APIRouter, HTTPException
from services.db import get_watchlist, get_historical_data, save_prediction
from services.data_ingest import AlpacaDataClient, AlphaVantageClient
from services.feature_engineering import process_stock_data, get_model_input_features
from services.scaler import FeatureScaler
from state import state
import pandas as pd

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/ingest/all")
async def ingest_all():
    # 1. Get Watchlist
    tickers = await get_watchlist()
    if not tickers:
        return {"message": "Watchlist is empty"}
    
    # 2. Ingest Data
    try:
        client = AlpacaDataClient()
        await client.fetch_all_data(tickers)
    except Exception as e:
        logger.warning(f"Batch data ingestion failed: {e}. Attempting individual fallback...")
        # Fallback: Try fetching each ticker individually (which has AV fallback)
        for ticker in tickers:
            try:
                # Reuse the single ingestion logic
                await ingest_data(ticker)
            except Exception as inner_e:
                logger.error(f"Fallback ingestion failed for {ticker}: {inner_e}")
    
    # 3. Run Predictions
    predictions = []
    
    # Check if model is loaded
    if state.lstm_model is None:
        logger.warning("LSTM model not initialized, skipping predictions")
        return {"message": "Ingestion complete, but model not initialized", "predictions": []}

    for ticker in tickers:
        try:
            historical_data = await get_historical_data(ticker, limit=100)
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
                
                if len(processed_df) >= 60:
                    # Get latest sequence of 60 steps
                    seq_data = processed_df.iloc[-60:][feature_cols].values
                    
                    # Scale
                    if state.scaler:
                        scaled_seq = state.scaler.transform(seq_data)
                        
                        # Transpose to [1, 37, 60] (Batch, Channels, SeqLen)
                        # CORRECTION: Model expects (Batch, SeqLen, Features) -> (1, 60, 37)
                        input_tensor = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0)
                        
                        logger.info(f"Predicting for {ticker} with input shape {input_tensor.shape}")
                        
                        # Use the global model from state
                        prediction = state.lstm_model.predict(input_tensor, state.device)
                        
                        # Save prediction
                        await save_prediction(ticker, float(prediction), datetime.now())
                        predictions.append({"ticker": ticker, "prediction": float(prediction)})
        except Exception as e:
            logger.error(f"Error predicting for {ticker}: {e}")
            
    return {"message": "Ingestion and prediction complete", "predictions": predictions}

@router.post("/ingest/{ticker}")
async def ingest_data(ticker: str):
    try:
        client = AlpacaDataClient()
        await client.fetch_data(ticker)
        return {"message": f"Ingestion started for {ticker}"}
    except Exception as e:
        logger.warning(f"Alpaca ingestion failed for {ticker}: {e}. Attempting Alpha Vantage...")
        try:
            av_client = AlphaVantageClient()
            await av_client.fetch_data(ticker)
            return {"message": f"Ingestion started for {ticker} (via Alpha Vantage)"}
        except Exception as av_e:
            logger.error(f"Ingestion failed for {ticker} (Both providers): {av_e}")
            raise HTTPException(status_code=400, detail=f"Alpaca failed: {e}. Alpha Vantage failed: {av_e}")
