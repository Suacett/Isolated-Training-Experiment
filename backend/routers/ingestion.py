import logging
import torch
import numpy as np
from datetime import datetime
from fastapi import APIRouter, HTTPException
from services.db import get_watchlist, get_historical_data, save_prediction
from services.data_ingest import AlpacaDataClient
from state import state

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
        logger.error(f"Data ingestion failed: {e}")
        return {"message": "Data ingestion failed", "error": str(e)}
    
    # 3. Run Predictions
    predictions = []
    
    # Check if model is loaded
    if state.lstm_model is None:
        logger.warning("LSTM model not initialized, skipping predictions")
        return {"message": "Ingestion complete, but model not initialized", "predictions": []}

    for ticker in tickers:
        try:
            historical_data = await get_historical_data(ticker, limit=60)
            if len(historical_data) == 60:
                data_array = np.array([[d.open, d.high, d.low, d.close, d.volume] for d in historical_data], dtype=np.float32)
                input_tensor = torch.tensor(data_array)
                
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
        logger.error(f"Ingestion failed for {ticker}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
