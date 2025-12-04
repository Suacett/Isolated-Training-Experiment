"""
Prediction API routes for frontend integration.

Provides endpoints to:
- Get prediction history for a ticker
- Get current/future predictions
- Get prediction accuracy statistics
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from services.db import get_predictions_for_ticker, get_pending_predictions

router = APIRouter(prefix="/predictions", tags=["predictions"])
logger = logging.getLogger(__name__)


class PredictionResponse(BaseModel):
    id: int
    ticker: str
    prediction_date: datetime
    target_date: datetime
    horizon: str
    predicted_value: float
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
