import logging
import numpy as np
from fastapi import APIRouter, HTTPException
from services.db import get_historical_data, get_predictions_for_ticker
from services.intrinsic import IntrinsicCalculator
from datetime import datetime
import pandas as pd
from state import state

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/dashboard/{ticker}")
async def get_dashboard_data(ticker: str):
    """
    Get historical price data with predictions and intrinsic values for the dashboard chart.

    Returns:
        {
            "history": [
                {
                    "date": "2024-01-01",
                    "open": 150.0,
                    "high": 152.0,
                    "low": 149.0,
                    "close": 151.0,
                    "predicted_close": 152.5,  # null if no prediction
                    "intrinsic_value": 145.0
                }
            ]
        }
    """
    try:
        logger.info(f"Fetching dashboard data for {ticker}")

        # Fetch historical price data (up to 20000 days for MAX view ~80 years)
        historical_data = await get_historical_data(ticker, limit=20000)

        if not historical_data:
            logger.warning(f"No historical data found for {ticker}")
            raise HTTPException(status_code=404, detail=f"No historical data for {ticker}")

        logger.info(f"Found {len(historical_data)} historical data points for {ticker}")

        # Fetch all predictions for this ticker (1-day horizon only)
        predictions = await get_predictions_for_ticker(ticker, horizon="1d", limit=20000)
        logger.info(f"Found {len(predictions)} predictions for {ticker}")

        # Create a prediction lookup dict by target_date
        # Store both log return and current price for conversion
        prediction_lookup = {}
        for pred in predictions:
            # Convert target_date to date string (YYYY-MM-DD)
            target_date_str = pred.target_date.date().isoformat()
            # Convert log return to predicted price: predicted_price = current_price * exp(log_return)
            predicted_price = pred.current_price * np.exp(pred.predicted_value)
            prediction_lookup[target_date_str] = predicted_price

        # Calculate intrinsic value (once per ticker, only if Alpha Vantage is enabled)
        # Note: This may be slow as it fetches EPS data from Alpha Vantage
        # Returns None if calculation fails to avoid destroying graph scale
        intrinsic_value = None
        intrinsic_breakdown = None
        if state.alpha_vantage_enabled:
            try:
                intrinsic_calc = IntrinsicCalculator()
                intrinsic_value = await intrinsic_calc.calculate(ticker, use_live_bond_yield=True)
                
                # Get breakdown for tooltip
                intrinsic_breakdown = {
                    "eps": intrinsic_calc.last_eps,
                    "growth_rate": intrinsic_calc.last_growth_rate,
                    "bond_yield": intrinsic_calc.last_bond_yield,
                    "intrinsic_value": intrinsic_value,
                    "is_estimated": intrinsic_calc.is_estimated
                }

                if intrinsic_value is None:
                    logger.warning(f"Intrinsic value calculation returned None for {ticker} - will not plot on chart")
                elif intrinsic_value <= 0:
                    logger.warning(f"Intrinsic value calculation returned invalid value for {ticker}: {intrinsic_value} - setting to None")
                    intrinsic_value = None
            except Exception as e:
                logger.warning(f"Failed to calculate intrinsic value for {ticker}: {e} - setting to None")
                intrinsic_value = None
        else:
            logger.debug(f"Skipping intrinsic value calculation for {ticker} (Alpha Vantage disabled)")

        # Fetch SPY data for comparison (if ticker is not SPY itself)
        spy_lookup = {}
        if ticker.upper() != "SPY":
            try:
                spy_data = await get_historical_data("SPY", limit=5000)
                if spy_data:
                    for sp in spy_data:
                        date_str = sp.timestamp.date().isoformat()
                        spy_lookup[date_str] = float(sp.close)
                    logger.info(f"Loaded {len(spy_lookup)} SPY data points for comparison")
            except Exception as e:
                logger.warning(f"Failed to fetch SPY data for comparison: {e}")

        # Build response
        history = []
        for data_point in historical_data:
            date_str = data_point.timestamp.date().isoformat()

            # Check if there's a prediction for this date
            predicted_close = prediction_lookup.get(date_str)
            
            # Get SPY close for this date
            spy_close = spy_lookup.get(date_str)

            history.append({
                "date": date_str,
                "open": float(data_point.open),
                "high": float(data_point.high),
                "low": float(data_point.low),
                "close": float(data_point.close),
                "predicted_close": float(predicted_close) if predicted_close is not None else None,
                "intrinsic_value": float(intrinsic_value) if intrinsic_value is not None else None,
                "spy_close": spy_close
            })

        return {"history": history, "intrinsic_breakdown": intrinsic_breakdown}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching dashboard data for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
