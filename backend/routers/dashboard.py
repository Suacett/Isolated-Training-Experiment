import logging
import numpy as np
from fastapi import APIRouter, HTTPException
from services.db import get_historical_data, get_predictions_for_ticker, get_latest_close, get_watchlist_item
from services.intrinsic import IntrinsicCalculator
from services.risk_management import get_regime_exposure
from datetime import datetime
import pandas as pd
from services.feature_engineering_v9 import (
    compute_v9_features,
    process_spy_data,
    process_vix_data,
    get_v9_feature_names,
)
from state import state

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/market-status")
async def get_market_status():
    """
    Get current market regime status (Bull, Bear, Crisis) based on SPY trend and VIX.
    """
    try:
        # Fetch SPY and VIX data
        spy_data = await get_historical_data("SPY", limit=300)
        vix_data = await get_historical_data("^VIX", limit=50)
        
        # fallback if ^VIX not found (Yahoo often uses ^VIX)
        if not vix_data:
             vix_data = await get_historical_data("VIX", limit=50)

        current_date = pd.Timestamp.now()
        
        # Prepare DataFrames
        spy_df = pd.DataFrame([{
            "date": d.timestamp,
            "close": d.close
        } for d in spy_data]) if spy_data else pd.DataFrame()
        
        vix_df = pd.DataFrame([{
            "date": d.timestamp,
            "close": d.close
        } for d in vix_data]) if vix_data else None
        
        if spy_df.empty:
            return {
                "regime": "Unknown",
                "exposure": 1.0,
                "vix": None,
                "details": "Insufficient SPY data"
            }
            
        exposure, regime_label = get_regime_exposure(spy_df, vix_df, current_date)
        
        current_vix = vix_df['close'].iloc[-1] if vix_df is not None and not vix_df.empty else None
        
        return {
            "regime": regime_label,
            "exposure": exposure,
            "vix": current_vix,
            "timestamp": current_date.isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching market status: {e}")
        return {
            "regime": "Error",
            "exposure": 1.0, 
            "error": str(e)
        }


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
            # Convert log return to predicted price: predicted_price = current_price * np.exp(pred.predicted_value)
            predicted_price = pred.current_price * np.exp(pred.predicted_value)
            prediction_lookup[target_date_str] = predicted_price

        # Check V9
        is_v9_model = state.lstm_model is not None and type(state.lstm_model).__name__ == "TransformerRankModel"
        current_rank = None
        
        if is_v9_model:
            # For V9, we don't predict prices for the graph, we just get the current rank
            # Fetch macro data for feature engineering
            try:
                 # Fetch macro data
                spy_hist = await get_historical_data("SPY", limit=500)
                vix_hist = await get_historical_data("^VIX", limit=500)
                if not vix_hist:
                     vix_hist = await get_historical_data("VIX", limit=500)
                
                spy_data_v9 = None
                vix_data_v9 = None

                if spy_hist:
                    spy_df = pd.DataFrame([{
                        "date": d.timestamp, "open": d.open, "high": d.high, "low": d.low, "close": d.close, "volume": d.volume
                    } for d in spy_hist])
                    spy_data_v9 = process_spy_data(spy_df)
                    
                if vix_hist:
                    vix_df = pd.DataFrame([{
                        "date": d.timestamp, "open": d.open, "high": d.high, "low": d.low, "close": d.close, "volume": d.volume
                    } for d in vix_hist])
                    vix_data_v9 = process_vix_data(vix_df)
                    
                # Process features for current ticker
                df = pd.DataFrame([{
                    "date": d.timestamp, "open": d.open, "high": d.high, "low": d.low, "close": d.close, "volume": d.volume
                } for d in historical_data])
                
                processed_df = compute_v9_features(df, spy_data_v9, vix_data_v9)
                v9_min_len = 60
                
                if len(processed_df) >= v9_min_len:
                     v9_features = get_v9_feature_names()
                     # Get last window
                     window_seq = processed_df.iloc[-v9_min_len:][v9_features].values
                     
                     if state.scaler:
                         window_seq = state.scaler.transform(window_seq)
                         
                     input_tensor = torch.tensor(window_seq, dtype=torch.float32).unsqueeze(0)
                     rank_score = state.lstm_model.predict(input_tensor).item()
                     current_rank = rank_score * 100
                     
            except Exception as e:
                logger.error(f"V9 Rank calc failed in detail view: {e}")

        # Calculate intrinsic value (once per ticker, only if Alpha Vantage is enabled AND ticker is favorite)
        # Note: This may be slow as it fetches EPS data from Alpha Vantage
        # Returns None if calculation fails to avoid destroying graph scale
        intrinsic_value = None
        intrinsic_breakdown = None
        if state.alpha_vantage_enabled:
            # Only calculate for favorite tickers to conserve API limits
            watchlist_item = await get_watchlist_item(ticker)
            if watchlist_item and watchlist_item.is_favorite:
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
                logger.debug(f"Skipping intrinsic value for {ticker} (not marked as favorite)")
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

        return {"history": history, "intrinsic_breakdown": intrinsic_breakdown, "rank": current_rank}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching dashboard data for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
