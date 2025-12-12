#!/usr/bin/env python3
"""
V7 Model Audit Script - Detect "Lazy Prediction" Pattern

This script mathematically proves if V6/V7 are just predicting "yesterday's price + noise"
rather than learning actual patterns.

The Diagnostic:
- If correlation between (Prediction_t) and (Actual_t-1) is > 0.95, the model is "lazy"
- A truly predictive model should have lower correlation with yesterday and higher
  correlation with actual future prices

Usage:
    docker exec proxmox_stock_backend python -m scripts.audit_v7
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import torch

backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from services.feature_engineering import process_stock_data, get_model_input_features_v7
from sklearn.preprocessing import StandardScaler
import pickle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Test tickers - mix of different behaviors
TEST_TICKERS = ["SPY", "AAPL", "MSFT", "GOOGL", "TSLA", "AMD", "NVDA", "META"]


def get_db_url():
    url = os.getenv("DATABASE_URL", "")
    return url.replace("postgresql+asyncpg://", "postgresql://")


def load_stock_data(ticker: str):
    """Load OHLCV data for a single stock."""
    import psycopg2
    conn = psycopg2.connect(get_db_url())
    cur = conn.cursor()
    cur.execute("""
        SELECT timestamp, open, high, low, close, volume 
        FROM stock_prices WHERE ticker = %s ORDER BY timestamp
    """, (ticker,))
    records = cur.fetchall()
    cur.close()
    conn.close()
    
    if records:
        return pd.DataFrame(records, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    return None


def load_v7_model():
    """Load the trained V7 model."""
    model_path = backend_path / "models" / "lstm_model_v7.pth"
    scaler_path = backend_path / "models" / "scaler_v7.pkl"
    
    if not model_path.exists():
        logger.error(f"V7 model not found at {model_path}")
        return None, None
        
    from services.lstm_model_v7 import LSTMModelV7
    model = LSTMModelV7.load(str(model_path))
    model.eval()
    
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
            scaler = scaler_data['scaler']
    else:
        logger.warning("Scaler not found, using fresh StandardScaler")
        scaler = StandardScaler()
        
    return model, scaler


def run_lazy_prediction_audit():
    """
    The "Lazy Prediction" Audit
    
    If a model just learns to predict "yesterday's price + small mean",
    then corr(Prediction_t, Actual_t-1) will be very high (>0.95).
    
    A good model should:
    - Have moderate correlation with yesterday (< 0.8)
    - Have higher correlation with actual future price
    - Show diverse predictions, not just echoing the past
    """
    logger.info("=" * 60)
    logger.info("🔍 V7 LAZY PREDICTION AUDIT")
    logger.info("=" * 60)
    
    model, scaler = load_v7_model()
    if model is None:
        return
    
    feature_cols = get_model_input_features_v7()
    window_size = 60
    device = model.device
    
    results = []
    
    for ticker in TEST_TICKERS:
        logger.info(f"\nAnalyzing {ticker}...")
        
        df = load_stock_data(ticker)
        if df is None or len(df) < 500:
            logger.warning(f"  Skipping {ticker} - insufficient data")
            continue
            
        # Process data
        try:
            processed = process_stock_data(df.copy(), create_targets=True)
        except Exception as e:
            logger.warning(f"  Skipping {ticker} - processing error: {e}")
            continue
            
        for col in feature_cols:
            if col not in processed.columns:
                processed[col] = 0.0
        
        # Only use last 500 days for analysis
        processed = processed.tail(500).copy()
        
        if len(processed) < window_size + 10:
            continue
            
        # Generate predictions
        features = processed[feature_cols].values
        closes = processed['close'].values
        
        # Scale features
        features_flat = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        features_scaled = scaler.transform(features_flat)
        
        predictions_1d = []
        actuals_1d = []
        yesterdays = []
        
        for i in range(window_size, len(features_scaled) - 1):  # -1 for actual future
            window = features_scaled[i - window_size:i]
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred = model(x)
                pred_1d = pred[0, 0].item()  # 1-day prediction (log return)
            
            # Convert log return prediction to price
            yesterday_close = closes[i - 1]
            predicted_price = yesterday_close * np.exp(pred_1d)
            
            actual_price = closes[i]
            
            predictions_1d.append(predicted_price)
            actuals_1d.append(actual_price)
            yesterdays.append(yesterday_close)
        
        predictions_1d = np.array(predictions_1d)
        actuals_1d = np.array(actuals_1d)
        yesterdays = np.array(yesterdays)
        
        # === THE KEY METRICS ===
        
        # 1. Correlation: Prediction vs Yesterday (THE LAZY SCORE)
        corr_pred_yesterday, _ = pearsonr(predictions_1d, yesterdays)
        
        # 2. Correlation: Prediction vs Actual
        corr_pred_actual, _ = pearsonr(predictions_1d, actuals_1d)
        
        # 3. Correlation: Yesterday vs Actual (baseline - how autocorrelated is the stock?)
        corr_yesterday_actual, _ = pearsonr(yesterdays, actuals_1d)
        
        # 4. Mean Absolute Prediction Change (is it predicting the same thing?)
        pred_changes = np.abs(np.diff(predictions_1d))
        mean_pred_change = np.mean(pred_changes)
        actual_changes = np.abs(np.diff(actuals_1d))
        mean_actual_change = np.mean(actual_changes)
        change_ratio = mean_pred_change / (mean_actual_change + 1e-9)
        
        # 5. Prediction std vs Actual std
        pred_std = np.std(predictions_1d)
        actual_std = np.std(actuals_1d)
        std_ratio = pred_std / (actual_std + 1e-9)
        
        results.append({
            "ticker": ticker,
            "corr_pred_yesterday": corr_pred_yesterday,
            "corr_pred_actual": corr_pred_actual,
            "corr_yesterday_actual": corr_yesterday_actual,
            "change_ratio": change_ratio,
            "std_ratio": std_ratio,
            "is_lazy": corr_pred_yesterday > 0.95
        })
        
        logger.info(f"  Corr(Prediction, Yesterday): {corr_pred_yesterday:.4f}")
        logger.info(f"  Corr(Prediction, Actual):    {corr_pred_actual:.4f}")
        logger.info(f"  Corr(Yesterday, Actual):     {corr_yesterday_actual:.4f}")
        logger.info(f"  Change Ratio (pred/actual):  {change_ratio:.4f}")
        
        if corr_pred_yesterday > 0.95:
            logger.warning(f"  ⚠️  LAZY: Model is just echoing yesterday's price!")
        elif corr_pred_yesterday > 0.90:
            logger.warning(f"  🟡 BORDERLINE: Model shows high correlation with yesterday")
        else:
            logger.info(f"  ✅ Model shows independent predictions")
    
    # === SUMMARY ===
    logger.info("\n" + "=" * 60)
    logger.info("📊 AUDIT SUMMARY")
    logger.info("=" * 60)
    
    if not results:
        logger.error("No results to analyze")
        return
        
    df_results = pd.DataFrame(results)
    
    avg_lazy_score = df_results['corr_pred_yesterday'].mean()
    lazy_count = df_results['is_lazy'].sum()
    
    logger.info(f"\nAverage Lazy Score (Corr Pred-Yesterday): {avg_lazy_score:.4f}")
    logger.info(f"Lazy Stocks ({'>'}0.95): {lazy_count}/{len(df_results)}")
    
    logger.info("\nInterpretation:")
    if avg_lazy_score > 0.95:
        logger.error("🚨 MODEL IS LAZY: Just predicting yesterday's price with tiny changes")
        logger.error("   This is exactly what the video warned about!")
        logger.error("   The sliding-window approach has caused memorization.")
    elif avg_lazy_score > 0.90:
        logger.warning("🟡 MODEL IS BORDERLINE LAZY: High correlation with yesterday")
        logger.warning("   Consider switching to V8 classification approach.")
    else:
        logger.info("✅ Model shows some independent predictive power")
        logger.info("   It's not just echoing yesterday's price.")
    
    # Print table
    print("\n" + "=" * 80)
    print(f"{'Ticker':<8} {'Pred-Yes':<12} {'Pred-Act':<12} {'Yes-Act':<12} {'Lazy?':<8}")
    print("=" * 80)
    for r in results:
        lazy_mark = "⚠️ YES" if r['is_lazy'] else "✅ NO"
        print(f"{r['ticker']:<8} {r['corr_pred_yesterday']:<12.4f} {r['corr_pred_actual']:<12.4f} {r['corr_yesterday_actual']:<12.4f} {lazy_mark:<8}")
    print("=" * 80)


def run_overlapping_window_audit():
    """
    Check if training data has overlapping windows.
    
    With stride=1, consecutive samples share 59/60 days of data.
    This means the model can memorize chart shapes instead of learning patterns.
    """
    logger.info("\n" + "=" * 60)
    logger.info("🔍 OVERLAPPING WINDOW AUDIT (V7 Training)")
    logger.info("=" * 60)
    
    # Check V7 training script for stride
    v7_script = backend_path / "scripts" / "train_model_v7.py"
    
    if v7_script.exists():
        with open(v7_script, 'r') as f:
            content = f.read()
            
        if "stride" in content.lower():
            logger.info("V7 training script contains 'stride' logic")
        else:
            logger.warning("⚠️ V7 training script does NOT mention 'stride'")
            logger.warning("   This means it's using stride=1 (sliding window)")
            logger.warning("   Consecutive samples share 98% of data → MEMORIZATION RISK")
            
        # Check for the sliding window pattern
        if "range(window_size, len(" in content or "range(self.window_size, len(" in content:
            if ", stride)" not in content and ", 126)" not in content:
                logger.error("🚨 V7 uses sliding window with stride=1!")
                logger.error("   This is the 'memorization bug' the video warned about.")
    else:
        logger.warning(f"V7 training script not found at {v7_script}")


if __name__ == "__main__":
    run_lazy_prediction_audit()
    run_overlapping_window_audit()
