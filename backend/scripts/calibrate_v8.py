#!/usr/bin/env python3
"""
V8 Buy Threshold Calibration Script

The V8 model outputs probabilities (0.0 to 1.0), but we need to find the optimal
threshold for converting these to Buy/Hold signals.

Problem: At threshold 0.5, precision may be 0% (no trades).
Solution: Grid search thresholds to find the best Precision/Recall tradeoff.

Usage:
    docker exec proxmox_stock_backend python -m scripts.calibrate_v8
    docker exec proxmox_stock_backend python -m scripts.calibrate_v8 --horizon 1w
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import torch
import pickle

backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from services.lstm_model_v8_class import LSTMModelV8Class
from services.feature_engineering import process_stock_data, get_model_input_features_v7

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Config - must match training
CONFIG = {
    "WINDOW_SIZE": 60,
    "MAX_HORIZON": 126,
    "MIN_RECORDS": 500,
    "TRAIN_CUTOFF": 0.80,
    "THRESHOLD_SIGMA": 1.0,
    "MAX_STOCKS": 200,  # Use first 200 for speed
}

MODEL_PATH = backend_path / "models" / "lstm_model_v8_class.pth"
SCALER_PATH = backend_path / "models" / "scaler_v8_class.pkl"
THRESHOLDS_PATH = backend_path / "models" / "thresholds_v8_class.pkl"

HORIZONS = ["1d", "1w", "1m", "6m"]


def get_db_url():
    url = os.getenv("DATABASE_URL", "")
    return url.replace("postgresql+asyncpg://", "postgresql://")


def get_ticker_list():
    """Get tickers with sufficient history."""
    import psycopg2
    conn = psycopg2.connect(get_db_url())
    cur = conn.cursor()
    cur.execute("""
        SELECT ticker, COUNT(*) as count 
        FROM stock_prices GROUP BY ticker 
        HAVING COUNT(*) >= %s ORDER BY count DESC LIMIT %s
    """, (CONFIG["MIN_RECORDS"], CONFIG["MAX_STOCKS"]))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return [(row[0], row[1]) for row in results]


def load_single_stock(ticker: str):
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


def load_validation_data(feature_cols: list, scaler: StandardScaler, class_thresholds: dict):
    """
    Load validation data using EXACT same non-overlapping window logic as training.
    Returns (X_val, y_val_binary, y_val_raw)
    """
    tickers = get_ticker_list()
    logger.info(f"Loading validation data from {len(tickers)} stocks...")
    
    all_X_val = []
    all_y_val_raw = []
    
    for i, (ticker, _) in enumerate(tickers):
        if (i + 1) % 50 == 0:
            logger.info(f"  Processing {i+1}/{len(tickers)}...")
        
        df = load_single_stock(ticker)
        if df is None:
            continue
        
        try:
            processed = process_stock_data(df.copy(), create_targets=True)
        except Exception as e:
            logger.exception(f"Error processing {ticker} (Target generation failed): {e}")
            continue
        
        min_required = CONFIG["WINDOW_SIZE"] + CONFIG["MAX_HORIZON"] + 10
        if len(processed) < min_required:
            continue
        
        for col in feature_cols:
            if col not in processed.columns:
                processed[col] = 0.0
        
        features = processed[feature_cols].values
        targets = processed[['Target_1d', 'Target_1w', 'Target_1m', 'Target_6m']].values
        
        # NON-OVERLAPPING WINDOWS (must match training)
        X_list, y_list = [], []
        stride = CONFIG["MAX_HORIZON"]
        
        for idx in range(CONFIG["WINDOW_SIZE"], len(features), stride):
            target = targets[idx]
            if np.isnan(target).any():
                continue
            X_list.append(features[idx - CONFIG["WINDOW_SIZE"]:idx])
            y_list.append(target)
        
        if not X_list:
            continue
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Only take VALIDATION portion (last 20%)
        split_idx = int(len(X) * CONFIG["TRAIN_CUTOFF"])
        X_val = X[split_idx:]
        y_val = y[split_idx:]
        
        if len(X_val) > 0:
            # Scale features
            N, T, F = X_val.shape
            X_flat = np.nan_to_num(X_val.reshape(-1, F), nan=0.0, posinf=1e6, neginf=-1e6)
            X_scaled = scaler.transform(X_flat).reshape(N, T, F)
            X_scaled = np.clip(X_scaled, -10.0, 10.0)
            
            all_X_val.append(X_scaled.astype(np.float32))
            all_y_val_raw.append(y_val.astype(np.float32))
    
    if not all_X_val or not all_y_val_raw:
        logger.warning("No validation data was collected. Skipping calibration steps.")
        return np.array([]), np.array([]), np.array([])
        
    X_val = np.vstack(all_X_val)
    y_val_raw = np.vstack(all_y_val_raw)
    
    # Convert to binary using training thresholds
    y_val_binary = np.zeros_like(y_val_raw)
    for i in range(4):
        y_val_binary[:, i] = (y_val_raw[:, i] > class_thresholds[i]).astype(float)
    
    return X_val, y_val_binary, y_val_raw


def run_calibration(horizon_idx: int = 0, horizon_name: str = "1d"):
    """
    Grid search thresholds to find optimal buy threshold.
    """
    logger.info("=" * 70)
    logger.info(f"🎯 V8 THRESHOLD CALIBRATION (Horizon: {horizon_name})")
    logger.info("=" * 70)
    
    # Load model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    logger.info(f"Loading V8 model from {MODEL_PATH}...")
    model = LSTMModelV8Class.load(str(MODEL_PATH))
    model.eval()
    device = model.device
    logger.info(f"Model loaded on {device}")
    
    # Load scaler
    if not SCALER_PATH.exists():
        logger.error(f"Scaler not found at {SCALER_PATH}")
        return
    
    with open(SCALER_PATH, 'rb') as f:
        scaler_data = pickle.load(f)
        scaler = scaler_data['scaler']
        feature_cols = scaler_data['feature_names']
    
    # Load thresholds
    if not THRESHOLDS_PATH.exists():
        logger.error(f"Thresholds not found at {THRESHOLDS_PATH}")
        return
    
    with open(THRESHOLDS_PATH, 'rb') as f:
        class_thresholds = pickle.load(f)
    
    logger.info(f"Loaded scaler with {len(feature_cols)} features")
    logger.info(f"Class thresholds: {class_thresholds}")
    
    # Load validation data
    X_val, y_val_binary, y_val_raw = load_validation_data(feature_cols, scaler, class_thresholds)
    
    # Handle empty validation data
    if len(X_val) == 0:
        logger.warning("No validation data available. Skipping calibration.")
        return
        
    logger.info(f"Validation samples: {len(X_val)}")
    
    # Get model probabilities
    logger.info("\nRunning inference on validation set...")
    all_probs = []
    
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            batch = torch.tensor(X_val[i:i+batch_size], dtype=torch.float32).to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    
    probs = np.vstack(all_probs)
    logger.info(f"Inference complete. Probability range: [{probs.min():.3f}, {probs.max():.3f}]")
    
    # Extract target horizon
    y_true = y_val_binary[:, horizon_idx]
    y_probs = probs[:, horizon_idx]
    
    logger.info(f"\n📊 Ground truth: {y_true.sum():.0f}/{len(y_true)} positives ({y_true.mean()*100:.1f}%)")
    
    # Grid search thresholds
    logger.info("\n🔍 Grid searching thresholds from 0.10 to 0.90...")
    
    results = []
    for threshold in np.arange(0.10, 0.91, 0.01):
        y_pred = (y_probs >= threshold).astype(int)
        
        trade_count = y_pred.sum()
        
        if trade_count == 0:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            # Precision = Win Rate (of trades taken)
            precision = precision_score(y_true, y_pred, zero_division=0)
            # Recall = Opportunity Capture (of all good trades, how many did we take?)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results.append({
            "threshold": threshold,
            "precision": precision * 100,
            "recall": recall * 100,
            "f1": f1 * 100,
            "trade_count": int(trade_count),
            "total_samples": len(y_true)
        })
    
    # Sort by precision (highest first)
    results_df = pd.DataFrame(results)
    results_df = results_df[results_df["trade_count"] > 0]  # Only show thresholds with trades
    results_df = results_df.sort_values("precision", ascending=False)
    
    # Print results
    logger.info("\n" + "=" * 80)
    logger.info(f"📋 THRESHOLD CALIBRATION RESULTS (Horizon: {horizon_name})")
    logger.info("=" * 80)
    print()
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Trades':<12}")
    print("-" * 60)
    
    for _, row in results_df.head(20).iterrows():
        print(f"{row['threshold']:<12.2f} {row['precision']:<12.1f}% {row['recall']:<12.1f}% {row['f1']:<12.1f}% {row['trade_count']:<12}")
    
    print("-" * 60)
    
    # Find best threshold (highest precision with at least 10 trades)
    usable = results_df[results_df["trade_count"] >= 10]
    if len(usable) > 0:
        best = usable.iloc[0]
        logger.info(f"\n🏆 RECOMMENDED THRESHOLD: {best['threshold']:.2f}")
        logger.info(f"   Precision (Win Rate): {best['precision']:.1f}%")
        logger.info(f"   Recall: {best['recall']:.1f}%")
        logger.info(f"   Trade Count: {best['trade_count']}")
    else:
        logger.warning("No threshold found with >= 10 trades. Model may need more training.")
    
    # Also show the "balanced" threshold (best F1)
    f1_sorted = results_df.sort_values("f1", ascending=False)
    if len(f1_sorted) > 0:
        best_f1 = f1_sorted.iloc[0]
        logger.info(f"\n⚖️ BEST BALANCED (F1): {best_f1['threshold']:.2f}")
        logger.info(f"   Precision: {best_f1['precision']:.1f}%, Recall: {best_f1['recall']:.1f}%, Trades: {best_f1['trade_count']}")
    
    logger.info("\n" + "=" * 80)
    
    return results_df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Calibrate V8 buy threshold")
    parser.add_argument("--horizon", type=str, default="1d", choices=["1d", "1w", "1m", "6m"],
                        help="Which prediction horizon to calibrate")
    args = parser.parse_args()
    
    horizon_idx = HORIZONS.index(args.horizon)
    run_calibration(horizon_idx=horizon_idx, horizon_name=args.horizon)


if __name__ == "__main__":
    main()
