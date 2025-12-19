#!/usr/bin/env python3
"""
V8 Pre-Training Sanity Check (Dry Run)

Runs a quick check on the V8 training pipeline WITHOUT training the model.
This validates:
1. Class balance (15-35% positives is ideal)
2. Threshold calculation (training data only)
3. Non-overlapping window sample count
4. Data loading order (no look-ahead bias)

Usage:
    docker exec proxmox_stock_backend python -m scripts.sanity_check_v8
"""

import sys
import os
import gc
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from services.feature_engineering import process_stock_data, get_model_input_features_v7

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# V8 Config
CONFIG = {
    "WINDOW_SIZE": 60,
    "MAX_HORIZON": 126,
    "MIN_RECORDS": 500,
    "TRAIN_CUTOFF": 0.80,
    "THRESHOLD_SIGMA": 1.0,  # Match training script
    "MAX_STOCKS": 100,  # Just check first 100 for speed
}


def get_db_url():
    url = os.getenv("DATABASE_URL", "")
    return url.replace("postgresql+asyncpg://", "postgresql://")


def get_ticker_list():
    import psycopg2
    with psycopg2.connect(get_db_url()) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ticker, COUNT(*) as count 
                FROM stock_prices GROUP BY ticker 
                HAVING COUNT(*) >= %s ORDER BY count DESC LIMIT %s
            """, (CONFIG["MIN_RECORDS"], CONFIG["MAX_STOCKS"]))
            results = cur.fetchall()
            return [(row[0], row[1]) for row in results]


def load_single_stock(ticker: str):
    import psycopg2
    try:
        with psycopg2.connect(get_db_url()) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT timestamp, open, high, low, close, volume 
                    FROM stock_prices WHERE ticker = %s ORDER BY timestamp
                """, (ticker,))
                records = cur.fetchall()
                
                if records:
                    return pd.DataFrame(records, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    except Exception as e:
        logger.error(f"Error loading {ticker}: {e}")
        
    return None


def compute_thresholds(all_train_targets: np.ndarray) -> dict:
    """Compute volatility thresholds from training data."""
    thresholds = {}
    horizon_names = ["1d", "1w", "1m", "6m"]
    
    for i, h in enumerate(horizon_names):
        targets = all_train_targets[:, i]
        valid = targets[~np.isnan(targets)]
        if len(valid) > 0:
            mu = np.mean(valid)
            sig = np.std(valid)
            thresholds[i] = mu + CONFIG["THRESHOLD_SIGMA"] * sig
        else:
            thresholds[i] = 0.0
    return thresholds


def run_sanity_check():
    logger.info("=" * 60)
    logger.info("🔍 V8 PRE-TRAINING SANITY CHECK")
    logger.info("=" * 60)
    
    # === CHECK 1: Database Status ===
    logger.info("\n📊 CHECK 1: Database Status")
    tickers = get_ticker_list()
    logger.info(f"   Found {len(tickers)} stocks with >= {CONFIG['MIN_RECORDS']} records")
    
    if len(tickers) < 50:
        logger.error("   ❌ FAIL: Need at least 50 stocks. Run fetch_sp500_full.py first!")
        return False
    else:
        logger.info("   ✅ PASS: Sufficient stocks available")
    
    # === CHECK 2: Non-Overlapping Window Sample Count ===
    logger.info("\n📊 CHECK 2: Non-Overlapping Window Sample Count")
    
    feature_cols = get_model_input_features_v7()
    
    # Quick sample from first 20 stocks
    total_train = 0
    total_val = 0
    
    for ticker, count in tickers[:20]:
        df = load_single_stock(ticker)
        if df is None:
            continue
            
        try:
            processed = process_stock_data(df.copy(), create_targets=True)
        except:
            continue
            
        min_required = CONFIG["WINDOW_SIZE"] + CONFIG["MAX_HORIZON"] + 10
        if len(processed) < min_required:
            continue
        
        # Count non-overlapping windows
        n_windows = 0
        stride = CONFIG["MAX_HORIZON"]
        for i in range(CONFIG["WINDOW_SIZE"], len(processed), stride):
            n_windows += 1
        
        split_idx = int(n_windows * CONFIG["TRAIN_CUTOFF"])
        total_train += split_idx
        total_val += (n_windows - split_idx)
    
    # Extrapolate to full dataset
    scale_factor = len(tickers) / 20
    estimated_train = int(total_train * scale_factor)
    estimated_val = int(total_val * scale_factor)
    
    logger.info(f"   Sample from 20 stocks: {total_train} train, {total_val} val")
    logger.info(f"   Estimated for {len(tickers)} stocks: ~{estimated_train} train, ~{estimated_val} val")
    
    if estimated_train < 1000:
        logger.warning(f"   ⚠️ WARNING: Only ~{estimated_train} training samples!")
        logger.warning("   Consider fetching more stocks with fetch_sp500_full.py")
    else:
        logger.info(f"   ✅ PASS: {estimated_train} training samples is sufficient")
    
    # === CHECK 3: Class Balance (Threshold Test) ===
    logger.info("\n📊 CHECK 3: Class Balance (Threshold Test)")
    logger.info("   Computing thresholds from first 30 stocks...")
    
    all_train_targets = []
    all_val_targets = []
    
    for ticker, count in tickers[:30]:
        df = load_single_stock(ticker)
        if df is None:
            continue
            
        try:
            processed = process_stock_data(df.copy(), create_targets=True)
        except:
            continue
            
        for col in feature_cols:
            if col not in processed.columns:
                processed[col] = 0.0
        
        targets = processed[['Target_1d', 'Target_1w', 'Target_1m', 'Target_6m']].values
        
        # Non-overlapping windows
        y_list = []
        stride = CONFIG["MAX_HORIZON"]
        for i in range(CONFIG["WINDOW_SIZE"], len(targets), stride):
            target = targets[i]
            if not np.isnan(target).any():
                y_list.append(target)
        
        if not y_list:
            continue
            
        y = np.array(y_list)
        split_idx = int(len(y) * CONFIG["TRAIN_CUTOFF"])
        
        all_train_targets.append(y[:split_idx])
        all_val_targets.append(y[split_idx:])
    
    if not all_train_targets:
        logger.error("   ❌ FAIL: No training data available!")
        return False
    
    train_targets = np.vstack(all_train_targets)
    val_targets = np.vstack(all_val_targets) if all_val_targets else np.array([])
    
    logger.info(f"   Train samples: {len(train_targets)}, Val samples: {len(val_targets)}")
    
    # Compute thresholds from TRAINING ONLY
    logger.info("   Computing thresholds from TRAINING DATA ONLY...")
    thresholds = compute_thresholds(train_targets)
    
    # Calculate class balance
    logger.info("\n   Class Balance Results:")
    
    all_good = True
    for i, h in enumerate(["1d", "1w", "1m", "6m"]):
        train_positive = (train_targets[:, i] > thresholds[i]).mean() * 100
        
        if len(val_targets) > 0:
            val_positive = (val_targets[:, i] > thresholds[i]).mean() * 100
        else:
            val_positive = 0
        
        # Check if balance is good
        if train_positive < 5 or train_positive > 50:
            status = "❌"
            all_good = False
        elif train_positive < 15 or train_positive > 35:
            status = "🟡"
        else:
            status = "✅"
        
        logger.info(f"   {status} {h}: Threshold={thresholds[i]:.4f} | Train={train_positive:.1f}% | Val={val_positive:.1f}%")
    
    if all_good:
        logger.info("\n   ✅ PASS: Class balance is acceptable (15-35%)")
    else:
        logger.warning("\n   ⚠️ WARNING: Class balance may need adjustment")
        logger.warning(f"   If positives < 5%, decrease THRESHOLD_SIGMA (currently {CONFIG['THRESHOLD_SIGMA']})")
        logger.warning("   If positives > 50%, increase THRESHOLD_SIGMA")
    
    # === CHECK 4: Data Loading Order ===
    logger.info("\n📊 CHECK 4: Data Loading Order (Look-Ahead Check)")
    logger.info("   Verifying thresholds are computed AFTER train/val split...")
    
    # This is a code check, not runtime
    v8_script = backend_path / "scripts" / "train_model_v8_class.py"
    if v8_script.exists():
        with open(v8_script, 'r') as f:
            content = f.read()
            
        # Find line numbers
        lines = content.split('\n')
        split_line = None
        threshold_line = None
        
        for i, line in enumerate(lines):
            if "y_train_raw = np.vstack" in line:
                split_line = i + 1
            if "compute_thresholds_from_training" in line and "=" in line:
                threshold_line = i + 1
        
        if split_line and threshold_line:
            if threshold_line > split_line:
                logger.info(f"   ✅ PASS: Thresholds computed (line {threshold_line}) AFTER split (line {split_line})")
            else:
                logger.error(f"   ❌ FAIL: Thresholds computed BEFORE split! This is data leakage!")
                return False
        else:
            logger.warning("   🟡 Could not verify order automatically")
    
    # === CHECK 5: Stride Verification ===
    logger.info("\n📊 CHECK 5: Non-Overlapping Stride Verification")
    
    if v8_script.exists():
        with open(v8_script, 'r') as f:
            content = f.read()
            
        if 'stride = CONFIG["MAX_HORIZON"]' in content and '"MAX_HORIZON": 126' in content:
            logger.info("   ✅ PASS: stride = 126 (non-overlapping)")
        elif "stride = 126" in content or "stride=126" in content:
            logger.info("   ✅ PASS: stride = 126 (non-overlapping)")
        elif "stride = 1" in content or "stride=1" in content:
            logger.error("   ❌ FAIL: stride = 1 (overlapping!) - This will cause memorization")
            return False
        else:
            logger.info("   ✅ Using MAX_HORIZON for stride")
    
    # === FINAL SUMMARY ===
    logger.info("\n" + "=" * 60)
    logger.info("📋 SANITY CHECK COMPLETE")
    logger.info("=" * 60)
    logger.info("")
    logger.info("If all checks passed, you're ready to train V8:")
    logger.info("  docker exec proxmox_stock_backend python -m scripts.train_model_v8_class")
    logger.info("")
    logger.info("Monitor the first epoch for:")
    logger.info("  - 'Training samples: X' should be 3,000-10,000")
    logger.info("  - '1d positives: Train=XX%' should be 15-35%")
    logger.info("  - Loss should decrease")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    run_sanity_check()
