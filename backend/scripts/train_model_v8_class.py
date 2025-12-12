#!/usr/bin/env python3
"""
Training Script v8 - CLASSIFICATION MODEL (Regime Detection)

Key differences from V7:
- Non-overlapping windows: Stride = 126 days (max horizon) for truly independent samples
- Classification targets: Binary (1 if return > μ + 2σ, else 0)
- Loss: BCEWithLogitsLoss
- Thresholds computed ONLY on training data (no future peeking)
- Increased dropout (0.5) for sparse data
- Uses ALL available tickers to maximize sample count

CRITICAL FIX: Thresholds are computed on training data ONLY to prevent data leakage.

Usage:
    docker exec proxmox_stock_backend python -m scripts.train_model_v8_class
"""

import sys
import os
import gc
import logging
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
import pickle

backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from services.feature_engineering import process_stock_data, get_model_input_features_v7
from services.lstm_model_v8_class import LSTMModelV8Class

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_v8_class.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# === CONFIG ===
CONFIG = {
    "WINDOW_SIZE": int(os.getenv("WINDOW_SIZE", "60")),
    "MAX_HORIZON": 126,  # 6 trading months - stride for non-overlapping
    "BATCH_SIZE": int(os.getenv("BATCH_SIZE", "64")),  # Smaller due to less data
    "HIDDEN_DIM": int(os.getenv("HIDDEN_DIM", "128")),
    "NUM_LAYERS": int(os.getenv("NUM_LAYERS", "2")),
    "NUM_ATTENTION_HEADS": int(os.getenv("NUM_ATTENTION_HEADS", "4")),
    "DROPOUT": float(os.getenv("DROPOUT", "0.5")),  # INCREASED for sparse data
    "MAX_STOCKS": int(os.getenv("MAX_STOCKS", "9999")),  # Use ALL stocks
    "MIN_RECORDS": int(os.getenv("MIN_RECORDS", "500")),
    "EPOCHS": int(os.getenv("EPOCHS", "300")),
    "PATIENCE": int(os.getenv("PATIENCE", "50")),  # More patience for sparse data
    "LEARNING_RATE": float(os.getenv("LEARNING_RATE", "0.0001")),
    "TRAIN_CUTOFF": 0.80,
    "THRESHOLD_SIGMA": 2.0,  # Mean + 2σ for buy signals
}

MODEL_SAVE_PATH = backend_path / "models" / "lstm_model_v8_class.pth"
MODEL_SAVE_PATH_BEST_AUC = backend_path / "models" / "lstm_model_v8_class_auc.pth"
SCALER_SAVE_PATH = backend_path / "models" / "scaler_v8_class.pkl"
THRESHOLDS_SAVE_PATH = backend_path / "models" / "thresholds_v8_class.pkl"
TEMP_DATA_DIR = Path("/tmp/training_data_v8_class")

# Horizon mapping
HORIZONS = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
    "6m": 126,
}


def get_db_url():
    url = os.getenv("DATABASE_URL", "")
    return url.replace("postgresql+asyncpg://", "postgresql://")


def get_ticker_list():
    """Get ALL tickers with sufficient history."""
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


def compute_thresholds_from_training(all_train_targets: np.ndarray) -> dict:
    """
    Compute volatility thresholds from TRAINING DATA ONLY.
    
    CRITICAL: This prevents data leakage by not peeking at future/validation data.
    
    Returns:
        Dictionary with thresholds for each horizon: {0: threshold_1d, 1: threshold_1w, ...}
    """
    thresholds = {}
    horizon_names = ["1d", "1w", "1m", "6m"]
    
    for i, h in enumerate(horizon_names):
        targets = all_train_targets[:, i]
        # Remove NaN
        valid_targets = targets[~np.isnan(targets)]
        if len(valid_targets) > 0:
            mu = np.mean(valid_targets)
            sig = np.std(valid_targets)
            thresholds[i] = mu + CONFIG["THRESHOLD_SIGMA"] * sig
            logger.info(f"  Horizon {h}: μ={mu:.4f}, σ={sig:.4f}, threshold={thresholds[i]:.4f}")
        else:
            thresholds[i] = 0.0
            logger.warning(f"  Horizon {h}: No valid targets, using threshold=0")
    
    return thresholds


def create_classification_targets(targets: np.ndarray, thresholds: dict) -> np.ndarray:
    """
    Convert regression targets (log returns) to binary classification targets.
    
    Args:
        targets: Array of shape (N, 4) with log returns for each horizon
        thresholds: Dictionary mapping horizon index to threshold value
        
    Returns:
        Binary array of shape (N, 4) where 1 = buy signal, 0 = no signal
    """
    binary = np.zeros_like(targets)
    for i in range(targets.shape[1]):
        binary[:, i] = (targets[:, i] > thresholds[i]).astype(float)
    return binary


def process_single_stock_v8(
    ticker: str, 
    df: pd.DataFrame, 
    feature_cols: list, 
    scaler: StandardScaler,
    is_training: bool = True
):
    """
    Process a single stock with NON-OVERLAPPING windows.
    
    Key difference from V7: Stride = MAX_HORIZON (126 days)
    This ensures truly independent samples with no label overlap.
    """
    try:
        processed = process_stock_data(df.copy(), create_targets=True)
        
        # Need enough data for window + max horizon
        min_required = CONFIG["WINDOW_SIZE"] + CONFIG["MAX_HORIZON"] + 10
        if len(processed) < min_required:
            return None, None, None, None
        
        for col in feature_cols:
            if col not in processed.columns:
                processed[col] = 0.0
        
        features = processed[feature_cols].values
        targets = processed[['Target_1d', 'Target_1w', 'Target_1m', 'Target_6m']].values
        
        # === NON-OVERLAPPING WINDOWS ===
        # Stride = MAX_HORIZON to ensure no overlap between input OR output
        X_list, y_list = [], []
        stride = CONFIG["MAX_HORIZON"]
        
        for i in range(CONFIG["WINDOW_SIZE"], len(features), stride):
            target = targets[i]
            if np.isnan(target).any():
                continue
            X_list.append(features[i - CONFIG["WINDOW_SIZE"]:i])
            y_list.append(target)
        
        if not X_list:
            return None, None, None, None
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Time-based split (no shuffle!)
        split_idx = int(len(X) * CONFIG["TRAIN_CUTOFF"])
        
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]
        
        # Scale features
        N_train, T, F = X_train.shape
        X_train_flat = np.nan_to_num(X_train.reshape(-1, F), nan=0.0, posinf=1e6, neginf=-1e6)
        X_train_scaled = scaler.transform(X_train_flat).reshape(N_train, T, F)
        
        if len(X_val) > 0:
            N_val = len(X_val)
            X_val_flat = np.nan_to_num(X_val.reshape(-1, F), nan=0.0, posinf=1e6, neginf=-1e6)
            X_val_scaled = scaler.transform(X_val_flat).reshape(N_val, T, F)
        else:
            X_val_scaled, y_val = None, None
        
        return (
            X_train_scaled.astype(np.float32) if X_train_scaled is not None else None,
            y_train.astype(np.float32) if y_train is not None else None,
            X_val_scaled.astype(np.float32) if X_val_scaled is not None else None,
            y_val.astype(np.float32) if y_val is not None else None
        )
    except Exception as e:
        logger.warning(f"Failed to process {ticker}: {e}")
        return None, None, None, None


class ClassificationDataset(Dataset):
    """Dataset for V8 classification training."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ClassificationMetrics:
    """Track classification metrics during training."""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.all_preds = []
        self.all_targets = []
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        # Convert logits to probabilities
        probs = torch.sigmoid(preds)
        self.all_preds.append(probs.cpu().numpy())
        self.all_targets.append(targets.cpu().numpy())
        
    def get_metrics(self) -> dict:
        if not self.all_preds:
            return {"auc": 0.5, "precision": 0.0, "recall": 0.0, "positive_rate": 0.0}
        
        preds = np.vstack(self.all_preds)
        targets = np.vstack(self.all_targets)
        
        # Focus on 1-day horizon (column 0)
        pred_1d = preds[:, 0]
        target_1d = targets[:, 0]
        
        # AUC-ROC
        try:
            auc = roc_auc_score(target_1d, pred_1d)
        except:
            auc = 0.5
        
        # Precision/Recall at threshold 0.5
        pred_binary = (pred_1d > 0.5).astype(int)
        try:
            precision = precision_score(target_1d, pred_binary, zero_division=0)
            recall = recall_score(target_1d, pred_binary, zero_division=0)
        except:
            precision, recall = 0.0, 0.0
        
        # Class balance (how many positives)
        positive_rate = target_1d.mean() * 100
        
        return {
            "auc": auc,
            "precision": precision * 100,
            "recall": recall * 100,
            "positive_rate": positive_rate
        }
    
    def log_summary(self, prefix: str = "Val"):
        metrics = self.get_metrics()
        logger.info(f"{prefix} Metrics:")
        logger.info(f"  AUC: {metrics['auc']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.1f}%")
        logger.info(f"  Recall: {metrics['recall']:.1f}%")
        logger.info(f"  Positive Rate: {metrics['positive_rate']:.1f}%")


def train_pipeline():
    """Main training pipeline for V8 classification model."""
    
    # GPU Check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🚀 GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    else:
        logger.warning("⚠️ NO GPU!")
    
    logger.info("="*60)
    logger.info("Training V8 - CLASSIFICATION MODEL (Regime Detection)")
    logger.info(f"Config: {CONFIG}")
    logger.info("="*60)
    
    tickers = get_ticker_list()
    logger.info(f"Found {len(tickers)} stocks with >= {CONFIG['MIN_RECORDS']} records")
    
    feature_cols = get_model_input_features_v7()
    n_features = len(feature_cols)
    logger.info(f"Using {n_features} features (v7 feature set)")
    
    # === PHASE 1: Fit Scaler on Training Data ===
    logger.info("\n📊 Phase 1: Fitting scaler on training data only...")
    sample_data = []
    for ticker, _ in tickers[:100]:
        df = load_single_stock(ticker)
        if df is not None:
            processed = process_stock_data(df.copy(), create_targets=False)
            for col in feature_cols:
                if col not in processed.columns:
                    processed[col] = 0.0
            train_end = int(len(processed) * CONFIG["TRAIN_CUTOFF"])
            sample_data.append(processed[feature_cols].values[:train_end])
    
    scaler = StandardScaler()
    all_sample = np.vstack(sample_data)
    all_sample = np.nan_to_num(all_sample, nan=0.0, posinf=1e6, neginf=-1e6)
    scaler.fit(all_sample)
    logger.info(f"✅ Scaler fitted on {len(all_sample):,} training samples")
    
    del sample_data, all_sample
    gc.collect()
    
    # === PHASE 2: Process All Stocks with Non-Overlapping Windows ===
    logger.info("\n📊 Phase 2: Processing stocks with non-overlapping windows...")
    
    all_X_train, all_y_train = [], []
    all_X_val, all_y_val = [], []
    
    for i, (ticker, _) in enumerate(tickers):
        if (i + 1) % 100 == 0:
            logger.info(f"Processing stock {i+1}/{len(tickers)}...")
        
        df = load_single_stock(ticker)
        if df is None:
            continue
            
        X_train, y_train, X_val, y_val = process_single_stock_v8(
            ticker, df, feature_cols, scaler
        )
        
        if X_train is not None and len(X_train) > 0:
            all_X_train.append(X_train)
            all_y_train.append(y_train)
        
        if X_val is not None and len(X_val) > 0:
            all_X_val.append(X_val)
            all_y_val.append(y_val)
    
    X_train = np.vstack(all_X_train)
    y_train_raw = np.vstack(all_y_train)  # Raw log returns
    X_val = np.vstack(all_X_val)
    y_val_raw = np.vstack(all_y_val)  # Raw log returns
    
    logger.info(f"📉 Non-Overlapping Window Stats:")
    logger.info(f"   Training samples: {len(X_train):,}")
    logger.info(f"   Validation samples: {len(X_val):,}")
    
    # === PHASE 3: Compute Thresholds from TRAINING DATA ONLY ===
    logger.info("\n🎯 Phase 3: Computing thresholds from TRAINING DATA ONLY...")
    thresholds = compute_thresholds_from_training(y_train_raw)
    
    # Convert to binary classification targets
    y_train = create_classification_targets(y_train_raw, thresholds)
    y_val = create_classification_targets(y_val_raw, thresholds)
    
    # Log class balance
    for i, h in enumerate(["1d", "1w", "1m", "6m"]):
        train_pos = y_train[:, i].mean() * 100
        val_pos = y_val[:, i].mean() * 100
        logger.info(f"   {h} positives: Train={train_pos:.1f}%, Val={val_pos:.1f}%")
    
    # === PHASE 4: Train Classification Model ===
    logger.info("\n🚀 Phase 4: Training classification model...")
    
    train_dataset = ClassificationDataset(X_train, y_train)
    val_dataset = ClassificationDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["BATCH_SIZE"], 
        shuffle=True,  # Can shuffle since non-overlapping
        num_workers=0, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG["BATCH_SIZE"], 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )
    
    model = LSTMModelV8Class(
        input_dim=n_features,
        hidden_dim=CONFIG["HIDDEN_DIM"],
        num_layers=CONFIG["NUM_LAYERS"],
        output_dim=4,
        dropout=CONFIG["DROPOUT"],
        window_size=CONFIG["WINDOW_SIZE"],
        num_attention_heads=CONFIG["NUM_ATTENTION_HEADS"]
    )
    
    model.enable_checkpointing()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    
    # BCEWithLogitsLoss - the mathematically correct choice
    criterion = nn.BCEWithLogitsLoss()
    
    grad_scaler = GradScaler('cuda')
    device = model.device
    
    metrics_tracker = ClassificationMetrics()
    
    best_val_loss = float('inf')
    best_auc = 0.0
    patience_counter = 0
    
    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
            
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        metrics_tracker.reset()
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                with autocast('cuda'):
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                
                val_loss += loss.item()
                metrics_tracker.update(logits, y_batch)
        
        val_loss /= len(val_loader)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        metrics = metrics_tracker.get_metrics()
        
        logger.info(
            f"Epoch {epoch+1}/{CONFIG['EPOCHS']} - "
            f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
            f"AUC: {metrics['auc']:.4f}, Prec: {metrics['precision']:.1f}%, "
            f"LR: {current_lr:.6f}"
        )
        
        # Log detailed metrics every 10 epochs
        if (epoch + 1) % 10 == 0:
            metrics_tracker.log_summary("Val")
        
        # === DUAL CHECKPOINTING ===
        saved_something = False
        
        # Priority 1: Best AUC
        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            model.save(str(MODEL_SAVE_PATH_BEST_AUC))
            logger.info(f"🏆 NEW BEST AUC: {best_auc:.4f} (saved)")
            patience_counter = 0
            saved_something = True
            
        # Priority 2: Best Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(str(MODEL_SAVE_PATH))
            logger.info(f"📉 NEW BEST LOSS: {best_val_loss:.4f} (saved)")
            patience_counter = 0
            saved_something = True
            
        if not saved_something:
            patience_counter += 1
            if patience_counter >= CONFIG["PATIENCE"]:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Save scaler and thresholds
    scaler_data = {'scaler': scaler, 'feature_names': feature_cols, 'version': 'v8-class'}
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler_data, f)
    
    # IMPORTANT: Save thresholds so they can be used for inference
    with open(THRESHOLDS_SAVE_PATH, 'wb') as f:
        pickle.dump(thresholds, f)
    
    logger.info("="*60)
    logger.info("🎉 Training Complete!")
    logger.info(f"Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"Best AUC: {best_auc:.4f}")
    logger.info(f"Training samples: {len(X_train):,} (non-overlapping)")
    logger.info(f"Thresholds saved to: {THRESHOLDS_SAVE_PATH}")
    logger.info("="*60)


if __name__ == "__main__":
    try:
        train_pipeline()
    except KeyboardInterrupt:
        logger.info("Training interrupted")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
