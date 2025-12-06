#!/usr/bin/env python3
"""
Training Script v6 - ENHANCED MODEL

Improvements over v5:
- Bigger LSTM (128 hidden units)
- Lower learning rate for finer tuning
- Stricter time-based train/val split (no data leakage)
- Longer training with more patience

Usage:
    docker exec proxmox_stock_backend python -m scripts.train_model_v6
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
import torch
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
import pickle

backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from services.feature_engineering import process_stock_data, get_model_input_features
from services.lstm_model import LSTMModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_v6.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# === CONFIG ===
CONFIG = {
    "WINDOW_SIZE": int(os.getenv("WINDOW_SIZE", "60")),
    "BATCH_SIZE": int(os.getenv("BATCH_SIZE", "256")),
    "HIDDEN_DIM": int(os.getenv("HIDDEN_DIM", "128")),  # Bigger model!
    "ACCUMULATION_STEPS": int(os.getenv("ACCUMULATION_STEPS", "2")),
    "MAX_STOCKS": int(os.getenv("MAX_STOCKS", "9999")),
    "MAX_RECORDS_PER_STOCK": int(os.getenv("MAX_RECORDS", "9999")),
    "MIN_RECORDS": int(os.getenv("MIN_RECORDS", "200")),
    "EPOCHS": int(os.getenv("EPOCHS", "200")),
    "PATIENCE": int(os.getenv("PATIENCE", "25")),
    "LEARNING_RATE": float(os.getenv("LEARNING_RATE", "0.0003")),  # Lower LR
    "TRAIN_CUTOFF": 0.80,  # Use first 80% of each stock's data for training
}

MODEL_SAVE_PATH = backend_path / "models" / "lstm_model_v6.pth"
SCALER_SAVE_PATH = backend_path / "models" / "scaler_v6.pkl"
TEMP_DATA_DIR = Path("/tmp/training_data_v6")


def get_db_url():
    url = os.getenv("DATABASE_URL", "")
    return url.replace("postgresql+asyncpg://", "postgresql://")


def get_ticker_list():
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
    
    if len(records) > CONFIG["MAX_RECORDS_PER_STOCK"]:
        records = records[-CONFIG["MAX_RECORDS_PER_STOCK"]:]
    
    if records:
        return pd.DataFrame(records, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    return None


def process_single_stock(ticker: str, df: pd.DataFrame, feature_cols: list, scaler: StandardScaler):
    """Process one stock with STRICT time-based split."""
    try:
        processed = process_stock_data(df.copy(), create_targets=True)
        
        if len(processed) < CONFIG["WINDOW_SIZE"] + 10:
            return None, None, None, None
        
        for col in feature_cols:
            if col not in processed.columns:
                processed[col] = 0.0
        
        features = processed[feature_cols].values
        targets = processed[['Target_1d', 'Target_1w', 'Target_1m', 'Target_6m']].values
        
        # Create windows
        X_list, y_list = [], []
        for i in range(CONFIG["WINDOW_SIZE"], len(features)):
            target = targets[i]
            if np.isnan(target).any():
                continue
            X_list.append(features[i - CONFIG["WINDOW_SIZE"]:i])
            y_list.append(target)
        
        if not X_list:
            return None, None, None, None
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # STRICT time-based split - first 80% train, last 20% val
        # This prevents any data leakage from future to past
        split_idx = int(len(X) * CONFIG["TRAIN_CUTOFF"])
        
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]
        
        # Scale
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
            X_train_scaled.astype(np.float32), 
            y_train.astype(np.float32),
            X_val_scaled.astype(np.float32) if X_val_scaled is not None else None,
            y_val.astype(np.float32) if y_val is not None else None
        )
    except Exception as e:
        logger.warning(f"Failed to process {ticker}: {e}")
        return None, None, None, None


class MemmapDataset(Dataset):
    def __init__(self, X_path: Path, y_path: Path, n_samples: int, n_features: int):
        self.X = np.memmap(X_path, dtype=np.float32, mode='r', 
                          shape=(n_samples, CONFIG["WINDOW_SIZE"], n_features))
        self.y = np.memmap(y_path, dtype=np.float32, mode='r', shape=(n_samples, 4))
        self.n_samples = n_samples
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx].copy()), torch.from_numpy(self.y[idx].copy())


# Global counters
_train_offset = 0
_val_offset = 0

def _flush_to_memmap(X_chunks, y_chunks, prefix: str, n_features: int):
    global _train_offset, _val_offset
    
    if not X_chunks:
        return
    
    X = np.vstack(X_chunks)
    y = np.vstack(y_chunks)
    n = len(X)
    
    X_path = TEMP_DATA_DIR / f"{prefix}_X.npy"
    y_path = TEMP_DATA_DIR / f"{prefix}_y.npy"
    info_path = TEMP_DATA_DIR / f"{prefix}_info.npy"
    
    if prefix == "train":
        offset = _train_offset
        _train_offset += n
        total = _train_offset
    else:
        offset = _val_offset
        _val_offset += n
        total = _val_offset
    
    if offset == 0:
        X_mm = np.memmap(X_path, dtype=np.float32, mode='w+', 
                        shape=(n, CONFIG["WINDOW_SIZE"], n_features))
        y_mm = np.memmap(y_path, dtype=np.float32, mode='w+', shape=(n, 4))
    else:
        old_X = np.memmap(X_path, dtype=np.float32, mode='r',
                         shape=(offset, CONFIG["WINDOW_SIZE"], n_features))
        old_y = np.memmap(y_path, dtype=np.float32, mode='r', shape=(offset, 4))
        
        new_X = np.memmap(X_path.with_suffix('.tmp'), dtype=np.float32, mode='w+',
                         shape=(total, CONFIG["WINDOW_SIZE"], n_features))
        new_y = np.memmap(y_path.with_suffix('.tmp'), dtype=np.float32, mode='w+',
                         shape=(total, 4))
        
        new_X[:offset] = old_X[:]
        new_y[:offset] = old_y[:]
        
        del old_X, old_y
        X_path.unlink()
        y_path.unlink()
        X_path.with_suffix('.tmp').rename(X_path)
        y_path.with_suffix('.tmp').rename(y_path)
        
        X_mm = np.memmap(X_path, dtype=np.float32, mode='r+',
                        shape=(total, CONFIG["WINDOW_SIZE"], n_features))
        y_mm = np.memmap(y_path, dtype=np.float32, mode='r+', shape=(total, 4))
    
    X_mm[offset:offset+n] = X
    y_mm[offset:offset+n] = y
    X_mm.flush()
    y_mm.flush()
    
    np.save(info_path, np.array([total]))


def train_pipeline():
    global _train_offset, _val_offset
    _train_offset = 0
    _val_offset = 0
    
    # GPU Check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🚀 GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    else:
        logger.warning("⚠️ NO GPU!")
    
    logger.info("="*60)
    logger.info("Training v6 - ENHANCED MODEL (128 hidden)")
    logger.info(f"Config: {CONFIG}")
    logger.info("="*60)
    
    if TEMP_DATA_DIR.exists():
        shutil.rmtree(TEMP_DATA_DIR)
    TEMP_DATA_DIR.mkdir(parents=True)
    
    tickers = get_ticker_list()
    logger.info(f"Found {len(tickers)} stocks")
    
    feature_cols = get_model_input_features()
    n_features = len(feature_cols)
    
    # Phase 1: Fit scaler
    logger.info("Phase 1: Fitting scaler...")
    sample_data = []
    for ticker, _ in tickers[:100]:
        df = load_single_stock(ticker)
        if df is not None:
            processed = process_stock_data(df.copy(), create_targets=False)
            for col in feature_cols:
                if col not in processed.columns:
                    processed[col] = 0.0
            # Only use train portion for scaler
            train_end = int(len(processed) * CONFIG["TRAIN_CUTOFF"])
            sample_data.append(processed[feature_cols].values[:train_end])
    
    scaler = StandardScaler()
    all_sample = np.vstack(sample_data)
    all_sample = np.nan_to_num(all_sample, nan=0.0, posinf=1e6, neginf=-1e6)
    scaler.fit(all_sample)
    logger.info(f"Scaler fitted on {len(all_sample)} samples")
    del sample_data, all_sample
    gc.collect()
    
    # Phase 2: Process and save
    logger.info("Phase 2: Processing stocks...")
    
    train_X_chunks, train_y_chunks = [], []
    val_X_chunks, val_y_chunks = [], []
    
    for i, (ticker, _) in enumerate(tickers):
        if (i + 1) % 50 == 0:
            logger.info(f"Processing stock {i+1}/{len(tickers)}...")
        
        df = load_single_stock(ticker)
        if df is None:
            continue
            
        X_train, y_train, X_val, y_val = process_single_stock(ticker, df, feature_cols, scaler)
        
        if X_train is not None and len(X_train) > 0:
            train_X_chunks.append(X_train)
            train_y_chunks.append(y_train)
        
        if X_val is not None and len(X_val) > 0:
            val_X_chunks.append(X_val)
            val_y_chunks.append(y_val)
        
        if len(train_X_chunks) >= 50:
            _flush_to_memmap(train_X_chunks, train_y_chunks, "train", n_features)
            _flush_to_memmap(val_X_chunks, val_y_chunks, "val", n_features)
            train_X_chunks, train_y_chunks = [], []
            val_X_chunks, val_y_chunks = [], []
            gc.collect()
    
    _flush_to_memmap(train_X_chunks, train_y_chunks, "train", n_features)
    _flush_to_memmap(val_X_chunks, val_y_chunks, "val", n_features)
    gc.collect()
    
    train_info = np.load(TEMP_DATA_DIR / "train_info.npy")
    val_info = np.load(TEMP_DATA_DIR / "val_info.npy")
    n_train, n_val = int(train_info[0]), int(val_info[0])
    
    logger.info(f"Training: {n_train}, Validation: {n_val}")
    
    # Phase 3: Train
    logger.info("Phase 3: Training ENHANCED model...")
    
    train_dataset = MemmapDataset(TEMP_DATA_DIR / "train_X.npy", TEMP_DATA_DIR / "train_y.npy", n_train, n_features)
    val_dataset = MemmapDataset(TEMP_DATA_DIR / "val_X.npy", TEMP_DATA_DIR / "val_y.npy", n_val, n_features)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=0, pin_memory=True)
    
    model = LSTMModel(
        input_dim=n_features,
        hidden_dim=CONFIG["HIDDEN_DIM"],  # Bigger!
        output_dim=4,
        dropout=0.3,
        window_size=CONFIG["WINDOW_SIZE"]
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = torch.nn.MSELoss()
    grad_scaler = GradScaler('cuda')
    device = model.device
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            
            with autocast('cuda'):
                out = model(X)
                loss = criterion(out, y) / CONFIG["ACCUMULATION_STEPS"]
            
            grad_scaler.scale(loss).backward()
            
            if (batch_idx + 1) % CONFIG["ACCUMULATION_STEPS"] == 0:
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * CONFIG["ACCUMULATION_STEPS"]
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                with autocast('cuda'):
                    out = model(X)
                    loss = criterion(out, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"Epoch {epoch+1}/{CONFIG['EPOCHS']} - Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {current_lr:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save(str(MODEL_SAVE_PATH))
            logger.info(f"✓ Best model saved (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["PATIENCE"]:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    scaler_data = {'scaler': scaler, 'feature_names': feature_cols}
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler_data, f)
    
    shutil.rmtree(TEMP_DATA_DIR)
    
    logger.info("="*60)
    logger.info("🎉 Training Complete!")
    logger.info(f"Best Val Loss: {best_val_loss:.6f}")
    logger.info("="*60)
    logger.info("To activate:")
    logger.info("  docker exec proxmox_stock_backend cp /app/models/lstm_model_v6.pth /app/models/lstm_model_v2.pth")
    logger.info("  docker exec proxmox_stock_backend cp /app/models/scaler_v6.pkl /app/models/scaler_v2.pkl")
    logger.info("  docker compose restart backend")


if __name__ == "__main__":
    try:
        train_pipeline()
    except KeyboardInterrupt:
        logger.info("Training interrupted")
        if TEMP_DATA_DIR.exists():
            shutil.rmtree(TEMP_DATA_DIR)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        if TEMP_DATA_DIR.exists():
            shutil.rmtree(TEMP_DATA_DIR)
        sys.exit(1)
