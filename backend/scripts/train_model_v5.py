#!/usr/bin/env python3
"""
Memory-Efficient Training Script v5 - STREAMING VERSION

Uses ALL stocks by streaming data from disk instead of holding in RAM.

Phase 1: Save processed windows to disk (numpy memmap)
Phase 2: Stream from disk during training

Usage:
    docker exec proxmox_stock_backend python -m scripts.train_model_v5
"""

import sys
import os
import gc
import logging
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
import pickle

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from services.feature_engineering import process_stock_data, get_model_input_features
from services.lstm_model import LSTMModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_v5.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# === CONFIG ===
CONFIG = {
    "WINDOW_SIZE": int(os.getenv("WINDOW_SIZE", "60")),
    "BATCH_SIZE": int(os.getenv("BATCH_SIZE", "256")),  # Larger batch for GPU
    "ACCUMULATION_STEPS": int(os.getenv("ACCUMULATION_STEPS", "2")),
    "MAX_STOCKS": int(os.getenv("MAX_STOCKS", "9999")),  # Unlimited
    "MAX_RECORDS_PER_STOCK": int(os.getenv("MAX_RECORDS", "9999")),  # All history
    "MIN_RECORDS": int(os.getenv("MIN_RECORDS", "200")),
    "EPOCHS": int(os.getenv("EPOCHS", "150")),
    "PATIENCE": int(os.getenv("PATIENCE", "20")),
    "LEARNING_RATE": float(os.getenv("LEARNING_RATE", "0.0005")),
    "TRAIN_SPLIT": 0.85,
}

MODEL_SAVE_PATH = backend_path / "models" / "lstm_model_v5.pth"
SCALER_SAVE_PATH = backend_path / "models" / "scaler_v5.pkl"
TEMP_DATA_DIR = Path("/tmp/training_data")


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
    """Process one stock and return scaled windows + targets."""
    try:
        processed = process_stock_data(df.copy(), create_targets=True)
        
        if len(processed) < CONFIG["WINDOW_SIZE"] + 10:
            return None, None
        
        # Ensure columns exist
        for col in feature_cols:
            if col not in processed.columns:
                processed[col] = 0.0
        
        # Get features and targets
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
            return None, None
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Scale features
        N, T, F = X.shape
        X_flat = np.nan_to_num(X.reshape(-1, F), nan=0.0, posinf=1e6, neginf=-1e6)
        X_scaled = scaler.transform(X_flat).reshape(N, T, F)
        
        return X_scaled.astype(np.float32), y.astype(np.float32)
    except Exception as e:
        logger.warning(f"Failed to process {ticker}: {e}")
        return None, None


class MemmapDataset(Dataset):
    """Dataset that reads from memory-mapped numpy arrays."""
    
    def __init__(self, X_path: Path, y_path: Path, n_samples: int):
        self.X = np.memmap(X_path, dtype=np.float32, mode='r', 
                          shape=(n_samples, CONFIG["WINDOW_SIZE"], 37))
        self.y = np.memmap(y_path, dtype=np.float32, mode='r',
                          shape=(n_samples, 4))
        self.n_samples = n_samples
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx].copy()), torch.from_numpy(self.y[idx].copy())


def train_pipeline():
    # === GPU CHECK ===
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🚀 GPU DETECTED: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    else:
        logger.warning("⚠️ NO GPU - Training will be slow!")
    
    logger.info("="*60)
    logger.info("Streaming Training v5 - MAXIMUM DATA")
    logger.info(f"Config: {CONFIG}")
    logger.info("="*60)
    
    # Setup temp directory
    if TEMP_DATA_DIR.exists():
        shutil.rmtree(TEMP_DATA_DIR)
    TEMP_DATA_DIR.mkdir(parents=True)
    
    # Get tickers
    tickers = get_ticker_list()
    logger.info(f"Found {len(tickers)} stocks")
    
    feature_cols = get_model_input_features()
    n_features = len(feature_cols)
    
    # === PHASE 1: Fit scaler on sample of data ===
    logger.info("Phase 1: Fitting scaler on sample...")
    sample_data = []
    for ticker, _ in tickers[:100]:  # Use first 100 for scaler
        df = load_single_stock(ticker)
        if df is not None:
            processed = process_stock_data(df.copy(), create_targets=False)
            for col in feature_cols:
                if col not in processed.columns:
                    processed[col] = 0.0
            sample_data.append(processed[feature_cols].values[:int(len(processed)*0.8)])
    
    scaler = StandardScaler()
    all_sample = np.vstack(sample_data)
    all_sample = np.nan_to_num(all_sample, nan=0.0, posinf=1e6, neginf=-1e6)
    scaler.fit(all_sample)
    logger.info(f"Scaler fitted on {len(all_sample)} samples")
    del sample_data, all_sample
    gc.collect()
    
    # === PHASE 2: Process all stocks and save to disk ===
    logger.info("Phase 2: Processing stocks and saving to disk...")
    
    train_X_chunks, train_y_chunks = [], []
    val_X_chunks, val_y_chunks = [], []
    
    for i, (ticker, _) in enumerate(tickers):
        if (i + 1) % 50 == 0:
            logger.info(f"Processing stock {i+1}/{len(tickers)}...")
        
        df = load_single_stock(ticker)
        if df is None:
            continue
            
        X, y = process_single_stock(ticker, df, feature_cols, scaler)
        if X is None:
            continue
        
        # Split train/val
        split = int(len(X) * CONFIG["TRAIN_SPLIT"])
        train_X_chunks.append(X[:split])
        train_y_chunks.append(y[:split])
        val_X_chunks.append(X[split:])
        val_y_chunks.append(y[split:])
        
        # Periodically flush to disk to avoid RAM buildup
        if len(train_X_chunks) >= 50:
            # Append to memmap files
            _flush_to_memmap(train_X_chunks, train_y_chunks, "train", n_features)
            _flush_to_memmap(val_X_chunks, val_y_chunks, "val", n_features)
            train_X_chunks, train_y_chunks = [], []
            val_X_chunks, val_y_chunks = [], []
            gc.collect()
    
    # Final flush
    _flush_to_memmap(train_X_chunks, train_y_chunks, "train", n_features)
    _flush_to_memmap(val_X_chunks, val_y_chunks, "val", n_features)
    del train_X_chunks, train_y_chunks, val_X_chunks, val_y_chunks
    gc.collect()
    
    # Get sizes
    train_info = np.load(TEMP_DATA_DIR / "train_info.npy")
    val_info = np.load(TEMP_DATA_DIR / "val_info.npy")
    n_train, n_val = int(train_info[0]), int(val_info[0])
    
    logger.info(f"Training samples: {n_train}, Validation samples: {n_val}")
    
    # === PHASE 3: Train with streaming data ===
    logger.info("Phase 3: Training model...")
    
    train_dataset = MemmapDataset(
        TEMP_DATA_DIR / "train_X.npy",
        TEMP_DATA_DIR / "train_y.npy",
        n_train
    )
    val_dataset = MemmapDataset(
        TEMP_DATA_DIR / "val_X.npy", 
        TEMP_DATA_DIR / "val_y.npy",
        n_val
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], 
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"],
                            shuffle=False, num_workers=0, pin_memory=True)
    
    model = LSTMModel(
        input_dim=n_features,
        hidden_dim=64,
        output_dim=4,
        dropout=0.3,
        window_size=CONFIG["WINDOW_SIZE"]
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"])
    criterion = torch.nn.MSELoss()
    grad_scaler = GradScaler('cuda')
    device = model.device
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(CONFIG["EPOCHS"]):
        # Train
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
        
        # Validate
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
        
        logger.info(f"Epoch {epoch+1}/{CONFIG['EPOCHS']} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
        
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
    
    # Save scaler
    scaler_data = {'scaler': scaler, 'feature_names': feature_cols}
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler_data, f)
    
    # Cleanup temp files
    shutil.rmtree(TEMP_DATA_DIR)
    
    logger.info("="*60)
    logger.info("🎉 Training Complete!")
    logger.info(f"Best Val Loss: {best_val_loss:.6f}")
    logger.info("="*60)
    logger.info("To activate:")
    logger.info("  docker exec proxmox_stock_backend cp /app/models/lstm_model_v5.pth /app/models/lstm_model_v2.pth")
    logger.info("  docker exec proxmox_stock_backend cp /app/models/scaler_v5.pkl /app/models/scaler_v2.pkl")
    logger.info("  docker compose restart backend")


# Global counters for memmap
_train_offset = 0
_val_offset = 0

def _flush_to_memmap(X_chunks, y_chunks, prefix: str, n_features: int):
    """Append chunks to memory-mapped files on disk."""
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
    
    # Create or extend memmap
    if offset == 0:
        # Create new files
        X_mm = np.memmap(X_path, dtype=np.float32, mode='w+', 
                        shape=(n, CONFIG["WINDOW_SIZE"], n_features))
        y_mm = np.memmap(y_path, dtype=np.float32, mode='w+', shape=(n, 4))
    else:
        # Extend existing - need to create new larger file
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
