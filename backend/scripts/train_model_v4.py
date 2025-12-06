#!/usr/bin/env python3
"""
Memory-Efficient Training Script v4

Uses all 505 stocks from database with memory optimization:
- Batch stock processing (50 at a time)
- Gradient accumulation (effective batch=256)
- Mixed precision (FP16) for RTX 4080 Super
- Explicit memory management

Usage:
    docker exec proxmox_stock_backend python -m scripts.train_model_v4

Environment variables:
    BATCH_SIZE=64         # GPU batch size
    ACCUMULATION_STEPS=4  # Effective batch = BATCH_SIZE * ACCUMULATION_STEPS
    STOCKS_PER_BATCH=50   # Stocks loaded into memory at once
    MAX_STOCKS=500        # Total stocks to use
    EPOCHS=200
    PATIENCE=25
"""

import sys
import os
import gc
import logging
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
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
        logging.FileHandler("training_v4.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# === CONFIG from environment ===
CONFIG = {
    "WINDOW_SIZE": int(os.getenv("WINDOW_SIZE", "60")),
    "BATCH_SIZE": int(os.getenv("BATCH_SIZE", "64")),
    "ACCUMULATION_STEPS": int(os.getenv("ACCUMULATION_STEPS", "4")),
    "STOCKS_PER_BATCH": int(os.getenv("STOCKS_PER_BATCH", "50")),
    "MAX_STOCKS": int(os.getenv("MAX_STOCKS", "500")),
    "MAX_RECORDS_PER_STOCK": int(os.getenv("MAX_RECORDS", "4000")),
    "MIN_RECORDS": int(os.getenv("MIN_RECORDS", "500")),
    "EPOCHS": int(os.getenv("EPOCHS", "200")),
    "PATIENCE": int(os.getenv("PATIENCE", "25")),
    "LEARNING_RATE": float(os.getenv("LEARNING_RATE", "0.001")),
    "TRAIN_SPLIT": 0.8,
    "VAL_SPLIT": 0.1,
}

# Paths
MODEL_SAVE_PATH = backend_path / "models" / "lstm_model_v4.pth"
SCALER_SAVE_PATH = backend_path / "models" / "scaler_v4.pkl"


def clear_memory():
    """Aggressively clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_db_url():
    """Get sync database URL from environment."""
    url = os.getenv("DATABASE_URL", "")
    # Convert async URL to sync
    return url.replace("postgresql+asyncpg://", "postgresql://")


def get_ticker_list():
    """Get list of tickers with sufficient data using sync connection."""
    import psycopg2
    
    conn = psycopg2.connect(get_db_url())
    cur = conn.cursor()
    
    cur.execute("""
        SELECT ticker, COUNT(*) as count 
        FROM stock_prices 
        GROUP BY ticker 
        HAVING COUNT(*) >= %s 
        ORDER BY count DESC 
        LIMIT %s
    """, (CONFIG["MIN_RECORDS"], CONFIG["MAX_STOCKS"]))
    
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    return [(row[0], row[1]) for row in results]


def load_stock_batch(tickers: list):
    """Load a batch of stocks using sync connection."""
    import psycopg2
    
    conn = psycopg2.connect(get_db_url())
    cur = conn.cursor()
    
    datasets = []
    
    for ticker in tickers:
        try:
            cur.execute("""
                SELECT ticker, timestamp, open, high, low, close, volume 
                FROM stock_prices 
                WHERE ticker = %s 
                ORDER BY timestamp
            """, (ticker,))
            
            records = cur.fetchall()
            
            if len(records) > CONFIG["MAX_RECORDS_PER_STOCK"]:
                records = records[-CONFIG["MAX_RECORDS_PER_STOCK"]:]
            
            if len(records) > 0:
                df = pd.DataFrame(records, columns=['ticker', 'date', 'open', 'high', 'low', 'close', 'volume'])
                df = df.drop(columns=['ticker'])
                datasets.append((ticker, df))
                logger.debug(f"Loaded {ticker}: {len(df)} records")
        except Exception as e:
            logger.warning(f"Failed to load {ticker}: {e}")
    
    cur.close()
    conn.close()
    
    return datasets


def process_stock_batch(datasets, feature_cols, scaler=None):
    """Process a batch of stocks into training windows."""
    all_X = []
    all_y = []
    scaler_data = []
    
    for ticker, df in datasets:
        try:
            processed = process_stock_data(df.copy(), create_targets=True)
            
            if len(processed) < CONFIG["WINDOW_SIZE"] + 10:
                continue
            
            # Ensure all columns exist
            for col in feature_cols:
                if col not in processed.columns:
                    processed[col] = 0.0
            
            # Store for scaler fitting
            if scaler is None:
                train_size = int(len(processed) * CONFIG["TRAIN_SPLIT"])
                scaler_data.append(processed[feature_cols].iloc[:train_size].values)
            
            # Create windows
            features = processed[feature_cols].values
            targets = processed[['Target_1d', 'Target_1w', 'Target_1m', 'Target_6m']].values
            
            for i in range(CONFIG["WINDOW_SIZE"], len(features)):
                window = features[i - CONFIG["WINDOW_SIZE"]:i]
                target = targets[i]
                
                # Skip if NaN in target
                if np.isnan(target).any():
                    continue
                    
                all_X.append(window)
                all_y.append(target)
                
        except Exception as e:
            logger.warning(f"Failed to process {ticker}: {e}")
    
    return np.array(all_X) if all_X else None, np.array(all_y) if all_y else None, scaler_data


class MemoryEfficientTrainer:
    """Trainer with gradient accumulation and mixed precision."""
    
    def __init__(self, model: LSTMModel, learning_rate: float, accumulation_steps: int):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()
        self.device = model.device
        self.accumulation_steps = accumulation_steps
        self.scaler = GradScaler()  # For mixed precision
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.to(self.device)
            y = y.to(self.device)
            
            # Mixed precision forward pass
            with autocast():
                outputs = self.model(X)
                loss = self.criterion(outputs, y) / self.accumulation_steps
            
            # Scaled backward pass
            self.scaler.scale(loss).backward()
            
            # Accumulate gradients
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
        
        # Handle remaining gradients
        if num_batches % self.accumulation_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                
                with autocast():
                    outputs = self.model(X)
                    loss = self.criterion(outputs, y)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0


def train_pipeline():
    logger.info("="*60)
    logger.info("Memory-Efficient Training v4")
    logger.info(f"Config: {CONFIG}")
    logger.info("="*60)
    
    # Get ticker list
    tickers = get_ticker_list()
    logger.info(f"Found {len(tickers)} stocks with >= {CONFIG['MIN_RECORDS']} records")
    
    feature_cols = get_model_input_features()
    n_features = len(feature_cols)
    
    # Phase 1: Fit scaler on all training data
    logger.info("Phase 1: Fitting scaler...")
    all_scaler_data = []
    
    for i in range(0, len(tickers), CONFIG["STOCKS_PER_BATCH"]):
        batch_tickers = [t[0] for t in tickers[i:i + CONFIG["STOCKS_PER_BATCH"]]]
        logger.info(f"Loading stocks {i+1}-{i+len(batch_tickers)} for scaler fitting...")
        
        datasets = load_stock_batch(batch_tickers)
        _, _, scaler_data = process_stock_batch(datasets, feature_cols, scaler=None)
        
        for data in scaler_data:
            if len(data) > 0:
                all_scaler_data.append(data)
        
        # Clear memory
        del datasets
        clear_memory()
    
    # Fit scaler
    scaler = StandardScaler()
    all_data = np.vstack(all_scaler_data)
    all_data = np.nan_to_num(all_data, nan=0.0, posinf=1e6, neginf=-1e6)
    scaler.fit(all_data)
    logger.info(f"Scaler fitted on {len(all_data)} samples")
    
    del all_scaler_data, all_data
    clear_memory()
    
    # Phase 2: Create training/validation data
    logger.info("Phase 2: Creating training data...")
    all_X_train, all_y_train = [], []
    all_X_val, all_y_val = [], []
    
    for i in range(0, len(tickers), CONFIG["STOCKS_PER_BATCH"]):
        batch_tickers = [t[0] for t in tickers[i:i + CONFIG["STOCKS_PER_BATCH"]]]
        logger.info(f"Processing stocks {i+1}-{i+len(batch_tickers)}...")
        
        datasets = load_stock_batch(batch_tickers)
        X_batch, y_batch, _ = process_stock_batch(datasets, feature_cols, scaler=scaler)
        
        if X_batch is not None and len(X_batch) > 0:
            # Scale features
            N, T, F = X_batch.shape
            X_flat = X_batch.reshape(-1, F)
            X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=1e6, neginf=-1e6)
            X_scaled = scaler.transform(X_flat).reshape(N, T, F)
            
            # Split train/val
            n = len(X_scaled)
            train_end = int(n * CONFIG["TRAIN_SPLIT"])
            val_end = int(n * (CONFIG["TRAIN_SPLIT"] + CONFIG["VAL_SPLIT"]))
            
            all_X_train.append(X_scaled[:train_end])
            all_y_train.append(y_batch[:train_end])
            all_X_val.append(X_scaled[train_end:val_end])
            all_y_val.append(y_batch[train_end:val_end])
        
        del datasets, X_batch, y_batch
        clear_memory()
    
    # Combine all batches
    X_train = np.vstack(all_X_train).astype(np.float32)
    y_train = np.vstack(all_y_train).astype(np.float32)
    X_val = np.vstack(all_X_val).astype(np.float32)
    y_val = np.vstack(all_y_val).astype(np.float32)
    
    del all_X_train, all_y_train, all_X_val, all_y_val
    clear_memory()
    
    logger.info(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["BATCH_SIZE"], 
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG["BATCH_SIZE"], 
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )
    
    # Phase 3: Train model
    logger.info("Phase 3: Training model...")
    
    model = LSTMModel(
        input_dim=n_features,
        hidden_dim=64,
        output_dim=4,
        dropout=0.3,
        window_size=CONFIG["WINDOW_SIZE"]
    )
    
    trainer = MemoryEfficientTrainer(
        model, 
        CONFIG["LEARNING_RATE"],
        CONFIG["ACCUMULATION_STEPS"]
    )
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(CONFIG["EPOCHS"]):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{CONFIG['EPOCHS']} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            model.save(str(MODEL_SAVE_PATH))
            logger.info(f"✓ Best model saved (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["PATIENCE"]:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Clear GPU cache each epoch
        clear_memory()
    
    # Save scaler
    scaler_data = {'scaler': scaler, 'feature_names': feature_cols}
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler_data, f)
    logger.info(f"Scaler saved to {SCALER_SAVE_PATH}")
    
    logger.info("="*60)
    logger.info("🎉 Training Complete!")
    logger.info(f"Best Val Loss: {best_val_loss:.6f}")
    logger.info(f"Best Epoch: {best_epoch}")
    logger.info("="*60)
    logger.info("To activate:")
    logger.info("  docker exec proxmox_stock_backend cp /app/models/lstm_model_v4.pth /app/models/lstm_model_v2.pth")
    logger.info("  docker exec proxmox_stock_backend cp /app/models/scaler_v4.pkl /app/models/scaler_v2.pkl")
    logger.info("  docker compose restart backend")


if __name__ == "__main__":
    try:
        train_pipeline()
    except KeyboardInterrupt:
        logger.info("Training interrupted")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
