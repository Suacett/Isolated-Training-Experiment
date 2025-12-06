#!/usr/bin/env python3
"""
HYBRID Training Script - Best of Both Worlds

Uses LEGACY hyperparameters (proven to work):
- Window size: 90 days
- StandardScaler (not MinMax)
- Time-weighted loss (decay 0.002)
- Patience: 25
- Batch size: 128

But with YOUR current features (37 features from database).

Usage:
    docker exec proxmox_stock_backend python -m scripts.train_model_hybrid
"""

import sys
import os
import logging
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset
import pickle

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from services.feature_engineering import process_stock_data, get_model_input_features
from services.lstm_model import LSTMModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_hybrid.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# === HYBRID CONFIG - Legacy hyperparams + current features ===
CONFIG = {
    "WINDOW_SIZE": 90,  # Legacy uses 90!
    "BATCH_SIZE": 128,
    "EPOCHS": 200,
    "PATIENCE": 25,
    "LEARNING_RATE": 0.001,
    "DECAY_FACTOR": 0.002,
    "TRAIN_SPLIT": 0.64,  # 80% of 80%
    "VAL_SPLIT": 0.16,    # 20% of 80%
    "TEST_SPLIT": 0.20,
    "MIN_RECORDS": 500,
    "MAX_STOCKS": 100,  # Reduced to prevent OOM
    "MAX_RECORDS_PER_STOCK": 2500,  # Reduced to prevent OOM
}

# Paths
MODEL_SAVE_PATH = backend_path / "models" / "lstm_model_hybrid.pth"
SCALER_SAVE_PATH = backend_path / "models" / "scaler_hybrid.pkl"


class HybridStockDataset(Dataset):
    """Dataset with time-weighted sampling"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, dates: np.ndarray = None, 
                 use_time_weights: bool = True, decay_factor: float = 0.002):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.dates = dates
        self.use_time_weights = use_time_weights
        self.decay_factor = decay_factor
        
        # Pre-compute weights
        if use_time_weights and dates is not None:
            self.weights = self._compute_time_weights()
        else:
            self.weights = torch.ones(len(X), dtype=torch.float32)
    
    def _compute_time_weights(self) -> torch.Tensor:
        """Exponential time decay - more weight to recent data"""
        weights = []
        now = pd.Timestamp.now()
        for date in self.dates:
            try:
                date_ago = (now - pd.to_datetime(date)).days
                weight = np.exp(-self.decay_factor * max(0, date_ago))
            except:
                weight = 1.0
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.weights[idx]


class HybridTrainer:
    """Trainer with weighted MSE loss"""
    
    def __init__(self, model: LSTMModel, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.device = model.device
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_weight = 0.0
        
        for X, y, weights in train_loader:
            X = X.to(self.device)
            y = y.to(self.device)
            weights = weights.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(X)
            
            # Weighted loss
            loss_per_sample = self.criterion(outputs, y).mean(dim=1)
            weighted_loss = (loss_per_sample * weights).sum() / weights.sum()
            
            weighted_loss.backward()
            self.optimizer.step()
            
            total_loss += weighted_loss.item() * weights.sum().item()
            total_weight += weights.sum().item()
        
        return total_loss / total_weight if total_weight > 0 else 0
    
    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for X, y, _ in val_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                outputs = self.model(X)
                loss = torch.nn.functional.mse_loss(outputs, y)
                total_loss += loss.item() * len(X)
                total_samples += len(X)
        
        return total_loss / total_samples if total_samples > 0 else 0
    
    def fit(self, train_loader, val_loader, epochs, patience, model_path):
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                self.model.save(model_path)
                logger.info(f"✓ Best model saved (val_loss={val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        return {"best_val_loss": best_val_loss, "best_epoch": best_epoch}


async def load_data_from_database():
    """Load data from TimescaleDB"""
    from services.db import AsyncSessionLocal, StockPrice
    from sqlalchemy import select, func
    
    datasets = []
    
    async with AsyncSessionLocal() as session:
        ticker_counts = await session.execute(
            select(StockPrice.ticker, func.count(StockPrice.ticker).label('count'))
            .group_by(StockPrice.ticker)
            .having(func.count(StockPrice.ticker) >= CONFIG["MIN_RECORDS"])
            .order_by(func.count(StockPrice.ticker).desc())
            .limit(CONFIG["MAX_STOCKS"])
        )
        
        tickers = [(row[0], row[1]) for row in ticker_counts.fetchall()]
        logger.info(f"Found {len(tickers)} stocks with >= {CONFIG['MIN_RECORDS']} records")
        
        for ticker, count in tickers:
            try:
                result = await session.execute(
                    select(StockPrice)
                    .where(StockPrice.ticker == ticker)
                    .order_by(StockPrice.timestamp)
                )
                records = result.scalars().all()
                
                if len(records) > CONFIG["MAX_RECORDS_PER_STOCK"]:
                    records = records[-CONFIG["MAX_RECORDS_PER_STOCK"]:]
                
                df = pd.DataFrame([{
                    'date': r.timestamp,
                    'open': r.open,
                    'high': r.high,
                    'low': r.low,
                    'close': r.close,
                    'volume': r.volume
                } for r in records])
                
                datasets.append((ticker, df))
                logger.info(f"Loaded {ticker}: {len(df)} records")
                
            except Exception as e:
                logger.warning(f"Failed to load {ticker}: {e}")
    
    return datasets


def process_datasets(datasets):
    """Process and create training data with 90-day windows"""
    all_X = []
    all_y = []
    all_dates = []
    feature_cols = None
    scaler_data = []
    
    for ticker, df in datasets:
        try:
            # Process features (creates ~37 features)
            processed = process_stock_data(df.copy(), create_targets=True)
            
            if len(processed) < CONFIG["WINDOW_SIZE"] + 10:
                continue
            
            # Get feature columns
            if feature_cols is None:
                feature_cols = get_model_input_features()
            
            # Ensure all columns exist
            for col in feature_cols:
                if col not in processed.columns:
                    processed[col] = 0.0
            
            # Store for scaler fitting (only training portion)
            train_size = int(len(processed) * CONFIG["TRAIN_SPLIT"])
            scaler_data.append(processed[feature_cols].iloc[:train_size].values)
            
            # Create windows
            features = processed[feature_cols].values
            targets = processed[['Target_1d', 'Target_1w', 'Target_1m', 'Target_6m']].values
            dates = processed['date'].values if 'date' in processed.columns else np.arange(len(processed))
            
            for i in range(CONFIG["WINDOW_SIZE"], len(features)):
                window = features[i - CONFIG["WINDOW_SIZE"]:i]
                target = targets[i]
                date = dates[i]
                
                all_X.append(window)
                all_y.append(target)
                all_dates.append(date)
                
        except Exception as e:
            logger.warning(f"Failed to process {ticker}: {e}")
    
    logger.info(f"Created {len(all_X)} windows from {len(datasets)} stocks")
    
    return np.array(all_X), np.array(all_y), np.array(all_dates), scaler_data, feature_cols


def train_pipeline():
    logger.info("="*60)
    logger.info("HYBRID Training Pipeline")
    logger.info(f"Window: {CONFIG['WINDOW_SIZE']}, Using StandardScaler")
    logger.info("="*60)
    
    # Load data
    datasets = asyncio.run(load_data_from_database())
    if len(datasets) < 10:
        logger.error("Not enough data!")
        sys.exit(1)
    
    # Process into windows
    X_all, y_all, dates_all, scaler_data, feature_cols = process_datasets(datasets)
    
    # Fit StandardScaler on training portion
    scaler = StandardScaler()
    all_train_data = np.vstack([d for d in scaler_data if len(d) > 0])
    all_train_data = np.nan_to_num(all_train_data, nan=0.0, posinf=1e6, neginf=-1e6)
    scaler.fit(all_train_data)
    logger.info(f"Fitted StandardScaler on {len(all_train_data)} samples")
    
    # Scale all windows
    n_samples, window_size, n_features = X_all.shape
    X_flat = X_all.reshape(-1, n_features)
    X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=1e6, neginf=-1e6)
    X_scaled = scaler.transform(X_flat)
    X_scaled = X_scaled.reshape(n_samples, window_size, n_features)
    
    # Split data
    n = len(X_scaled)
    train_end = int(n * CONFIG["TRAIN_SPLIT"])
    val_end = int(n * (CONFIG["TRAIN_SPLIT"] + CONFIG["VAL_SPLIT"]))
    
    X_train, y_train, dates_train = X_scaled[:train_end], y_all[:train_end], dates_all[:train_end]
    X_val, y_val, dates_val = X_scaled[train_end:val_end], y_all[train_end:val_end], dates_all[train_end:val_end]
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Create datasets
    train_dataset = HybridStockDataset(X_train, y_train, dates_train, use_time_weights=True, decay_factor=CONFIG["DECAY_FACTOR"])
    val_dataset = HybridStockDataset(X_val, y_val, dates_val, use_time_weights=False)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)
    
    # Initialize model
    model = LSTMModel(
        input_dim=n_features,
        hidden_dim=64,
        output_dim=4,
        dropout=0.3,
        window_size=CONFIG["WINDOW_SIZE"]
    )
    logger.info(f"Model: {n_features} features, {CONFIG['WINDOW_SIZE']} window")
    
    # Train
    trainer = HybridTrainer(model, learning_rate=CONFIG["LEARNING_RATE"])
    history = trainer.fit(train_loader, val_loader, CONFIG["EPOCHS"], CONFIG["PATIENCE"], str(MODEL_SAVE_PATH))
    
    # Save scaler in correct format
    scaler_data = {'scaler': scaler, 'feature_names': feature_cols}
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler_data, f)
    logger.info(f"Scaler saved to {SCALER_SAVE_PATH}")
    
    logger.info("="*60)
    logger.info("🎉 Hybrid Training Complete!")
    logger.info(f"Best Val Loss: {history['best_val_loss']:.6f}")
    logger.info(f"Best Epoch: {history['best_epoch']}")
    logger.info("="*60)
    logger.info("To activate:")
    logger.info("  cp /app/models/lstm_model_hybrid.pth /app/models/lstm_model_v2.pth")
    logger.info("  cp /app/models/scaler_hybrid.pkl /app/models/scaler_v2.pkl")
    logger.info("  docker compose restart backend")


if __name__ == "__main__":
    try:
        train_pipeline()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)
