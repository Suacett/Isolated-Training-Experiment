#!/usr/bin/env python3
"""
LEGACY-ACCURATE Training Script

This script trains the LSTM using the EXACT settings from the original author's 
forecast.ipynb notebook, for maximum accuracy comparison.

Key Changes from v3:
- Window size: 90 (not 60)
- Uses StandardScaler (not MinMaxScaler)
- MC Samples: 25 (not 50)
- Uses original legacy preprocessed data files
- Decay factor: 0.002

Usage:
    docker exec proxmox_stock_backend python -m scripts.train_model_legacy
"""

import sys
import os
import logging
import glob
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from services.lstm_model import LSTMModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_legacy.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# === LEGACY CONFIG (from forecast.ipynb) ===
CONFIG = {
    "WINDOW_SIZE": 90,  # Legacy uses 90, not 60!
    "TRAIN_VAL_FRAC": 0.8,
    "VAL_FRAC_WITHIN_TRAIN": 0.2,
    "MC_DROPOUT_SAMPLES": 25,
    "BATCH_SIZE": 128,
    "EPOCHS": 200,
    "PATIENCE": 25,
    "LEARNING_RATE": 0.001,
    "DECAY_FACTOR": 0.002,
    "EXCLUDED_COLS": ["date", "Target_1d", "Target_1w", "Target_1m", "Target_6m"],
}

# Paths
LEGACY_DATA_DIR = Path("/app/legacy/LSTM_AI_Stock_Predictor/TrainingData/indicators_data/processed/stocksData")
MODEL_SAVE_PATH = backend_path / "models" / "lstm_model_legacy.pth"
SCALER_SAVE_PATH = backend_path / "models" / "scaler_legacy.pkl"


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add target columns (log returns for multiple horizons)"""
    df = df.copy()
    df["Target_1d"] = np.log(df["close"].shift(-1) / df["close"])
    df["Target_1w"] = np.log(df["close"].shift(-5) / df["close"])
    df["Target_1m"] = np.log(df["close"].shift(-21) / df["close"])
    df["Target_6m"] = np.log(df["close"].shift(-126) / df["close"])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


class LegacyStockDataset(Dataset):
    """Dataset matching legacy StockDataGenerator"""
    
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
            date_ago = (now - pd.to_datetime(date)).days
            weight = np.exp(-self.decay_factor * date_ago)
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.weights[idx]


class LegacyLSTMTrainer:
    """Trainer with weighted loss matching legacy behavior"""
    
    def __init__(self, model: LSTMModel, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss(reduction='none')  # Per-sample loss
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
            loss_per_sample = self.criterion(outputs, y).mean(dim=1)  # Mean over horizons
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
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                # Save best model
                self.model.save(model_path)
                logger.info(f"✓ New best model saved (val_loss={val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        return {"best_val_loss": best_val_loss, "best_epoch": best_epoch}


def load_legacy_data():
    """Load and process data exactly like the legacy notebook"""
    logger.info(f"Loading data from {LEGACY_DATA_DIR}")
    
    csv_files = sorted(glob.glob(str(LEGACY_DATA_DIR / "*_daily_processed.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {LEGACY_DATA_DIR}")
    
    logger.info(f"Found {len(csv_files)} stock files")
    
    # Step 1: Determine global date range
    global_min_date, global_max_date = None, None
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, parse_dates=["date"])
        if global_min_date is None or df["date"].min() < global_min_date:
            global_min_date = df["date"].min()
        if global_max_date is None or df["date"].max() > global_max_date:
            global_max_date = df["date"].max()
    
    train_val_cutoff = global_min_date + (global_max_date - global_min_date) * CONFIG["TRAIN_VAL_FRAC"]
    train_cutoff = global_min_date + (train_val_cutoff - global_min_date) * (1 - CONFIG["VAL_FRAC_WITHIN_TRAIN"])
    
    logger.info(f"Date range: {global_min_date} to {global_max_date}")
    logger.info(f"Train cutoff: {train_cutoff}, Val cutoff: {train_val_cutoff}")
    
    # Step 2: Fit scaler on training data (like legacy)
    scaler = StandardScaler()
    feature_cols = None
    scaler_inputs = []
    
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").dropna()
        df = add_targets(df)
        if df.empty:
            continue
        
        # Get feature columns (exclude date and targets)
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in CONFIG["EXCLUDED_COLS"]]
        
        train_rows = df[df["date"] < train_cutoff]
        if len(train_rows) > 0:
            scaler_inputs.append(train_rows[feature_cols].values)
    
    X_all = np.vstack(scaler_inputs)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=1e6, neginf=-1e6)
    scaler.fit(X_all)
    logger.info(f"Fitted StandardScaler on {len(X_all)} samples, {len(feature_cols)} features")
    
    # Step 3: Create windowed data
    X_train, y_train, dates_train = [], [], []
    X_val, y_val, dates_val = [], [], []
    X_test, y_test, dates_test = [], [], []
    
    for csv_path in csv_files:
        stock_name = Path(csv_path).stem
        df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        df = add_targets(df)
        if df.empty:
            continue
        
        # Ensure all feature columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        # Scale features
        features_scaled = scaler.transform(df[feature_cols].values)
        dates = df["date"].values
        targets = df[["Target_1d", "Target_1w", "Target_1m", "Target_6m"]].values
        
        # Create windows (window_size + 1 like legacy)
        for i in range(CONFIG["WINDOW_SIZE"], len(features_scaled)):
            window = features_scaled[i - CONFIG["WINDOW_SIZE"]:i + 1]  # +1 to include current
            target = targets[i]
            date = dates[i]
            
            date_ts = pd.to_datetime(date)
            
            if date_ts < train_cutoff:
                X_train.append(window)
                y_train.append(target)
                dates_train.append(date)
            elif date_ts < train_val_cutoff:
                X_val.append(window)
                y_val.append(target)
                dates_val.append(date)
            else:
                X_test.append(window)
                y_test.append(target)
                dates_test.append(date)
    
    logger.info(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
    
    return (
        np.array(X_train), np.array(y_train), np.array(dates_train),
        np.array(X_val), np.array(y_val), np.array(dates_val),
        scaler, feature_cols
    )


def train_pipeline():
    logger.info("="*60)
    logger.info("LEGACY-ACCURATE Training Pipeline")
    logger.info(f"Window: {CONFIG['WINDOW_SIZE']}, MC: {CONFIG['MC_DROPOUT_SAMPLES']}")
    logger.info("="*60)
    
    # Load data
    X_train, y_train, dates_train, X_val, y_val, dates_val, scaler, feature_cols = load_legacy_data()
    
    # Create datasets
    train_dataset = LegacyStockDataset(
        X_train, y_train, dates_train, 
        use_time_weights=True, 
        decay_factor=CONFIG["DECAY_FACTOR"]
    )
    val_dataset = LegacyStockDataset(
        X_val, y_val, dates_val,
        use_time_weights=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)
    
    # Initialize model with legacy settings
    # Window is 91 (90+1), features same as feature_cols length
    input_dim = len(feature_cols)
    window_size = CONFIG["WINDOW_SIZE"] + 1  # +1 like legacy!
    
    logger.info(f"Model: input_dim={input_dim}, window_size={window_size}")
    
    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=1,
        output_dim=4,  # 4 horizons
        dropout=0.3,
        window_size=window_size
    )
    
    # Train
    trainer = LegacyLSTMTrainer(model, learning_rate=CONFIG["LEARNING_RATE"])
    history = trainer.fit(
        train_loader, val_loader,
        epochs=CONFIG["EPOCHS"],
        patience=CONFIG["PATIENCE"],
        model_path=str(MODEL_SAVE_PATH)
    )
    
    # Save scaler
    import pickle
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {SCALER_SAVE_PATH}")
    
    logger.info("="*60)
    logger.info("🎉 Legacy Training Complete!")
    logger.info(f"Best Val Loss: {history['best_val_loss']:.6f}")
    logger.info(f"Best Epoch: {history['best_epoch']}")
    logger.info(f"Model: {MODEL_SAVE_PATH}")
    logger.info(f"Scaler: {SCALER_SAVE_PATH}")
    logger.info("="*60)
    logger.info("")
    logger.info("📋 To activate this model:")
    logger.info("cp /app/models/lstm_model_legacy.pth /app/models/lstm_model_v2.pth")
    logger.info("cp /app/models/scaler_legacy.pkl /app/models/scaler_v2.pkl")
    logger.info("docker compose restart backend")
    logger.info("="*60)


if __name__ == "__main__":
    try:
        train_pipeline()
    except KeyboardInterrupt:
        logger.info("Training interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
