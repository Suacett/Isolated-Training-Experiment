#!/usr/bin/env python3
"""
Training Script v8 - CLASSIFICATION MODEL (Regime Detection)
REWRITTEN FOR STRICT V4 LEGACY COMPLIANCE

Key V4 Corrections Applied:
  1. WINDOW_SIZE = 90 (not 60)
  2. Window slice: features[i - WINDOW : i + 1] (91 timesteps, includes current day)
  3. THRESHOLD_SIGMA = 2.0 (strict sniper standard, ~2.5% positives)
  4. Simple Architecture: Conv1D(32) -> LSTM(64) -> Dense(32) -> Dense(4)
  5. Standard training params: LR=0.001, BATCH=128

Usage:
    docker exec proxmox_stock_backend python -m scripts.train_model_v8_class
"""

import sys
import os
import gc
import logging
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from services.feature_engineering import process_stock_data, get_model_input_features_v7

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_v8_class.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# === CONFIG (V4 COMPLIANT) ===
CONFIG = {
    # ===== V4 CRITICAL FIXES =====
    "WINDOW_SIZE": 90,              # V4 uses 90, not 60
    "THRESHOLD_SIGMA": 2.0,         # V4 uses 2.0 (sniper standard), not 1.0
    
    # ===== V4 Training Params =====
    "MAX_HORIZON": 126,             # 6 trading months - stride for non-overlapping
    "BATCH_SIZE": 128,              # V4 standard batch size
    "LEARNING_RATE": 0.001,         # V4 standard LR (not 1e-5)
    "EPOCHS": 500,                  # V4 max epochs
    "PATIENCE": 25,                 # V4 early stopping patience
    "TRAIN_CUTOFF": 0.80,
    
    # ===== V4 Architecture (Simple) =====
    "CONV_FILTERS": 32,             # V4 uses 32, not 64
    "LSTM_UNITS": 64,               # V4 uses 64, not 128
    "DENSE_UNITS": 32,              # V4 uses 32, not 64
    "DROPOUT": 0.3,                 # V4 uses 0.3, not 0.5
    
    # ===== Data Settings =====
    "MAX_STOCKS": int(os.getenv("MAX_STOCKS", "9999")),
    "MIN_RECORDS": int(os.getenv("MIN_RECORDS", "500")),
}

MODEL_SAVE_PATH = backend_path / "models" / "lstm_model_v8_class.pth"
MODEL_SAVE_PATH_BEST_AUC = backend_path / "models" / "lstm_model_v8_class_auc.pth"
SCALER_SAVE_PATH = backend_path / "models" / "scaler_v8_class.pkl"
THRESHOLDS_SAVE_PATH = backend_path / "models" / "thresholds_v8_class.pkl"

# Horizon mapping
HORIZONS = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
    "6m": 126,
}


def get_device() -> torch.device:
    """Dynamic device selection - never hardcode cuda:0"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# V4-COMPLIANT SIMPLE MODEL ARCHITECTURE
# =============================================================================

class MCDropout(nn.Module):
    """Monte Carlo Dropout: Active during both training and inference (for uncertainty)."""
    def __init__(self, p: float = 0.3):
        super().__init__()
        self.p = p
        self.force_dropout = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active = self.training or self.force_dropout
        return nn.functional.dropout(x, p=self.p, training=active)


class LSTMClassifierV4(nn.Module):
    """
    V4-Compliant Simple LSTM Classifier
    
    Architecture (matches legacy run_forecast_v4.ipynb):
        Conv1D(32, k=3) -> BatchNorm -> MCDropout(0.3)
        LSTM(64, 1-layer, unidirectional)
        MCDropout(0.3)
        Dense(32, ReLU) -> Dense(4, Logits)
    
    Parameter count: ~50k (vs 500k+ for BiLSTM+Attention)
    """
    
    def __init__(
        self,
        input_dim: int,
        window_size: int = 91,  # WINDOW_SIZE + 1 (includes current day)
        conv_filters: int = 32,
        lstm_units: int = 64,
        dense_units: int = 32,
        dropout: float = 0.3,
        output_dim: int = 4,
        device: torch.device = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.window_size = window_size
        self.device = device or get_device()
        
        # Layer 1: Conv1D Block
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_filters,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.dropout1 = MCDropout(dropout)
        
        # Layer 2: Simple LSTM (unidirectional, 1 layer)
        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0
        )
        self.dropout2 = MCDropout(dropout)
        
        # Layer 3: Dense layers
        self.fc1 = nn.Linear(lstm_units, dense_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_units, output_dim)  # Output logits
        
        self.to(self.device)
        self._count_parameters()
        
    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Model Parameters: {total:,} total, {trainable:,} trainable")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - returns RAW LOGITS (not probabilities).
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Logits tensor of shape (batch, 4)
        """
        if x.device != self.device:
            x = x.to(self.device)
        
        # Conv1D expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        
        # Back to (batch, seq_len, channels) for LSTM
        x = x.permute(0, 2, 1)
        
        # LSTM - only take final hidden state
        lstm_out, (h_n, _) = self.lstm(x)
        x = h_n[-1]  # Last layer's hidden state
        
        x = self.dropout2(x)
        
        # Dense layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # Raw logits
        
        return x
    
    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Inference method - returns probabilities via sigmoid."""
        x = x.to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
        self.eval()
        self.dropout1.force_dropout = False
        self.dropout2.force_dropout = False
        
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
            
        return probabilities
    
    def save(self, path: str) -> None:
        """Save model weights and config."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'window_size': self.window_size,
            'version': 'v8-class-v4compliant'
        }, path)
        logger.info(f"Model saved to {path}")
        
    @classmethod
    def load(cls, path: str, device: torch.device = None) -> 'LSTMClassifierV4':
        """Load model from file."""
        if device is None:
            device = get_device()
            
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        model = cls(
            input_dim=checkpoint['input_dim'],
            window_size=checkpoint['window_size'],
            device=device
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")
        
        return model


# =============================================================================
# DATA LOADING
# =============================================================================

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


# =============================================================================
# THRESHOLD COMPUTATION (V4 COMPLIANT: 2-SIGMA)
# =============================================================================

def compute_thresholds_from_training(all_train_targets: np.ndarray) -> dict:
    """
    Compute volatility thresholds from TRAINING DATA ONLY.
    
    V4 COMPLIANCE: Uses 2.0 sigma (sniper standard)
    This gives ~2.5% positive rate (top-tier signals only)
    """
    thresholds = {}
    horizon_names = ["1d", "1w", "1m", "6m"]
    
    logger.info(f"Computing thresholds with SIGMA = {CONFIG['THRESHOLD_SIGMA']}")
    
    for i, h in enumerate(horizon_names):
        targets = all_train_targets[:, i]
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
    """Convert regression targets (log returns) to binary classification targets."""
    binary = np.zeros_like(targets)
    for i in range(targets.shape[1]):
        binary[:, i] = (targets[:, i] > thresholds[i]).astype(float)
    return binary


# =============================================================================
# DATA PROCESSING (V4 COMPLIANT: WINDOW + 1)
# =============================================================================

def process_single_stock_v8(
    ticker: str, 
    df: pd.DataFrame, 
    feature_cols: list, 
    scaler: StandardScaler,
):
    """
    Process a single stock with V4-compliant windowing.
    
    V4 CRITICAL FIX:
        Window slice is [i - WINDOW_SIZE : i + 1] which includes 91 timesteps!
        This means the model sees data up to and including the current day.
    """
    try:
        processed = process_stock_data(df.copy(), create_targets=True)
        
        window = CONFIG["WINDOW_SIZE"]
        stride = CONFIG["MAX_HORIZON"]
        
        # V4 FIX: Need WINDOW_SIZE + MAX_HORIZON + 1 for proper indexing
        min_required = window + stride + 1
        if len(processed) < min_required:
            return None, None, None, None
        
        for col in feature_cols:
            if col not in processed.columns:
                processed[col] = 0.0
        
        features = processed[feature_cols].values
        targets = processed[['Target_1d', 'Target_1w', 'Target_1m', 'Target_6m']].values
        
        # === V4-COMPLIANT NON-OVERLAPPING WINDOWS ===
        X_list, y_list = [], []
        
        # V4 FIX: Window is [i - window : i + 1] which gives 91 timesteps!
        for i in range(window, len(features), stride):
            target = targets[i]
            if np.isnan(target).any():
                continue
            
            # V4 CRITICAL: Include current day in window (i+1 NOT i)
            window_slice = features[i - window : i + 1]  # Shape: (91, n_features)
            
            if len(window_slice) != window + 1:
                continue  # Skip incomplete windows
                
            X_list.append(window_slice)
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
        X_train_scaled = np.clip(X_train_scaled, -10.0, 10.0)
        
        if len(X_val) > 0:
            N_val = len(X_val)
            X_val_flat = np.nan_to_num(X_val.reshape(-1, F), nan=0.0, posinf=1e6, neginf=-1e6)
            X_val_scaled = scaler.transform(X_val_flat).reshape(N_val, T, F)
            X_val_scaled = np.clip(X_val_scaled, -10.0, 10.0)
        else:
            X_val_scaled, y_val = None, None
        
        # Clip targets to prevent extreme values
        y_train = np.clip(y_train, -5.0, 5.0)
        if y_val is not None:
            y_val = np.clip(y_val, -5.0, 5.0)
        
        return (
            X_train_scaled.astype(np.float32) if X_train_scaled is not None else None,
            y_train.astype(np.float32) if y_train is not None else None,
            X_val_scaled.astype(np.float32) if X_val_scaled is not None else None,
            y_val.astype(np.float32) if y_val is not None else None
        )
    except Exception as e:
        logger.warning(f"Failed to process {ticker}: {e}")
        return None, None, None, None


# =============================================================================
# DATASET & METRICS
# =============================================================================

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
        
        try:
            auc = roc_auc_score(target_1d, pred_1d)
        except:
            auc = 0.5
        
        pred_binary = (pred_1d > 0.5).astype(int)
        try:
            precision = precision_score(target_1d, pred_binary, zero_division=0)
            recall = recall_score(target_1d, pred_binary, zero_division=0)
        except:
            precision, recall = 0.0, 0.0
        
        positive_rate = target_1d.mean() * 100
        
        return {
            "auc": auc,
            "precision": precision * 100,
            "recall": recall * 100,
            "positive_rate": positive_rate
        }


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def train_pipeline():
    """Main training pipeline for V8 classification model (V4 Compliant)."""
    
    # GPU Check
    device = get_device()
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🚀 GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    else:
        logger.warning("⚠️ NO GPU!")
    
    logger.info("=" * 60)
    logger.info("Training V8 - CLASSIFICATION MODEL (V4 LEGACY COMPLIANT)")
    logger.info("=" * 60)
    logger.info("V4 Compliance Checklist:")
    logger.info(f"  ✓ WINDOW_SIZE:      {CONFIG['WINDOW_SIZE']} (V4 standard)")
    logger.info(f"  ✓ THRESHOLD_SIGMA:  {CONFIG['THRESHOLD_SIGMA']} (sniper standard)")
    logger.info(f"  ✓ LEARNING_RATE:    {CONFIG['LEARNING_RATE']} (V4 standard)")
    logger.info(f"  ✓ BATCH_SIZE:       {CONFIG['BATCH_SIZE']} (V4 standard)")
    logger.info(f"  ✓ Architecture:     Conv1D(32)->LSTM(64)->Dense(32)->Dense(4)")
    logger.info("=" * 60)
    
    tickers = get_ticker_list()
    logger.info(f"Found {len(tickers)} stocks with >= {CONFIG['MIN_RECORDS']} records")
    
    feature_cols = get_model_input_features_v7()
    n_features = len(feature_cols)
    logger.info(f"Using {n_features} features")
    
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
    
    # === PHASE 2: Process All Stocks with V4-Compliant Windows ===
    logger.info("\n📊 Phase 2: Processing stocks with V4-compliant windows...")
    logger.info(f"   Window: {CONFIG['WINDOW_SIZE']} days + 1 (current day) = {CONFIG['WINDOW_SIZE'] + 1} timesteps")
    
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
    y_train_raw = np.vstack(all_y_train)
    X_val = np.vstack(all_X_val)
    y_val_raw = np.vstack(all_y_val)
    
    logger.info(f"\n📉 V4-Compliant Window Stats:")
    logger.info(f"   Training samples: {len(X_train):,}")
    logger.info(f"   Validation samples: {len(X_val):,}")
    logger.info(f"   Sequence length: {X_train.shape[1]} (WINDOW+1)")
    
    # === PHASE 3: Compute V4 Thresholds (2-SIGMA) ===
    logger.info("\n🎯 Phase 3: Computing V4 thresholds (2-SIGMA)...")
    thresholds = compute_thresholds_from_training(y_train_raw)
    
    # Convert to binary classification targets
    y_train = create_classification_targets(y_train_raw, thresholds)
    y_val = create_classification_targets(y_val_raw, thresholds)
    
    # Log class balance (should be ~2-3% with 2-sigma)
    logger.info("\n⚖️ Class Balance (V4 2-Sigma Standard):")
    for i, h in enumerate(["1d", "1w", "1m", "6m"]):
        train_pos = y_train[:, i].mean() * 100
        val_pos = y_val[:, i].mean() * 100
        logger.info(f"   {h} positives: Train={train_pos:.1f}%, Val={val_pos:.1f}%")
    
    # === PHASE 4: Train V4-Compliant Model ===
    logger.info("\n🚀 Phase 4: Training V4-compliant simple model...")
    
    train_dataset = ClassificationDataset(X_train, y_train)
    val_dataset = ClassificationDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["BATCH_SIZE"], 
        shuffle=True,
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
    
    # V4-Compliant Simple Model
    model = LSTMClassifierV4(
        input_dim=n_features,
        window_size=CONFIG["WINDOW_SIZE"] + 1,  # V4: 91 timesteps
        conv_filters=CONFIG["CONV_FILTERS"],
        lstm_units=CONFIG["LSTM_UNITS"],
        dense_units=CONFIG["DENSE_UNITS"],
        dropout=CONFIG["DROPOUT"],
        output_dim=4,
        device=device
    )
    
    # V4-Compliant Optimizer: Standard Adam with 0.001 LR
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["LEARNING_RATE"])
    
    # No fancy scheduler - V4 uses ReduceLROnPlateau equivalent
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # === CLASS IMBALANCE FIX (Aggressive Weighting) ===
    # With 2-sigma, expect ~2-3% positives, so weight should be ~30-50x
    pos_counts = y_train.sum(axis=0)
    neg_counts = len(y_train) - pos_counts
    pos_weights = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32).to(device)
    
    logger.info(f"\n⚖️ Class Weights (Aggressive for 2-Sigma):")
    for i, h in enumerate(["1d", "1w", "1m", "6m"]):
        logger.info(f"   {h}: {pos_weights[i]:.1f}x weight on positives")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
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
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        metrics_tracker.reset()
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    metrics_tracker.update(logits, y_batch)
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        metrics = metrics_tracker.get_metrics()
        
        logger.info(
            f"Epoch {epoch+1}/{CONFIG['EPOCHS']} - "
            f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
            f"AUC: {metrics['auc']:.4f}, Pos: {metrics['positive_rate']:.1f}%, "
            f"LR: {current_lr:.6f}"
        )
        
        # === DUAL CHECKPOINTING ===
        saved_something = False
        
        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            model.save(str(MODEL_SAVE_PATH_BEST_AUC))
            logger.info(f"🏆 NEW BEST AUC: {best_auc:.4f}")
            patience_counter = 0
            saved_something = True
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(str(MODEL_SAVE_PATH))
            logger.info(f"📉 NEW BEST LOSS: {best_val_loss:.4f}")
            patience_counter = 0
            saved_something = True
            
        if not saved_something:
            patience_counter += 1
            if patience_counter >= CONFIG["PATIENCE"]:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Save scaler and thresholds
    scaler_data = {'scaler': scaler, 'feature_names': feature_cols, 'version': 'v8-class-v4compliant'}
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler_data, f)
    
    with open(THRESHOLDS_SAVE_PATH, 'wb') as f:
        pickle.dump(thresholds, f)
    
    logger.info("=" * 60)
    logger.info("🎉 Training Complete (V4 Compliant)!")
    logger.info(f"Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"Best AUC: {best_auc:.4f}")
    logger.info(f"Training samples: {len(X_train):,}")
    logger.info(f"Model: {MODEL_SAVE_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        train_pipeline()
    except KeyboardInterrupt:
        logger.info("Training interrupted")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
