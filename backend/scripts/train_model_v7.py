#!/usr/bin/env python3
"""
Training Script v7 - ENHANCED MODEL WITH ATTENTION

Improvements over v6:
- Bidirectional LSTM with Temporal Attention
- DirectionalLoss: MSE + penalty for wrong-sign predictions
- Weighted sampling: high-volatility days get more attention
- Confusion matrix logging per epoch
- Gradient clipping to prevent exploding gradients
- CPU offload support for large batches
- Cosine annealing learning rate schedule

Usage:
    docker exec proxmox_stock_backend python -m scripts.train_model_v7
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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.amp import autocast, GradScaler
import pickle

backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from services.feature_engineering import process_stock_data, get_model_input_features_v7
from services.lstm_model_v7 import LSTMModelV7

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_v7.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# === CONFIG ===
CONFIG = {
    "WINDOW_SIZE": int(os.getenv("WINDOW_SIZE", "60")),
    "BATCH_SIZE": int(os.getenv("BATCH_SIZE", "128")),  # Smaller for attention model
    "HIDDEN_DIM": int(os.getenv("HIDDEN_DIM", "128")),
    "NUM_LAYERS": int(os.getenv("NUM_LAYERS", "2")),
    "NUM_ATTENTION_HEADS": int(os.getenv("NUM_ATTENTION_HEADS", "4")),
    "ACCUMULATION_STEPS": int(os.getenv("ACCUMULATION_STEPS", "4")),  # More accumulation
    "MAX_STOCKS": int(os.getenv("MAX_STOCKS", "9999")),
    "MAX_RECORDS_PER_STOCK": int(os.getenv("MAX_RECORDS", "9999")),
    "MIN_RECORDS": int(os.getenv("MIN_RECORDS", "500")),  # More history required
    "EPOCHS": int(os.getenv("EPOCHS", "300")),  # More epochs
    "PATIENCE": int(os.getenv("PATIENCE", "30")),
    "LEARNING_RATE": float(os.getenv("LEARNING_RATE", "0.0001")),  # Lower LR for attention
    "DIRECTION_WEIGHT": float(os.getenv("DIRECTION_WEIGHT", "0.5")),
    "TRAIN_CUTOFF": 0.80,
    "USE_CPU_OFFLOAD": os.getenv("USE_CPU_OFFLOAD", "1").lower() in ("1", "true", "yes"),
    "GRADIENT_CLIP_NORM": float(os.getenv("GRADIENT_CLIP_NORM", "1.0")),
}

MODEL_SAVE_PATH = backend_path / "models" / "lstm_model_v7.pth"  # Will point to BEST ACCURACY
MODEL_SAVE_PATH_LOSS = backend_path / "models" / "lstm_model_v7_loss.pth"  # Backup: Best Loss
SCALER_SAVE_PATH = backend_path / "models" / "scaler_v7.pkl"
TEMP_DATA_DIR = Path("/tmp/training_data_v7")


class DirectionalLoss(nn.Module):
    """
    MSE + penalty for wrong direction predictions.
    
    Direction matters more than magnitude for trading!
    If actual is +2% and we predict -1%, that's worse than predicting +5%.
    
    Args:
        direction_weight: Weight for direction penalty (0.5 = equal to MSE)
        volatility_boost: Extra penalty multiplier for high-volatility predictions
    """
    def __init__(self, direction_weight: float = 0.5, volatility_boost: float = 2.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.direction_weight = direction_weight
        self.volatility_boost = volatility_boost
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Per-sample MSE
        mse_loss = self.mse(pred, target).mean(dim=1)  # Mean across horizons
        
        # Direction penalty: 1 if signs don't match, 0 if they do
        # Focus on 1-day prediction (index 0)
        pred_sign = torch.where(pred[:, 0] >= 0, torch.ones_like(pred[:, 0]), -torch.ones_like(pred[:, 0]))
        target_sign = torch.where(target[:, 0] >= 0, torch.ones_like(target[:, 0]), -torch.ones_like(target[:, 0]))
        direction_wrong = (pred_sign != target_sign).float()
        
        # Volatility weighting: penalize more on high-movement days
        target_magnitude = torch.abs(target[:, 0])
        volatility_weight = 1.0 + self.volatility_boost * target_magnitude
        
        # Weighted direction loss
        direction_loss = direction_wrong * volatility_weight
        
        # Combined loss
        total_loss = mse_loss + self.direction_weight * direction_loss
        
        return total_loss.mean()


class ConfusionTracker:
    """
    Tracks confusion matrix statistics during training.
    
    Logs:
    - True Up / False Up (predicted up correctly / predicted up incorrectly)
    - True Down / False Down (predicted down correctly / predicted down incorrectly)
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.true_up = 0
        self.false_up = 0
        self.true_down = 0
        self.false_down = 0
        
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update confusion matrix with batch predictions."""
        # Focus on 1-day predictions
        pred_up = (pred[:, 0] > 0).cpu()
        target_up = (target[:, 0] > 0).cpu()
        
        self.true_up += ((pred_up) & (target_up)).sum().item()
        self.false_up += ((pred_up) & (~target_up)).sum().item()
        self.true_down += ((~pred_up) & (~target_up)).sum().item()
        self.false_down += ((~pred_up) & (target_up)).sum().item()
        
    def get_metrics(self) -> dict:
        total = self.true_up + self.false_up + self.true_down + self.false_down
        if total == 0:
            return {"accuracy": 0, "up_accuracy": 0, "down_accuracy": 0, "bias": 0}
        
        accuracy = (self.true_up + self.true_down) / total
        
        # Up prediction accuracy
        up_total = self.true_up + self.false_up
        up_accuracy = self.true_up / up_total if up_total > 0 else 0
        
        # Down prediction accuracy
        down_total = self.true_down + self.false_down
        down_accuracy = self.true_down / down_total if down_total > 0 else 0
        
        # Bias: positive = bullish bias, negative = bearish bias
        predicted_up_pct = up_total / total if total > 0 else 0.5
        actual_up_pct = (self.true_up + self.false_down) / total if total > 0 else 0.5
        bias = (predicted_up_pct - actual_up_pct) * 100
        
        return {
            "accuracy": accuracy * 100,
            "up_accuracy": up_accuracy * 100,
            "down_accuracy": down_accuracy * 100,
            "bias": bias,
            "confusion": {
                "true_up": self.true_up,
                "false_up": self.false_up,
                "true_down": self.true_down,
                "false_down": self.false_down
            }
        }
        
    def log_summary(self, prefix: str = "Val"):
        metrics = self.get_metrics()
        logger.info(f"{prefix} Confusion Matrix:")
        logger.info(f"  True Up: {self.true_up}, False Up: {self.false_up}")
        logger.info(f"  True Down: {self.true_down}, False Down: {self.false_down}")
        logger.info(f"  Direction Accuracy: {metrics['accuracy']:.1f}%")
        logger.info(f"  Up Accuracy: {metrics['up_accuracy']:.1f}%, Down Accuracy: {metrics['down_accuracy']:.1f}%")
        logger.info(f"  Bias: {metrics['bias']:+.1f}%")


def get_db_url():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL environment variable must be set")
    return url.replace("postgresql+asyncpg://", "postgresql://")


def get_ticker_list():
    """Get list of available tickers from DB."""
    import psycopg2
    try:
        with psycopg2.connect(get_db_url()) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ticker, COUNT(*) as count 
                    FROM stock_prices GROUP BY ticker 
                    HAVING COUNT(*) >= %s ORDER BY count DESC LIMIT %s
                """, (CONFIG["MIN_RECORDS"], CONFIG["MAX_STOCKS"]))

 def get_ticker_list():
     import psycopg2
-    conn = psycopg2.connect(get_db_url())
-    cur = conn.cursor()
-    cur.execute("""
-        SELECT ticker, COUNT(*) as count 
-        FROM stock_prices GROUP BY ticker 
-        HAVING COUNT(*) >= %s ORDER BY count DESC LIMIT %s
-    """, (CONFIG["MIN_RECORDS"], CONFIG["MAX_STOCKS"]))
-    results = cur.fetchall()
-    cur.close()
-    conn.close()
-    return [(row[0], row[1]) for row in results]
+    with psycopg2.connect(get_db_url()) as conn:
+        with conn.cursor() as cur:
+            cur.execute("""
+                SELECT ticker, COUNT(*) as count 
+                FROM stock_prices GROUP BY ticker 
+                HAVING COUNT(*) >= %s ORDER BY count DESC LIMIT %s
+            """, (CONFIG["MIN_RECORDS"], CONFIG["MAX_STOCKS"]))
+            results = cur.fetchall()
+    return [(row[0], row[1]) for row in results]                results = cur.fetchall()
        return [(row[0], row[1]) for row in results]
    except Exception as e:
        logger.error(f"Failed to get ticker list: {e}")
        return []


def load_single_stock(ticker: str):
    """Load stock data from DB."""
    import psycopg2
    try:
        with psycopg2.connect(get_db_url()) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT timestamp, open, high, low, close, volume 
                    FROM stock_prices WHERE ticker = %s ORDER BY timestamp
                """, (ticker,))
                records = cur.fetchall()
        
        if len(records) > CONFIG["MAX_RECORDS_PER_STOCK"]:
            records = records[-CONFIG["MAX_RECORDS_PER_STOCK"]:]
        
        if records:
            return pd.DataFrame(records, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    except Exception as e:
        logger.error(f"Failed to load data for {ticker}: {e}")
    return None


def compute_sample_weights(targets: np.ndarray) -> np.ndarray:
    """
    Compute sample weights based on target magnitude.
    High-volatility days get more weight.
    
    weight = 1 + abs(target_1d) * boost_factor
    """
    # Use 1-day target (column 0)
    magnitudes = np.abs(targets[:, 0])
    
    # Normalize magnitudes
    if magnitudes.max() > 0:
        normalized = magnitudes / magnitudes.max()
    else:
        normalized = np.ones_like(magnitudes)
    
    # Weight: min 1, max 10
    weights = 1.0 + 9.0 * normalized
    
    return weights


def process_single_stock(ticker: str, df: pd.DataFrame, feature_cols: list, scaler: StandardScaler):
    """Process one stock with STRICT time-based split."""
    try:
        processed = process_stock_data(df.copy(), create_targets=True)
        
        if len(processed) < CONFIG["WINDOW_SIZE"] + 10:
            return None, None, None, None, None
        
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
            return None, None, None, None, None
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # ========================================================
        # STRICT TIME-SERIES SPLIT (NO SHUFFLE!)
        # Training: First 80% of data (older dates)
        # Validation: Last 20% of data (newer dates)
        # This prevents data leakage from future to past
        # ========================================================
        split_idx = int(len(X) * CONFIG["TRAIN_CUTOFF"])
        
        X_train, y_train = X[:split_idx], y[:split_idx]  # Older data
        X_val, y_val = X[split_idx:], y[split_idx:]      # Newer data (never shuffled!)
        
        # Compute sample weights for training data
        train_weights = compute_sample_weights(y_train)
        
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
            train_weights.astype(np.float32),
            X_val_scaled.astype(np.float32) if X_val_scaled is not None else None,
            y_val.astype(np.float32) if y_val is not None else None
        )
    except Exception as e:
        logger.warning(f"Failed to process {ticker}: {e}")
        return None, None, None, None, None


class MemmapDataset(Dataset):
    """Memory-mapped dataset for large training data."""
    def __init__(self, X_path: Path, y_path: Path, weights_path: Path, n_samples: int, n_features: int):
        self.X = np.memmap(X_path, dtype=np.float32, mode='r', 
                          shape=(n_samples, CONFIG["WINDOW_SIZE"], n_features))
        self.y = np.memmap(y_path, dtype=np.float32, mode='r', shape=(n_samples, 4))
        self.weights = np.memmap(weights_path, dtype=np.float32, mode='r', shape=(n_samples,))
        self.n_samples = n_samples
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx].copy()), 
            torch.from_numpy(self.y[idx].copy()),
            self.weights[idx]
        )


class MemmapDatasetVal(Dataset):
    """Validation dataset (no weights)."""
    def __init__(self, X_path: Path, y_path: Path, n_samples: int, n_features: int):
        self.X = np.memmap(X_path, dtype=np.float32, mode='r', 
                          shape=(n_samples, CONFIG["WINDOW_SIZE"], n_features))
        self.y = np.memmap(y_path, dtype=np.float32, mode='r', shape=(n_samples, 4))
        self.n_samples = n_samples
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx].copy()), torch.from_numpy(self.y[idx].copy())


# Global counters for memmap
_train_offset = 0
_val_offset = 0

def _flush_to_memmap(X_chunks, y_chunks, weights_chunks, prefix: str, n_features: int):
    """Flush data chunks to memory-mapped files."""
    global _train_offset, _val_offset
    
    if not X_chunks:
        return
    
    X = np.vstack(X_chunks)
    y = np.vstack(y_chunks)
    n = len(X)
    
    X_path = TEMP_DATA_DIR / f"{prefix}_X.npy"
    y_path = TEMP_DATA_DIR / f"{prefix}_y.npy"
    weights_path = TEMP_DATA_DIR / f"{prefix}_weights.npy"
    info_path = TEMP_DATA_DIR / f"{prefix}_info.npy"
    
    if prefix == "train":
        weights = np.concatenate(weights_chunks)
        offset = _train_offset
        _train_offset += n
        total = _train_offset
    else:
        weights = None
        offset = _val_offset
        _val_offset += n
        total = _val_offset
    
    if offset == 0:
        X_mm = np.memmap(X_path, dtype=np.float32, mode='w+', 
                        shape=(n, CONFIG["WINDOW_SIZE"], n_features))
        y_mm = np.memmap(y_path, dtype=np.float32, mode='w+', shape=(n, 4))
        if weights is not None:
            w_mm = np.memmap(weights_path, dtype=np.float32, mode='w+', shape=(n,))
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
        
        if weights is not None:
            old_w = np.memmap(weights_path, dtype=np.float32, mode='r', shape=(offset,))
            new_w = np.memmap(weights_path.with_suffix('.tmp'), dtype=np.float32, mode='w+',
                             shape=(total,))
            new_w[:offset] = old_w[:]
            del old_w
            weights_path.unlink()
            weights_path.with_suffix('.tmp').rename(weights_path)
            w_mm = np.memmap(weights_path, dtype=np.float32, mode='r+', shape=(total,))
        
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
    
    if weights is not None:
        w_mm[offset:offset+n] = weights
        w_mm.flush()
    
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
    logger.info("Training v7 - ATTENTION MODEL")
    logger.info(f"Config: {CONFIG}")
    logger.info("="*60)
    
    if TEMP_DATA_DIR.exists():
        shutil.rmtree(TEMP_DATA_DIR)
    TEMP_DATA_DIR.mkdir(parents=True)
    
    tickers = get_ticker_list()
    logger.info(f"Found {len(tickers)} stocks with >= {CONFIG['MIN_RECORDS']} records")
    
    feature_cols = get_model_input_features_v7()
    n_features = len(feature_cols)
    logger.info(f"Using {n_features} features (v7)")
    
    # Phase 1: Fit scaler on training data only
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
    
    # Log scaler statistics for key features
    logger.info(f"✅ Scaler fitted on {len(all_sample):,} training samples ONLY (no validation data)")
    logger.info(f"   Feature count: {len(feature_cols)}")
    logger.info(f"   Mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
    logger.info(f"   Scale range: [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")
    
    del sample_data, all_sample
    gc.collect()
    
    # Phase 2: Process and save
    logger.info("Phase 2: Processing stocks...")
    
    train_X_chunks, train_y_chunks, train_w_chunks = [], [], []
    val_X_chunks, val_y_chunks = [], []
    
    for i, (ticker, _) in enumerate(tickers):
        if (i + 1) % 50 == 0:
            logger.info(f"Processing stock {i+1}/{len(tickers)}...")
        
        df = load_single_stock(ticker)
        if df is None:
            continue
            
        X_train, y_train, w_train, X_val, y_val = process_single_stock(ticker, df, feature_cols, scaler)
        
        if X_train is not None and len(X_train) > 0:
            train_X_chunks.append(X_train)
            train_y_chunks.append(y_train)
            train_w_chunks.append(w_train)
        
        if X_val is not None and len(X_val) > 0:
            val_X_chunks.append(X_val)
            val_y_chunks.append(y_val)
        
        if len(train_X_chunks) >= 50:
            _flush_to_memmap(train_X_chunks, train_y_chunks, train_w_chunks, "train", n_features)
            _flush_to_memmap(val_X_chunks, val_y_chunks, None, "val", n_features)
            train_X_chunks, train_y_chunks, train_w_chunks = [], [], []
            val_X_chunks, val_y_chunks = [], []
            gc.collect()
    
    _flush_to_memmap(train_X_chunks, train_y_chunks, train_w_chunks, "train", n_features)
    _flush_to_memmap(val_X_chunks, val_y_chunks, None, "val", n_features)
    gc.collect()
    
    train_info_path = TEMP_DATA_DIR / "train_info.npy"
    val_info_path = TEMP_DATA_DIR / "val_info.npy"
    if not train_info_path.exists() or not val_info_path.exists():
        logger.error("No training data was generated. Check stock processing logs.")
        shutil.rmtree(TEMP_DATA_DIR)
        sys.exit(1)
        
    train_info = np.load(train_info_path)
    val_info = np.load(val_info_path)
    n_train, n_val = int(train_info[0]), int(val_info[0])
    
    logger.info(f"Training samples: {n_train}, Validation samples: {n_val}")
    
    # Phase 3: Train with weighted sampling
    logger.info("Phase 3: Training ATTENTION model...")
    
    train_dataset = MemmapDataset(
        TEMP_DATA_DIR / "train_X.npy", 
        TEMP_DATA_DIR / "train_y.npy",
        TEMP_DATA_DIR / "train_weights.npy",
        n_train, n_features
    )
    val_dataset = MemmapDatasetVal(
        TEMP_DATA_DIR / "val_X.npy", 
        TEMP_DATA_DIR / "val_y.npy",
        n_val, n_features
    )
    
    # Create weighted sampler
    sample_weights = np.memmap(TEMP_DATA_DIR / "train_weights.npy", dtype=np.float32, mode='r', shape=(n_train,))
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights.copy()),
        num_samples=n_train,
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["BATCH_SIZE"], 
        sampler=sampler,
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
    
    model = LSTMModelV7(
        input_dim=n_features,
        hidden_dim=CONFIG["HIDDEN_DIM"],
        num_layers=CONFIG["NUM_LAYERS"],
        output_dim=4,
        dropout=0.3,
        window_size=CONFIG["WINDOW_SIZE"],
        num_attention_heads=CONFIG["NUM_ATTENTION_HEADS"],
        use_cpu_offload=CONFIG["USE_CPU_OFFLOAD"]
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.enable_checkpointing()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=0.01)
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    
    criterion = DirectionalLoss(
        direction_weight=CONFIG["DIRECTION_WEIGHT"],
        volatility_boost=2.0
    )
    
    device = torch.device('cuda' if use_cuda else 'cpu')
    # Define use_amp for mixed precision training if available
    use_amp = use_cuda  
    grad_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    model.to(device)
    
    confusion_tracker = ConfusionTracker()
    
    best_val_loss = float('inf')
    best_accuracy = 0.0
    patience_counter = 0
    
    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (X, y, _) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            
            with torch.cuda.amp.autocast(enabled=use_cuda):
                out = model(X)
                loss = criterion(out, y) / CONFIG["ACCUMULATION_STEPS"]
            
            grad_scaler.scale(loss).backward()
            
            if (batch_idx + 1) % CONFIG["ACCUMULATION_STEPS"] == 0:
                # Gradient clipping
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["GRADIENT_CLIP_NORM"])
                
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * CONFIG["ACCUMULATION_STEPS"]
        
        train_loss /= len(train_loader)
        
        # Validation with confusion matrix
        model.eval()
        val_loss = 0.0
        confusion_tracker.reset()
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                if use_amp:
                    with autocast('cuda'):
                        out = model(X)
                        loss = criterion(out, y)
                else:
                    out = model(X)
                    loss = criterion(out, y)
                val_loss += loss.item()
                confusion_tracker.update(out, y)
        
        val_loss /= len(val_loader)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        metrics = confusion_tracker.get_metrics()
        
        logger.info(
            f"Epoch {epoch+1}/{CONFIG['EPOCHS']} - "
            f"Train: {train_loss:.6f}, Val: {val_loss:.6f}, "
            f"Acc: {metrics['accuracy']:.1f}%, Bias: {metrics['bias']:+.1f}%, "
            f"LR: {current_lr:.6f}"
        )
        
        # Log confusion matrix every 10 epochs
        if (epoch + 1) % 10 == 0:
            confusion_tracker.log_summary("Val")
        
        # === DUAL CHECKPOINTING STRATEGY ===
        # Priority 1: Best Directional Accuracy (The "Winner" logic)
        # Priority 2: Lowest Validation Loss (The "Safety" logic)
        
        saved_something = False
        
        # 1. Check Accuracy (Primary Goal)
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            # Save as the MAIN model
            model.save(str(MODEL_SAVE_PATH))
            logger.info(f"🏆 NEW BEST ACCURACY: {best_accuracy:.1f}% (saved to v7.pth)")
            # Reset patience only on accuracy improvement (since that's what we want)
            patience_counter = 0
            saved_something = True
            
        # 2. Check Loss (Secondary Goal)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save as the BACKUP/LOSS model
            model.save(str(MODEL_SAVE_PATH_LOSS))
            logger.info(f"📉 NEW BEST LOSS: {best_val_loss:.6f} (saved to v7_loss.pth)")
            # Only reset patience if we want to wait for loss convergence too
            # For trading, we care about accuracy, so we rely on strict accuracy patience?
            # Or we keep standard patience logic: if EITHER improves, we keep going.
            patience_counter = 0
            saved_something = True
            
        if not saved_something:
            patience_counter += 1
            if patience_counter >= CONFIG["PATIENCE"]:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Save scaler
    scaler_data = {'scaler': scaler, 'feature_names': feature_cols, 'version': 'v7'}
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler_data, f)
    
    # Cleanup
    shutil.rmtree(TEMP_DATA_DIR)
    
    logger.info("="*60)
    logger.info("🎉 Training Complete!")
    logger.info(f"Best Val Loss: {best_val_loss:.6f}")
    logger.info(f"Best Accuracy: {best_accuracy:.1f}%")
    logger.info("="*60)
    logger.info("To activate v7 as default:")
    logger.info("  docker exec proxmox_stock_backend cp /app/models/lstm_model_v7.pth /app/models/lstm_model_v2.pth")
    logger.info("  docker exec proxmox_stock_backend cp /app/models/scaler_v7.pkl /app/models/scaler_v2.pkl")
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
