#!/usr/bin/env python3
"""
V9 Training Script - Transformer Relative Strength Ranking
MEMORY-OPTIMIZED VERSION with Lazy Loading

This version uses LazyRankDataset to avoid OOM by:
- Storing only raw DataFrames in memory (~500MB)
- Slicing windows on-demand in DataLoader workers
- Using num_workers=4 for parallel window creation

Usage:
    docker exec proxmox_stock_backend python -m scripts.train_model_v9

Memory Footprint:
    - Raw Data: ~500MB (500 stocks × 5000 days × 20 cols × 8 bytes)
    - Batch: ~60MB (256 × 60 × 12 × 4 bytes)
    - Model: ~2MB
    - Total: <2GB RAM, <1GB VRAM per batch
"""

import sys
import os
import gc
import logging
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict, Optional

backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from services.feature_engineering_v9 import (
    compute_v9_features,
    compute_future_return,
    compute_cross_sectional_ranks,
    process_spy_data,
    process_vix_data,
    get_v9_feature_names,
    V9_FEATURE_NAMES,
    N_FEATURES,
)
from services.transformer_model import TransformerRankModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_v9.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Architecture
    "WINDOW_SIZE": 60,
    "D_MODEL": 64,
    "NHEAD": 4,
    "NUM_LAYERS": 2,
    "DIM_FEEDFORWARD": 256,
    "DROPOUT": 0.2,
    
    # Training
    "BATCH_SIZE": int(os.getenv("BATCH_SIZE", "256")),
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 1e-4,
    "EPOCHS": int(os.getenv("EPOCHS", "100")),
    "PATIENCE": 15,
    "TRAIN_CUTOFF": 0.8,
    
    # Data - Memory Optimized
    "STRIDE": 1,
    "MAX_STOCKS": int(os.getenv("MAX_STOCKS", "9999")),
    "MIN_RECORDS": 500,
    "FUTURE_HORIZON": 5,
    "NUM_WORKERS": 4,  # Parallel data loading
}

MODEL_SAVE_PATH = backend_path / "models" / "transformer_v9.pth"
MODEL_SAVE_PATH_BEST = backend_path / "models" / "transformer_v9_best.pth"
SCALER_SAVE_PATH = backend_path / "models" / "scaler_v9.pkl"


# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def get_db_url():
    url = os.getenv("DATABASE_URL", "")
    return url.replace("postgresql+asyncpg://", "postgresql://")


def get_ticker_list():
    """Get all tickers with sufficient history."""
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


def load_single_stock(ticker: str) -> Optional[pd.DataFrame]:
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


def load_macro_data():
    """Load SPY and VIX data for macro context."""
    spy_df = load_single_stock('SPY')
    vix_df = load_single_stock('^VIX')
    
    if vix_df is None:
        vix_df = load_single_stock('VIX')
    
    spy_data = None
    vix_data = None
    
    if spy_df is not None:
        spy_data = process_spy_data(spy_df)
        logger.info(f"✅ Loaded SPY data: {len(spy_df)} days")
    else:
        logger.warning("⚠️ SPY data not found")
    
    if vix_df is not None:
        vix_data = process_vix_data(vix_df)
        logger.info(f"✅ Loaded VIX data: {len(vix_df)} days")
    else:
        logger.warning("⚠️ VIX data not found")
    
    return spy_data, vix_data


# =============================================================================
# LAZY LOADING DATASET
# =============================================================================

class LazyRankDataset(Dataset):
    """
    Memory-efficient dataset that creates windows on-demand.
    
    Instead of pre-allocating all windows (~20GB), we:
    1. Store processed DataFrames for each stock (~500MB total)
    2. Store an index mapping: [(stock_idx, row_start), ...]
    3. Create windows on-the-fly in __getitem__
    
    This reduces memory usage by ~40x while only adding ~10% training time
    due to parallel DataLoader workers.
    """
    
    def __init__(
        self,
        stock_data: List[np.ndarray],  # List of (N, features) arrays per stock
        target_data: List[np.ndarray],  # List of (N,) target arrays per stock
        index_mapping: List[Tuple[int, int]],  # [(stock_idx, row_idx), ...]
        window_size: int = 60,
    ):
        """
        Args:
            stock_data: List of feature arrays, one per stock
            target_data: List of target rank arrays, one per stock
            index_mapping: List of (stock_index, row_index) tuples
            window_size: Number of timesteps in each window
        """
        self.stock_data = stock_data
        self.target_data = target_data
        self.index_mapping = index_mapping
        self.window_size = window_size
        
    def __len__(self) -> int:
        return len(self.index_mapping)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a single window on-demand.
        
        Returns:
            X: Window tensor of shape (window_size, n_features)
            y: Target rank scalar
        """
        stock_idx, row_idx = self.index_mapping[idx]
        
        # Slice window from stock data
        window = self.stock_data[stock_idx][row_idx:row_idx + self.window_size]
        target = self.target_data[stock_idx][row_idx + self.window_size - 1]
        
        # Convert to tensors
        X = torch.from_numpy(window).float()
        y = torch.tensor([target], dtype=torch.float32)
        
        return X, y


# =============================================================================
# DATA PROCESSING
# =============================================================================

def process_all_stocks(
    tickers: List[Tuple[str, int]],
    spy_data: Optional[Dict],
    vix_data: Optional[pd.Series],
) -> Tuple[List[pd.DataFrame], StandardScaler]:
    """
    Process all stocks and fit scaler.
    
    Returns:
        stock_dfs: List of processed DataFrames (one per stock)
        scaler: Fitted StandardScaler
    """
    stock_dfs = []
    scaler_samples = []
    
    for i, (ticker, _) in enumerate(tickers):
        if (i + 1) % 100 == 0:
            logger.info(f"Processing stock {i+1}/{len(tickers)}...")
        
        df = load_single_stock(ticker)
        if df is None:
            continue
        
        try:
            # Compute V9 features
            df = compute_v9_features(df, spy_data, vix_data)
            
            # Compute future return for ranking
            df = compute_future_return(df, horizon=CONFIG["FUTURE_HORIZON"])
            
            # Add ticker
            df['ticker'] = ticker
            
            # Drop rows without features
            df = df.dropna(subset=V9_FEATURE_NAMES + ['future_ret_5d'])
            
            if len(df) > CONFIG["WINDOW_SIZE"] + 10:
                stock_dfs.append(df)
                # Sample for scaler (first 80% = training data)
                train_end = int(len(df) * CONFIG["TRAIN_CUTOFF"])
                scaler_samples.append(df[V9_FEATURE_NAMES].values[:train_end])
        except Exception as e:
            logger.warning(f"Failed to process {ticker}: {e}")
            continue
    
    logger.info(f"Successfully processed {len(stock_dfs)} stocks")
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    all_samples = np.vstack(scaler_samples)
    all_samples = np.nan_to_num(all_samples, nan=0.0, posinf=1e6, neginf=-1e6)
    scaler.fit(all_samples)
    
    del scaler_samples, all_samples
    gc.collect()
    
    return stock_dfs, scaler


def compute_global_ranks(stock_dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Compute cross-sectional ranks across all stocks for each date.
    """
    # Combine all stocks temporarily for ranking
    all_data = pd.concat(stock_dfs, ignore_index=True)
    
    # Compute cross-sectional ranks
    all_data = compute_cross_sectional_ranks(all_data)
    
    # Split back into individual stocks
    result_dfs = []
    for ticker in [df['ticker'].iloc[0] for df in stock_dfs]:
        ticker_df = all_data[all_data['ticker'] == ticker].copy()
        result_dfs.append(ticker_df.sort_values('date').reset_index(drop=True))
    
    del all_data
    gc.collect()
    
    return result_dfs


def prepare_lazy_datasets(
    stock_dfs: List[pd.DataFrame],
    scaler: StandardScaler,
) -> Tuple[LazyRankDataset, LazyRankDataset]:
    """
    Prepare training and validation datasets using lazy loading.
    
    Memory efficient: Only stores numpy arrays, creates windows on-demand.
    """
    window_size = CONFIG["WINDOW_SIZE"]
    stride = CONFIG["STRIDE"]
    train_cutoff = CONFIG["TRAIN_CUTOFF"]
    
    # Convert DataFrames to numpy arrays and build index mappings
    train_stock_data = []
    train_target_data = []
    train_index_mapping = []
    
    val_stock_data = []
    val_target_data = []
    val_index_mapping = []
    
    for stock_idx, df in enumerate(stock_dfs):
        # Scale features
        features = df[V9_FEATURE_NAMES].values
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        features = scaler.transform(features)
        features = np.clip(features, -10, 10).astype(np.float32)
        
        targets = df['target_rank'].values.astype(np.float32)
        
        # Time-based split
        split_idx = int(len(df) * train_cutoff)
        
        # Training indices
        train_features = features[:split_idx]
        train_targets = targets[:split_idx]
        
        if len(train_features) > window_size:
            train_stock_data.append(train_features)
            train_target_data.append(train_targets)
            train_stock_idx = len(train_stock_data) - 1
            
            # Create index mapping for this stock (stride=1)
            for row_start in range(0, len(train_features) - window_size, stride):
                train_index_mapping.append((train_stock_idx, row_start))
        
        # Validation indices
        val_features = features[split_idx:]
        val_targets = targets[split_idx:]
        
        if len(val_features) > window_size:
            val_stock_data.append(val_features)
            val_target_data.append(val_targets)
            val_stock_idx = len(val_stock_data) - 1
            
            for row_start in range(0, len(val_features) - window_size, stride):
                val_index_mapping.append((val_stock_idx, row_start))
    
    logger.info(f"Training samples: {len(train_index_mapping):,}")
    logger.info(f"Validation samples: {len(val_index_mapping):,}")
    
    # Calculate memory usage
    train_mem = sum(arr.nbytes for arr in train_stock_data) / 1e6
    val_mem = sum(arr.nbytes for arr in val_stock_data) / 1e6
    logger.info(f"Memory usage: Train={train_mem:.1f}MB, Val={val_mem:.1f}MB")
    
    train_dataset = LazyRankDataset(
        stock_data=train_stock_data,
        target_data=train_target_data,
        index_mapping=train_index_mapping,
        window_size=window_size,
    )
    
    val_dataset = LazyRankDataset(
        stock_data=val_stock_data,
        target_data=val_target_data,
        index_mapping=val_index_mapping,
        window_size=window_size,
    )
    
    return train_dataset, val_dataset


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    n_batches = 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                n_batches += 1
                all_preds.append(preds.cpu())
                all_targets.append(y_batch.cpu())
    
    if not all_preds:
        return float('inf'), 0.0
    
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    
    # Compute correlation
    correlation = np.corrcoef(preds.flatten(), targets.flatten())[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    
    return total_loss / max(n_batches, 1), correlation


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def train_pipeline():
    """Main training pipeline for V9 Transformer with lazy loading."""
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🚀 GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    else:
        logger.warning("⚠️ No GPU detected")
    
    logger.info("=" * 70)
    logger.info("V9 TRANSFORMER - MEMORY OPTIMIZED (Lazy Loading)")
    logger.info("=" * 70)
    logger.info(f"Window Size: {CONFIG['WINDOW_SIZE']} days")
    logger.info(f"Features: {N_FEATURES}")
    logger.info(f"Batch Size: {CONFIG['BATCH_SIZE']}")
    logger.info(f"DataLoader Workers: {CONFIG['NUM_WORKERS']}")
    logger.info("=" * 70)
    
    # Load tickers
    tickers = get_ticker_list()
    logger.info(f"Found {len(tickers)} stocks")
    
    # Load macro data
    spy_data, vix_data = load_macro_data()
    
    # Process all stocks and fit scaler
    logger.info("\n📊 Phase 1: Processing all stocks...")
    stock_dfs, scaler = process_all_stocks(tickers, spy_data, vix_data)
    
    # Compute global ranks
    logger.info("\n📊 Phase 2: Computing cross-sectional ranks...")
    stock_dfs = compute_global_ranks(stock_dfs)
    
    # Prepare lazy datasets
    logger.info("\n📊 Phase 3: Creating lazy datasets...")
    train_dataset, val_dataset = prepare_lazy_datasets(stock_dfs, scaler)
    
    # Free DataFrames - we only need numpy arrays now
    del stock_dfs
    gc.collect()
    
    # Create data loaders with parallel workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True,
        num_workers=CONFIG["NUM_WORKERS"],
        pin_memory=True,
        persistent_workers=True if CONFIG["NUM_WORKERS"] > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=False,
        num_workers=CONFIG["NUM_WORKERS"],
        pin_memory=True,
        persistent_workers=True if CONFIG["NUM_WORKERS"] > 0 else False,
    )
    
    # Create model
    model = TransformerRankModel(
        input_dim=N_FEATURES,
        d_model=CONFIG["D_MODEL"],
        nhead=CONFIG["NHEAD"],
        num_layers=CONFIG["NUM_LAYERS"],
        dim_feedforward=CONFIG["DIM_FEEDFORWARD"],
        dropout=CONFIG["DROPOUT"],
        device=device
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["LEARNING_RATE"],
        weight_decay=CONFIG["WEIGHT_DECAY"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    logger.info("\n🚀 Starting training...")
    best_val_loss = float('inf')
    best_correlation = 0.0
    patience_counter = 0
    
    for epoch in range(CONFIG["EPOCHS"]):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, correlation = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(
            f"Epoch {epoch+1}/{CONFIG['EPOCHS']} - "
            f"Train: {train_loss:.6f}, Val: {val_loss:.6f}, "
            f"Corr: {correlation:.4f}, LR: {current_lr:.6f}"
        )
        
        # Checkpointing
        saved = False
        
        if correlation > best_correlation:
            best_correlation = correlation
            model.save(str(MODEL_SAVE_PATH_BEST))
            logger.info(f"🏆 NEW BEST CORRELATION: {best_correlation:.4f}")
            patience_counter = 0
            saved = True
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(str(MODEL_SAVE_PATH))
            logger.info(f"📉 NEW BEST LOSS: {best_val_loss:.6f}")
            patience_counter = 0
            saved = True
        
        if not saved:
            patience_counter += 1
            if patience_counter >= CONFIG["PATIENCE"]:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Save scaler
    scaler_data = {
        'scaler': scaler,
        'feature_names': V9_FEATURE_NAMES,
        'version': 'v9-lazy'
    }
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler_data, f)
    
    logger.info("=" * 70)
    logger.info("🎉 TRAINING COMPLETE!")
    logger.info(f"Best Validation Loss: {best_val_loss:.6f}")
    logger.info(f"Best Correlation: {best_correlation:.4f}")
    logger.info(f"Model: {MODEL_SAVE_PATH}")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        train_pipeline()
    except KeyboardInterrupt:
        logger.info("Training interrupted")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
