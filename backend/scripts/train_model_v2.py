#!/usr/bin/env python3
"""
Script to train the LSTM model using the NEW S&P 500 dataset.

Usage:
    docker exec proxmox_stock_backend python -m scripts.train_model_v2
"""

import sys
import logging
import pandas as pd
from pathlib import Path
from typing import List, Tuple

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from services.training import prepare_training_data, LSTMTrainer
from services.lstm_model import LSTMModel
from services.scaler import FeatureScaler
from torch.utils.data import DataLoader
from services.training import StockDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_v2.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

DATA_DIR = backend_path / "data" / "training_raw"
MODEL_SAVE_PATH = backend_path / "models" / "lstm_model_v2.pth"
SCALER_SAVE_PATH = backend_path / "models" / "scaler_v2.pkl"


def load_new_training_data(data_dir: Path) -> List[Tuple[str, pd.DataFrame]]:
    """
    Load new training data CSVs ({ticker}.csv).
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    stock_files = list(data_dir.glob("*.csv"))
    if not stock_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    logger.info(f"Found {len(stock_files)} stock files in {data_dir}")

    datasets = []
    for file_path in stock_files:
        ticker = file_path.stem  # No _daily suffix anymore
        try:
            df = pd.read_csv(file_path, parse_dates=['date'])
            # Ensure columns are lowercase (should be already, but just in case)
            df.columns = [c.lower() for c in df.columns]
            datasets.append((ticker, df))
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")

    logger.info(f"Successfully loaded {len(datasets)} datasets")
    return datasets


def train_pipeline():
    logger.info("="*50)
    logger.info("Starting LSTM Training Pipeline (New Dataset)")
    logger.info("="*50)

    # 1. Load data
    datasets = load_new_training_data(DATA_DIR)

    # 2. Prepare training data
    # Using smaller batch size for larger dataset
    window_size = 60
    batch_size = 64  # Reduced from 128 for memory
    epochs = 100     # Reduced - more data means fewer epochs needed
    learning_rate = 0.001

    # Limit to newer data to reduce memory (last 5 years per stock = ~1250 days)
    MAX_ROWS_PER_STOCK = 1250
    limited_datasets = []
    for ticker, df in datasets:
        if len(df) > MAX_ROWS_PER_STOCK:
            df = df.tail(MAX_ROWS_PER_STOCK).reset_index(drop=True)
        limited_datasets.append((ticker, df))
    
    logger.info(f"Limited datasets to last {MAX_ROWS_PER_STOCK} rows each")

    X_train, y_train, X_val, y_val, train_weights = prepare_training_data(
        limited_datasets,
        window_size=window_size
    )

    # 3. Fit scaler
    scaler = FeatureScaler(scaler_path=str(SCALER_SAVE_PATH))
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    scaler.save()
    logger.info(f"Scaler saved to {SCALER_SAVE_PATH}")

    # 4. Create datasets/loaders
    train_dataset = StockDataset(X_train_scaled, y_train, window_size, train_weights)
    val_dataset = StockDataset(X_val_scaled, y_val, window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 5. Initialize model
    input_dim = X_train.shape[1]
    model = LSTMModel(input_dim=input_dim, window_size=window_size)
    logger.info(f"Model initialized with {input_dim} input features")

    # 6. Train
    trainer = LSTMTrainer(model, scaler, learning_rate=learning_rate)
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        patience=25,
        save_best=True,
        model_path=str(MODEL_SAVE_PATH)
    )

    logger.info("="*50)
    logger.info("Training Complete!")
    logger.info(f"Best Val Loss: {history['best_val_loss']:.6f}")
    logger.info(f"Model saved to: {MODEL_SAVE_PATH}")
    logger.info("="*50)


if __name__ == "__main__":
    try:
        train_pipeline()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
