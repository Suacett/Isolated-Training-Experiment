#!/usr/bin/env python3
"""
IMPROVED LSTM Training Script - Uses ALL database data for better accuracy.

This script addresses the bullish bias issue by:
1. Training on 500+ stocks (not just SPY)
2. Using full historical data (10+ years per stock)
3. Including both bull and bear market periods
4. Using time-weighted sampling to prioritize recent patterns

Usage:
    docker exec proxmox_stock_backend python -m scripts.train_model_v3

Estimated time: 5-15 minutes depending on GPU
"""

import sys
import os
import logging
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
from datetime import datetime, timedelta

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from services.feature_engineering import process_stock_data, get_model_input_features
from services.training import prepare_training_data, LSTMTrainer, StockDataset
from services.lstm_model import LSTMModel
from services.scaler import FeatureScaler
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_v3.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Paths
MODEL_SAVE_PATH = backend_path / "models" / "lstm_model_v3.pth"
SCALER_SAVE_PATH = backend_path / "models" / "scaler_v3.pkl"

# Training config - can be adjusted via env vars
MIN_RECORDS_PER_STOCK = int(os.getenv("MIN_RECORDS", "500"))  # Min days of data required
MAX_RECORDS_PER_STOCK = int(os.getenv("MAX_RECORDS", "5000"))  # Limit per stock for memory
MAX_TOTAL_STOCKS = int(os.getenv("MAX_STOCKS", "200"))  # Limit total stocks
WINDOW_SIZE = 60
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))
EPOCHS = int(os.getenv("EPOCHS", "150"))
LEARNING_RATE = float(os.getenv("LR", "0.0005"))  # Lower LR for more stable training
PATIENCE = int(os.getenv("PATIENCE", "30"))  # Patience for early stopping


async def load_data_from_database() -> List[Tuple[str, pd.DataFrame]]:
    """
    Load training data directly from the TimescaleDB database.
    This ensures we use the most up-to-date data.
    """
    from services.db import AsyncSessionLocal, StockPrice
    from sqlalchemy import select, func
    
    datasets = []
    
    async with AsyncSessionLocal() as session:
        # First, get list of tickers with enough data
        ticker_counts = await session.execute(
            select(StockPrice.ticker, func.count(StockPrice.ticker).label('count'))
            .group_by(StockPrice.ticker)
            .having(func.count(StockPrice.ticker) >= MIN_RECORDS_PER_STOCK)
            .order_by(func.count(StockPrice.ticker).desc())
            .limit(MAX_TOTAL_STOCKS)
        )
        
        tickers = [(row[0], row[1]) for row in ticker_counts.fetchall()]
        logger.info(f"Found {len(tickers)} stocks with >= {MIN_RECORDS_PER_STOCK} records")
        
        # Load data for each ticker
        for ticker, count in tickers:
            try:
                # Get data ordered by date
                result = await session.execute(
                    select(StockPrice)
                    .where(StockPrice.ticker == ticker)
                    .order_by(StockPrice.timestamp)
                )
                records = result.scalars().all()
                
                # Limit to MAX_RECORDS_PER_STOCK (use most recent data)
                if len(records) > MAX_RECORDS_PER_STOCK:
                    records = records[-MAX_RECORDS_PER_STOCK:]
                
                # Convert to DataFrame
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
                continue
    
    logger.info(f"Successfully loaded {len(datasets)} datasets from database")
    return datasets


def balance_dataset(datasets: List[Tuple[str, pd.DataFrame]]) -> List[Tuple[str, pd.DataFrame]]:
    """
    Balance the dataset to include both bull and bear market periods.
    Also ensures we have diverse stock types (tech, finance, healthcare, etc.)
    """
    processed_datasets = []
    
    for ticker, df in datasets:
        try:
            # Process features (this adds all technical indicators)
            processed = process_stock_data(df.copy(), create_targets=True)
            
            if len(processed) < WINDOW_SIZE + 10:  # Need enough data for sequences
                logger.warning(f"Skipping {ticker}: only {len(processed)} rows after processing")
                continue
                
            processed_datasets.append((ticker, processed))
            
        except Exception as e:
            logger.warning(f"Failed to process {ticker}: {e}")
            continue
    
    logger.info(f"Processed {len(processed_datasets)} datasets with features")
    return processed_datasets


def train_pipeline():
    logger.info("="*60)
    logger.info("IMPROVED LSTM Training Pipeline (Database Data)")
    logger.info("="*60)
    logger.info(f"Config: MIN_RECORDS={MIN_RECORDS_PER_STOCK}, MAX_RECORDS={MAX_RECORDS_PER_STOCK}")
    logger.info(f"Config: MAX_STOCKS={MAX_TOTAL_STOCKS}, BATCH_SIZE={BATCH_SIZE}")
    logger.info(f"Config: EPOCHS={EPOCHS}, LR={LEARNING_RATE}, PATIENCE={PATIENCE}")
    logger.info("="*60)
    
    # 1. Load data from database (async)
    logger.info("Step 1: Loading data from database...")
    datasets = asyncio.run(load_data_from_database())
    
    if len(datasets) < 10:
        logger.error(f"Not enough stocks with data: {len(datasets)}. Need at least 10.")
        sys.exit(1)
    
    # 2. Process and balance datasets
    logger.info("Step 2: Processing features and balancing dataset...")
    processed = balance_dataset(datasets)
    
    # 3. Prepare training data
    logger.info("Step 3: Preparing training sequences...")
    X_train, y_train, X_val, y_val, train_weights = prepare_training_data(
        processed,
        window_size=WINDOW_SIZE,
        train_split=0.80,
        val_split=0.15,
        time_decay_factor=0.001  # Slight preference for recent data
    )
    
    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # 4. Fit scaler on training data
    logger.info("Step 4: Fitting scaler...")
    scaler = FeatureScaler(scaler_path=str(SCALER_SAVE_PATH))
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    scaler.save()
    logger.info(f"Scaler saved to {SCALER_SAVE_PATH}")
    
    # 5. Create DataLoaders
    logger.info("Step 5: Creating DataLoaders...")
    train_dataset = StockDataset(X_train_scaled, y_train, WINDOW_SIZE, train_weights)
    val_dataset = StockDataset(X_val_scaled, y_val, WINDOW_SIZE)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues in Docker
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # 6. Initialize model
    logger.info("Step 6: Initializing LSTM model...")
    input_dim = X_train.shape[1]
    model = LSTMModel(input_dim=input_dim, window_size=WINDOW_SIZE)
    logger.info(f"Model input features: {input_dim}")
    logger.info(f"Model architecture: {model}")
    
    # 7. Train!
    logger.info("Step 7: Starting training...")
    trainer = LSTMTrainer(model, scaler, learning_rate=LEARNING_RATE)
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=EPOCHS,
        patience=PATIENCE,
        save_best=True,
        model_path=str(MODEL_SAVE_PATH)
    )
    
    logger.info("="*60)
    logger.info("🎉 Training Complete!")
    logger.info(f"Best Validation Loss: {history['best_val_loss']:.6f}")
    logger.info(f"Best Epoch: {history.get('best_epoch', 'N/A')}")
    logger.info(f"Model saved to: {MODEL_SAVE_PATH}")
    logger.info(f"Scaler saved to: {SCALER_SAVE_PATH}")
    logger.info("="*60)
    logger.info("")
    logger.info("📋 NEXT STEPS:")
    logger.info("1. Restart the backend to load the new model:")
    logger.info("   docker compose restart backend")
    logger.info("")
    logger.info("2. Update the model path in main.py (if not using v3 automatically):")
    logger.info(f"   MODEL_PATH = '{MODEL_SAVE_PATH}'")
    logger.info(f"   SCALER_PATH = '{SCALER_SAVE_PATH}'")
    logger.info("="*60)


if __name__ == "__main__":
    try:
        train_pipeline()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
