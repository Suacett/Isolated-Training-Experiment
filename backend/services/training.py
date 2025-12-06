"""
Training pipeline for the LSTM model using legacy training data.

This module implements the training logic ported from the legacy notebook:
legacy/LSTM_AI_Stock_Predictor/forecast.ipynb
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from datetime import datetime

from services.lstm_model import LSTMModel, get_device
from services.scaler import FeatureScaler
from services.feature_engineering import (
    process_stock_data,
    get_model_input_features,
    get_feature_columns
)

logger = logging.getLogger(__name__)


class StockDataset(Dataset):
    """
    PyTorch Dataset for stock data windows.

    Creates sliding windows of historical data for time series prediction.
    """

    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        window_size: int = 60,
        time_weights: Optional[np.ndarray] = None
    ):
        """
        Args:
            data: Feature matrix (n_samples, n_features)
            targets: Target matrix (n_samples, n_horizons)
            window_size: Number of timesteps in each window
            time_weights: Optional weights for time-based weighting
        """
        self.data = data
        self.targets = targets
        self.window_size = window_size
        self.time_weights = time_weights

        # Create windows
        self.windows, self.window_targets, self.weights = self._create_windows()

    def _create_windows(self):
        """Create sliding windows from data"""
        windows = []
        targets = []
        weights = []

        for i in range(len(self.data) - self.window_size):
            # Window: [i:i+window_size]
            window = self.data[i:i + self.window_size]
            target = self.targets[i + self.window_size]

            windows.append(window)
            targets.append(target)

            if self.time_weights is not None:
                weights.append(self.time_weights[i + self.window_size])
            else:
                weights.append(1.0)

        return (
            np.array(windows, dtype=np.float32),
            np.array(targets, dtype=np.float32),
            np.array(weights, dtype=np.float32)
        )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return (
            self.windows[idx],
            self.window_targets[idx],
            self.weights[idx]
        )


class LSTMTrainer:
    """Handles training of the LSTM model"""

    def __init__(
        self,
        model: LSTMModel,
        scaler: FeatureScaler,
        learning_rate: float = 0.001,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.scaler = scaler
        self.device = device if device else get_device()
        self.model.to(self.device)

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss(reduction='none')  # Per-sample loss for weighting

        self.train_losses = []
        self.val_losses = []

    def train_epoch(
        self,
        train_loader: DataLoader,
        use_time_weighting: bool = True
    ) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for X_batch, y_batch, weights_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            weights_batch = weights_batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            # X_batch shape: (batch, window_size, features)
            # Model handles the transpose internally, no need to transpose here
            predictions = self.model(X_batch)

            # Compute loss
            loss_per_sample = self.criterion(predictions, y_batch).mean(dim=1)  # Average over horizons

            if use_time_weighting:
                weighted_loss = (loss_per_sample * weights_batch).mean()
            else:
                weighted_loss = loss_per_sample.mean()

            # Backward pass
            weighted_loss.backward()
            self.optimizer.step()

            total_loss += weighted_loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self, val_loader: DataLoader) -> float:
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for X_batch, y_batch, _ in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Model handles transpose internally
                predictions = self.model(X_batch)

                loss = self.criterion(predictions, y_batch).mean()
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 200,
        patience: int = 25,
        save_best: bool = True,
        model_path: str = "backend/models/lstm_model.pth"
    ) -> dict:
        """
        Train the model with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            patience: Early stopping patience
            save_best: Whether to save best model
            model_path: Path to save model

        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0

        logger.info(f"Starting training for up to {epochs} epochs")
        logger.info(f"Early stopping patience: {patience}")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, use_time_weighting=True)
            val_loss = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0

                if save_best:
                    self.model.save(model_path)
                    logger.info(f"Saved best model with val_loss={val_loss:.6f}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                logger.info(f"Best epoch was {best_epoch+1} with val_loss={best_val_loss:.6f}")
                break

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss
        }


def load_legacy_training_data(
    legacy_data_dir: str = "legacy/LSTM_AI_Stock_Predictor/TrainingData/indicators_data/raw/stocksData"
) -> List[Tuple[str, pd.DataFrame]]:
    """
    Load legacy training data CSVs.

    Returns:
        List of (ticker, dataframe) tuples
    """
    data_dir = Path(legacy_data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Legacy data directory not found: {data_dir}")

    stock_files = list(data_dir.glob("*_daily.csv"))
    if not stock_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    logger.info(f"Found {len(stock_files)} stock files in legacy data")

    datasets = []
    for file_path in stock_files:
        ticker = file_path.stem.replace("_daily", "")
        try:
            df = pd.read_csv(file_path, parse_dates=['date'])
            datasets.append((ticker, df))
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")

    logger.info(f"Successfully loaded {len(datasets)} datasets")
    return datasets


def prepare_training_data(
    datasets: List[Tuple[str, pd.DataFrame]],
    window_size: int = 60,
    train_split: float = 0.80,
    val_split: float = 0.16,
    time_decay_factor: float = 0.002
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare training data from multiple stocks.

    Args:
        datasets: List of (ticker, dataframe) tuples
        window_size: Window size for sequences
        train_split: Training data fraction
        val_split: Validation data fraction
        time_decay_factor: Exponential decay for time weighting

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, time_weights)
    """
    all_features = []
    all_targets = []
    all_weights = []

    target_columns = ['Target_1d', 'Target_1w', 'Target_1m', 'Target_6m']
    feature_columns = get_model_input_features()

    logger.info(f"Processing {len(datasets)} stocks for training data preparation")

    for ticker, raw_df in datasets:
        try:
            # Process stock data (adds technical indicators)
            # Note: For legacy data, insider and sentiment should already be in the CSV
            # If not, we pass None and they'll be filled with defaults
            processed_df = process_stock_data(raw_df, create_targets=True)

            if len(processed_df) < window_size + 126:  # Need enough data for 6m target
                logger.warning(f"Skipping {ticker}: insufficient data ({len(processed_df)} rows)")
                continue

            # Extract features and targets
            features = processed_df[feature_columns].values
            targets = processed_df[target_columns].values

            # Time-based weighting (exponential decay, more recent = higher weight)
            n_samples = len(features)
            weights = np.exp(time_decay_factor * np.arange(n_samples))
            weights = weights / weights.sum()  # Normalize

            all_features.append(features)
            all_targets.append(targets)
            all_weights.append(weights)

        except Exception as e:
            logger.error(f"Failed to process {ticker}: {e}")
            continue

    if not all_features:
        raise ValueError("No valid training data could be prepared")

    # Concatenate all stocks
    X = np.vstack(all_features)
    y = np.vstack(all_targets)
    weights = np.concatenate(all_weights)

    logger.info(f"Total samples: {len(X)}, Features: {X.shape[1]}")

    # Train/val split (time-based, not random)
    n_samples = len(X)
    train_end = int(n_samples * train_split)
    val_end = int(n_samples * (train_split + val_split))

    X_train = X[:train_end]
    y_train = y[:train_end]
    train_weights = weights[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    logger.info(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    return X_train, y_train, X_val, y_val, train_weights


def train_lstm_from_legacy(
    legacy_data_dir: str = "legacy/LSTM_AI_Stock_Predictor/TrainingData/indicators_data/raw/stocksData",
    model_save_path: str = "backend/models/lstm_model.pth",
    scaler_save_path: str = "backend/models/scaler.pkl",
    window_size: int = 60,
    batch_size: int = 128,
    epochs: int = 200,
    learning_rate: float = 0.001
) -> dict:
    """
    Complete training pipeline using legacy data.

    Args:
        legacy_data_dir: Path to legacy CSV files
        model_save_path: Path to save trained model
        scaler_save_path: Path to save scaler
        window_size: Sequence length
        batch_size: Training batch size
        epochs: Maximum epochs
        learning_rate: Learning rate

    Returns:
        Training history dictionary
    """
    logger.info("="*50)
    logger.info("Starting LSTM Training Pipeline")
    logger.info("="*50)

    # 1. Load legacy data
    datasets = load_legacy_training_data(legacy_data_dir)

    # 2. Prepare training data
    X_train, y_train, X_val, y_val, train_weights = prepare_training_data(
        datasets,
        window_size=window_size
    )

    # 3. Fit scaler on training data
    scaler = FeatureScaler(scaler_path=scaler_save_path)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    scaler.save()

    logger.info("Scaler fitted and saved")

    # 4. Create datasets and loaders
    train_dataset = StockDataset(X_train_scaled, y_train, window_size, train_weights)
    val_dataset = StockDataset(X_val_scaled, y_val, window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Created dataloaders: Train={len(train_dataset)}, Val={len(val_dataset)}")

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
        model_path=model_save_path
    )

    logger.info("="*50)
    logger.info("Training Complete!")
    logger.info(f"Best Val Loss: {history['best_val_loss']:.6f}")
    logger.info(f"Model saved to: {model_save_path}")
    logger.info(f"Scaler saved to: {scaler_save_path}")
    logger.info("="*50)

    return history
