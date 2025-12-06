#!/usr/bin/env python3
"""
Script to train the LSTM model using legacy training data.

Usage:
    python backend/scripts/train_model.py
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

import logging
from services.training import train_lstm_from_legacy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backend/training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Starting LSTM model training from legacy data")

    try:
        history = train_lstm_from_legacy(
            legacy_data_dir="legacy/LSTM_AI_Stock_Predictor/TrainingData/indicators_data/raw/stocksData",
            model_save_path="backend/models/lstm_model.pth",
            scaler_save_path="backend/models/scaler.pkl",
            window_size=60,
            batch_size=128,
            epochs=200,
            learning_rate=0.001
        )

        logger.info("\nTraining Summary:")
        logger.info(f"Best validation loss: {history['best_val_loss']:.6f}")
        logger.info(f"Best epoch: {history['best_epoch']}")
        logger.info(f"Total epochs: {len(history['train_losses'])}")

        print("\n" + "="*50)
        print("Training completed successfully!")
        print(f"Model saved to: backend/models/lstm_model.pth")
        print(f"Scaler saved to: backend/models/scaler.pkl")
        print("="*50)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
