"""
StandardScaler wrapper for feature normalization.

Handles fitting, saving, and loading of the scaler used for preprocessing.
CRITICAL: The same scaler must be used for training and inference.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


class FeatureScaler:
    """
    Wrapper around StandardScaler for feature normalization.

    Ensures consistent scaling between training and inference.
    """

    def __init__(self, scaler_path: str = "backend/models/scaler.pkl"):
        self.scaler_path = Path(scaler_path)
        self.scaler = None
        self.is_fitted = False
        self.feature_names = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], feature_names: Optional[list] = None):
        """
        Fit the scaler on training data.

        Args:
            X: Training data (n_samples, n_features)
            feature_names: List of feature names (optional, for validation)
        """
        from sklearn.preprocessing import StandardScaler

        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values

        self.scaler = StandardScaler()
        self.scaler.fit(X)
        self.is_fitted = True
        self.feature_names = feature_names

        logger.info(f"Scaler fitted on {X.shape[0]} samples with {X.shape[1]} features")

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform data using fitted scaler.

        Args:
            X: Data to transform

        Returns:
            Scaled data
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform. Call fit() or load() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.scaler.transform(X)

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], feature_names: Optional[list] = None) -> np.ndarray:
        """
        Fit scaler and transform data in one step.

        Args:
            X: Training data
            feature_names: List of feature names (optional)

        Returns:
            Scaled data
        """
        self.fit(X, feature_names)
        return self.transform(X)

    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Inverse transform (convert back to original scale).

        Args:
            X: Scaled data

        Returns:
            Data in original scale
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse_transform.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.scaler.inverse_transform(X)

    def save(self, path: Optional[Path] = None):
        """
        Save scaler to disk.

        Args:
            path: Path to save scaler (default: self.scaler_path)
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted scaler")

        save_path = path if path is not None else self.scaler_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)

        logger.info(f"Scaler saved to {save_path}")

    def load(self, path: Optional[Path] = None) -> bool:
        """
        Load scaler from disk.

        Args:
            path: Path to load scaler from (default: self.scaler_path)

        Returns:
            True if loaded successfully, False otherwise
        """
        load_path = path if path is not None else self.scaler_path

        if not load_path.exists():
            logger.warning(f"Scaler file not found: {load_path}")
            return False

        try:
            with open(load_path, 'rb') as f:
                data = pickle.load(f)

            self.scaler = data['scaler']
            self.feature_names = data.get('feature_names')
            self.is_fitted = True

            logger.info(f"Scaler loaded from {load_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load scaler: {e}")
            return False

    def get_stats(self) -> dict:
        """
        Get scaler statistics (mean and std for each feature).

        Returns:
            Dictionary with mean and std arrays
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted first")

        return {
            'mean': self.scaler.mean_,
            'std': self.scaler.scale_,
            'feature_names': self.feature_names
        }


# Global scaler instance
global_scaler = FeatureScaler()
