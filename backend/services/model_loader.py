"""
Multi-Model Loader

Loads multiple LSTM models for comparison in the Model Playground.
Supports VRAM -> System RAM fallback for memory management.
"""

import logging
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import gc

from services.lstm_model import LSTMModel
from services.lstm_model_v7 import LSTMModelV7
from services.scaler import FeatureScaler
from services.model_metadata import get_available_models, get_model_path, get_scaler_path, ModelInfo

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"


@dataclass
class LoadedModel:
    """Container for a loaded model and its scaler."""
    version: str
    model: LSTMModel
    scaler: Optional[FeatureScaler]
    device: str  # 'cuda' or 'cpu'
    info: ModelInfo


class MultiModelLoader:
    """
    Manages loading multiple models with automatic VRAM -> RAM fallback.
    """
    
    def __init__(self):
        self.loaded_models: Dict[str, LoadedModel] = {}
        self.primary_device = self._detect_device()
        self._enabled = False
        
    def _detect_device(self) -> str:
        """Detect available device."""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _get_gpu_memory_free(self) -> float:
        """Get free GPU memory in GB."""
        if not torch.cuda.is_available():
            return 0.0
        try:
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            return free_memory / (1024 ** 3)  # Convert to GB
        except Exception:
            return 0.0
    
    def _estimate_model_size_gb(self) -> float:
        """Estimate memory needed per model (rough estimate)."""
        # LSTM with ~37 input features, 256 hidden dim, 2 layers
        # Rough estimate: ~100-200MB per model
        return 0.2  # 200MB
    
    @property
    def is_enabled(self) -> bool:
        return self._enabled
    
    def enable(self):
        """Enable the multi-model loader."""
        self._enabled = True
        logger.info("🔓 Model Playground ENABLED")
    
    def disable(self):
        """Disable and unload all models."""
        self._enabled = False
        self.unload_all()
        logger.info("🔒 Model Playground DISABLED")
    
    def load_all_models(self) -> Dict[str, LoadedModel]:
        """
        Load all available models with VRAM -> RAM fallback.
        
        Strategy:
        1. Try to load as many models on GPU as possible
        2. When VRAM is low, fall back to CPU for remaining models
        """
        if not self._enabled:
            logger.warning("Model Playground is disabled. Enable it first.")
            return {}
        
        available = get_available_models()
        logger.info(f"🔄 Loading {len(available)} models...")
        
        # Sort so we load most important models to GPU first (v7, v6, v5, v3)
        priority_order = ["v7", "v6", "v5", "v3", "v2", "hybrid", "legacy", "v4", "pre_stationarity", "backup", "default"]
        available_sorted = sorted(
            available,
            key=lambda m: priority_order.index(m.version) if m.version in priority_order else 99
        )
        
        for info in available_sorted:
            if info.version in self.loaded_models:
                logger.debug(f"Model {info.version} already loaded")
                continue
            
            self._load_single_model(info)
        
        logger.info(f"✅ Loaded {len(self.loaded_models)} models")
        self._log_memory_usage()
        
        return self.loaded_models
    
    def _load_single_model(self, info: ModelInfo) -> bool:
        """Load a single model with device fallback."""
        model_path = get_model_path(info.version)
        scaler_path = get_scaler_path(info.version)
        
        if not model_path:
            logger.warning(f"Model file not found for {info.version}")
            return False
        
        # Determine device (GPU if enough memory, else CPU)
        device = self.primary_device
        if device == "cuda":
            free_gb = self._get_gpu_memory_free()
            model_size = self._estimate_model_size_gb()
            
            if free_gb < model_size + 0.5:  # Keep 0.5GB buffer
                logger.info(f"⚠️ Low VRAM ({free_gb:.2f}GB free), loading {info.version} to CPU")
                device = "cpu"
        
        try:
            # Load model - use LSTMModelV7 for v7, LSTMModel for others
            if info.version == "v7":
                # LSTMModelV7.load expects torch.device, not string
                device_obj = torch.device(device)
                model = LSTMModelV7.load(str(model_path), device=device_obj)
                logger.info(f"Loading V7 attention model with 41 features")
            else:
                model = LSTMModel.load(str(model_path), device=device)
            
            # Load scaler
            scaler = None
            if scaler_path:
                scaler = FeatureScaler(str(scaler_path))
                if not scaler.load():
                    logger.warning(f"Failed to load scaler for {info.version}")
                    scaler = None
            
            self.loaded_models[info.version] = LoadedModel(
                version=info.version,
                model=model,
                scaler=scaler,
                device=device,
                info=info
            )
            
            logger.info(f"✅ Loaded {info.display_name} on {device.upper()}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load {info.version}: {e}")
            return False
    
    def unload_model(self, version: str):
        """Unload a specific model to free memory."""
        if version in self.loaded_models:
            loaded = self.loaded_models.pop(version)
            del loaded.model
            del loaded.scaler
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"🗑️ Unloaded model {version}")
    
    def unload_all(self):
        """Unload all models."""
        versions = list(self.loaded_models.keys())
        for version in versions:
            self.unload_model(version)
        logger.info("🗑️ Unloaded all models")
    
    def get_model(self, version: str) -> Optional[LoadedModel]:
        """Get a specific loaded model."""
        return self.loaded_models.get(version)
    
    def get_prediction(
        self, 
        version: str, 
        features: torch.Tensor,
        current_price: float
    ) -> Optional[Dict[str, Any]]:
        """
        Get prediction from a specific model.
        
        Args:
            version: Model version
            features: Preprocessed feature tensor
            current_price: Current stock price for converting log returns
            
        Returns:
            Dictionary with predictions for each horizon
        """
        loaded = self.loaded_models.get(version)
        if not loaded:
            return None
        
        try:
            model = loaded.model
            model.eval()
            
            with torch.no_grad():
                # Move features to model's device
                features = features.to(loaded.device)
                
                # Get raw predictions (log returns for each horizon)
                predictions = model(features)
                
                # Convert to prices
                results = {}
                horizon_labels = ["1d", "1w", "1m", "6m"]
                
                for i, label in enumerate(horizon_labels):
                    if i < predictions.shape[1]:
                        log_return = predictions[0, i].item()
                        predicted_price = current_price * (1 + log_return)
                        change_pct = log_return * 100
                        
                        results[label] = {
                            "price": round(predicted_price, 2),
                            "change_pct": round(change_pct, 2),
                            "log_return": round(log_return, 4),
                            "direction": "bullish" if log_return > 0 else "bearish"
                        }
                
                return results
                
        except Exception as e:
            logger.error(f"Prediction error for {version}: {e}")
            return None
    
    def calculate_recent_accuracy(
        self,
        version: str,
        df: Any, # pandas DataFrame
        feature_cols: List[str],
        days: int = 10
    ) -> Optional[float]:
        """
        Calculate directional accuracy over the recent 'days' period.
        
        Args:
            version: Model version
            df: Processed DataFrame with features and 'close' price
            feature_cols: List of feature column names
            days: Number of days to backtest
            
        Returns:
            Accuracy score (0.0 to 100.0) or None if failed
        """
        loaded = self.loaded_models.get(version)
        if not loaded:
            return None
            
        try:
            model = loaded.model
            scaler = loaded.scaler
            window_size = getattr(model, 'window_size', 60)
            
            # Ensure enough data: need window_size + days
            if len(df) < window_size + days + 1:
                return None
                
            correct = 0
            total = 0
            
            # Use a single batch for efficiency if possible, but loop is safer for logic
            # Loop through the last 'days' entries (excluding today T, so T-days to T-1)
            # We want to predict T-days+1 ... T
            
            # Indices:
            # We want to predict prices at indices: -days, -days+1, ..., -1
            # Inputs end at indices: -days-1, -days, ..., -2
            
            for i in range(days):
                # Target index (the day we are predicting)
                target_idx = -(days - i) 
                # Input end index (the day before)
                input_idx = target_idx - 1
                
                # Actual movement: Price(target) - Price(input)
                actual_close = float(df.iloc[target_idx]["close"])
                prev_close = float(df.iloc[input_idx]["close"])
                actual_move = actual_close - prev_close
                
                # Prepare features
                if window_size > 1:
                    # Window: [input_idx - window_size + 1 : input_idx + 1]
                    # e.g. if input_idx is -1 (yesterday), we want up to -1 inclusive
                    # iloc slice is exclusive at end, so +1
                    # But wait, iloc negative slicing:
                    # if input_idx is -1, input_idx+1 is 0 (which is wrong for slice end)
                    # if input_idx is -2, input_idx+1 is -1
                    
                    start_pos = len(df) + input_idx - window_size + 1
                    end_pos = len(df) + input_idx + 1
                    
                    window_features = df.iloc[start_pos:end_pos][feature_cols].values
                else:
                    window_features = df.iloc[input_idx][feature_cols].values.reshape(1, -1)
                
                if scaler:
                    window_features = scaler.transform(window_features)
                
                features_tensor = torch.tensor(window_features, dtype=torch.float32).unsqueeze(0)
                
                # Predict
                # Note: get_prediction returns dict with '1d' key
                # We can call model directly for speed
                model.eval()
                with torch.no_grad():
                    features_tensor = features_tensor.to(loaded.device)
                    pred_log_return = model(features_tensor)[0, 0].item()
                
                # Predicted move direction matches log return sign
                # (log_return > 0) == (actual_move > 0)
                
                pred_bullish = pred_log_return > 0
                actual_bullish = actual_move > 0
                
                if pred_bullish == actual_bullish:
                    correct += 1
                total += 1
            
            if total == 0:
                return 0.0
                
            return (correct / total) * 100.0
            
        except Exception as e:
            logger.error(f"Accuracy calc error for {version}: {e}")
            return None

    def _log_memory_usage(self):
        """Log current memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            logger.info(f"📊 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        gpu_models = sum(1 for m in self.loaded_models.values() if m.device == "cuda")
        cpu_models = sum(1 for m in self.loaded_models.values() if m.device == "cpu")
        logger.info(f"📊 Models: {gpu_models} on GPU, {cpu_models} on CPU")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current loader status."""
        return {
            "enabled": self._enabled,
            "loaded_count": len(self.loaded_models),
            "models": {
                version: {
                    "display_name": loaded.info.display_name,
                    "device": loaded.device,
                    "has_scaler": loaded.scaler is not None
                }
                for version, loaded in self.loaded_models.items()
            },
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory_free_gb": round(self._get_gpu_memory_free(), 2) if torch.cuda.is_available() else 0
        }


# Global singleton instance
model_loader = MultiModelLoader()
