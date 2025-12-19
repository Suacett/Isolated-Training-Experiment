"""
LSTM Model v8 - Classification Architecture (V4 Legacy Compliant)

This module provides the V4-compliant simple LSTM classifier for regime detection.
The model is designed for BINARY CLASSIFICATION of "buy signals" rather than
regression-based price prediction.

V4 Compliance:
    - Simple architecture: Conv1D(32) -> LSTM(64) -> Dense(32) -> Dense(4)
    - Window size: 91 (90 + 1 for current day)
    - Dropout: 0.3 (not 0.5)
    - ~50k parameters (not 500k+)

Key Differences from V7:
    - Output: 4 class probabilities (one per horizon: 1d, 1w, 1m, 6m)
    - Loss: BCEWithLogitsLoss (during training, raw logits; during inference, sigmoid)
    - Target: Binary (1 = return > threshold, 0 = otherwise)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Dynamic device selection - never hardcode cuda:0"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MCDropout(nn.Module):
    """
    Monte Carlo Dropout: Applies dropout during training.
    For inference, applies dropout ONLY if force_dropout is True (for uncertainty estimation).
    """
    def __init__(self, p: float = 0.3):
        super().__init__()
        self.p = p
        self.force_dropout = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active = self.training or self.force_dropout
        return F.dropout(x, p=self.p, training=active)


class LSTMModelV8Class(nn.Module):
    """
    V4-Compliant Simple LSTM Classifier (Regime Detection)
    
    This model outputs BUY PROBABILITIES for each time horizon:
    - Class 0: No exceptional move expected
    - Class 1: Exceptional positive move expected (return > threshold)
    
    Architecture (matches legacy run_forecast_v4.ipynb):
        Conv1D(32, k=3) -> BatchNorm -> MCDropout(0.3)
        LSTM(64, 1-layer, unidirectional)
        MCDropout(0.3)
        Dense(32, ReLU) -> Dense(4, Logits)
    
    The threshold is computed as μ + 2σ based on training data only (sniper standard).
    """
    
    # Classification thresholds
    PROBABILITY_THRESHOLD = 0.70  # Only act if P(buy) > 70%
    MC_SAMPLES = 25  # V4 uses 25
    
    def __init__(
        self,
        input_dim: int = 41,
        hidden_dim: int = 64,      # V4: 64 (not 128)
        num_layers: int = 1,       # V4: 1 layer (not 2)
        output_dim: int = 4,
        dropout: float = 0.3,      # V4: 0.3 (not 0.5)
        window_size: int = 91,     # V4: 90 + 1 = 91
        num_attention_heads: int = 0,  # V4: No attention
        use_cpu_offload: bool = False,
        device: Optional[torch.device] = None
    ):
        super(LSTMModelV8Class, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.window_size = window_size
        self.dropout_p = dropout
        self.num_attention_heads = num_attention_heads
        self.use_cpu_offload = use_cpu_offload
        
        if device is None:
            self.device = get_device()
        else:
            self.device = device
            
        if self.device.type == 'cuda':
            # Use the actual device index assigned (torch.cuda.current_device() is often default, index is better)
            dev_idx = self.device.index if self.device.index is not None else torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(dev_idx)
            gpu_mem = torch.cuda.get_device_properties(dev_idx).total_memory / 1e9
            logger.info(f"🚀 LSTMModelV8Class (V4) on {gpu_name} (index {dev_idx}) - {gpu_mem:.1f} GB")
        else:
            logger.warning("⚠️ LSTMModelV8Class running on CPU")
        
        # === V4 ARCHITECTURE ===
        
        # Layer 1: Conv1D Block (32 filters, not 64)
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.mc_dropout1 = MCDropout(dropout)
        
        # Layer 2: Simple LSTM (64 units, 1 layer, unidirectional)
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0
        )
        self.mc_dropout2 = MCDropout(dropout)
        
        # Layer 3: Dense layers (32 units, not 64)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)
        
        self.relu = nn.ReLU()
        self.to(self.device)
        
        # Log parameter count
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"V4 Model Parameters: {total_params:,} (target: ~50k)")
        
    def enable_checkpointing(self):
        """No-op for V4 simple model (not needed)."""
        pass
        
    def disable_checkpointing(self):
        """No-op for V4 simple model."""
        pass
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - returns RAW LOGITS (not probabilities).
        Apply sigmoid only during inference.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Logits tensor of shape (batch, 4)
        """
        if self.use_cpu_offload and x.device.type == 'cpu':
            x = x.to(self.device)
        elif x.device != self.device:
            x = x.to(self.device)
        
        # Conv1D expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.mc_dropout1(x)
        
        # Back to (batch, seq_len, channels) for LSTM
        x = x.permute(0, 2, 1)
        
        # Simple LSTM - take final hidden state
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]  # Last layer's hidden state
        
        x = self.mc_dropout2(x)
        
        # Dense layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # Raw logits
        
        return x
    
    def predict_probabilities(self, x: torch.Tensor, device: torch.device = None) -> torch.Tensor:
        """Inference method - returns PROBABILITIES via sigmoid."""
        if device is None:
            device = self.device
        x = x.to(device)
        
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
        self.eval()
        self.mc_dropout1.force_dropout = False
        self.mc_dropout2.force_dropout = False
        
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
            
        return probabilities
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        device: torch.device = None,
        n_samples: int = None
    ) -> dict:
        """
        Monte Carlo prediction with uncertainty estimation.
        Returns probabilities with confidence intervals.
        """
        if n_samples is None:
            n_samples = self.MC_SAMPLES
        if device is None:
            device = self.device
            
        x = x.to(device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
        # Deterministic prediction
        self.eval()
        self.mc_dropout1.force_dropout = False
        self.mc_dropout2.force_dropout = False
        with torch.no_grad():
            deterministic_logits = self.forward(x)
            deterministic_probs = torch.sigmoid(deterministic_logits)
            
        # MC Sampling
        self.mc_dropout1.force_dropout = True
        self.mc_dropout2.force_dropout = True
        
        all_probs = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                probs = torch.sigmoid(logits)
                all_probs.append(probs)
                
        self.mc_dropout1.force_dropout = False
        self.mc_dropout2.force_dropout = False
        
        all_probs = torch.stack(all_probs, dim=0)
        mean_probs = all_probs.mean(dim=0)
        std_probs = all_probs.std(dim=0)
        
        return {
            "probabilities": mean_probs.cpu().numpy(),
            "std": std_probs.cpu().numpy(),
            "buy_signals": (mean_probs >= self.PROBABILITY_THRESHOLD).cpu().numpy(),
            "confidence": 1.0 - std_probs.cpu().numpy(),
        }
    
    def save(self, path: str) -> None:
        """Save model weights and config."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'output_dim': self.output_dim,
            'window_size': self.window_size,
            'dropout_p': self.dropout_p,
            'num_attention_heads': self.num_attention_heads,
            'version': 'v8-class-v4compliant'
        }, path)
        logger.info(f"Model saved to {path}")
        
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None, use_cpu_offload: bool = False) -> 'LSTMModelV8Class':
        """Load model from file."""
        if device is None:
            device = get_device()
            
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        
        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint.get('hidden_dim', 64),
            num_layers=checkpoint.get('num_layers', 1),
            output_dim=checkpoint.get('output_dim', 4),  # Default 4 horizons for old checkpoints
            window_size=checkpoint['window_size'],
            dropout=checkpoint.get('dropout_p', 0.3),
            num_attention_heads=checkpoint.get('num_attention_heads', 0),
            use_cpu_offload=use_cpu_offload,
            device=device
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")
        
        return model
