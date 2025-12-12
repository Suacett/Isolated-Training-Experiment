"""
LSTM Model v8 - Classification Architecture (Regime Detection)

This model is designed for BINARY CLASSIFICATION of "buy signals" rather than
regression-based price prediction. It inherits the attention architecture from V7
but changes the output layer and loss paradigm.

Key Differences from V7:
- Output: 4 class probabilities (one per horizon: 1d, 1w, 1m, 6m)
- Loss: BCEWithLogitsLoss (during training, raw logits; during inference, sigmoid)
- Dropout: Increased to 0.5 to prevent overfitting on sparse data
- Target: Binary (1 = return > threshold, 0 = otherwise)

Architecture:
    1. Conv1D (64 filters, kernel=3, relu)
    2. BatchNorm1D
    3. MC Dropout (0.5)
    4. Bidirectional LSTM (2 layers, 128 hidden)
    5. Temporal Attention (weighted timestep importance)
    6. MC Dropout (0.5)
    7. Dense (64, relu) with residual
    8. Dense (4, linear) -> logits for BCEWithLogitsLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple
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
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.force_dropout = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active = self.training or self.force_dropout
        return F.dropout(x, p=self.p, training=active)


class TemporalAttention(nn.Module):
    """
    Temporal Attention: Learns which timesteps are most important for prediction.
    
    With sparse non-overlapping windows, attention is CRITICAL for finding
    patterns across the 60-day window without relying on recency bias.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = self.head_dim ** 0.5
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        attended = self.out_proj(attended)
        
        return attended, attention_weights


class LSTMModelV8Class(nn.Module):
    """
    Classification LSTM Model v8 - Regime Detection
    
    This model outputs BUY PROBABILITIES for each time horizon:
    - Class 0: No exceptional move expected
    - Class 1: Exceptional positive move expected (return > threshold)
    
    The threshold is computed as μ + 2σ based on training data only.
    """
    
    # Classification thresholds
    PROBABILITY_THRESHOLD = 0.70  # Only act if P(buy) > 70%
    MC_SAMPLES = 50
    
    def __init__(
        self,
        input_dim: int = 41,  # Same as V7
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 4,  # 4 horizons: 1d, 1w, 1m, 6m
        dropout: float = 0.5,  # INCREASED from 0.3 to prevent overfitting
        window_size: int = 60,
        num_attention_heads: int = 4,
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
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"🚀 LSTMModelV8Class initializing on {gpu_name} ({gpu_mem:.1f} GB VRAM)")
        else:
            logger.warning("⚠️ LSTMModelV8Class running on CPU - training will be slow")
        
        print(f"Initializing LSTMModelV8Class (Classification) on device: {self.device}")
        
        # === LAYER 1: Conv1D Block ===
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        # === LAYER 2: MC Dropout (0.5 for sparse data) ===
        self.mc_dropout1 = MCDropout(dropout)
        
        # === LAYER 3: Bidirectional LSTM ===
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        lstm_output_dim = hidden_dim * 2  # Bidirectional
        
        # === LAYER 4: Temporal Attention ===
        self.attention = TemporalAttention(lstm_output_dim, num_heads=num_attention_heads)
        self.attention_ln = nn.LayerNorm(lstm_output_dim)
        
        # === LAYER 5: MC Dropout ===
        self.mc_dropout2 = MCDropout(dropout)
        
        # === LAYER 6: Dense with Residual ===
        self.fc1 = nn.Linear(lstm_output_dim, 64)
        self.fc1_ln = nn.LayerNorm(64)
        self.residual_proj = nn.Linear(lstm_output_dim, 64) if lstm_output_dim != 64 else nn.Identity()
        
        # === LAYER 7: Classification Output (Logits) ===
        # NO ACTIVATION HERE - BCEWithLogitsLoss handles sigmoid internally
        self.fc2 = nn.Linear(64, output_dim)
        
        self.relu = nn.ReLU()
        self.to(self.device)
        self._use_checkpointing = False
        
    def enable_checkpointing(self):
        self._use_checkpointing = True
        logger.info("✅ Gradient checkpointing enabled")
        
    def disable_checkpointing(self):
        self._use_checkpointing = False
        
    def _lstm_forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - returns RAW LOGITS (not probabilities).
        Apply sigmoid only during inference.
        """
        if self.use_cpu_offload and x.device.type == 'cpu':
            x = x.to(self.device)
        elif x.device != self.device:
            x = x.to(self.device)
        
        batch_size = x.shape[0]
        
        # Conv1D Block
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.mc_dropout1(x)
        x = x.permute(0, 2, 1)
        
        # Bidirectional LSTM
        if self._use_checkpointing and self.training:
            lstm_out = checkpoint(self._lstm_forward, x, use_reentrant=False)
        else:
            lstm_out, _ = self.lstm(x)
        
        # Temporal Attention
        attended, _ = self.attention(lstm_out)
        attended = self.attention_ln(attended + lstm_out)
        pooled = attended.mean(dim=1)
        
        # Dropout
        pooled = self.mc_dropout2(pooled)
        
        # Dense with Residual
        residual = self.residual_proj(pooled)
        out = self.fc1(pooled)
        out = self.relu(out)
        out = self.fc1_ln(out)
        out = out + residual
        
        # Output (LOGITS, not probabilities)
        out = self.fc2(out)
        
        return out
    
    def predict_probabilities(self, x: torch.Tensor, device: torch.device = None) -> torch.Tensor:
        """
        Inference method - returns PROBABILITIES via sigmoid.
        """
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
            "confidence": 1.0 - std_probs.cpu().numpy(),  # Lower std = higher confidence
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
            'version': 'v8-class'
        }, path)
        print(f"Model saved to {path}")
        
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None, use_cpu_offload: bool = False) -> 'LSTMModelV8Class':
        """Load model from file."""
        if device is None:
            device = get_device()
            
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint.get('num_layers', 2),
            output_dim=checkpoint['output_dim'],
            window_size=checkpoint['window_size'],
            dropout=checkpoint['dropout_p'],
            num_attention_heads=checkpoint.get('num_attention_heads', 4),
            use_cpu_offload=use_cpu_offload,
            device=device
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")
        
        return model
