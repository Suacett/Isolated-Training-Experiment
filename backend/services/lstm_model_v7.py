"""
LSTM Model v7 - Enhanced Architecture with Attention and CPU Offloading

Improvements over v6:
- Temporal Attention mechanism for weighted timestep importance
- Bidirectional LSTM (2 layers) for pattern recognition
- Residual connections for gradient flow
- CPU offload support for large batch training on limited VRAM
- Gradient checkpointing for memory efficiency

Architecture:
    1. Conv1D (64 filters, kernel=3, relu)
    2. BatchNorm1D
    3. MC Dropout (0.3)
    4. Bidirectional LSTM (2 layers, 128 hidden)
    5. Temporal Attention (weighted timestep importance)
    6. MC Dropout (0.3)
    7. Dense (64, relu) with residual
    8. Dense (output_dim, linear)
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
    def __init__(self, p: float = 0.3):
        super().__init__()
        self.p = p
        self.force_dropout = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active = self.training or self.force_dropout
        return F.dropout(x, p=self.p, training=active)


class TemporalAttention(nn.Module):
    """
    Temporal Attention: Learns which timesteps are most important for prediction.
    
    In a 60-day window, day 55 (earnings release) might be more important than day 60.
    A standard LSTM forgets day 55 by day 60. Attention allows looking back directly.
    
    Input: (batch, seq_len, hidden_dim)
    Output: (batch, seq_len, hidden_dim) with attention-weighted features
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
        """
        Args:
            x: (batch, seq_len, hidden_dim) - LSTM sequence output
            
        Returns:
            attended: (batch, seq_len, hidden_dim) - attention-weighted sequence
            attention_weights: (batch, num_heads, seq_len, seq_len) - for visualization
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.query(x)  # (batch, seq_len, hidden)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: Q @ K^T / sqrt(d_k)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        attended = self.out_proj(attended)
        
        return attended, attention_weights


class LSTMModelV7(nn.Module):
    """
    Enhanced LSTM Model v7 with Attention and CPU Offloading
    
    Architecture:
        1. Conv1D (64 filters, kernel=3) -> BatchNorm -> ReLU
        2. MC Dropout (0.3)
        3. Bidirectional LSTM (2 layers, hidden_dim units)
        4. Temporal Multi-Head Attention (4 heads)
        5. MC Dropout (0.3)
        6. Dense (64) -> ReLU with residual connection
        7. Dense (output_dim)
    
    Features:
        - CPU offloading for large batches when VRAM is limited
        - Gradient checkpointing for memory efficiency
        - Multi-head temporal attention
    """
    
    # Legacy constants
    CONFIDENCE_THRESHOLD = 0.70
    CONFIDENCE_Z = 1.5
    MC_SAMPLES = 50
    
    def __init__(
        self,
        input_dim: int = 45,  # Increased from 37 to 45 features
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 4,  # Multi-horizon: 1d, 1w, 1m, 6m
        dropout: float = 0.3,
        window_size: int = 60,
        num_attention_heads: int = 4,
        use_cpu_offload: bool = False,
        device: Optional[torch.device] = None
    ):
        super(LSTMModelV7, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.window_size = window_size
        self.dropout_p = dropout
        self.num_attention_heads = num_attention_heads
        self.use_cpu_offload = use_cpu_offload
        
        # Device handling
        if device is None:
            self.device = get_device()
        else:
            self.device = device
            
        # Log GPU info
        if self.device.type == 'cuda':
            # Use the resolved index
            idx = self.device.index if self.device.index is not None else 0
            
            gpu_name = torch.cuda.get_device_name(idx)
            gpu_mem = torch.cuda.get_device_properties(idx).total_memory / 1e9
            logger.info(f"🚀 LSTMModelV7 initializing on {gpu_name} (Device {idx}, {gpu_mem:.1f} GB VRAM)")
            if use_cpu_offload:
                logger.info(f"📤 CPU offload enabled - will use system RAM for overflow")
        else:
            logger.warning("⚠️ LSTMModelV7 running on CPU - training will be slow")
        
        print(f"Initializing LSTMModelV7 on device: {self.device}")
        
        # === LAYER 1: Conv1D Block ===
        # Larger conv for better pattern detection
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        # === LAYER 2: MC Dropout ===
        self.mc_dropout1 = MCDropout(dropout)
        
        # === LAYER 3: Bidirectional LSTM ===
        # Bidirectional doubles the output dimension
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # LSTM output is hidden_dim * 2 due to bidirectional
        lstm_output_dim = hidden_dim * 2
        
        # === LAYER 4: Temporal Attention ===
        self.attention = TemporalAttention(lstm_output_dim, num_heads=num_attention_heads)
        self.attention_ln = nn.LayerNorm(lstm_output_dim)  # Post-attention normalization
        
        # === LAYER 5: MC Dropout ===
        self.mc_dropout2 = MCDropout(dropout)
        
        # === LAYER 6: Dense with Residual ===
        self.fc1 = nn.Linear(lstm_output_dim, 64)
        self.fc1_ln = nn.LayerNorm(64)
        
        # Projection for residual (if dimensions don't match)
        self.residual_proj = nn.Linear(lstm_output_dim, 64) if lstm_output_dim != 64 else nn.Identity()
        
        # === LAYER 7: Output ===
        self.fc2 = nn.Linear(64, output_dim)
        
        self.relu = nn.ReLU()
        
        # Move to device
        self.to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        self._use_checkpointing = False
        
    def enable_checkpointing(self):
        """Enable gradient checkpointing to reduce VRAM usage during training."""
        self._use_checkpointing = True
        logger.info("✅ Gradient checkpointing enabled")
        
    def disable_checkpointing(self):
        """Disable gradient checkpointing for faster inference."""
        self._use_checkpointing = False
        
    def _lstm_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Separate LSTM forward for checkpointing."""
        out, _ = self.lstm(x)
        return out
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Output tensor of shape (batch, output_dim)
        """
        # CPU offload: move to GPU in chunks if needed
        if self.use_cpu_offload and x.device.type == 'cpu':
            x = x.to(self.device)
        elif x.device != self.device:
            x = x.to(self.device)
        
        batch_size = x.shape[0]
        
        # === Conv1D Block ===
        # Conv1D expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.mc_dropout1(x)
        
        # Back to (batch, seq_len, features) for LSTM
        x = x.permute(0, 2, 1)
        
        # === Bidirectional LSTM ===
        if self._use_checkpointing and self.training:
            # Gradient checkpointing saves memory but increases compute
            lstm_out = checkpoint(self._lstm_forward, x, use_reentrant=False)
        else:
            lstm_out, _ = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_dim * 2)
        
        # === Temporal Attention ===
        # Attention over the FULL sequence, not just last hidden state
        attended, _ = self.attention(lstm_out)
        attended = self.attention_ln(attended + lstm_out)  # Residual connection
        
        # Pool attended sequence: mean pooling over time
        # This gives us (batch, hidden_dim * 2)
        pooled = attended.mean(dim=1)
        
        # === Dropout ===
        pooled = self.mc_dropout2(pooled)
        
        # === Dense with Residual ===
        residual = self.residual_proj(pooled)
        out = self.fc1(pooled)
        out = self.relu(out)
        out = self.fc1_ln(out)
        out = out + residual  # Residual connection
        
        # === Output ===
        out = self.fc2(out)
        
        return out
    
    def predict(self, x: torch.Tensor, device: torch.device = None) -> float:
        """
        Single prediction helper. Returns the 1-day forecast (index 0).
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
            output = self.forward(x)
            
        return output[0, 0].item()
    
    def predict_batch(self, x: torch.Tensor, device: torch.device = None) -> torch.Tensor:
        """
        Batch prediction for multiple windows.
        Supports CPU offloading for large batches.
        """
        if device is None:
            device = self.device
            
        self.eval()
        self.mc_dropout1.force_dropout = False
        self.mc_dropout2.force_dropout = False
        
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        # CPU offload: process in chunks to avoid OOM
        if self.use_cpu_offload and x.shape[0] > 512:
            results = []
            chunk_size = 256
            for i in range(0, x.shape[0], chunk_size):
                chunk = x[i:i+chunk_size].to(device)
                with torch.no_grad():
                    out = self.forward(chunk)
                results.append(out[:, 0].cpu())
                # Clear GPU cache after each chunk
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            return torch.cat(results, dim=0)
        else:
            x = x.to(device)
            with torch.no_grad():
                output = self.forward(x)
            return output[:, 0]
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        device: torch.device = None,
        n_samples: int = None
    ) -> dict:
        """
        Monte Carlo prediction with uncertainty estimation.
        """
        if n_samples is None:
            n_samples = self.MC_SAMPLES
        if device is None:
            device = self.device
            
        x = x.to(device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
        # 1. Deterministic Prediction
        self.eval()
        self.mc_dropout1.force_dropout = False
        self.mc_dropout2.force_dropout = False
        with torch.no_grad():
            deterministic_pred = self.forward(x)
            
        # 2. MC Sampling for uncertainty
        self.mc_dropout1.force_dropout = True
        self.mc_dropout2.force_dropout = True
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                output = self.forward(x)
                predictions.append(output)
                
        self.mc_dropout1.force_dropout = False
        self.mc_dropout2.force_dropout = False
        
        predictions = torch.stack(predictions, dim=0)
        std_pred = predictions.std(dim=0)
        mean_pred = deterministic_pred
        
        adjusted_pred = mean_pred - self.CONFIDENCE_Z * std_pred
        
        eps = 1e-6
        z_score = mean_pred / (std_pred + eps)
        confidence = 0.5 * (1 + torch.erf(z_score / np.sqrt(2)))
        
        return {
            "prediction": mean_pred[:, 0].cpu().numpy(),
            "prediction_all_horizons": mean_pred.cpu().numpy(),
            "std": std_pred[:, 0].cpu().numpy(),
            "adjusted_prediction": adjusted_pred[:, 0].cpu().numpy(),
            "confidence": confidence[:, 0].cpu().numpy(),
            "should_act": (confidence[:, 0] >= self.CONFIDENCE_THRESHOLD).cpu().numpy()
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
            'version': 'v7'
        }, path)
        print(f"Model saved to {path}")
        
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None, use_cpu_offload: bool = False) -> 'LSTMModelV7':
        """
        Load model from file.
        WARNING: Loading checkpoints carries security risks. Only load from trusted sources.
        """
        if device is None:
            device = get_device()
            
        ckpt = torch.load(path, map_location=device, weights_only=True)
        
        # Validate keys
        expected_keys = {'model_state_dict', 'input_dim', 'hidden_dim', 'output_dim', 'window_size'}
        if not expected_keys.issubset(ckpt.keys()):
             missing_keys = expected_keys - ckpt.keys()
             raise ValueError(f"Checkpoint {path} missing required keys: {missing_keys}")

        model = cls(
            input_dim=ckpt['input_dim'],
            hidden_dim=ckpt['hidden_dim'],
            num_layers=ckpt.get('num_layers', 2),
            output_dim=ckpt['output_dim'],
            window_size=ckpt['window_size'],
            dropout=ckpt.get('dropout_p', 0.0),  # Use get() with default 0.0
            num_attention_heads=ckpt.get('num_attention_heads', 4),
            use_cpu_offload=use_cpu_offload,
            device=device
        )
        
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Model loaded from {path}")
        
        return model
