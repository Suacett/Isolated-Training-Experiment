"""
V9 Transformer Encoder for Relative Strength Ranking

Architecture:
    Input: (batch, seq_len=60, features=12)
    
    1. InputEmbedding: Linear(12 → 64) - Project features to d_model
    2. PositionalEncoding: Sine/Cosine - Inject sequence position
    3. TransformerEncoder: 2 layers, 4 heads, d_ff=256, dropout=0.2
    4. GlobalMeanPooling: Average across sequence dimension
    5. OutputHead: Linear(64 → 1) → Sigmoid - Predict rank 0.0-1.0

Why Transformer?
    - Captures long-range dependencies (patterns across 60 days)
    - No vanishing gradient (unlike LSTM)
    - Parallel computation (faster training)
    - Learns complex interactions between macro/micro features

Usage:
    model = TransformerRankModel(input_dim=12, d_model=64)
    x = torch.randn(32, 60, 12)  # (batch, seq, features)
    rank = model(x)  # (batch, 1) in range [0, 1]
"""

import math
import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Dynamic device selection."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# POSITIONAL ENCODING (Standard Sine/Cosine)
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    Standard sine/cosine positional encoding from "Attention Is All You Need".
    
    Injects position information into the sequence so the Transformer
    knows which timestep each feature vector belongs to.
    """
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
# TRANSFORMER RANK MODEL
# =============================================================================

class TransformerRankModel(nn.Module):
    """
    Time-Series Transformer for Relative Strength Ranking.
    
    Predicts a stock's relative rank (0.0 = worst, 1.0 = best) among
    all stocks for the next 5 days.
    
    Architecture:
        Input Embedding → Positional Encoding → Transformer Encoder →
        Global Mean Pooling → Output Head (Sigmoid)
    """
    
    def __init__(
        self,
        input_dim: int = 12,       # Number of input features
        d_model: int = 64,         # Transformer embedding dimension
        nhead: int = 4,            # Number of attention heads
        num_layers: int = 2,       # Number of Transformer encoder layers
        dim_feedforward: int = 256, # Feedforward network dimension
        dropout: float = 0.2,      # Dropout rate
        max_seq_len: int = 100,    # Maximum sequence length
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.max_seq_len = max_seq_len
        
        self.device = device or get_device()
        
        # === Layer 1: Input Embedding ===
        # Project raw features to d_model dimensions
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # === Layer 2: Positional Encoding ===
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # === Layer 3: Transformer Encoder ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Use (batch, seq, feature) format
            activation='gelu',  # GELU activation (modern choice)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # === Layer 4: Output Head ===
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),  # Output in [0, 1] range for rank
        )
        
        self.to(self.device)
        self._log_params()
        
    def _log_params(self):
        """Log model parameter count."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"TransformerRankModel: {total:,} total, {trainable:,} trainable params")
        print(f"TransformerRankModel initialized: {trainable:,} trainable parameters")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
               e.g., (256, 60, 12) for batch of 256, 60-day sequences, 12 features
        
        Returns:
            Predicted ranks of shape (batch, 1) in range [0, 1]
        """
        if x.device != self.device:
            x = x.to(self.device)
        
        # Input embedding: (batch, seq, input_dim) -> (batch, seq, d_model)
        x = self.input_embedding(x)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Global mean pooling: (batch, seq, d_model) -> (batch, d_model)
        x = x.mean(dim=1)
        
        # Output head: (batch, d_model) -> (batch, 1)
        x = self.output_head(x)
        
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference method.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted ranks
        """
        self.eval()
        with torch.no_grad():
            if x.device != self.device:
                x = x.to(self.device)
            if x.dim() == 2:
                x = x.unsqueeze(0)  # Add batch dimension
            return self.forward(x)
    
    def predict_batch(
        self,
        X: torch.Tensor,
        batch_size: int = 256
    ) -> torch.Tensor:
        """
        Batch prediction for large datasets.
        
        Args:
            X: Input tensor of shape (N, seq_len, features)
            batch_size: Batch size for inference
            
        Returns:
            Predicted ranks of shape (N, 1)
        """
        self.eval()
        all_preds = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                if batch.device != self.device:
                    batch = batch.to(self.device)
                preds = self.forward(batch)
                all_preds.append(preds.cpu())
        
        return torch.cat(all_preds, dim=0)
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout_rate,
            'max_seq_len': self.max_seq_len,
            'version': 'v9-transformer',
        }, path)
        print(f"Model saved to {path}")
        
    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[torch.device] = None
    ) -> 'TransformerRankModel':
        """Load model from checkpoint."""
        if device is None:
            device = get_device()
            
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        model = cls(
            input_dim=checkpoint['input_dim'],
            d_model=checkpoint['d_model'],
            nhead=checkpoint['nhead'],
            num_layers=checkpoint['num_layers'],
            dim_feedforward=checkpoint['dim_feedforward'],
            dropout=checkpoint['dropout'],
            max_seq_len=checkpoint.get('max_seq_len', 100),
            device=device
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")
        
        return model


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_model(
    input_dim: int = 12,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.2,
    device: Optional[torch.device] = None
) -> TransformerRankModel:
    """Factory function to create a TransformerRankModel with default settings."""
    return TransformerRankModel(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        device=device
    )


if __name__ == "__main__":
    # Quick test
    print("Testing TransformerRankModel...")
    
    model = TransformerRankModel(input_dim=12, d_model=64)
    
    # Test forward pass
    x = torch.randn(32, 60, 12)  # batch=32, seq=60, features=12
    out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    print("✅ Test passed!")
