import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np

def get_device() -> torch.device:
    """Dynamic device selection - never hardcode cuda:0"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MCDropout(nn.Module):
    """
    Monte Carlo Dropout: Applies dropout during training.
    For inference:
    - If model.train(), applies dropout (standard training)
    - If model.eval(), applies dropout ONLY if force_dropout is True
    """
    def __init__(self, p: float = 0.3):
        super().__init__()
        self.p = p
        self.force_dropout = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply dropout if training OR if force_dropout is enabled (for MC sampling)
        active = self.training or self.force_dropout
        return F.dropout(x, p=self.p, training=active)


class LSTMModel(nn.Module):
    """
    Strict port of the legacy Keras LSTM model from forecasting_backtest_Predictor_v2.py

    Architecture:
        1. Conv1D (32 filters, kernel=3, relu)
        2. BatchNorm1D
        3. MC Dropout (0.3) - always active for uncertainty estimation
        4. LSTM (64 hidden units)
        5. MC Dropout (0.3)
        6. Dense (32, relu)
        7. Dense (output_dim, linear) - outputs multi-horizon predictions

    Legacy hyperparameters:
        - CONFIDENCE_THRESHOLD = 0.70 (only act if P(up) > 70%)
        - CONFIDENCE_Z = 1.5 (for uncertainty adjustment)
        - Multi-horizon: 1d, 1w (5d), 1m (21d), 6m (126d)
    """

    # Legacy constants from forecasting_backtest_Predictor_v2.py
    CONFIDENCE_THRESHOLD = 0.70
    CONFIDENCE_Z = 1.5
    MC_SAMPLES = 50  # Number of forward passes for uncertainty estimation (legacy: 50)

    def __init__(
        self,
        input_dim: int = 39,  # 39 features (technical indicators + alternative data)
        hidden_dim: int = 64,
        num_layers: int = 1,
        output_dim: int = 4,  # Multi-horizon: 1d, 1w, 1m, 6m
        dropout: float = 0.3,
        window_size: int = 60,
        device: Optional[torch.device] = None
    ):
        super(LSTMModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.window_size = window_size
        self.dropout_p = dropout

        # Device handling - dynamic selection
        if device is None:
            self.device = get_device()
        else:
            self.device = device

        print(f"Initializing LSTM on device: {self.device}")

        # Conv1D: in_channels=input_dim, out_channels=32, kernel_size=3
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3)
        self.bn = nn.BatchNorm1d(32)

        # MC Dropout layers (always active)
        self.mc_dropout1 = MCDropout(dropout)
        self.mc_dropout2 = MCDropout(dropout)

        # LSTM: input_size=32 (from Conv1D), hidden_size=64
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Dense layers
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)

        self.relu = nn.ReLU()

        # Move to device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, seq_len, features)

        Returns:
            Output tensor of shape (batch, output_dim)
        """
        # Ensure on correct device
        x = x.to(self.device)

        # Conv1D expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)

        # Conv1D -> ReLU -> BatchNorm
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.bn(x)

        # MC Dropout 1
        x = self.mc_dropout1(x)

        # Back to (batch, seq_len, features) for LSTM
        x = x.permute(0, 2, 1)

        # LSTM - take output from all timesteps, then select last
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last timestep

        # MC Dropout 2
        out = self.mc_dropout2(out)

        # Dense layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out

    def predict(self, x: torch.Tensor, device: torch.device) -> float:
        """
        Single prediction helper. Returns the 1-day forecast (index 0).

        Args:
            x: Input tensor of shape (seq_len, features) or (batch, seq_len, features)
            device: Target device

        Returns:
            Single float prediction for 1-day horizon
        """
        x = x.to(device)

        # Add batch dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # Single forward pass (Deterministic if eval mode)
        self.eval()
        # Ensure dropout is off for single prediction
        self.mc_dropout1.force_dropout = False
        self.mc_dropout2.force_dropout = False
        
        with torch.no_grad():
            output = self.forward(x)

        # Return 1-day forecast (index 0)
        return output[0, 0].item()

    def predict_batch(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Batch prediction for multiple windows. Returns 1-day forecasts for all inputs.

        Args:
            x: Input tensor of shape (batch, seq_len, features)
            device: Target device

        Returns:
            Tensor of shape (batch,) with 1-day predictions
        """
        x = x.to(device)

        # Ensure 3D
        if x.dim() == 2:
            x = x.unsqueeze(0)

        self.eval()
        self.mc_dropout1.force_dropout = False
        self.mc_dropout2.force_dropout = False
        
        with torch.no_grad():
            output = self.forward(x)

        # Return 1-day forecast (column 0) for all batch items
        return output[:, 0]

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        device: torch.device,
        n_samples: int = None
    ) -> dict:
        """
        Monte Carlo prediction with uncertainty estimation.
        Runs multiple forward passes and computes mean/std.

        This implements the legacy uncertainty logic from forecasting_backtest_Predictor_v2.py:
        - adjust_prediction: pred - CONFIDENCE_Z * std
        - Confidence: P(return > 0) using normal CDF

        Args:
            x: Input tensor
            device: Target device
            n_samples: Number of MC samples (default: MC_SAMPLES)

        Returns:
            Dict with predictions, uncertainties, and confidence scores
        """
        if n_samples is None:
            n_samples = self.MC_SAMPLES

        x = x.to(device)
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # 1. Deterministic Prediction (Dropout OFF)
        self.eval()
        self.mc_dropout1.force_dropout = False
        self.mc_dropout2.force_dropout = False
        with torch.no_grad():
            deterministic_pred = self.forward(x) # (batch, output_dim)

        # 2. Uncertainty Estimation (Dropout ON)
        self.mc_dropout1.force_dropout = True
        self.mc_dropout2.force_dropout = True

        # Collect MC samples
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                output = self.forward(x)
                predictions.append(output)
        
        # Disable MC Dropout after sampling
        self.mc_dropout1.force_dropout = False
        self.mc_dropout2.force_dropout = False

        # Stack: (n_samples, batch, output_dim)
        predictions = torch.stack(predictions, dim=0)

        # Compute statistics
        # mean_pred = predictions.mean(dim=0)  # OLD: Mean of samples (Noisy)
        std_pred = predictions.std(dim=0)    # (batch, output_dim)

        # Use deterministic prediction as the primary value
        mean_pred = deterministic_pred

        # Adjusted prediction (legacy: pred - CONFIDENCE_Z * std)
        adjusted_pred = mean_pred - self.CONFIDENCE_Z * std_pred

        # Confidence: P(prediction > 0) assuming normal distribution
        # Using standard normal CDF: P(X > 0) = 1 - Phi(-mean/std) = Phi(mean/std)
        eps = 1e-6
        z_score = mean_pred / (std_pred + eps)
        # Approximate normal CDF using error function
        confidence = 0.5 * (1 + torch.erf(z_score / np.sqrt(2)))

        return {
            "prediction": mean_pred[:, 0].cpu().numpy(),  # 1-day forecast (Deterministic)
            "prediction_all_horizons": mean_pred.cpu().numpy(),
            "std": std_pred[:, 0].cpu().numpy(),
            "adjusted_prediction": adjusted_pred[:, 0].cpu().numpy(),
            "confidence": confidence[:, 0].cpu().numpy(),
            "should_act": (confidence[:, 0] >= self.CONFIDENCE_THRESHOLD).cpu().numpy()
        }

    def save(self, path: str) -> None:
        """
        Save model weights to a .pth file.

        Args:
            path: File path to save weights (should end in .pth)
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'window_size': self.window_size,
            'dropout_p': self.dropout_p,
        }, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'LSTMModel':
        """
        Load model weights from a .pth file.

        Args:
            path: File path to load weights from
            device: Target device (default: auto-detect)

        Returns:
            LSTMModel instance with loaded weights
        """
        if device is None:
            device = get_device()

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            output_dim=checkpoint['output_dim'],
            window_size=checkpoint['window_size'],
            dropout=checkpoint['dropout_p'],
            device=device
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")

        return model
