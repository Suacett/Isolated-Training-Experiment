import torch
import torch.nn as nn
import torch.nn.functional as F

class MCDropout(nn.Module):
    """
    Monte Carlo Dropout: Applies dropout even during inference (eval mode).
    This allows estimating uncertainty by running multiple forward passes.
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)

class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int = 64, output_dim: int = 4, dropout_rate: float = 0.3, device: torch.device = None):
        """
        LSTM Model ported from legacy Keras implementation.
        Architecture: Conv1D -> BatchNorm -> MCDropout -> LSTM -> MCDropout -> Dense -> Dense
        """
        super().__init__()
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Initializing LSTM on device: {self.device}")
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        
        # Conv1D layer: Maps input_dim features to 32 channels
        # Kernel size 3 reduces sequence length by 2
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.mc_dropout1 = MCDropout(dropout_rate)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, batch_first=True)
        self.mc_dropout2 = MCDropout(dropout_rate)
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)
        
        self.to(self.device)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # Move to device
        x = x.to(self.device)
        
        # Conv1d expects (batch_size, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.mc_dropout1(x)
        
        # LSTM expects (batch_size, seq_len, input_dim)
        # Permute back: (batch_size, new_seq_len, 32)
        x = x.permute(0, 2, 1)
        
        # LSTM output: (batch_size, seq_len, hidden_dim)
        # We take the last time step output
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        
        out = self.mc_dropout2(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        
        return out
