import torch
import torch.nn as nn
import torch.nn.functional as F

class MCDropout(nn.Dropout):
    """
    Monte Carlo Dropout layer.
    Always applies dropout, even during evaluation, to estimate uncertainty.
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.dropout(input, self.p, True, self.inplace)

class LSTMModel(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        window_size: int, 
        conv_filters: int = 32, 
        lstm_units: int = 64, 
        dropout_rate: float = 0.3
    ):
        """
        PyTorch implementation of the legacy Keras LSTM model.
        
        Architecture:
        1. Conv1D
        2. BatchNorm
        3. MC Dropout
        4. LSTM
        5. MC Dropout
        6. Dense (ReLU)
        7. Dense (Linear, 4 outputs)
        """
        super(LSTMModel, self).__init__()
        
        self.input_dim = input_dim
        self.window_size = window_size
        
        # Conv1D: Input (Batch, Input_Dim, Seq_Len) -> Output (Batch, Filters, Seq_Len')
        # Note: We will transpose input in forward() to match PyTorch convention
        self.conv1 = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=conv_filters, 
            kernel_size=3, 
            padding=1 # Same padding to keep length roughly similar if needed, or valid
        )
        
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.dropout1 = MCDropout(dropout_rate)
        
        # LSTM: Input (Seq_Len, Batch, Input_Size) or (Batch, Seq_Len, Input_Size)
        # We use batch_first=True
        self.lstm = nn.LSTM(
            input_size=conv_filters, 
            hidden_size=lstm_units, 
            batch_first=True
        )
        
        self.dropout2 = MCDropout(dropout_rate)
        
        self.fc1 = nn.Linear(lstm_units, 32)
        self.fc2 = nn.Linear(32, 4) # 4 Time horizons: 1d, 1w, 1m, 6m

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Input_Dim)
        
        # PyTorch Conv1d expects (Batch, Channels, Length)
        x = x.transpose(1, 2) # -> (Batch, Input_Dim, Seq_Len)
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        
        # LSTM expects (Batch, Seq_Len, Features)
        x = x.transpose(1, 2) # -> (Batch, Seq_Len, Filters)
        
        # LSTM output: output, (h_n, c_n)
        # output shape: (Batch, Seq_Len, Hidden_Size)
        # We only want the last time step
        lstm_out, _ = self.lstm(x)
        
        # Take last time step
        last_step = lstm_out[:, -1, :]
        
        x = self.dropout2(last_step)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

def get_device():
    """Dynamic device detection as per project rules."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    return device
