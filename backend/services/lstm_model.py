import torch
import torch.nn as nn
import torch.nn.functional as F

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMModel(nn.Module):
    """
    Strict port of the legacy Keras LSTM model.
    Architecture:
        1. Conv1D (32 filters, kernel=3, relu)
        2. BatchNorm
        3. MC Dropout (0.3)
        4. LSTM (64 units)
        5. MC Dropout (0.3)
        6. Dense (32, relu)
        7. Dense (4, linear)
    """
    def __init__(self, input_dim=39, hidden_dim=64, num_layers=1, output_dim=4, dropout=0.3, window_size=90):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        
        # Conv1D: in_channels=input_dim, out_channels=32, kernel_size=3
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3)
        self.bn = nn.BatchNorm1d(32)
        
        # LSTM: input_size=32 (from Conv1D), hidden_size=64
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)
        
        self.dropout_p = dropout
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # Conv1D expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.bn(x)
        
        # Back to (batch, seq_len, features) for LSTM
        x = x.permute(0, 2, 1)
        
        # MC Dropout 1
        x = F.dropout(x, p=self.dropout_p, training=True)
        
        # LSTM
        # out: (batch, seq_len, hidden_dim)
        out, _ = self.lstm(x)
        
        # Take the last time step
        out = out[:, -1, :]
        
        # MC Dropout 2
        out = F.dropout(out, p=self.dropout_p, training=True)
        
        # Dense layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out

    def predict(self, x, device):
        """
        Helper for inference. Returns the 1-day forecast (index 0) for compatibility.
        """
        self.eval()
        # Ensure x is on the correct device
        x = x.to(device)
        
        # Add batch dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
        with torch.no_grad():
            # For MC Dropout, we technically should run multiple times and average,
            # but for single point prediction in this strict port context, 
            # we might just run it once or follow the legacy 'predict' behavior.
            # Legacy code used `model.predict(X_val)` which in Keras usually turns off dropout 
            # UNLESS `training=True` was passed in `call`.
            # The legacy MCDropout class forced `training=True`.
            # So we should keep dropout ON even in inference (which forward() does via F.dropout(..., training=True)).
            output = self.forward(x)
            
        # Return the first output (1d forecast)
        return output[0, 0].item()
