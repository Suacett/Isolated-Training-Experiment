import pytest
import torch
from backend.services.lstm import LSTMModel

def test_lstm_initialization_and_forward_pass(capsys):
    # Parameters
    input_dim = 10
    seq_len = 30
    batch_size = 5
    output_dim = 4
    
    # Initialize model
    model = LSTMModel(input_dim=input_dim, seq_len=seq_len, output_dim=output_dim)
    
    # Check if print statement works
    captured = capsys.readouterr()
    assert "Initializing LSTM on device:" in captured.out
    
    # Create dummy input
    # Shape: (batch_size, seq_len, input_dim)
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    output = model(x)
    
    # Verify output shape
    assert output.shape == (batch_size, output_dim)
    
    # Verify device placement
    assert next(model.parameters()).device.type == model.device.type
    if torch.cuda.is_available():
        assert model.device.type == "cuda"
    else:
        assert model.device.type == "cpu"

def test_lstm_device_selection():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = LSTMModel(input_dim=5, seq_len=10, device=device)
        assert next(model.parameters()).is_cuda
    
    device = torch.device("cpu")
    model = LSTMModel(input_dim=5, seq_len=10, device=device)
    assert not next(model.parameters()).is_cuda
