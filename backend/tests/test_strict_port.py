import pytest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.lstm_model import LSTMModel
from services.intrinsic import IntrinsicCalculator

def test_lstm_architecture():
    # Legacy params: input_dim=39, window_size=90
    model = LSTMModel(input_dim=39, window_size=90)
    
    # Create dummy input: (batch_size=1, seq_len=91, features=39)
    # Note: Keras Conv1D input_shape=(window+1, features) -> (91, 39)
    dummy_input = torch.randn(1, 91, 39)
    
    # Forward pass
    output = model(dummy_input)
    
    # Check output shape: (batch_size, output_dim=4)
    assert output.shape == (1, 4), f"Expected output shape (1, 4), got {output.shape}"
    
    # Check if dropout is active (output should be non-deterministic even in eval mode if we forced it)
    # Our implementation forces training=True in F.dropout, so it should be stochastic.
    out1 = model(dummy_input)
    out2 = model(dummy_input)
    assert not torch.allclose(out1, out2), "Dropout should be active (MC Dropout)"

def test_intrinsic_formula():
    calc = IntrinsicCalculator()
    
    # Test case
    eps = 10.0
    g = 0.05 # 5%
    yield_val = 4.4
    
    # Formula: eps * (8.5 + 2 * (g * 100)) * (yield / 4.4)
    # Expected: 10 * (8.5 + 2 * 5) * (4.4 / 4.4)
    #         = 10 * (8.5 + 10) * 1
    #         = 10 * 18.5
    #         = 185.0
    
    val = calc.calculate_graham(eps, g, yield_val)
    assert val == 185.0, f"Expected 185.0, got {val}"
    
    # Test case 2: Yield 2.2
    # Expected: 10 * 18.5 * (2.2 / 4.4) = 185 * 0.5 = 92.5
    val2 = calc.calculate_graham(eps, g, 2.2)
    assert val2 == 92.5, f"Expected 92.5, got {val2}"
