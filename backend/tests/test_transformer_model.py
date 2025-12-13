"""
Safety Net Tests - Transformer Model (V9)

Tests for backend/services/transformer_model.py to ensure refactoring
doesn't break the V9 ranking model.

These tests verify:
- Model initialization with correct architecture
- Forward pass produces valid output shapes
- Output values are in valid range [0, 1]
- Save/load functionality works correctly
- Batch inference handles large inputs
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile


# Import the module under test
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.transformer_model import (
    TransformerRankModel,
    PositionalEncoding,
    get_device,
    create_model,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_input():
    """Create sample input tensor for model testing."""
    # (batch_size=32, seq_len=60, features=12)
    return torch.randn(32, 60, 12)


@pytest.fixture
def single_sample():
    """Create single sample input for edge case testing."""
    # (seq_len=60, features=12) - no batch dimension
    return torch.randn(60, 12)


@pytest.fixture
def model():
    """Create a TransformerRankModel instance for testing."""
    return TransformerRankModel(
        input_dim=12,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.2,
        device=torch.device("cpu")  # Force CPU for testing
    )


# =============================================================================
# TESTS: get_device()
# =============================================================================

def test_get_device_returns_valid_device():
    """Test that get_device returns a valid torch device."""
    device = get_device()
    
    assert isinstance(device, torch.device)
    assert device.type in ["cuda", "cpu"]


# =============================================================================
# TESTS: PositionalEncoding
# =============================================================================

def test_positional_encoding_output_shape():
    """Test that PositionalEncoding preserves input shape."""
    pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.1)
    x = torch.randn(16, 60, 64)  # (batch, seq, d_model)
    
    output = pe(x)
    
    assert output.shape == x.shape


def test_positional_encoding_adds_position_info():
    """Test that PositionalEncoding modifies input values."""
    pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
    pe.eval()  # Disable dropout
    
    x = torch.zeros(1, 10, 64)
    output = pe(x)
    
    # Output should be non-zero (positional encoding added)
    assert not torch.allclose(output, x)


def test_positional_encoding_different_positions():
    """Test that different positions get different encodings."""
    pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
    pe.eval()
    
    x = torch.zeros(1, 5, 64)
    output = pe(x)
    
    # Each position should have different encoding
    pos0 = output[0, 0, :]
    pos1 = output[0, 1, :]
    
    assert not torch.allclose(pos0, pos1)


# =============================================================================
# TESTS: TransformerRankModel Initialization
# =============================================================================

def test_model_initialization(model):
    """Test that model initializes without errors."""
    assert isinstance(model, nn.Module)
    assert model.input_dim == 12
    assert model.d_model == 64


def test_model_has_expected_layers(model):
    """Test that model has all expected components."""
    assert hasattr(model, "input_embedding")
    assert hasattr(model, "pos_encoder")
    assert hasattr(model, "transformer_encoder")
    assert hasattr(model, "output_head")


def test_model_parameter_count(model):
    """Test that model has reasonable parameter count."""
    total_params = sum(p.numel() for p in model.parameters())
    
    # Should be in reasonable range for small transformer
    assert 10_000 < total_params < 1_000_000


def test_model_default_device():
    """Test model uses GPU if available, else CPU."""
    model = TransformerRankModel(input_dim=12)
    
    if torch.cuda.is_available():
        assert model.device.type == "cuda"
    else:
        assert model.device.type == "cpu"


# =============================================================================
# TESTS: Forward Pass
# =============================================================================

def test_forward_output_shape(model, sample_input):
    """Test that forward pass produces correct output shape."""
    output = model(sample_input)
    
    # Output should be (batch_size, 1)
    assert output.shape == (32, 1)


def test_forward_output_range(model, sample_input):
    """Test that output values are in [0, 1] range (sigmoid)."""
    model.eval()
    
    with torch.no_grad():
        output = model(sample_input)
    
    assert (output >= 0).all(), "Output should be >= 0"
    assert (output <= 1).all(), "Output should be <= 1"


def test_forward_different_batch_sizes(model):
    """Test forward pass with different batch sizes."""
    for batch_size in [1, 16, 64, 128]:
        x = torch.randn(batch_size, 60, 12)
        output = model(x)
        
        assert output.shape == (batch_size, 1)


def test_forward_different_sequence_lengths(model):
    """Test forward pass with different sequence lengths."""
    for seq_len in [10, 30, 60, 90]:
        x = torch.randn(8, seq_len, 12)
        output = model(x)
        
        assert output.shape == (8, 1)


def test_forward_deterministic_in_eval_mode(model, sample_input):
    """Test that eval mode produces deterministic output."""
    model.eval()
    
    with torch.no_grad():
        output1 = model(sample_input.clone())
        output2 = model(sample_input.clone())
    
    assert torch.allclose(output1, output2)


# =============================================================================
# TESTS: predict()
# =============================================================================

def test_predict_batched_input(model, sample_input):
    """Test predict method with batched input."""
    output = model.predict(sample_input)
    
    assert output.shape == (32, 1)


def test_predict_single_input(model, single_sample):
    """Test predict method with single sample (no batch dim)."""
    output = model.predict(single_sample)
    
    # Should add batch dimension and return (1, 1)
    assert output.shape == (1, 1)


def test_predict_sets_eval_mode(model, sample_input):
    """Test that predict sets model to eval mode."""
    model.train()  # Start in training mode
    
    _ = model.predict(sample_input)
    
    assert not model.training


# =============================================================================
# TESTS: predict_batch()
# =============================================================================

def test_predict_batch_large_input(model):
    """Test batch prediction with large input."""
    large_input = torch.randn(1000, 60, 12)
    
    output = model.predict_batch(large_input, batch_size=128)
    
    assert output.shape == (1000, 1)


def test_predict_batch_smaller_than_batch_size(model):
    """Test batch prediction when input is smaller than batch size."""
    small_input = torch.randn(50, 60, 12)
    
    output = model.predict_batch(small_input, batch_size=256)
    
    assert output.shape == (50, 1)


def test_predict_batch_returns_cpu_tensor(model):
    """Test that predict_batch returns CPU tensor."""
    x = torch.randn(100, 60, 12)
    
    output = model.predict_batch(x)
    
    assert output.device.type == "cpu"


# =============================================================================
# TESTS: Save and Load
# =============================================================================

def test_save_creates_file(model):
    """Test that save creates a checkpoint file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_model.pth"
        
        model.save(str(path))
        
        assert path.exists()


def test_load_restores_architecture(model):
    """Test that load restores model architecture."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_model.pth"
        model.save(str(path))
        
        loaded_model = TransformerRankModel.load(str(path), device=torch.device("cpu"))
        
        assert loaded_model.input_dim == model.input_dim
        assert loaded_model.d_model == model.d_model
        assert loaded_model.nhead == model.nhead
        assert loaded_model.num_layers == model.num_layers


def test_load_produces_same_output(model, sample_input):
    """Test that loaded model produces same output as original."""
    model.eval()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_model.pth"
        model.save(str(path))
        
        with torch.no_grad():
            original_output = model(sample_input)
        
        loaded_model = TransformerRankModel.load(str(path), device=torch.device("cpu"))
        loaded_model.eval()
        
        with torch.no_grad():
            loaded_output = loaded_model(sample_input)
        
        assert torch.allclose(original_output, loaded_output)


# =============================================================================
# TESTS: create_model() Factory
# =============================================================================

def test_create_model_default_params():
    """Test factory function with default parameters."""
    model = create_model()
    
    assert model.input_dim == 12
    assert model.d_model == 64


def test_create_model_custom_params():
    """Test factory function with custom parameters."""
    model = create_model(
        input_dim=8,
        d_model=128,
        nhead=8,
        num_layers=4
    )
    
    assert model.input_dim == 8
    assert model.d_model == 128
    assert model.nhead == 8
    assert model.num_layers == 4


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

def test_model_handles_extreme_values(model):
    """Test model handles extreme input values."""
    # Very large values
    large_input = torch.randn(8, 60, 12) * 1000
    output = model.predict(large_input)
    
    # Should still be in [0, 1] due to sigmoid
    assert (output >= 0).all()
    assert (output <= 1).all()


def test_model_handles_zeros(model):
    """Test model handles zero inputs."""
    zero_input = torch.zeros(8, 60, 12)
    output = model.predict(zero_input)
    
    assert output.shape == (8, 1)
    assert not torch.isnan(output).any()


def test_model_handles_nan_in_input(model):
    """Test model behavior with NaN values (should propagate)."""
    nan_input = torch.randn(8, 60, 12)
    nan_input[0, 0, 0] = float("nan")
    
    output = model.predict(nan_input)
    
    # NaN should propagate to output
    assert torch.isnan(output[0]).any()


def test_gradient_flow(model, sample_input):
    """Test that gradients flow through the model."""
    model.train()
    sample_input.requires_grad = True
    
    output = model(sample_input)
    loss = output.sum()
    loss.backward()
    
    assert sample_input.grad is not None
    assert not torch.isnan(sample_input.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
