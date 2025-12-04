import pytest
import torch
from backend.services.lstm_model import LSTMModel, get_device


def test_lstm_initialization_and_forward_pass(capsys):
    """Test model initializes correctly and produces expected output shape."""
    # Parameters
    input_dim = 10
    seq_len = 30
    batch_size = 5
    output_dim = 4

    # Initialize model (consolidated API uses window_size instead of seq_len)
    model = LSTMModel(input_dim=input_dim, window_size=seq_len, output_dim=output_dim)

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
    """Test that model respects explicit device selection."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = LSTMModel(input_dim=5, window_size=10, device=device)
        assert next(model.parameters()).is_cuda

    device = torch.device("cpu")
    model = LSTMModel(input_dim=5, window_size=10, device=device)
    assert not next(model.parameters()).is_cuda


def test_predict_single():
    """Test single prediction returns a float."""
    model = LSTMModel(input_dim=5, window_size=60, device=torch.device("cpu"))
    x = torch.randn(60, 5)  # (seq_len, features)

    prediction = model.predict(x, torch.device("cpu"))

    assert isinstance(prediction, float)


def test_predict_batch():
    """Test batch prediction returns correct shape."""
    model = LSTMModel(input_dim=5, window_size=60, device=torch.device("cpu"))
    batch_size = 10
    x = torch.randn(batch_size, 60, 5)  # (batch, seq_len, features)

    predictions = model.predict_batch(x, torch.device("cpu"))

    assert predictions.shape == (batch_size,)


def test_predict_with_uncertainty():
    """Test Monte Carlo uncertainty estimation."""
    model = LSTMModel(input_dim=5, window_size=60, device=torch.device("cpu"))
    x = torch.randn(3, 60, 5)  # Small batch

    result = model.predict_with_uncertainty(x, torch.device("cpu"), n_samples=5)

    assert "prediction" in result
    assert "std" in result
    assert "confidence" in result
    assert "should_act" in result
    assert len(result["prediction"]) == 3
    assert all(0 <= c <= 1 for c in result["confidence"])


def test_get_device():
    """Test get_device helper function."""
    device = get_device()
    assert device.type in ("cuda", "cpu")


def test_model_save_and_load(tmp_path):
    """Test saving and loading model weights."""
    # Create and configure model
    model = LSTMModel(input_dim=5, window_size=60, device=torch.device("cpu"))

    # Create some test input
    x = torch.randn(1, 60, 5)

    # Save model
    save_path = tmp_path / "test_model.pth"
    model.save(str(save_path))

    assert save_path.exists()

    # Load model
    loaded_model = LSTMModel.load(str(save_path), torch.device("cpu"))

    # Verify architecture matches
    assert loaded_model.input_dim == model.input_dim
    assert loaded_model.hidden_dim == model.hidden_dim
    assert loaded_model.output_dim == model.output_dim
    assert loaded_model.window_size == model.window_size

    # Note: Due to MC Dropout, predictions won't be identical
    # but the model should load without errors
    loaded_pred = loaded_model.predict(x, torch.device("cpu"))
    assert isinstance(loaded_pred, float)
