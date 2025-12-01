import torch
import pytest
from services.lstm_model import LSTMModel, get_device

def test_lstm_predict():
    device = get_device()
    model = LSTMModel(input_dim=5, window_size=60)
    model.to(device)
    
    # Create dummy input: (Seq_Len, Input_Dim)
    dummy_input = torch.randn(60, 5)
    
    prediction = model.predict(dummy_input, device)
    
    assert isinstance(prediction, float)
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    test_lstm_predict()
