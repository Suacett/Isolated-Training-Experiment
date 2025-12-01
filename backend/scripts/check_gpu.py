import torch
import sys
import os

# Add backend to path to import services
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.lstm_model import get_device, LSTMModel

def check_gpu():
    print("Checking GPU availability...")
    if torch.cuda.is_available():
        print(f"✅ CUDA is available.")
        print(f"   Device Count: {torch.cuda.device_count()}")
        print(f"   Current Device: {torch.cuda.current_device()}")
        print(f"   Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA is NOT available. Running on CPU.")

    device = get_device()
    
    # Create a dummy model to verify it moves to device
    model = LSTMModel(input_dim=10, window_size=90).to(device)
    print(f"✅ Model successfully moved to {device}")
    
    # Create dummy input
    dummy_input = torch.randn(1, 90, 10).to(device)
    output = model(dummy_input)
    print(f"✅ Forward pass successful. Output shape: {output.shape}")

if __name__ == "__main__":
    check_gpu()
