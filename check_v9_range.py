
import torch
import sys
from pathlib import Path

# Setup backend path
BACKEND_DIR = Path("/home/brock/Ai trading/Isolated-Training-Experiment/backend")
sys.path.insert(0, str(BACKEND_DIR))

from services.transformer_model import TransformerRankModel

def check_model():
    model_path = BACKEND_DIR / "models" / "transformer_v9_best.pth"
    device = torch.device("cpu")
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = TransformerRankModel.load(str(model_path), device=device)
    model.eval()
    
    # Create dummy input: (batch=1, seq=60, features=12)
    x = torch.randn(1, 60, 12)
    
    with torch.no_grad():
        out = model(x)
        print(f"Model output: {out}")
        print(f"Min: {out.min().item():.4f}, Max: {out.max().item():.4f}")
        
        if out.min() < 0 or out.max() > 1:
            print("🚨 BUG CONFIRMED: Model output is OUTSIDE [0, 1] range! Missing Sigmoid.")
        else:
            print("✅ Model output is within [0, 1] range. Sigmoid is active.")

if __name__ == "__main__":
    check_model()
