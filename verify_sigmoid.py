
import torch
import sys
import os

# Try to find the file manually
paths = [
    "/home/brock/Ai trading/Isolated-Training-Experiment/backend/services/transformer_model.py",
    "backend/services/transformer_model.py"
]

for p in paths:
    if os.path.exists(p):
        print(f"Found file at: {p}")
        with open(p, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "Sigmoid" in line:
                    print(f"L{i+1}: {line.strip()}")

# Test output
sys.path.insert(0, "/home/brock/Ai trading/Isolated-Training-Experiment/backend")
from services.transformer_model import TransformerRankModel

model = TransformerRankModel(input_dim=12, d_model=64)
x = torch.randn(1, 60, 12)
out = model(x)
print(f"Model output: {out}")
print(f"Range: {out.min().item()} to {out.max().item()}")
