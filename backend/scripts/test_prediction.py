#!/usr/bin/env python3
"""Quick test script to verify the trained model works"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from services.lstm_model import LSTMModel
from services.scaler import FeatureScaler

print("Loading model and scaler...")
model = LSTMModel(input_dim=37, window_size=60)
model.load("backend/models/lstm_model.pth")
model.eval()

scaler = FeatureScaler("backend/models/scaler.pkl")
scaler.load()

print("✓ Model and scaler loaded successfully!")
print(f"✓ Model on device: {model.device}")

# Create dummy data (60 timesteps, 37 features)
dummy_data = np.random.randn(60, 37)
scaled_data = scaler.transform(dummy_data)

# Convert to tensor
X = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0)  # (1, 60, 37)

# Predict
with torch.no_grad():
    predictions = model(X)

print(f"\n✓ Prediction successful!")
print(f"  Input shape: {X.shape}")
print(f"  Output shape: {predictions.shape}")
print(f"  Predictions (log returns):")
print(f"    1-day:   {predictions[0, 0].item():.6f}")
print(f"    1-week:  {predictions[0, 1].item():.6f}")
print(f"    1-month: {predictions[0, 2].item():.6f}")
print(f"    6-month: {predictions[0, 3].item():.6f}")
print("\n🎉 Model is working correctly!")
