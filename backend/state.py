from typing import Optional
import torch
from services.lstm_model import LSTMModel

class GlobalState:
    lstm_model: Optional[LSTMModel] = None
    device: Optional[torch.device] = None
    scaler: Optional[any] = None  # FeatureScaler instance
    alpha_vantage_enabled: bool = False  # Whether Alpha Vantage is properly configured

state = GlobalState()
