import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import torch
import numpy as np

# Need to import app after mocking potentially, or patch where it's used
from backend.main import app

client = TestClient(app)

# Mock DB functions
@pytest.fixture
def mock_db_functions():
    with patch("backend.main.get_latest_close", new_callable=AsyncMock) as mock_close, \
         patch("backend.main.get_historical_data", new_callable=AsyncMock) as mock_hist:
        yield mock_close, mock_hist

# Mock LSTM Model
@pytest.fixture
def mock_lstm():
    with patch("backend.main.lstm_model") as mock:
        mock.predict_batch = MagicMock()
        yield mock

def test_get_dashboard_detail(mock_db_functions, mock_lstm):
    mock_close, mock_hist = mock_db_functions
    
    # Mock latest close
    mock_close.return_value = 150.0
    
    # Mock historical data (150 days)
    # Create dummy objects with attributes
    class StockData:
        def __init__(self, date, open, high, low, close, volume):
            self.timestamp = date
            self.open = open
            self.high = high
            self.low = low
            self.close = close
            self.volume = volume

    start_date = datetime(2023, 1, 1)
    data = []
    for i in range(150):
        d = start_date + timedelta(days=i)
        data.append(StockData(d, 100.0 + i, 105.0 + i, 95.0 + i, 102.0 + i, 1000.0))
    
    mock_hist.return_value = data
    
    # Mock LSTM predictions (should be called with batch size 90)
    # 150 total, window 60 -> 150 - 60 = 90 predictions
    mock_lstm.predict_batch.return_value = torch.tensor([103.0 + i for i in range(90)])
    
    response = client.get("/dashboard/AAPL")
    assert response.status_code == 200
    
    json_data = response.json()
    assert "history" in json_data
    history = json_data["history"]
    
    # Should return last 90 days
    assert len(history) == 90
    
    # Check first item in history (which corresponds to index 60 in data)
    first_item = history[0]
    assert first_item["date"] == (start_date + timedelta(days=60)).strftime("%Y-%m-%d")
    assert first_item["predicted_close"] == 103.0
    assert first_item["intrinsic_value"] == 142.0
    
    # Check last item
    last_item = history[-1]
    assert last_item["date"] == (start_date + timedelta(days=149)).strftime("%Y-%m-%d")
    assert last_item["predicted_close"] == 103.0 + 89
