import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app, WatchlistItem

client = TestClient(app)

@pytest.fixture
def mock_db_session():
    with patch("services.db.AsyncSessionLocal") as mock:
        yield mock

@pytest.fixture
def mock_alpaca():
    with patch("services.data_ingest.AlpacaDataClient") as mock:
        yield mock

def test_watchlist_crud(mock_db_session):
    # Mock DB execution
    mock_session = AsyncMock()
    mock_db_session.return_value.__aenter__.return_value = mock_session
    
    # 1. Add to Watchlist
    response = client.post("/watchlist", json={"ticker": "AAPL"})
    assert response.status_code == 200
    assert response.json() == {"message": "Added AAPL to watchlist"}
    
    # 2. Get Watchlist (Mock return)
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = ["AAPL"]
    mock_session.execute.return_value = mock_result
    
    response = client.get("/watchlist")
    assert response.status_code == 200
    assert response.json() == {"watchlist": ["AAPL"]}

@patch("main.get_watchlist")
@patch("main.AlpacaDataClient")
@patch("main.get_historical_data")
@patch("main.lstm_model")
@patch("main.save_prediction")
def test_ingest_all(mock_save_pred, mock_lstm, mock_hist_data, mock_alpaca_client, mock_get_watchlist):
    # Setup mocks
    mock_get_watchlist.side_effect = AsyncMock(return_value=["AAPL"])
    mock_hist_data.side_effect = AsyncMock(return_value=[])
    mock_save_pred.side_effect = AsyncMock()
    
    mock_client_instance = MagicMock()
    mock_client_instance.fetch_all_data = AsyncMock(return_value=None)
    mock_alpaca_client.return_value = mock_client_instance
    
    print(f"DEBUG: mock_get_watchlist: {mock_get_watchlist}")
    
    # Mock historical data (60 days)
    mock_data = []
    for i in range(60):
        mock_obj = MagicMock()
        mock_obj.open = 100.0
        mock_obj.high = 105.0
        mock_obj.low = 95.0
        mock_obj.close = 102.0
        mock_obj.volume = 1000.0
        mock_data.append(mock_obj)
    
    # Update return value for hist data
    mock_hist_data.side_effect = AsyncMock(return_value=mock_data)
    
    mock_lstm.predict.return_value = 105.0
    
    # Run Ingest All
    response = client.post("/ingest/all")
    
    # Verify
    assert response.status_code == 200
    assert response.json()["message"] == "Ingestion and prediction complete"
    assert len(response.json()["predictions"]) == 1
    assert response.json()["predictions"][0]["ticker"] == "AAPL"
    assert response.json()["predictions"][0]["prediction"] == 105.0
    
    # Verify calls
    mock_client_instance.fetch_all_data.assert_called_with(["AAPL"])
    mock_save_pred.assert_called_once()
