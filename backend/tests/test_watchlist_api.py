import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from backend.main import app, get_watchlist, add_watchlist_item, remove_watchlist_item

client = TestClient(app)

# Mock DB functions
@pytest.fixture
def mock_db_functions():
    with patch("backend.main.add_watchlist_item", new_callable=AsyncMock) as mock_add, \
         patch("backend.main.remove_watchlist_item", new_callable=AsyncMock) as mock_remove, \
         patch("backend.main.get_watchlist", new_callable=AsyncMock) as mock_get, \
         patch("backend.main.get_latest_close", new_callable=AsyncMock) as mock_close, \
         patch("backend.main.get_historical_data", new_callable=AsyncMock) as mock_hist:
        yield mock_add, mock_remove, mock_get, mock_close, mock_hist

# Mock Alpaca Client
@pytest.fixture
def mock_alpaca():
    with patch("backend.main.alpaca_client") as mock:
        mock.fetch_data = AsyncMock()
        yield mock

def test_add_watchlist_item(mock_db_functions, mock_alpaca):
    mock_add, _, _, _, _ = mock_db_functions

    response = client.post("/watchlist", json={"ticker": "AAPL"})
    assert response.status_code == 200
    assert response.json() == {"message": "Added AAPL to watchlist"}

    mock_add.assert_called_once_with("AAPL")

def test_remove_watchlist_item(mock_db_functions):
    _, mock_remove, _, _, _ = mock_db_functions

    response = client.delete("/watchlist/AAPL")
    assert response.status_code == 200
    assert response.json() == {"message": "Removed AAPL from watchlist"}

    mock_remove.assert_called_once_with("AAPL")

def test_get_dashboard(mock_db_functions):
    _, _, mock_get, mock_close, mock_hist = mock_db_functions
    
    mock_get.return_value = ["AAPL"]
    mock_close.return_value = 150.0
    mock_hist.return_value = [] # Empty for simplicity
    
    response = client.get("/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["ticker"] == "AAPL"
    assert data[0]["current_price"] == 150.0
