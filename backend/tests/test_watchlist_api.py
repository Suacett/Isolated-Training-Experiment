"""
Tests for watchlist API endpoints.

These tests verify CRUD operations on the watchlist.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

# Import app after patching to avoid startup issues
with patch("main.get_watchlist", new_callable=AsyncMock) as mock_wl:
    mock_wl.return_value = []
    from main import app

client = TestClient(app)


# Mock DB functions
@pytest.fixture
def mock_db_functions():
    with patch("main.add_watchlist_item", new_callable=AsyncMock) as mock_add, \
         patch("main.remove_watchlist_item", new_callable=AsyncMock) as mock_remove, \
         patch("main.get_watchlist", new_callable=AsyncMock) as mock_get, \
         patch("main.get_latest_close", new_callable=AsyncMock) as mock_close, \
         patch("main.get_historical_data", new_callable=AsyncMock) as mock_hist, \
         patch("main.get_watchlist_with_favorites", new_callable=AsyncMock) as mock_wl_fav, \
         patch("main.get_all_cached_intrinsic_values", new_callable=AsyncMock) as mock_intrinsic:
        mock_intrinsic.return_value = {}
        yield mock_add, mock_remove, mock_get, mock_close, mock_hist, mock_wl_fav


def test_add_watchlist_item(mock_db_functions):
    mock_add, _, _, _, _, _ = mock_db_functions

    response = client.post("/watchlist", json={"ticker": "AAPL"})
    assert response.status_code == 200
    assert response.json() == {"message": "Added AAPL to watchlist"}

    mock_add.assert_called_once_with("AAPL")


def test_remove_watchlist_item(mock_db_functions):
    _, mock_remove, _, _, _, _ = mock_db_functions

    response = client.delete("/watchlist/AAPL")
    assert response.status_code == 200
    assert response.json() == {"message": "Removed AAPL from watchlist"}

    mock_remove.assert_called_once_with("AAPL")


def test_get_dashboard(mock_db_functions):
    _, _, mock_get, mock_close, mock_hist, mock_wl_fav = mock_db_functions
    
    mock_wl_fav.return_value = [{"ticker": "AAPL", "is_favorite": True}]
    mock_close.return_value = 150.0
    mock_hist.return_value = []  # Empty for simplicity
    
    response = client.get("/dashboard")
    assert response.status_code == 200
    data = response.json()
    # Dashboard returns list of summaries - verify it works
    assert isinstance(data, list)

