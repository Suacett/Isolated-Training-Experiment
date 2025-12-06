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



# NOTE: test_ingest_all has been removed because it referenced obsolete components:
# - AlpacaDataClient (removed, now using YahooFinanceClient)
# - main.lstm_model (model is now accessed via state.lstm_model)
# The ingestion functionality is tested via the running application's API tests.
