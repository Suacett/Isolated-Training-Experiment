import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import asyncio

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from services.alpha_vantage_ingest import AlphaVantageClient

class TestAlphaVantageClient(unittest.TestCase):
    def setUp(self):
        os.environ["ALPHA_VANTAGE_KEY"] = "test_key"

    @patch("services.alpha_vantage_ingest.settings")
    @patch("services.alpha_vantage_ingest.requests.get")
    @patch("services.alpha_vantage_ingest.time.sleep")
    @patch("services.alpha_vantage_ingest.AsyncSessionLocal")
    def test_fetch_daily_data(self, mock_session, mock_sleep, mock_get, mock_settings):
        # Mock settings to return test API key
        mock_settings.ALPHA_VANTAGE_KEY = "test_key"

        # Mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Time Series (Daily)": {
                "2023-10-27": {
                    "1. open": "100.0",
                    "2. high": "105.0",
                    "3. low": "99.0",
                    "4. close": "102.0",
                    "5. volume": "10000"
                }
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Mock DB Session
        mock_db_session = MagicMock()
        # execute and commit must be awaitable
        mock_db_session.execute = MagicMock()
        mock_db_session.execute.return_value = asyncio.Future()
        mock_db_session.execute.return_value.set_result(None)
        
        mock_db_session.commit = MagicMock()
        mock_db_session.commit.return_value = asyncio.Future()
        mock_db_session.commit.return_value.set_result(None)

        mock_session.return_value.__aenter__.return_value = mock_db_session
        mock_session.return_value.__aexit__.return_value = asyncio.Future()
        mock_session.return_value.__aexit__.return_value.set_result(None)

        client = AlphaVantageClient()
        
        # Run async method
        asyncio.run(client.fetch_daily_data("TEST"))

        # Verify rate limit sleep was called (current rate limit is 12s)
        mock_sleep.assert_called_with(12)

        # Verify API called with correct params
        mock_get.assert_called_with(
            "https://www.alphavantage.co/query",
            params={
                "function": "TIME_SERIES_DAILY",
                "symbol": "TEST",
                "apikey": "test_key",
                "outputsize": "compact"  # Default is compact for daily fetches
            }
        )

        # Verify DB interaction (execute and commit called)
        self.assertTrue(mock_db_session.execute.called)
        self.assertTrue(mock_db_session.commit.called)

if __name__ == "__main__":
    unittest.main()
