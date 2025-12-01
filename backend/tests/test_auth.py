import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from utils import config_loader

client = TestClient(app)

@pytest.fixture
def mock_secrets_file(tmp_path):
    test_secrets = tmp_path / "secrets.json"
    with patch("utils.config_loader.SECRETS_FILE", str(test_secrets)):
        yield test_secrets

def test_auth_flow_new(mock_secrets_file):
    response = client.get("/status")
    assert response.status_code == 200
    
    payload = {"ALPACA_API_KEY": "test_api", "ALPACA_SECRET_KEY": "test_secret"}
    response = client.post("/settings/keys", json=payload)
    assert response.status_code == 200
    
    with open(mock_secrets_file, "r") as f:
        content = f.read()
        print(f"CONTENT: {content}")
        json_data = json.loads(content)
        assert json_data["ALPACA_API_KEY"] == "test_api"
