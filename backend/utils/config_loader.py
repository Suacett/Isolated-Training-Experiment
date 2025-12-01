import json
import os
from typing import Optional, Dict

SECRETS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "secrets.json")

def load_config() -> Optional[Dict[str, str]]:
    """
    Attempts to read backend/secrets.json.
    If the file doesn't exist or keys are missing, return None.
    """
    if not os.path.exists(SECRETS_FILE):
        return None
    
    try:
        with open(SECRETS_FILE, "r") as f:
            data = json.load(f)
            
        api_key = data.get("ALPACA_API_KEY")
        secret_key = data.get("ALPACA_SECRET_KEY")
        
        if not api_key or not secret_key:
            return None
            
        return {"ALPACA_API_KEY": api_key, "ALPACA_SECRET_KEY": secret_key}
    except Exception:
        return None

def save_config(api_key: str, secret_key: str):
    """
    Saves ALPACA_API_KEY and ALPACA_SECRET_KEY to backend/secrets.json.
    """
    data = {"ALPACA_API_KEY": api_key, "ALPACA_SECRET_KEY": secret_key}
    with open(SECRETS_FILE, "w") as f:
        json.dump(data, f, indent=4)

class Config:
    @property
    def ALPACA_API_KEY(self):
        config = load_config() or {}
        return config.get("ALPACA_API_KEY")

    @property
    def ALPACA_SECRET_KEY(self):
        config = load_config() or {}
        return config.get("ALPACA_SECRET_KEY")

Config = Config()
