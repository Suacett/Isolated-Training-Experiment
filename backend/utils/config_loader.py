"""
Configuration Loader - Simplified (No Alpaca)

Loads API keys and settings from secrets.json and environment variables.
Only Alpha Vantage needed for EPS data (intrinsic value).
Yahoo Finance requires no API key.
"""

import json
import os
from typing import Optional, Dict
from dotenv import load_dotenv

load_dotenv()

SECRETS_FILE = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "secrets.json"))


def load_config() -> Optional[Dict[str, str]]:
    """
    Attempts to read backend/secrets.json.
    Returns the full config dict, or None if file doesn't exist.
    """
    if not os.path.exists(SECRETS_FILE):
        return None

    try:
        with open(SECRETS_FILE, "r") as f:
            data = json.load(f)
        return data
    except Exception:
        return None


def save_config(alpha_vantage_key: str = None):
    """
    Saves API keys to backend/secrets.json.
    Only Alpha Vantage needed (Yahoo Finance is free).
    """
    existing = load_config() or {}

    if alpha_vantage_key is not None:
        existing["ALPHA_VANTAGE_KEY"] = alpha_vantage_key

    with open(SECRETS_FILE, "w") as f:
        json.dump(existing, f, indent=4)


def clear_config():
    """
    Clears all API keys from backend/secrets.json.
    """
    with open(SECRETS_FILE, "w") as f:
        json.dump({}, f, indent=4)


def is_alpha_vantage_configured() -> bool:
    """Check if Alpha Vantage API key is configured."""
    config = load_config() or {}
    key = config.get("ALPHA_VANTAGE_KEY") or os.getenv("ALPHA_VANTAGE_KEY")
    return bool(key and key != "your_alpha_vantage_key_here")


class Settings:
    """
    Settings class providing property access to API keys.
    Yahoo Finance requires no API key.
    Alpha Vantage needed only for EPS data (intrinsic value).
    """

    @property
    def ALPHA_VANTAGE_KEY(self) -> Optional[str]:
        config = load_config() or {}
        return config.get("ALPHA_VANTAGE_KEY") or os.getenv("ALPHA_VANTAGE_KEY")


settings = Settings()


def get_playground_enabled() -> bool:
    """
    Check if Model Playground mode is enabled.
    This setting persists across restarts via secrets.json.
    Defaults to False to prevent resource issues on weaker devices.
    """
    config = load_config() or {}
    return config.get("PLAYGROUND_ENABLED", False)


def set_playground_enabled(enabled: bool):
    """
    Enable or disable Model Playground mode.
    Persists to secrets.json for restart persistence.
    """
    existing = load_config() or {}
    existing["PLAYGROUND_ENABLED"] = enabled
    
    with open(SECRETS_FILE, "w") as f:
        json.dump(existing, f, indent=4)
