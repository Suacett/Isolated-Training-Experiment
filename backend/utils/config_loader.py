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


def save_config(alpha_vantage_key: str = None, alpha_vantage_keys: str = None):
    """
    Saves API keys to backend/secrets.json.
    Only Alpha Vantage needed (Yahoo Finance is free).

    Args:
        alpha_vantage_key: Single API key (legacy, backwards compatible)
        alpha_vantage_keys: Comma-separated list of API keys (preferred for multiple keys)
    """
    existing = load_config() or {}

    if alpha_vantage_key is not None:
        existing["ALPHA_VANTAGE_KEY"] = alpha_vantage_key

    if alpha_vantage_keys is not None:
        # Validate and save multiple keys
        keys_list = [k.strip() for k in alpha_vantage_keys.split(",") if k.strip()]
        if keys_list:
            existing["ALPHA_VANTAGE_KEYS"] = ",".join(keys_list)

    with open(SECRETS_FILE, "w") as f:
        json.dump(existing, f, indent=4)


def clear_config():
    """
    Clears all API keys from backend/secrets.json.
    """
    with open(SECRETS_FILE, "w") as f:
        json.dump({}, f, indent=4)


def is_alpha_vantage_configured() -> bool:
    """
    Check if Alpha Vantage API key(s) are configured.
    Returns True if either single or multiple keys are set.
    """
    config = load_config() or {}

    # Check for multiple keys first (preferred)
    keys_str = config.get("ALPHA_VANTAGE_KEYS") or os.getenv("ALPHA_VANTAGE_KEYS", "")
    if keys_str:
        keys_list = [k.strip() for k in keys_str.split(",") if k.strip()]
        if keys_list:
            return True

    # Fall back to single key (legacy)
    key = config.get("ALPHA_VANTAGE_KEY") or os.getenv("ALPHA_VANTAGE_KEY", "")
    return bool(key and key != "your_alpha_vantage_key_here")


class Settings:
    """
    Settings class providing property access to API keys.
    Yahoo Finance requires no API key.
    Alpha Vantage needed only for EPS data (intrinsic value).
    Supports both single key (ALPHA_VANTAGE_KEY) and multiple keys (ALPHA_VANTAGE_KEYS).
    """

    @property
    def ALPHA_VANTAGE_KEY(self) -> Optional[str]:
        config = load_config() or {}
        return config.get("ALPHA_VANTAGE_KEY") or os.getenv("ALPHA_VANTAGE_KEY")

    @property
    def ALPHA_VANTAGE_KEYS(self) -> Optional[str]:
        """Get comma-separated list of Alpha Vantage API keys."""
        config = load_config() or {}
        return config.get("ALPHA_VANTAGE_KEYS") or os.getenv("ALPHA_VANTAGE_KEYS")


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
