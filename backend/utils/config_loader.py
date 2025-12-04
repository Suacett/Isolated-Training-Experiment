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


def save_config(
    alpaca_api_key: str = None,
    alpaca_secret_key: str = None,
    alpha_vantage_key: str = None
):
    """
    Saves API keys to backend/secrets.json.
    Preserves existing keys if not provided.
    """
    # Load existing config
    existing = load_config() or {}

    # Update with new values (if provided)
    if alpaca_api_key is not None:
        existing["ALPACA_API_KEY"] = alpaca_api_key
    if alpaca_secret_key is not None:
        existing["ALPACA_SECRET_KEY"] = alpaca_secret_key
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


def is_alpaca_configured() -> bool:
    """Check if Alpaca credentials are configured."""
    config = load_config() or {}
    return bool(config.get("ALPACA_API_KEY") and config.get("ALPACA_SECRET_KEY"))


def is_alpha_vantage_configured() -> bool:
    """Check if Alpha Vantage API key is configured."""
    config = load_config() or {}
    return bool(config.get("ALPHA_VANTAGE_KEY"))


class Settings:
    """
    Settings class providing property access to API keys.
    Supports both Alpaca and Alpha Vantage.
    """

    @property
    def ALPACA_API_KEY(self) -> Optional[str]:
        config = load_config() or {}
        return config.get("ALPACA_API_KEY") or os.getenv("ALPACA_API_KEY")

    @property
    def ALPACA_SECRET_KEY(self) -> Optional[str]:
        config = load_config() or {}
        return config.get("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")

    @property
    def ALPHA_VANTAGE_KEY(self) -> Optional[str]:
        config = load_config() or {}
        return config.get("ALPHA_VANTAGE_KEY") or os.getenv("ALPHA_VANTAGE_KEY")


settings = Settings()
