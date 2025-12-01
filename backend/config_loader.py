import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Centralized configuration loader.
    """
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:password@timescaledb:5432/stock_db")
    
    @classmethod
    def validate(cls):
        """
        Validates that critical environment variables are set.
        """
        if not cls.ALPACA_API_KEY:
            raise ValueError("ALPACA_API_KEY is not set")
        if not cls.ALPACA_SECRET_KEY:
            raise ValueError("ALPACA_SECRET_KEY is not set")

# Validate on import (optional, but good for fail-fast)
# Config.validate()
