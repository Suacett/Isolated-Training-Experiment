"""Environment-based settings that can be overridden."""
import os
from pathlib import Path
from . import constants

# Environment-based settings (can be overridden)
API_BASE_URL = os.getenv("API_BASE_URL", constants.DEFAULT_API_BASE_URL)
PAPER_SESSION_ID = os.getenv("PAPER_SESSION_ID", constants.DEFAULT_PAPER_SESSION_ID)

# Database
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:password@timescaledb:5432/stock_db"
)

# Model paths (absolute)
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR_PATH = BASE_DIR / constants.MODELS_DIR
DATA_DIR_PATH = BASE_DIR / constants.DATA_DIR
