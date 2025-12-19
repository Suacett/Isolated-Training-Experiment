"""
Bootstrap Service - Handles application initialization, logging, and scheduler setup.
Extracted from main.py for cleaner backend architecture (Phase 5.1).
"""

import logging
import re
import sys
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config.constants import DEFAULT_TICKERS, PAPER_TRADING_SCHEDULE
from config.settings import MODELS_DIR_PATH
from services.db import get_watchlist, add_watchlist_item
from services.data_ingest import YahooFinanceClient
from services.device_utils import get_device
from services.scaler import FeatureScaler
from services.lstm_model import LSTMModel
from services.transformer_model import TransformerRankModel
from utils.config_loader import settings, is_alpha_vantage_configured
from state import state

logger = logging.getLogger(__name__)

# Model config defaults
MODEL_VERSIONS = ["v9", "v7", "v6", "v5", "v4", "v3", "v2"]


class APIKeyFilter(logging.Filter):
    """Filter to redact API keys from log messages."""
    def filter(self, record):
        msg = record.getMessage()
        if "apikey=" in msg:
            record.msg = re.sub(r'apikey=[^&"\s]+', 'apikey=REDACTED', msg)
            record.args = () 
        return True


def setup_logging(log_file: Path):
    """Configure system-wide logging with API key masking."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Apply filter to root logger handlers
    for handler in logging.getLogger().handlers:
        handler.addFilter(APIKeyFilter())
    
    logger.info("🚀 Logging initialized with API key redaction")


async def initialize_app_state():
    """
    Perform device detection, model/scaler loading, and initial state setup.
    """
    # 1. Device detection
    state.device = get_device()
    logger.info(f"Computing Device: {state.device}")

    # 2. Determine best available model/scaler
    model_weights_path, scaler_path, active_version = _resolve_model_paths()
    state.active_model_version = active_version # Store in state for routers

    # 3. Load Scaler
    state.scaler = FeatureScaler(str(scaler_path))
    if scaler_path.exists():
        if state.scaler.load():
            logger.info(f"✅ Loaded feature scaler: {scaler_path.name}")
        else:
            logger.warning(f"⚠️ Failed to load scaler: {scaler_path}")
    else:
        logger.warning(f"⚠️ Scaler not found at {scaler_path}")

    # 4. Load AI Model
    await _load_model(model_weights_path, active_version)

    # 5. Initialize Alpha Vantage Status
    _initialize_external_services()

    # 6. Seed Watchlist
    await _seed_watchlist()

    logger.info("✨ Application state initialization complete")


def _resolve_model_paths():
    """Find the highest version model and scaler in the models directory."""
    models_dir = MODELS_DIR_PATH
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Defaults
    weights_path = models_dir / "lstm_model_v2.pth"
    scaler_path = models_dir / "scaler_v2.pkl"
    version = "v2"

    for v in MODEL_VERSIONS:
        m_path = models_dir / ("transformer_v9.pth" if v == "v9" else f"lstm_model_{v}.pth")
        s_path = models_dir / f"scaler_{v}.pkl"
        
        if m_path.exists() and s_path.exists():
            weights_path, scaler_path, version = m_path, s_path, v
            break
            
    return weights_path, scaler_path, version


async def _load_model(path: Path, version: str):
    """Internal helper to load the specific model version."""
    if not path.exists():
        logger.warning(f"⚠️ No model found at {path}. Initializing empty model.")
        state.lstm_model = LSTMModel(input_dim=37, window_size=60, device=state.device)
        return

    try:
        if version == "v9":
            state.lstm_model = TransformerRankModel.load(str(path), device=state.device)
            logger.info(f"✅ Loaded V9 Transformer: {path.name}")
        else:
            state.lstm_model = LSTMModel.load(str(path), device=state.device)
            logger.info(f"✅ Loaded LSTM {version}: {path.name}")
    except Exception as e:
        logger.error(f"❌ Failed to load model {path.name}: {e}")
        state.lstm_model = LSTMModel(input_dim=37, window_size=60, device=state.device)


def _initialize_external_services():
    """Verify API keys and set relevant flags in GlobalState."""
    if is_alpha_vantage_configured():
        av_key = settings.ALPHA_VANTAGE_KEY
        if av_key and "placeholder" in av_key.lower():
            state.alpha_vantage_enabled = False
            logger.warning("⚠️ Alpha Vantage key is a placeholder - features disabled")
        else:
            state.alpha_vantage_enabled = True
            logger.info("✅ Alpha Vantage configured")
    else:
        state.alpha_vantage_enabled = False


async def _seed_watchlist():
    """Seed the database with default symbols if empty."""
    try:
        watchlist = await get_watchlist()
        if not watchlist:
            logger.info("📋 Watchlist empty - seeding defaults...")
            # yf = YahooFinanceClient() - Removed unused client
            for ticker in DEFAULT_TICKERS:
                await add_watchlist_item(ticker)
                # Async ingestion is preferred here but for startup we just ensure they exist
                # Detailed ingestion usually happens in background
    except Exception as e:
        logger.warning(f"⚠️ Watchlist seed failed: {e}")


def setup_scheduler() -> AsyncIOScheduler:
    """Initialize and start the background task scheduler."""
    scheduler = AsyncIOScheduler()
    
    # Add Daily Paper Trading Run
    scheduler.add_job(
        _run_daily_paper_trading,
        CronTrigger(
            hour=PAPER_TRADING_SCHEDULE["HOUR"],
            minute=PAPER_TRADING_SCHEDULE["MINUTE"],
            day_of_week=PAPER_TRADING_SCHEDULE["DAY_OF_WEEK"],
            timezone=PAPER_TRADING_SCHEDULE["TIMEZONE"]
        ),
        id="daily_paper_trading",
        name="V9 Paper Trading Daily Run"
    )
    
    scheduler.start()
    logger.info("📅 Background scheduler started")
    return scheduler


async def _run_daily_paper_trading():
    """Executor for the paper trader script using non-blocking Popen."""
    logger.info("🤖 Starting scheduled paper trading run...")
    try:
        # Use create_subprocess_exec for true async non-blocking execution
        process = await asyncio.create_subprocess_exec(
            "python", "-m", "scripts.paper_trader_v9", "--run-day",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            cwd="/app"
        )
        logger.info(f"✅ Paper trading process started (PID: {process.pid})")
    except Exception as e:
        logger.error(f"❌ Error starting scheduled paper trade: {e}")
