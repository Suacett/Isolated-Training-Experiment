"""
Proxmox AI Stock Predictor - Main FastAPI Application
Lightweight entry point refactored in Phase 5.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from state import state
from services import bootstrap
from routers import (
    ingestion, predictions, dashboard, backtest, 
    stocks, forecasts, playground, paper, 
    portfolio_comparison, system, settings
)
from contextlib import asynccontextmanager
import os

# Initialize logging before any other operations
LOG_FILE = Path("backend.log")
bootstrap.setup_logging(LOG_FILE)
from services.bootstrap import logger

# CORS Configuration
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle application lifecycle using the modern lifespan pattern.
    Replaces deprecated @app.on_event("startup") and "shutdown".
    """
    # Startup
    await bootstrap.initialize_app_state()
    state.scheduler = bootstrap.setup_scheduler()
    
    yield
    
    # Shutdown
    if state.scheduler:
        try:
            if state.scheduler.running:
                state.scheduler.shutdown(wait=False)
                logger.info("Scheduler shut down successfully.")
        except Exception as e:
            logger.error(f"Error during scheduler shutdown: {e}")

# Global for scheduler to allow shutdown
state.scheduler = None

app = FastAPI(
    title="Proxmox AI Stock Predictor",
    description="LSTM-based stock prediction API with multi-horizon forecasting",
    version="1.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Modular Routers
# Note: prefixes are defined within the routers themselves usually,
# but for stocks we mount it twice for legacy /watchlist support.
app.include_router(stocks.router, prefix="/stocks")
app.include_router(stocks.router, prefix="/watchlist")

app.include_router(ingestion.router)      # /ingest
app.include_router(predictions.router)    # /predictions
app.include_router(dashboard.router)      # /dashboard
app.include_router(backtest.router)       # /backtest
app.include_router(forecasts.router)      # /forecasts
app.include_router(playground.router)     # /playground
app.include_router(paper.router)          # /paper
app.include_router(portfolio_comparison.router) # /compare
app.include_router(system.router)         # /system
app.include_router(settings.router)       # /settings

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Proxmox AI Stock Predictor Backend is running",
        "version": "1.1.0",
        "status": "ready"
    }
