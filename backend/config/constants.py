"""Centralized constants for the application."""

# API Configuration
DEFAULT_API_BASE_URL = "http://localhost:8000"

# Session Management
DEFAULT_PAPER_SESSION_ID = "v9_golden_2025"

# Signal Thresholds
SIGNAL_THRESHOLDS = {
    "BUY_MULTIPLIER": 1.02,
    "SELL_MULTIPLIER": 0.98,
    "SELL_RANK_THRESHOLD": 0.2,
    "DEEP_VALUE_MULTIPLIER": 0.5,
}

# ATR Risk Parameters
ATR_PARAMS = {
    "PERIOD": 14,
    "MULTIPLIER": 2.5,
    "FALLBACK_STOP": 0.88,
    "MIN_STOP": 0.97,
}

# Drawdown Thresholds (exposure scaling)
DRAWDOWN_THRESHOLDS = [
    (-0.15, 1.00),  # Under 15% DD: full exposure
    (-0.20, 0.80),  # 15-20% DD: reduce 20%
    (-0.25, 0.60),  # 20-25% DD: reduce 40%
    (-0.30, 0.40),  # 25-30% DD: reduce 60%
    (-1.00, 0.20),  # >30% DD: defensive
]

# Market Regime Detection
REGIME_PARAMS = {
    "SMA_PERIOD": 200,
    "VIX_LOW": 20,
    "VIX_MODERATE": 25,
    "VIX_CRISIS": 30,
    "RISK_FREE_RATE": 0.02,
}

# Model Configuration
MODEL_DIMENSIONS = {
    "v2": {"input_dim": 37, "window_size": 60},
    "v7": {"input_dim": 41, "window_size": 60},
    "v8": {"input_dim": 41, "window_size": 91},
    "v9": {"input_dim": 12, "window_size": 60},
}

# Paths (relative to backend/)
MODELS_DIR = "models"
DATA_DIR = "data"
LEGACY_DATA_PATH = "/app/legacy/LSTM_AI_Stock_Predictor/TrainingData/indicators_data/processed/stocksData"

# Scheduling
PAPER_TRADING_SCHEDULE = {
    "HOUR": 16,
    "MINUTE": 30,
    "DAY_OF_WEEK": "mon-fri",
    "TIMEZONE": "America/New_York",
}

# Default Values
DEFAULT_TICKERS = ["SPY", "BTC-USD", "AAPL", "AMD", "TSLA", "AMZN"]
INITIAL_PORTFOLIO_CASH = 10000.0
DEFAULT_REBALANCE_DAYS = 5

# Volatility & Risk Calculations
VOLATILITY_PARAMS = {
    "LOOKBACK_PERIOD": 20,
    "TRADING_DAYS_PER_YEAR": 252,
}
