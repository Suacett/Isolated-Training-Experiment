# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Proxmox AI Stock Predictor - A GPU-accelerated stock/crypto prediction platform using PyTorch LSTM models, FastAPI backend, Next.js frontend, and TimescaleDB for time-series storage. Data sourced from Alpaca Markets API.

## Commands

### Docker (Primary Development)
```bash
docker compose up -d --build    # Build and start all services
docker compose down             # Stop services
docker compose logs -f backend  # Follow backend logs
docker compose exec backend pytest  # Run tests in container
```

### Backend (Python/FastAPI)
```bash
# Run tests (from project root)
python -m pytest backend/tests/

# Run single test file
python -m pytest backend/tests/test_lstm.py -v

# Run specific test
python -m pytest backend/tests/test_lstm.py::test_lstm_model_output_shape -v

# Run tests with coverage
python -m pytest backend/tests/ --cov=backend/services

# Utility scripts
python backend/scripts/check_gpu.py   # Verify GPU availability
python backend/scripts/init_db.py     # Initialize database schema

# Manual backend testing
python -c "import torch; print(torch.cuda.is_available())"  # Check CUDA availability
```

### Frontend (Next.js)
```bash
cd frontend
npm run dev      # Development server (port 3000)
npm run build    # Production build
npm run lint     # ESLint
```

### Service Ports
- Frontend: 3001 (external) / 3000 (internal)
- Backend API: 8000
- PostgreSQL: 5432

## Architecture Overview

### Data Flow
```
Yahoo Finance (Primary) → YahooFinanceClient → TimescaleDB → FastAPI → Next.js Dashboard
       ↓ (Fallback)                                              ↑
Alpaca Markets API                                Alpha Vantage (Sentiment/EPS - favorites only)
```

### Backend Services (`backend/services/`)
- **data_ingest.py**: `YahooFinanceClient` (primary) and `AlpacaDataClient` (fallback) fetch OHLCV data, auto-detects crypto vs stocks
- **lstm_model.py**: PyTorch model (Conv1D → BatchNorm → MC Dropout → LSTM → Dense). 60-day input, 37 features, 4-step forecast
- **intrinsic.py**: Graham formula intrinsic value calculation
- **db.py**: SQLAlchemy async models (`StockPrice`, `Watchlist`, `Prediction`, `InsiderTrade`, `SentimentData`)
- **feature_engineering.py**: Computes 37 technical features from OHLCV + insider + sentiment data
- **insider_data.py**: SEC Form 4 scraper for insider trading data
- **sentiment_data.py**: Alpha Vantage News API for sentiment analysis
- **scaler.py**: StandardScaler wrapper for feature normalization
- **training.py**: Complete training pipeline using legacy data
- **reconciliation.py**: Grades past predictions against actual market data for accuracy tracking
- **model_loader.py**: Multi-model loader with VRAM → RAM fallback for Model Playground
- **model_metadata.py**: Model version registry and metadata management

### Signal Logic
```python
BUY:  prediction > current_price * 1.02
SELL: prediction < current_price * 0.98
HOLD: otherwise
```

### Frontend Components (`frontend/app/components/`)
- **SetupModal**: API key configuration
- **AddAssetBar**: Add ticker to watchlist
- **AssetTable**: Portfolio overview table
- **DetailDrawer**: Detailed chart/analysis view

### API Routes (`backend/routers/`)
- **stocks.py**:
  - `GET /stocks/` - List all watchlist tickers
  - `GET /stocks/with-favorites` - List tickers with favorite status
  - `GET /stocks/favorites` - List only favorite tickers
  - `GET /stocks/browse` - Browse categorized popular stocks (Tech, Finance, Healthcare, etc.)
  - `POST /stocks/` - Add ticker with Yahoo Finance validation and hybrid data ingestion
  - `PATCH /stocks/{ticker}/favorite` - Toggle favorite status
  - `DELETE /stocks/{ticker}?cascade=bool` - Remove ticker (optionally cascade delete all data)
- **dashboard.py**:
  - `GET /dashboard/{ticker}` - Historical data with predictions, intrinsic value, SPY comparison (up to 5000 points)
- **backtest.py**:
  - `POST /backtest/all?days=N` - Generate historical predictions for all tickers
  - `POST /backtest/{ticker}?days=N` - Generate historical predictions for chart visualization
- **ingestion.py**:
  - `POST /ingest/all` - Fetch data for all watchlist tickers and generate predictions
- **predictions.py**:
  - `GET /predictions/{ticker}` - Get prediction history for a ticker
  - `GET /predictions/{ticker}/stats` - Get prediction accuracy statistics
  - `GET /predictions/pending/validate` - Get predictions needing validation

### Utility Modules (`backend/utils/`)
- **config_loader.py**: Load/save API keys and settings from `secrets.json`

### Training Scripts (`backend/scripts/`)
- **train_model_v6.py**: RECOMMENDED - Train with all stocks, 128 hidden units, clean data
- **train_model_v5.py**: Train with all stocks, 64 hidden units
- **train_model_v4.py**: Train with 200 stocks (may OOM on low RAM)
- **train_model_v3.py**: Train with 100 stocks
- **train_model_legacy.py**: Original 31 stocks training script
- **check_gpu.py**: Verify CUDA availability
- **init_db.py**: Initialize database schema
- **clean_db.py**: Database maintenance utilities
- **migrate_*.py**: Database migration scripts

---

## Development Guidelines

### Technology Stack
- **Backend**: Python 3.10+, FastAPI (async), SQLAlchemy async, PyTorch CUDA
- **Frontend**: Next.js 16+ (App Router), TypeScript, Tailwind CSS, Recharts
- **Database**: PostgreSQL 16 + TimescaleDB
- **Data Source**: Alpaca Markets API (Official SDK)

### Legacy Source Material

The ML models are ported from `legacy/` directory notebooks:
- **Intrinsic Value**: `legacy/intrinsic-value-monitor/` - Graham formula implementation
- **LSTM Predictor**: `legacy/lstm_ai_stock_predictor/` - PyTorch model architecture and preprocessing

When modifying ML logic, reference these sources rather than inventing new approaches.

### Security Pattern ("Vault")

- API keys entered via Frontend Modal, stored in `backend/secrets.json` (git-ignored)
- Never store secrets in `.env` or commit to Git
- App starts in "Locked" state until keys are configured

### Data Conservation

- Check `StockPrice` table before hitting external API
- If data for current date exists, use cached DB record
- Treat all external APIs as rate-limited resources
- Data source priority: Yahoo Finance (primary) → Alpaca (fallback) → Alpha Vantage (sentiment/EPS only)

### Favorites System

- **Purpose**: Conserve Alpha Vantage API limits by only calculating intrinsic values for selected stocks
- **Behavior**:
  - Regular watchlist items: Yahoo Finance + Alpaca data only (fast, no API limits)
  - Favorite items: Also fetch EPS data from Alpha Vantage for intrinsic value calculations
- **Management**: Use `PATCH /stocks/{ticker}/favorite` to toggle favorite status
- **UI Indicator**: Stars or badges in the frontend mark favorite tickers

### Environment Configuration

- `.env` file contains sensitive configuration (git-ignored):
  - `ALPHA_VANTAGE_KEY`: Alpha Vantage API key for fallback data source
  - `POSTGRES_USER`: Database user (default: postgres)
  - `POSTGRES_PASSWORD`: Database password (default: password)
  - `ALPACA_BASE_URL`: Alpaca API endpoint
    - Paper Trading: `https://paper-api.alpaca.markets`
    - Live Trading: `https://api.alpaca.markets`
    - **Important**: Ensure your API keys (in `secrets.json`) match the endpoint (Paper keys won't work with Live endpoint)
- Frontend secrets (Alpaca API Key/Secret) stored in `backend/secrets.json` at runtime, not in environment

### Hardware Agnosticism

- Never hardcode `cuda:0` - use dynamic `torch.device()` detection
- Backend probes `torch.cuda.is_available()` at startup
- Falls back to CPU with warning if CUDA unavailable

### Coding Standards

**Backend (Python)**
- Type hints everywhere (`def func(x: int) -> str:`)
- Async-first: `async def` for routes and DB interactions
- Separate service logic (ML/math) from router logic (HTTP)
- Graceful degradation: if ML fails, return Intrinsic Value alone

**Frontend (TypeScript)**
- No `any` types - define interfaces for all API responses
- Atomic components preferred over monolithic pages
- Loading states required for all async actions

### TDD Workflow

1. Reference relevant `legacy/` source material
2. Write failing pytest in `backend/tests/`
3. Implement service code to pass test
4. Run full test suite: `docker compose exec backend pytest`

Hot reload is enabled via Docker volume mappings.

---

## Model Training

### Training Scripts

Multiple training scripts are available with different trade-offs:

| Script | Stocks | Memory | Speed | Use Case |
|--------|--------|--------|-------|----------|
| `train_model_v6.py` | **ALL** | **Streaming** | Med | **RECOMMENDED - Clean data & bigger model** |
| `train_model_v5.py` | ALL | Streaming | Med | Previous version (smaller model) |
| `train_model_v4.py` | 200 | ~8GB | Med | RAM-based, may OOM |

### Maximum Data Training (v6) - RECOMMENDED

```bash
# Train with ALL stocks, ALL history (recommended for best accuracy)
docker exec proxmox_stock_backend python -m scripts.train_model_v6

# After training, activate:
docker exec proxmox_stock_backend cp /app/models/lstm_model_v6.pth /app/models/lstm_model_v2.pth
docker exec proxmox_stock_backend cp /app/models/scaler_v6.pkl /app/models/scaler_v2.pkl
docker compose restart backend
```

**v6 Improvements:**
- **128 hidden units** (2x bigger than v5)
- **Strict time-based split** (prevents data leakage)
- **Cleaned Data:** Removed 35k duplicate records that caused "permabear" bias
- **Realistic Accuracy:** ~51-54% (vs inflated 68% on dirty data)

**Environment variables:**
- `BATCH_SIZE=64` - GPU batch size
- `ACCUMULATION_STEPS=4` - Gradient accumulation steps
- `STOCKS_PER_BATCH=50` - Stocks loaded at once
- `MAX_STOCKS=500` - Total stocks to use
- `EPOCHS=200` - Training epochs
- `PATIENCE=25` - Early stopping patience

### Activating a Trained Model

After training completes:
```bash
# Copy model files
docker exec proxmox_stock_backend cp /app/models/lstm_model_v4.pth /app/models/lstm_model_v2.pth
docker exec proxmox_stock_backend cp /app/models/scaler_v4.pkl /app/models/scaler_v2.pkl

# Restart backend
docker compose restart backend
```

### Accuracy Benchmarks

| Model | Training Data | SPY Accuracy | AAPL Accuracy |
|-------|--------------|--------------|---------------|
| v3 | 100 stocks | ~54% | ~45% |
| Legacy reference | 31 stocks | ~58% (claimed) | N/A |

**Note:** Accuracy varies by ~5% due to Monte Carlo Dropout randomness in inference.

### Model Architecture (37 Features)

**Input Features:**
- Log returns (5): Yesterday's OHLCV returns
- Moving averages (5): MA10/20/30, EMA10/30
- Time features (3): Day of week/month, month number
- Technical indicators (5): RSI, MACD, MACD Signal, Bollinger Bands
- Volatility (4): Multiple timeframes (10d, 20d, 30d)
- Volume indicators (2): OBV, abnormal volume
- Price patterns (5): Z-score, overnight gap, momentum, skewness, intraday range
- Insider trading (3): Shares, amount, buy/sell flag *(optional)*
- Sentiment (3): News sentiment score, article count, sentiment change *(optional)*

**Output Horizons:**
- `1d`: 1 day ahead (primary for accuracy calculation)
- `1w`: 5 days ahead
- `1m`: 21 days ahead
- `6m`: 126 days ahead

### Multi-API Key Support (Alpha Vantage)

For EPS/sentiment data, supports multiple keys with automatic rotation:

```bash
# In .env file
ALPHA_VANTAGE_KEYS=key1,key2,key3,key4,key5
```

Each key provides 5 calls/minute. Multiple keys multiply throughput.

### Important Notes

- **Scaler is Critical**: Predictions are meaningless without the fitted scaler from training
- **Feature Consistency**: Inference must use exact same features as training
- **Data Sources**: Training uses database data; production uses Yahoo (primary) + Alpaca (fallback)
- **GPU**: RTX 4080 Super (16GB) recommended for v4 training

---

## Model Playground

The Model Playground feature allows loading and comparing multiple LSTM model versions simultaneously.

### Key Features
- **Multi-Model Loading**: Load all available model versions (v6, v5, v4, v3, legacy, hybrid, etc.)
- **Memory Management**: Automatic VRAM → System RAM fallback when GPU memory is limited
- **Priority Loading**: Loads models to GPU in priority order (v6 → v5 → v3 → others)
- **Live Comparison**: Compare predictions across models for the same ticker
- **Accuracy Metrics**: Calculate recent accuracy for each model

### Architecture (`backend/services/model_loader.py`)
- **MultiModelLoader**: Singleton class managing all loaded models
- **LoadedModel**: Container for model + scaler + device info
- **Memory Estimation**: Monitors GPU memory and falls back to CPU when needed (~200MB per model)
- **Enable/Disable**: Toggle feature on/off to free resources when not needed

### Usage Pattern
```python
from services.model_loader import model_loader

# Enable and load all models
model_loader.enable()
model_loader.load_all_models()

# Get prediction from specific model
result = model_loader.get_prediction("v6", features, current_price)

# Check status
status = model_loader.get_status()

# Cleanup when done
model_loader.disable()
```

---

## DetailDrawer Component (Enhanced)

The DetailDrawer (`frontend/app/components/DetailDrawer.tsx`) provides detailed AI prediction visualization.

### Current Features (2025-12-06)
- **Forecast Cards**: Yesterday's result, Tomorrow, 1 Week, 1 Month, 6 Month predictions
- **Interactive Chart**: Multi-line with Price, AI Prediction, S&P 500, Intrinsic Value
- **View Modes**: Chart vs Table toggle
- **Percentage Mode**: Compare stock vs S&P 500 performance
- **Line Toggles**: Show/hide individual chart lines
- **Time Ranges**: 1M, 3M, 6M, 1Y, ALL (currently missing 1D, 1W)
- **Accuracy Stream**: Clickable bars showing daily prediction accuracy
- **Detail Modal**: Click any bar/row to see prediction details

### Backend Integration
- `GET /dashboard/{ticker}` - Historical data with predictions, intrinsic value, SPY comparison
- `GET /forecasts/{ticker}` - Multi-horizon AI predictions (1d, 1w, 1m, 6m)

---

## Outstanding Issues (TODO)

### DetailDrawer Issues
1. **Performance**: Graph laggy with ALL data (5000+ points) - reduce limit or downsample
2. **Missing Time Ranges**: Add 1D (1 day) and 1W (1 week) options
3. **Info Bubbles Blocked**: Forecast card tooltips cut off - fix z-index/overflow
4. **More Prediction Details Needed**:
   - Show expected % movement (not just price)
   - Explain confidence score meaning
   - Explain intrinsic value calculation
   - Show error rate distribution
5. **Model Loading Timing**: "No models loaded" errors appear during startup before models finish loading

### Sync Button Issue
- Settings "Sync All" button may be broken - needs investigation

### API Errors
- Some tickers getting 400 errors due to model loading timing (should resolve after startup completes)