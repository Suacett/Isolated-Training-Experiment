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
Alpaca Markets API → AlpacaDataClient → TimescaleDB → FastAPI → Next.js Dashboard
```

### Backend Services (`backend/services/`)
- **data_ingest.py**: `AlpacaDataClient` fetches OHLCV data, auto-detects crypto vs stocks
- **lstm_model.py**: PyTorch model (Conv1D → BatchNorm → MC Dropout → LSTM → Dense). 60-day input, 39 features, 4-step forecast
- **intrinsic.py**: Graham formula intrinsic value calculation
- **db.py**: SQLAlchemy async models (`StockPrice`, `Watchlist`, `Prediction`, `InsiderTrade`, `SentimentData`)
- **feature_engineering.py**: Computes 39 technical features from OHLCV + insider + sentiment data
- **insider_data.py**: SEC Form 4 scraper for insider trading data
- **sentiment_data.py**: Alpha Vantage News API for sentiment analysis
- **scaler.py**: StandardScaler wrapper for feature normalization
- **training.py**: Complete training pipeline using legacy data

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
- **ingestion.py**:
  - `POST /ingest/all` - Fetch data for all watchlist tickers and generate predictions
- **predictions.py**:
  - `GET /predictions/{ticker}` - Get prediction history for a ticker
  - `GET /predictions/{ticker}/stats` - Get prediction accuracy statistics
  - `GET /predictions/pending/validate` - Get predictions needing validation

### Utility Modules (`backend/utils/`)
- **config_loader.py**: Load/save API keys and settings from `secrets.json`

### Training Scripts (`backend/scripts/`)
- **train_model.py**: Train LSTM model using legacy training data (31 stocks, 39 features)
- **check_gpu.py**: Verify CUDA availability
- **init_db.py**: Initialize database schema

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
- Treat Alpaca API as a rate-limited resource
- Alpha Vantage Client provides fallback data ingestion when Alpaca is unavailable

### Environment Configuration

- `.env` file contains sensitive configuration (git-ignored):
  - `ALPHA_VANTAGE_KEY`: Alpha Vantage API key for fallback data source
  - `POSTGRES_USER`: Database user (default: postgres)
  - `POSTGRES_PASSWORD`: Database password (default: password)
- Frontend secrets (API keys) stored in `backend/secrets.json` at runtime, not in environment

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

### Training the LSTM Model

The model requires training before it can make predictions. Train using legacy data:

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Train model (1-3 hours on CPU, 15-30 min on GPU)
python backend/scripts/train_model.py
```

This creates:
- `backend/models/lstm_model.pth` - Trained model weights
- `backend/models/scaler.pkl` - Fitted StandardScaler (critical for inference)

### Model Architecture (39 Features)

**Input Features:**
- Log returns (5): Yesterday's OHLCV returns
- Moving averages (5): MA10/20/30, EMA10/30
- Time features (3): Day of week/month, month number
- Technical indicators (5): RSI, MACD, MACD Signal, Bollinger Bands
- Volatility (6): Multiple timeframes (5d, 10d, 20d, 30d)
- Volume indicators (2): OBV, abnormal volume
- Price patterns (5): Z-score, overnight gap, momentum, skewness, intraday range
- Insider trading (3): Shares, amount, buy/sell flag from SEC Form 4
- Sentiment (3): News sentiment score, article count, sentiment change

**Output Horizons:**
- `1d`: 1 day ahead
- `1w`: 5 days (1 week) ahead
- `1m`: 21 days (1 month) ahead
- `6m`: 126 days (6 months) ahead

### Training Data

Legacy data location: `legacy/LSTM_AI_Stock_Predictor/TrainingData/indicators_data/raw/stocksData/`
- 31+ stock CSV files
- ~61MB total
- Historical OHLCV from Alpha Vantage

### Important Notes

- **Scaler is Critical**: Predictions are meaningless without the fitted scaler from training
- **Feature Consistency**: Inference must use exact same 39 features as training
- **Alternative Data**: Insider (SEC) and sentiment (Alpha Vantage) data are optional but improve accuracy
- **Data Sources**: Training uses Alpha Vantage data; production uses Alpaca (primary) + Alpha Vantage (fallback)