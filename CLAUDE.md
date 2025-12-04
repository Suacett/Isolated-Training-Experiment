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

# Utility scripts
python backend/scripts/check_gpu.py   # Verify GPU availability
python backend/scripts/init_db.py     # Initialize database schema
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
- **lstm_model.py**: PyTorch model (Conv1D → BatchNorm → MC Dropout → LSTM → Dense). 60-day input, 4-step forecast
- **intrinsic.py**: Graham formula intrinsic value calculation
- **db.py**: SQLAlchemy async models (`StockPrice`, `Watchlist`, `Prediction`)

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