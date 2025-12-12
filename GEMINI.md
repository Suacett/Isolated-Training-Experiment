# Project: Proxmox AI Stock Predictor (Closed-Loop Paper Trading)

## 1. Infrastructure & Hardware (CRITICAL)
- **Environment:** Docker (via `docker-compose`).
- **Host:** Proxmox Server (128GB RAM, Ryzen 9 3900X).
- **GPU Acceleration:**
  - The system possesses either an NVIDIA T400 or RTX 4080 Super.
  - **NEVER** hardcode device IDs (e.g., `cuda:0`).
  - **ALWAYS** use dynamic detection: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
  - **Boot Check:** The Backend service must log the detected GPU name on startup.

## 2. Data Architecture
- **Database:** PostgreSQL with TimescaleDB extension (running in Docker).
- **Data Sources:**
  - Yahoo Finance (Primary) - Free, no API key required
  - Alpha Vantage (EPS/Intrinsic Value for favorites only)
- **Secrets:** API Keys stored in `backend/secrets.json` via frontend modal. NEVER commit keys.

## 3. UI/UX Guidelines (Minimalist Dark Design System)
- **Theme:** Minimalist Dark with Warm Amber Accent
- **Colors:**
  - Background: `#0A0A0F` (deep slate)
  - Background Alt: `#12121A` (headers, modals)
  - Accent: `#F59E0B` (warm amber)
  - Success: `#10B981` (green)
  - Error: `#F43F5E` (red)
- **Typography:**
  - Display: Space Grotesk (headings)
  - Body: Inter (UI text)
  - Mono: JetBrains Mono (prices, data)
- **Effects:**
  - Glass card effect with backdrop blur
  - Amber glow on hover for interactive elements
  - Scale-in and slide-in animations for modals

## 4. Coding Standards
1. **Modularity:** Break logic into `services/`, `routers/`, `utils/`.
2. **Type Hints:** Python type hints everywhere.
3. **Async-First:** `async def` for routes and DB interactions.
4. **No Hardcoded Devices:** Use dynamic `torch.device()` detection.

---

## 5. Backend Architecture

### API Routers (`backend/routers/`)
- **stocks.py**: Watchlist management, favorites
- **dashboard.py**: Historical data, predictions
- **ingestion.py**: Data fetching from external APIs
- **predictions.py**: Prediction history and stats
- **backtest.py**: Historical prediction generation
- **forecasts.py**: Multi-horizon AI forecasts
- **backtest_v9_portfolio.py**: V9 Portfolio simulation (Top-K Ranking)
- **playground.py**: Model Playground for comparing multiple models

### Services (`backend/services/`)
- **data_ingest.py**: Yahoo Finance client (primary data source)
- **lstm_model.py**: PyTorch LSTM model v6 (60-day window, 37 features)
- **lstm_model_v7.py**: Enhanced model with attention, biLSTM (41 features)
- **lstm_model_v8_class.py**: Classification model (binary buy signals, non-overlapping windows)
- **intrinsic.py**: Graham formula intrinsic value calculation
- **db.py**: SQLAlchemy async models
- **feature_engineering.py**: Technical feature computation (41 features for v7)
- **feature_engineering_v9.py**: V9 Stationary feature computation (12 features)
- **transformer_model.py**: V9 Transformer Encoder for Relative Strength Ranking
- **model_loader.py**: Multi-model loader for Model Playground
- **risk_management.py**: Market regime detection (VIX/SPY) and position sizing

---

## 6. Model Training

### Training Commands

```bash
# STATE-OF-THE-ART: V9 - Transformer Ranking Model
# Predicts "Relative Strength Rank" (0.0-1.0) among 500 stocks
docker exec proxmox_stock_backend python -m scripts.fetch_training_data --sp500  # Fetch Full S&P 500
docker exec proxmox_stock_backend python -m scripts.train_model_v9            # Train Transformer

# RECOMMENDED: v7 - Attention model with market features
docker exec proxmox_stock_backend python -m scripts.fetch_training_data  # Fetch 59 tickers
docker exec proxmox_stock_backend python -m scripts.train_model_v7       # Train v7 model

# EXPERIMENTAL: v8 - Classification model (Regime Detection)
# Uses non-overlapping windows (stride=126) and binary classification targets
docker exec proxmox_stock_backend python -m scripts.train_model_v8_class

# Previous: v6 - Standard LSTM
docker exec proxmox_stock_backend python -m scripts.train_model_v6

# Models auto-activate on restart (backend detects latest version)
docker compose restart backend
```

### Training Scripts

| Script | Model | Description |
|--------|-------|-------------|
| `train_model_v9.py` | **V9** | **STATE-OF-THE-ART** - Transformer Encoder, 12 Stationary Features, Relative Strength Ranking |
| `train_model_v7.py` | v7 | Attention + BiLSTM, 41 features, DirectionalLoss |
| `train_model_v8_class.py` | v8-class | **LEGACY** - Classification (BCELoss), non-overlapping windows |
| `train_model_v6.py` | v6 | Standard LSTM, 128 hidden, 37 features |
| `backtest_v9_portfolio.py` | - | **NEW** - Backtests Top-10 Ranking Strategy |
| `fetch_training_data.py` | - | Fetches 59 tickers (indices, sectors, top stocks) |
| `fetch_sp500_full.py` | - | **NEW** - Fetches full S&P 500 (500+ tickers) dynamically |
| `audit_v7.py` | - | Detects "lazy prediction" (memorization) in V6/V7 |
| `sanity_check_v8.py` | - | Pre-training validation for V8 |
| `calibrate_v8.py` | - | Find optimal buy threshold for V8 |

### Accuracy Results (2025-12-12)

| Model | SPY | AAPL | AMD | Bias | Notes |
|-------|-----|------|-----|------|-------|
| **v7 (Attention)** | 52.6% | TBD | TBD | -23.7% (Bearish) | **NEW** - BiLSTM + Attention |
| v6 (Clean Data) | 54.0% | 51.1% | 50.1% | +8% (Bullish) | Baseline |

### Risk Management Standards
- **Market Regime Detection**:
  - **Crisis** (VIX > 30): 100% Cash / Veto all buys
  - **Bear Market** (SPY < SMA200): Defensive (30% Exposure)
  - **Bull Market**:
    - Calm (VIX < 20): 100% Exposure
    - Moderate (VIX 20-25): 80% Exposure
    - Elevated (VIX 25-30): 60% Exposure

---

## 7. Frontend Components

### Key Components (`frontend/app/components/`)
- **AssetTable**: Portfolio overview with favorites toggle
- **DetailDrawer**: Chart visualization (1W, 1M, 3M, 6M, 1Y, 5Y, ALL)
- **SetupModal**: API key configuration
- **ModelPlayground**: Compare multiple model versions
- **AIPredictionPanel**: AI model inspector
- **MarketStatusHeader**: Displays current Regime (Bull/Bear) and VIX level

### Chart Features
- Multi-line: Price, AI Prediction, S&P 500, Intrinsic Value
- 500-point downsampling for performance
- Percentage mode for relative comparison
- Accuracy stream with daily prediction results

---

## 8. Quick Commands

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f backend

# Restart after code changes
docker compose restart backend

# Run tests
docker compose exec backend pytest -v
```
