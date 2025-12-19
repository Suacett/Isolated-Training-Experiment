# Proxmox AI Stock Predictor

A closed-loop AI-powered stock prediction and paper trading system with multi-model support,
designed for deployment on Proxmox with optional GPU acceleration.

## Features

- **V9 Transformer AI**: State-of-the-art relative strength ranking model (12 stationary features)
- **Multi-Model Playground**: Compare predictions across V6, V7, V8, and V9 models
- **Laboratory (Stress-Test)**: 20-year historical backtesting (2005-2024) across 9 strategic variants
- **Paper Trading Simulator**: Interactive dashboard with real-time portfolio tracking and daily rebalancing
- **Real-Time Dashboard**: Next.js frontend with interactive charts and prediction visualization
- **Intrinsic Value Analysis**: Graham Number calculations using Alpha Vantage EPS data
- **Multi-Asset Support**: Stocks, ETFs, and cryptocurrencies
- **GPU Acceleration**: NVIDIA GPU support for fast inference and training
- **Risk Management**: Market regime detection (VIX/SPY) and automatic stop-loss

## Quick Start

### One-Line Setup (Run on Proxmox Host)

```bash
bash -c "$(wget -qO- https://raw.githubusercontent.com/Suacett/Isolated-Training-Experiment/main/proxmox-install.sh)"
```

The installer will:
1. Create a new LXC container with optimized resources (8GB RAM, 50GB Disk)
2. **Auto-configure NVIDIA GPU passthrough** if a GPU is detected
3. Install Docker, NVIDIA Container Toolkit, and all repo dependencies
4. Start the full application (Frontend + Backend + DB)

### Alternate Installation (Pre-existing Linux/LXC)

If you already have a container or Ubuntu machine:

```bash
bash -c "$(wget -qO- https://raw.githubusercontent.com/Suacett/Isolated-Training-Experiment/main/install.sh)"
```

### Manual Installation

```bash
# Clone repository
git clone https://github.com/Suacett/Isolated-Training-Experiment.git
cd Isolated-Training-Experiment

# Configure environment
cp .env.example .env
nano .env  # Add your Alpha Vantage API key (optional)

# Start services
docker compose up -d --build

# Verify installation
curl http://localhost:8000/status
```

## Access

| Service | URL | Description |
|---------|-----|-------------|
| Dashboard | http://localhost:3001 | Web interface |
| API | http://localhost:8000 | REST endpoints |
| API Docs | http://localhost:8000/docs | Swagger UI |

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| POSTGRES_PASSWORD | Yes | Database password |
| ALPHA_VANTAGE_KEY | No | For EPS data and intrinsic value |
| ALPHA_VANTAGE_KEYS | No | Comma-separated keys for higher throughput |

Get a free Alpha Vantage key: https://www.alphavantage.co/support/#api-key

### Daily Data Sync

The system includes a daily sync script that fetches closing prices and generates predictions.

Add to crontab (runs at 6 AM):
```bash
0 6 * * * cd /opt/stock-predictor && docker exec proxmox_stock_backend python -m scripts.daily_sync
```

## Training the AI Model

### Quick Training

```bash
# V9 Transformer (STATE-OF-THE-ART - Relative Strength Ranking)
docker exec proxmox_stock_backend python -m scripts.train_model_v9

# V7 Attention LSTM (Recommended for price prediction)
docker exec proxmox_stock_backend python -m scripts.train_model_v7

# Restart backend to load new model
docker compose restart backend
```

### Training Options

| `train_model_v9` | Transformer | Relative strength ranking (12 features) |
| `seed_multi_portfolio` | Simulator | Populates Laboratory with 20-year variants |
| `train_model_v8_class` | Classification LSTM | Directional prediction (Regime Detection) |
| `train_model_v7` | Attention LSTM | Price prediction (41 features) |
| `train_model_v6` | Standard LSTM | Legacy (37 features) |

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed training instructions.

## GPU Setup

For GPU acceleration on Proxmox, see [docs/GPU_PASSTHROUGH.md](docs/GPU_PASSTHROUGH.md).

Verify GPU detection:
```bash
docker exec proxmox_stock_backend python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Project Structure

```
.
├── backend/                 # FastAPI Python backend
│   ├── main.py             # Application entry point
│   ├── routers/            # API endpoint modules
│   ├── services/           # Business logic
│   │   ├── lstm_model.py   # Neural network model
│   │   ├── data_ingest.py  # Yahoo Finance integration
│   │   ├── intrinsic.py    # Intrinsic value calculator
│   │   └── ...
│   └── scripts/            # Training and utility scripts
├── frontend/               # Next.js dashboard
│   └── app/                # App router pages and components
├── docs/                   # Documentation
│   ├── API.md             # API reference
│   ├── GPU_PASSTHROUGH.md # GPU setup guide
│   └── INSTALLATION.md    # Manual installation
├── docker-compose.yml      # Service orchestration
├── install.sh             # Automated installer
└── update.sh              # Update script
```

## Commands Reference

```bash
# Start services
docker compose up -d

# View logs
docker compose logs -f backend

# Update to latest version
./update.sh

# Run daily sync manually
docker exec proxmox_stock_backend python -m scripts.daily_sync

# Run tests
docker exec proxmox_stock_backend pytest tests/ -v

# Access backend shell
docker exec -it proxmox_stock_backend bash
```

## API Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| /status | GET | System health check |
| /dashboard | GET | All watchlist summaries |
| /dashboard/{ticker} | GET | Detailed ticker data |
| /forecasts/{ticker} | GET | Multi-horizon predictions |
| /stocks/ | GET/POST/DELETE | Watchlist management |
| /ingest/{ticker} | POST | Trigger data sync |
| /market-status | GET | Returns VIX level, regime (Normal/Volatile/Panic), and SPY trend |

See [docs/API.md](docs/API.md) for complete API reference.

## Model Performance

| Metric | V9 (Transformer) | V7 (LSTM) |
|--------|------------------|------------|
| Type | Ranking | Price Prediction |
| Features | 12 stationary | 41 technical |
| Window Size | 60 days | 60 days |
| Output | Rank Score (0-1) | Log Returns |
| Paper Trading | +93% (5Y) | N/A |

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `docker exec proxmox_stock_backend pytest tests/ -v`
5. Submit a pull request
