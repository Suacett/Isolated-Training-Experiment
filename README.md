# Proxmox AI Stock Predictor

A closed-loop LSTM-based stock prediction system with multi-horizon forecasting,
designed for deployment on Proxmox with optional GPU acceleration.

## Features

- **LSTM Neural Network**: 37-feature model predicting 1-day, 1-week, 1-month, and 6-month horizons
- **Real-Time Dashboard**: Next.js frontend with interactive charts and prediction visualization
- **Intrinsic Value Analysis**: Graham Number and DCF calculations using Alpha Vantage EPS data
- **Multi-Asset Support**: Stocks, ETFs, and cryptocurrencies
- **GPU Acceleration**: NVIDIA GPU support for fast inference and training
- **Automated Sync**: Daily data fetching and prediction generation
- **Risk Management**: Adaptive market regime detection (VIX/SPY) and volatility targeting

## Quick Start

### One-Line Installation (Proxmox/Debian/Ubuntu)

```bash
bash -c "$(wget -qO- https://raw.githubusercontent.com/Suacett/Isolated-Training-Experiment/MAIN-BRANCH/install.sh)"
```

The installer will:
1. Install Docker and Docker Compose
2. Configure NVIDIA Container Toolkit (if GPU detected)
3. Clone and configure the application
4. Start all services

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
# Train with maximum data (recommended)
docker exec proxmox_stock_backend python -m scripts.train_model_v6

# Restart backend to load new model
docker compose restart backend
```

### Training Options

| Script | Memory | Description |
|--------|--------|-------------|
| train_model_v6 | Streaming | Production - handles unlimited data |
| train_model_v5 | Streaming | Similar to v6, disk-based memmap |
| train_model_v4 | ~8GB RAM | In-memory, may OOM on large datasets |

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
| /market-status | GET | Current market regime (VIX/SPY) |

See [docs/API.md](docs/API.md) for complete API reference.

## Model Performance

| Metric | Value |
|--------|-------|
| Direction Accuracy (SPY) | ~54% |
| Direction Accuracy (Individual) | 50-55% |
| Prediction Horizons | 1d, 1w, 1m, 6m |
| Features | 37 technical indicators |
| Window Size | 60 trading days |

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `docker exec proxmox_stock_backend pytest tests/ -v`
5. Submit a pull request
