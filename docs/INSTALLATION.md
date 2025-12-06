# Installation Guide

This guide covers manual installation of the Proxmox AI Stock Predictor.

For automated installation, use the install script:
```bash
bash -c "$(wget -qO- https://raw.githubusercontent.com/Suacett/Isolated-Training-Experiment/main/install.sh)"
```

## System Requirements

- **OS**: Debian 11+, Ubuntu 22.04+, or Proxmox VE 7+
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 20GB minimum
- **GPU**: NVIDIA GPU recommended (T400, RTX series)

## Prerequisites

### 1. Docker Installation

```bash
# Update system
apt update && apt upgrade -y

# Install Docker dependencies
apt install -y ca-certificates curl gnupg lsb-release

# Add Docker repository
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/$(. /etc/os-release && echo $ID)/gpg | \
    gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/$(. /etc/os-release && echo $ID) \
    $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list

# Install Docker
apt update
apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Start Docker
systemctl enable docker
systemctl start docker
```

### 2. NVIDIA Container Toolkit (Optional but Recommended)

See [GPU_PASSTHROUGH.md](GPU_PASSTHROUGH.md) for complete GPU setup.

Quick install:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt update && apt install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

## Installation

### 1. Clone Repository

```bash
cd /opt
git clone https://github.com/Suacett/Isolated-Training-Experiment.git stock-predictor
cd stock-predictor
```

### 2. Configure Environment

```bash
cp .env.example .env
nano .env
```

Configure these variables:
```bash
# Required
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=stock_predictor

# Optional (for intrinsic value calculations)
ALPHA_VANTAGE_KEY=your_api_key
```

Get a free Alpha Vantage key at: https://www.alphavantage.co/support/#api-key

### 3. Build and Start Services

```bash
docker compose up -d --build
```

### 4. Verify Installation

```bash
# Check services are running
docker compose ps

# Check backend health
curl http://localhost:8000/status

# Check GPU detection (if applicable)
docker exec proxmox_stock_backend python -c "import torch; print(torch.cuda.is_available())"
```

## Post-Installation

### Access the Dashboard

Open in browser: http://localhost:3001

### Configure Daily Sync (Optional)

Add to crontab for daily data updates:
```bash
crontab -e
```

Add this line (runs at 6 AM):
```
0 6 * * * cd /opt/stock-predictor && docker exec proxmox_stock_backend python -m scripts.daily_sync >> /var/log/stock-predictor-sync.log 2>&1
```

### Train the AI Model

```bash
# Train with all available data
docker exec proxmox_stock_backend python -m scripts.train_model_v6

# Restart backend to load new model
docker compose restart backend
```

## Service Ports

| Service | Port | Description |
|---------|------|-------------|
| Frontend | 3001 | Next.js dashboard |
| Backend | 8000 | FastAPI REST API |
| Database | 5432 | PostgreSQL/TimescaleDB |

## Useful Commands

```bash
# View logs
docker compose logs -f

# Restart all services
docker compose restart

# Stop all services
docker compose down

# Update to latest version
./update.sh

# Rebuild containers
docker compose up -d --build
```

## Troubleshooting

### Backend fails to start
Check logs: `docker compose logs backend`

Common issues:
- Database not ready: Wait 30 seconds and restart backend
- Port conflict: Change ports in docker-compose.yml

### No GPU detected
See [GPU_PASSTHROUGH.md](GPU_PASSTHROUGH.md)

### Frontend not accessible
Check if port 3001 is open. For remote access, you may need to:
- Configure firewall: `ufw allow 3001`
- Use nginx reverse proxy for HTTPS
