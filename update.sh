#!/usr/bin/env bash

# ============================================================================
# Proxmox AI Stock Predictor - Update Script
# ============================================================================
#
# Updates the installation to the latest version from the repository.
#
# Usage:
#   cd /opt/stock-predictor && ./update.sh
#
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

msg_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
msg_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
msg_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "============================================"
echo "  Proxmox AI Stock Predictor - Update"
echo "============================================"
echo ""

# Check if git repository
if [ ! -d .git ]; then
    msg_warn "Not a git repository. Cannot update automatically."
    exit 1
fi

# Check for local changes
if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    msg_warn "Local changes detected. Stashing..."
    git stash
fi

# Pull latest changes
msg_info "Pulling latest changes from repository..."
git pull origin main

# Check if docker-compose.yml changed
if git diff --name-only HEAD@{1} HEAD 2>/dev/null | grep -q docker-compose.yml; then
    msg_info "Docker Compose configuration changed. Rebuilding..."
    docker compose build --no-cache
else
    msg_info "Rebuilding containers..."
    docker compose build
fi

# Check if requirements.txt changed
if git diff --name-only HEAD@{1} HEAD 2>/dev/null | grep -q requirements.txt; then
    msg_info "Python dependencies changed. Full rebuild required..."
    docker compose build --no-cache backend
fi

# Restart services
msg_info "Restarting services..."
docker compose down
docker compose up -d

# Wait for services to be ready
msg_info "Waiting for services to initialize..."
for i in {1..30}; do
    if curl -s http://localhost:8000/status > /dev/null 2>&1; then
        break
    fi
    sleep 2
done

# Check backend status
if curl -s http://localhost:8000/status > /dev/null 2>&1; then
    msg_ok "Backend is running"
    
    # Get status info
    STATUS=$(curl -s http://localhost:8000/status)
    MODEL_VER=$(echo "$STATUS" | grep -o '"model_version":"[^"]*"' | cut -d'"' -f4)
    
    if [ -n "$MODEL_VER" ]; then
        msg_ok "Active model version: $MODEL_VER"
    fi
else
    msg_warn "Backend may still be initializing. Check logs:"
    echo "  docker compose logs -f backend"
fi

# Check frontend
if curl -s http://localhost:3001 > /dev/null 2>&1; then
    msg_ok "Frontend is running"
else
    msg_warn "Frontend may still be starting..."
fi

echo ""
echo "============================================"
echo "  Update Complete!"
echo "============================================"
echo ""
echo "Dashboard: http://localhost:3001"
echo "API:       http://localhost:8000"
echo ""
echo "View logs: docker compose logs -f"
echo ""
