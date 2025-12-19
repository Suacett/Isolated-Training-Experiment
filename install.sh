#!/usr/bin/env bash

# ============================================================================
# Proxmox AI Stock Predictor - Installation Script
# ============================================================================
#
# This script installs the Proxmox AI Stock Predictor on a fresh Debian/Ubuntu
# system or Proxmox LXC container with NVIDIA GPU passthrough support.
#
# Usage:
#   bash -c "$(wget -qO- https://raw.githubusercontent.com/Suacett/Isolated-Training-Experiment/MAIN-BRANCH/install.sh)"
#
# Or locally:
#   chmod +x install.sh && ./install.sh
#
# Requirements:
#   - Debian 11+ or Ubuntu 22.04+
#   - NVIDIA GPU (for accelerated inference)
#   - At least 8GB RAM recommended
#   - 20GB disk space
#
# ============================================================================

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

INSTALL_DIR="/opt/stock-predictor"
REPO_URL="https://github.com/Suacett/Isolated-Training-Experiment.git"
BRANCH="MAIN-BRANCH"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

msg_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
msg_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
msg_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
msg_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_root() {
    if [[ $EUID -ne 0 ]]; then
        msg_error "This script must be run as root"
        exit 1
    fi
}

# =============================================================================
# SYSTEM DETECTION
# =============================================================================

detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VERSION_ID=$VERSION_ID
    else
        msg_error "Cannot detect OS. /etc/os-release not found."
        exit 1
    fi
    
    msg_info "Detected OS: $OS $VERSION_ID"
}

# =============================================================================
# DEPENDENCY INSTALLATION
# =============================================================================

install_dependencies() {
    msg_info "Updating system packages..."
    apt-get update -qq
    
    msg_info "Installing base dependencies..."
    apt-get install -y -qq \
        curl \
        wget \
        git \
        ca-certificates \
        gnupg \
        lsb-release \
        software-properties-common
    
    msg_ok "Base dependencies installed"
}

install_docker() {
    if command -v docker &> /dev/null; then
        msg_ok "Docker already installed: $(docker --version)"
        return
    fi
    
    msg_info "Installing Docker..."
    
    # Add Docker's official GPG key
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/$OS/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    
    # Set up repository
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/$OS \
        $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    
    # Start and enable Docker
    systemctl start docker
    systemctl enable docker
    
    msg_ok "Docker installed successfully"
}

install_nvidia_container_toolkit() {
    # Check if NVIDIA GPU is present
    if ! lspci | grep -i nvidia &> /dev/null; then
        msg_warn "No NVIDIA GPU detected. Skipping NVIDIA Container Toolkit installation."
        msg_warn "The system will run on CPU only (slower inference)."
        return
    fi
    
    # Check if nvidia-smi works
    if ! command -v nvidia-smi &> /dev/null; then
        msg_warn "NVIDIA driver not installed. Please install NVIDIA drivers first."
        msg_warn "For Proxmox GPU passthrough, refer to docs/GPU_PASSTHROUGH.md"
        return
    fi
    
    msg_info "NVIDIA GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    
    if command -v nvidia-container-cli &> /dev/null; then
        msg_ok "NVIDIA Container Toolkit already installed"
        return
    fi
    
    msg_info "Installing NVIDIA Container Toolkit..."
    
    # Add NVIDIA repository
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    apt-get update -qq
    apt-get install -y -qq nvidia-container-toolkit
    
    # Configure Docker to use NVIDIA runtime
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    
    msg_ok "NVIDIA Container Toolkit installed and configured"
}

# =============================================================================
# APPLICATION INSTALLATION
# =============================================================================

clone_repository() {
    if [ -d "$INSTALL_DIR" ]; then
        msg_warn "Installation directory already exists: $INSTALL_DIR"
        read -p "Remove existing installation and reinstall? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$INSTALL_DIR"
        else
            msg_info "Keeping existing installation. Run update.sh to update."
            return 1
        fi
    fi
    
    msg_info "Cloning repository to $INSTALL_DIR..."
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
    
    msg_ok "Repository cloned successfully"
}

configure_environment() {
    msg_info "Configuring environment..."
    
    cd "$INSTALL_DIR"
    
    if [ ! -f .env ]; then
        cp .env.example .env
        
        echo ""
        echo "============================================"
        echo "  CONFIGURATION"
        echo "============================================"
        echo ""
        echo "Alpha Vantage API Key is optional but recommended for:"
        echo "  - Intrinsic value calculations (EPS data)"
        echo "  - News sentiment analysis"
        echo ""
        echo "Get a free key at: https://www.alphavantage.co/support/#api-key"
        echo ""
        
        if [ -n "$AV_KEY" ]; then
            msg_info "Using Alpha Vantage key from environment/prompt"
        else
            if [ "$NON_INTERACTIVE" != "true" ]; then
                read -p "Enter Alpha Vantage API Key (or press Enter to skip): " AV_KEY
            fi
        fi
        
        if [ -n "$AV_KEY" ]; then
            sed -i "s/your_alpha_vantage_key_here/$AV_KEY/" .env
            msg_ok "Alpha Vantage key configured"
        else
            msg_warn "Alpha Vantage key not set. Intrinsic value calculations will use defaults."
        fi
        
        # Generate secure PostgreSQL password
        PG_PASS=$(openssl rand -base64 16 | tr -dc 'a-zA-Z0-9' | head -c 16)
        sed -i "s/POSTGRES_PASSWORD=password/POSTGRES_PASSWORD=$PG_PASS/" .env
        
        msg_ok "Environment configured with secure database password"
    else
        msg_ok "Environment file already exists"
    fi
}

start_services() {
    msg_info "Starting Docker services..."
    
    cd "$INSTALL_DIR"
    
    # Build and start containers
    docker compose up -d --build
    
    msg_ok "Services started"
    
    # Wait for backend to be ready
    msg_info "Waiting for backend to initialize..."
    
    for i in {1..30}; do
        if curl -s http://localhost:8000/status > /dev/null 2>&1; then
            msg_ok "Backend is ready"
            break
        fi
        sleep 2
    done
}

create_update_script() {
    msg_info "Creating update script..."
    
    cat > "$INSTALL_DIR/update.sh" << 'UPDATEEOF'
#!/usr/bin/env bash
# Proxmox AI Stock Predictor - Update Script

set -e

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$INSTALL_DIR"

echo "[INFO] Pulling latest changes..."
git pull origin MAIN-BRANCH

echo "[INFO] Rebuilding containers..."
docker compose build

echo "[INFO] Restarting services..."
docker compose up -d

echo "[INFO] Waiting for services..."
sleep 10

if curl -s http://localhost:8000/status > /dev/null 2>&1; then
    echo "[OK] Update complete! Services are running."
    echo ""
    echo "Access the dashboard at: http://localhost:3001"
else
    echo "[WARN] Backend may still be initializing. Check logs with:"
    echo "  docker compose logs -f backend"
fi
UPDATEEOF
    
    chmod +x "$INSTALL_DIR/update.sh"
    msg_ok "Update script created: $INSTALL_DIR/update.sh"
}

create_systemd_service() {
    msg_info "Creating systemd service for auto-start..."
    
    cat > /etc/systemd/system/stock-predictor.service << EOF
[Unit]
Description=Proxmox AI Stock Predictor
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl daemon-reload
    systemctl enable stock-predictor
    
    msg_ok "Systemd service created and enabled"
}

# =============================================================================
# MAIN INSTALLATION FLOW
# =============================================================================

main() {
    clear
    echo "============================================"
    echo "  Proxmox AI Stock Predictor Installer"
    echo "============================================"
    echo ""
    
    check_root
    detect_os
    
    echo ""
    echo "This script will install:"
    echo "  - Docker and Docker Compose"
    echo "  - NVIDIA Container Toolkit (if GPU detected)"
    echo "  - Proxmox AI Stock Predictor"
    echo ""
    echo "Installation directory: $INSTALL_DIR"
    echo ""
    
    if [ "$NON_INTERACTIVE" != "true" ]; then
        read -p "Continue with installation? [Y/n] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            msg_warn "Installation cancelled"
            exit 0
        fi
    fi
    
    install_dependencies
    install_docker
    install_nvidia_container_toolkit
    
    if clone_repository; then
        configure_environment
        create_update_script
        create_systemd_service
        start_services
    fi
    
    echo ""
    echo "============================================"
    echo "  Installation Complete!"
    echo "============================================"
    echo ""
    echo "Access the dashboard at:"
    echo "  http://$(hostname -I | awk '{print $1}'):3001"
    echo ""
    echo "Useful commands:"
    echo "  cd $INSTALL_DIR"
    echo "  docker compose logs -f         # View logs"
    echo "  ./update.sh                    # Update to latest version"
    echo "  docker compose restart         # Restart services"
    echo ""
    
    # Check GPU status
    if docker exec proxmox_stock_backend python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null; then
        GPU_NAME=$(docker exec proxmox_stock_backend python -c "import torch; print(torch.cuda.get_device_name(0))")
        echo "GPU Detected: $GPU_NAME"
    else
        echo "Note: Running on CPU. See docs/GPU_PASSTHROUGH.md for GPU setup."
    fi
    echo ""
}

main "$@"
