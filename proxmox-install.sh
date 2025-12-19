#!/usr/bin/env bash

# ============================================================================
# Proxmox AI Stock Predictor - Automated LXC Container Creation & Installation
# ============================================================================
#
# This script creates a new Proxmox LXC container and installs the stock
# predictor inside it automatically.
#
# Usage (run on Proxmox host):
#   bash -c "$(wget -qO- https://raw.githubusercontent.com/Suacett/Isolated-Training-Experiment/main/proxmox-install.sh)"
#
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

msg_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
msg_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
msg_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
msg_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running on Proxmox
if ! command -v pct &> /dev/null; then
    msg_error "This script must be run on a Proxmox VE host"
    exit 1
fi

clear
echo "============================================"
echo "  Proxmox AI Stock Predictor"
echo "  Automated LXC Container Installer"
echo "============================================"
echo ""

# Configuration
CTID=""
HOSTNAME="stock-predictor"
TEMPLATE_STORAGE="local"  # Templates are stored on 'local', not 'local-lvm'
ROOTFS_STORAGE="local-lvm"  # Rootfs goes on LVM
ROOTFS_SIZE=50
MEMORY=8192
SWAP=4096
CORES=4
PASSWORD=""
BRIDGE="vmbr0"

# Detect available storage for rootfs
if ! pvesm status | grep -q "local-lvm"; then
    ROOTFS_STORAGE="local"
    msg_warn "local-lvm not found, using 'local' for rootfs"
fi

# Get next available CT ID
msg_info "Finding next available container ID..."
CTID=$(pvesh get /cluster/nextid)
msg_ok "Container ID: $CTID"

# Ask for customizations
echo ""
echo "Container Configuration:"
echo "  ID: $CTID"
echo "  Hostname: $HOSTNAME"
echo "  Disk: ${ROOTFS_SIZE}GB"
echo "  RAM: ${MEMORY}MB"
echo "  CPU Cores: $CORES"
echo ""
read -p "Continue with these settings? [Y/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    read -p "Enter container ID [$CTID]: " CUSTOM_CTID
    CTID=${CUSTOM_CTID:-$CTID}
    
    read -p "Enter RAM in MB [$MEMORY]: " CUSTOM_MEM
    MEMORY=${CUSTOM_MEM:-$MEMORY}
    
    read -p "Enter CPU cores [$CORES]: " CUSTOM_CORES
    CORES=${CUSTOM_CORES:-$CORES}
fi

# Set root password
echo ""
read -s -p "Set root password for container: " PASSWORD
echo ""
read -s -p "Confirm password: " PASSWORD2
echo ""

if [ "$PASSWORD" != "$PASSWORD2" ]; then
    msg_error "Passwords do not match"
    exit 1
fi

# Optional Alpha Vantage Key
echo ""
read -p "Optional: Enter Alpha Vantage API Key (for intrinsic value): " AV_KEY
echo ""

# Check if template exists, download if not
TEMPLATE="$TEMPLATE_STORAGE:vztmpl/ubuntu-22.04-standard_22.04-1_amd64.tar.zst"
if ! pveam list $TEMPLATE_STORAGE 2>/dev/null | grep -q "ubuntu-22.04-standard"; then
    msg_info "Downloading Ubuntu 22.04 template..."
    pveam download $TEMPLATE_STORAGE ubuntu-22.04-standard_22.04-1_amd64.tar.zst
fi

# Create container
msg_info "Creating LXC container $CTID..."
pct create $CTID $TEMPLATE \
    --hostname $HOSTNAME \
    --rootfs $ROOTFS_STORAGE:$ROOTFS_SIZE \
    --memory $MEMORY \
    --swap $SWAP \
    --cores $CORES \
    --net0 name=eth0,bridge=$BRIDGE,ip=dhcp \
    --unprivileged 1 \
    --features nesting=1 \
    --password "$PASSWORD" \
    --onboot 1

msg_ok "Container created successfully"

# Check for NVIDIA GPU
if lspci | grep -i nvidia &> /dev/null; then
    msg_info "NVIDIA GPU detected. Configuring GPU passthrough..."
    
    # Detect NVIDIA major numbers
    NVIDIA_MAJOR=$(ls -l /dev/nvidia0 | awk '{print $5}' | cut -d, -f1)
    UVM_MAJOR=$(ls -l /dev/nvidia-uvm | awk '{print $5}' | cut -d, -f1)
    
    # Add GPU device mappings to container config
    [ -n "$NVIDIA_MAJOR" ] && echo "lxc.cgroup2.devices.allow: c $NVIDIA_MAJOR:* rwm" >> /etc/pve/lxc/${CTID}.conf
    [ -n "$UVM_MAJOR" ] && echo "lxc.cgroup2.devices.allow: c $UVM_MAJOR:* rwm" >> /etc/pve/lxc/${CTID}.conf
    
    echo "lxc.mount.entry: /dev/nvidia0 dev/nvidia0 none bind,optional,create=file" >> /etc/pve/lxc/${CTID}.conf
    echo "lxc.mount.entry: /dev/nvidiactl dev/nvidiactl none bind,optional,create=file" >> /etc/pve/lxc/${CTID}.conf
    echo "lxc.mount.entry: /dev/nvidia-uvm dev/nvidia-uvm none bind,optional,create=file" >> /etc/pve/lxc/${CTID}.conf
    echo "lxc.mount.entry: /dev/nvidia-modeset dev/nvidia-modeset none bind,optional,create=file" >> /etc/pve/lxc/${CTID}.conf
    echo "lxc.mount.entry: /dev/nvidia-uvm-tools dev/nvidia-uvm-tools none bind,optional,create=file" >> /etc/pve/lxc/${CTID}.conf
    
    msg_ok "GPU passthrough configured"
fi

# Start container
msg_info "Starting container..."
pct start $CTID

# Wait for container to be ready
msg_info "Waiting for container to initialize..."
sleep 10

# Install stock predictor inside container
msg_info "Installing Stock Predictor inside container..."
pct exec $CTID -- bash -c "NON_INTERACTIVE=true AV_KEY='$AV_KEY' wget -qO- https://raw.githubusercontent.com/Suacett/Isolated-Training-Experiment/main/install.sh | bash"

# Get container IP
CTIP=$(pct exec $CTID -- hostname -I | awk '{print $1}')

echo ""
echo "============================================"
echo "  Installation Complete!"
echo "============================================"
echo ""
echo "Container ID: $CTID"
echo "Container IP: $CTIP"
echo ""
echo "Access the dashboard at:"
echo "  http://${CTIP}:3001"
echo ""
echo "Useful commands:"
echo "  pct enter $CTID              # Enter container shell"
echo "  pct stop $CTID               # Stop container"
echo "  pct start $CTID              # Start container"
echo "  pct destroy $CTID            # Delete container"
echo ""
echo "Inside container:"
echo "  cd /opt/stock-predictor"
echo "  docker compose logs -f       # View logs"
echo "  ./update.sh                  # Update application"
echo ""
