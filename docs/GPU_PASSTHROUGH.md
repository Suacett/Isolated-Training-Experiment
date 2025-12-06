# GPU Passthrough Guide for Proxmox

This guide explains how to configure NVIDIA GPU passthrough for the Proxmox AI Stock Predictor.

## Overview

The AI model uses an LSTM neural network that benefits significantly from GPU acceleration.
This guide covers GPU passthrough for both Proxmox VMs and LXC containers.

## Hardware Requirements

- NVIDIA GPU (T400, RTX 4080, or similar)
- IOMMU-capable CPU (Intel VT-d or AMD-Vi)
- Proxmox VE 7.0 or later

## Option 1: LXC Container (Recommended)

LXC containers are lighter weight and share the host kernel, making GPU passthrough simpler.

### Step 1: Install NVIDIA Drivers on Proxmox Host

```bash
# Update packages
apt update && apt upgrade -y

# Install kernel headers
apt install pve-headers-$(uname -r) -y

# Download NVIDIA driver (check latest version)
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/535.183.01/NVIDIA-Linux-x86_64-535.183.01.run

# Install driver (skip kernel module signing)
chmod +x NVIDIA-Linux-x86_64-*.run
./NVIDIA-Linux-x86_64-*.run --no-kernel-module-signing

# Verify installation
nvidia-smi
```

### Step 2: Create LXC Container

Create an unprivileged Ubuntu 22.04 container with these settings:
- Memory: 8GB minimum (16GB recommended)
- Disk: 50GB minimum
- CPU: 4+ cores

### Step 3: Configure LXC for GPU Access

Edit the container configuration:

```bash
# Edit /etc/pve/lxc/<CTID>.conf
nano /etc/pve/lxc/100.conf
```

Add these lines:

```
# GPU Passthrough
lxc.cgroup2.devices.allow: c 195:* rwm
lxc.cgroup2.devices.allow: c 509:* rwm
lxc.mount.entry: /dev/nvidia0 dev/nvidia0 none bind,optional,create=file
lxc.mount.entry: /dev/nvidiactl dev/nvidiactl none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm dev/nvidia-uvm none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm-tools dev/nvidia-uvm-tools none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-modeset dev/nvidia-modeset none bind,optional,create=file
```

### Step 4: Install NVIDIA Drivers in Container

The container needs matching driver binaries (not kernel modules):

```bash
# Inside container
apt update
apt install -y wget

# Download same version as host
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/535.183.01/NVIDIA-Linux-x86_64-535.183.01.run

# Install with --no-kernel-module (uses host modules)
chmod +x NVIDIA-Linux-x86_64-*.run
./NVIDIA-Linux-x86_64-*.run --no-kernel-module

# Verify
nvidia-smi
```

### Step 5: Install NVIDIA Container Toolkit

```bash
# Add NVIDIA repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt update
apt install -y nvidia-container-toolkit

# Configure Docker
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

### Step 6: Verify GPU Access in Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

---

## Option 2: VM with PCI Passthrough

Full VM passthrough provides complete GPU isolation but is more complex.

### Step 1: Enable IOMMU

Edit GRUB configuration:

```bash
nano /etc/default/grub
```

Add to GRUB_CMDLINE_LINUX_DEFAULT:
- Intel: `intel_iommu=on iommu=pt`
- AMD: `amd_iommu=on iommu=pt`

```bash
update-grub
reboot
```

### Step 2: Blacklist Host Drivers

```bash
echo "blacklist nouveau" >> /etc/modprobe.d/blacklist.conf
echo "blacklist nvidia" >> /etc/modprobe.d/blacklist.conf
update-initramfs -u
reboot
```

### Step 3: Find GPU IOMMU Group

```bash
# Find your GPU's IOMMU group
find /sys/kernel/iommu_groups/ -type l | xargs -n 1 readlink -f | sort | grep -i nvidia
```

### Step 4: Configure VFIO

```bash
# Get GPU IDs
lspci -nn | grep NVIDIA
# Example output: 01:00.0 VGA compatible controller [0300]: NVIDIA Corporation [10de:2684]

# Add to /etc/modprobe.d/vfio.conf
echo "options vfio-pci ids=10de:2684,10de:22ba" > /etc/modprobe.d/vfio.conf

# Load VFIO modules
echo "vfio" >> /etc/modules
echo "vfio_iommu_type1" >> /etc/modules
echo "vfio_pci" >> /etc/modules

update-initramfs -u
reboot
```

### Step 5: Add GPU to VM

In Proxmox web UI:
1. Select VM > Hardware > Add > PCI Device
2. Select the NVIDIA GPU
3. Check "All Functions" and "ROM-Bar"
4. Check "PCI-Express" if available

### Step 6: Install NVIDIA Drivers in VM

Boot the VM and install NVIDIA drivers normally:

```bash
# Ubuntu/Debian
apt update
apt install nvidia-driver-535 nvidia-container-toolkit -y
reboot
```

---

## Verification

After setup, verify GPU is accessible:

```bash
# Check GPU detection
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Check in Stock Predictor container
docker exec proxmox_stock_backend python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
GPU: NVIDIA GeForce RTX 4080 SUPER
```

---

## Troubleshooting

### nvidia-smi: command not found
Driver not installed correctly. Reinstall NVIDIA driver.

### Failed to initialize NVML
In LXC: Check cgroup device permissions in container config.
In VM: Verify PCI passthrough configuration.

### CUDA out of memory
The model uses approximately 2-4GB VRAM. Check if other processes are using the GPU.

### Docker: could not select device driver "nvidia"
NVIDIA Container Toolkit not installed or configured. Run:
```bash
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

---

## Performance Notes

| Hardware | Inference Speed | Training Speed |
|----------|-----------------|----------------|
| CPU Only | ~500ms/prediction | ~3 hours |
| NVIDIA T400 | ~50ms/prediction | ~45 minutes |
| RTX 4080 Super | ~10ms/prediction | ~10 minutes |

GPU acceleration provides 10-50x speedup for both inference and training.
