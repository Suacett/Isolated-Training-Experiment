"""Unified device detection utilities for PyTorch model loading.

This module provides centralized device management to replace scattered
device detection logic across the codebase.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Unified device detection for all model loading.

    Args:
        prefer_cuda: Whether to prefer CUDA if available (default: True)

    Returns:
        torch.device object ('cuda' or 'cpu')
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"✅ Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device
    else:
        device = torch.device("cpu")
        if prefer_cuda:
            logger.warning("⚠️ CUDA not available, falling back to CPU")
        return device


def has_sufficient_gpu_memory(required_mb: int = 200) -> bool:
    """
    Check if GPU has enough free memory for model loading.

    Args:
        required_mb: Required memory in MB (default: 200MB)

    Returns:
        True if sufficient memory available, False otherwise
    """
    if not torch.cuda.is_available():
        return False

    try:
        torch.cuda.empty_cache()
        props = torch.cuda.get_device_properties(0)
        total_mem = props.total_memory
        allocated_mem = torch.cuda.memory_allocated(0)
        free_mem = total_mem - allocated_mem
        free_mem_mb = free_mem / (1024 * 1024)

        logger.debug(f"GPU Memory: {free_mem_mb:.0f}MB free / {total_mem/(1024*1024):.0f}MB total")

        return free_mem_mb >= required_mb
    except Exception as e:
        logger.warning(f"Failed to check GPU memory: {e}")
        return False


def get_device_info() -> dict:
    """
    Get comprehensive device information for logging/debugging.

    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["total_memory_gb"] = props.total_memory / (1024**3)
        info["allocated_memory_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
        info["free_memory_gb"] = (props.total_memory - torch.cuda.memory_allocated(0)) / (1024**3)

    return info
