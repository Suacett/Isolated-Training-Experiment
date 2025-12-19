from typing import Optional, Dict, Any
from datetime import datetime, timezone
import torch
import threading
from services.lstm_model import LSTMModel

# Cache TTL in seconds (1 hour for market data)
CACHE_TTL_SECONDS = 3600


class GlobalState:
    lstm_model: Optional[Any] = None
    device: Optional[torch.device] = None
    scaler: Optional[Any] = None
    alpha_vantage_enabled: bool = False
    
    def __init__(self):
        self._cache_lock = threading.Lock()
        self._spy_cache: Dict[str, Any] = {}
        self._vix_cache: Dict[str, Any] = {}
        self._spy_cache_time: Optional[datetime] = None
        self._vix_cache_time: Optional[datetime] = None

    def get_spy_cache(self) -> Optional[list]:
        """Get cached SPY data if still valid (< 1 hour old)."""
        with self._cache_lock:
            if self._spy_cache_time is None:
                return None
            age = (datetime.now(timezone.utc) - self._spy_cache_time).total_seconds()
            if age > CACHE_TTL_SECONDS:
                return None
            return self._spy_cache.get("data")

    def set_spy_cache(self, data: list):
        """Cache SPY data with current timestamp."""
        with self._cache_lock:
            self._spy_cache = {"data": data}
            self._spy_cache_time = datetime.now(timezone.utc)

    def get_vix_cache(self) -> Optional[list]:
        """Get cached VIX data if still valid (< 1 hour old)."""
        with self._cache_lock:
            if self._vix_cache_time is None:
                return None
            age = (datetime.now(timezone.utc) - self._vix_cache_time).total_seconds()
            if age > CACHE_TTL_SECONDS:
                return None
            return self._vix_cache.get("data")

    def set_vix_cache(self, data: list):
        """Cache VIX data with current timestamp."""
        with self._cache_lock:
            self._vix_cache = {"data": data}
            self._vix_cache_time = datetime.now(timezone.utc)


state = GlobalState()
