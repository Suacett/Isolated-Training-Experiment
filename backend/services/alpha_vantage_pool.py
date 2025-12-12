"""
Alpha Vantage API Key Pool Manager

Manages multiple API keys with rate limiting and round-robin rotation.
Supports 5 calls/minute per key. With 5 keys, throughput increases to 25 calls/minute.

Free tier limits:
- 5 calls per minute per key
- 500 calls per day per key
- With 5 keys: 25 calls/minute, 2500 calls/day
"""

import os
import time
import logging
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class APIKey:
    """Represents a single Alpha Vantage API key with rate limiting metadata."""
    key: str
    last_used: float = 0.0
    calls_in_minute: int = 0


class AlphaVantageKeyPool:
    """
    Manages a pool of Alpha Vantage API keys with automatic rotation
    and rate limiting to maximize throughput while respecting API limits.

    Features:
    - Round-robin key rotation
    - Per-key rate limiting (5 calls/minute)
    - Automatic cooldown when all keys exhausted
    - Fallback to single key if ALPHA_VANTAGE_KEYS not set
    """

    def __init__(self):
        """Initialize the key pool from environment variables."""
        self.keys: List[APIKey] = []
        self.current_index = 0
        self._load_keys()

    def _load_keys(self):
        """Load API keys from environment variables.

        Tries ALPHA_VANTAGE_KEYS (plural, comma-separated) first,
        then falls back to ALPHA_VANTAGE_KEY (singular) for backwards compatibility.
        """
        # Try ALPHA_VANTAGE_KEYS (plural) first
        keys_str = os.getenv("ALPHA_VANTAGE_KEYS", "")
        if keys_str and keys_str.strip():
            key_list = [k.strip() for k in keys_str.split(",") if k.strip()]
            self.keys = [APIKey(key=k) for k in key_list]
            logger.info(f"✅ Loaded {len(self.keys)} Alpha Vantage keys from ALPHA_VANTAGE_KEYS")
            return

        # Fallback to single key for backwards compatibility
        single_key = os.getenv("ALPHA_VANTAGE_KEY", "")
        if single_key and single_key.strip():
            self.keys = [APIKey(key=single_key)]
            logger.info("✅ Loaded 1 Alpha Vantage key from ALPHA_VANTAGE_KEY (legacy)")
            return

        logger.warning("⚠️  No Alpha Vantage keys configured. Set ALPHA_VANTAGE_KEY or ALPHA_VANTAGE_KEYS in environment.")

    def get_key(self) -> Optional[str]:
        """Get next available API key with rate limiting.

        Uses round-robin rotation to distribute load across all keys.
        Automatically waits if all keys are rate-limited.

        Returns:
            str: The next available API key
            None: If no keys are configured

        Raises:
            After multiple retries, if no key becomes available
        """
        if not self.keys:
            logger.warning("⚠️  No API keys available in pool")
            return None

        now = time.time()
        attempts = 0
        max_attempts = len(self.keys) * 3

        while attempts < max_attempts:
            # Try each key in round-robin order
            for _ in range(len(self.keys)):
                key = self.keys[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.keys)

                # Reset counter if minute has passed (60 seconds)
                if now - key.last_used > 60:
                    key.calls_in_minute = 0

                # Check if key is available (< 5 calls in last minute)
                if key.calls_in_minute < 5:
                    # Increment counter and update time
                    key.calls_in_minute += 1
                    key.last_used = now
                    return key.key

            attempts += 1

            # All keys exhausted, wait for cooldown
            if attempts < max_attempts:
                wait_time = 12 if attempts < max_attempts - 1 else 30
                logger.info(
                    f"🔄 All Alpha Vantage keys rate-limited ({len(self.keys)} keys × 5 calls/min). "
                    f"Waiting {wait_time}s before retry... (attempt {attempts}/{max_attempts})"
                )
                time.sleep(wait_time)
                now = time.time()

        logger.error(f"❌ Could not get available API key after {max_attempts} attempts")
        return None

    def reset_stats(self):
        """Reset all key usage statistics.

        Useful for manual testing or when you want to force a fresh start.
        Call this after making bulk API calls to clear the rate limiting state.
        """
        for key in self.keys:
            key.calls_in_minute = 0
            key.last_used = 0.0
        logger.info(f"🔄 Reset usage stats for {len(self.keys)} Alpha Vantage keys")

    def get_status(self) -> dict:
        """Get current pool status and statistics.

        Returns:
            dict: Status information including number of keys, current calls per key, etc.
        """
        now = time.time()
        key_stats = []

        for idx, key in enumerate(self.keys):
            # Reset counter if minute has passed
            if now - key.last_used > 60:
                calls_this_minute = 0
            else:
                calls_this_minute = key.calls_in_minute

            key_stats.append({
                "index": idx,
                "key_preview": f"{key.key[:10]}...{key.key[-4:]}",
                "calls_this_minute": calls_this_minute,
                "available": calls_this_minute < 5,
                "last_used_seconds_ago": round(now - key.last_used, 1)
            })

        return {
            "total_keys": len(self.keys),
            "max_calls_per_minute": len(self.keys) * 5,
            "keys": key_stats,
            "current_index": self.current_index
        }

    def get_key_count(self) -> int:
        """Get number of API keys in the pool.

        Returns:
            int: Number of configured keys (0 if none configured)
        """
        return len(self.keys)


# Global singleton instance
# Use this instance throughout the application instead of creating new ones
key_pool = AlphaVantageKeyPool()
