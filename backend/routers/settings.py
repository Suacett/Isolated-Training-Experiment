"""
Settings Router - For managing API keys and configuration.
Extracted from main.py (Phase 5.2).
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils.config_loader import settings, save_config, clear_config
from state import state

router = APIRouter(
    prefix="/settings",
    tags=["settings"]
)

logger = logging.getLogger(__name__)


class APIKeys(BaseModel):
    ALPHA_VANTAGE_KEY: Optional[str] = None
    ALPHA_VANTAGE_KEYS: Optional[str] = None  # Comma-separated for multiple keys


@router.post("/keys")
async def save_keys(keys: APIKeys):
    """Save API keys. Supports single key (legacy) or multiple keys (recommended)."""
    try:
        # Save both single and multiple keys
        save_config(
            alpha_vantage_key=keys.ALPHA_VANTAGE_KEY,
            alpha_vantage_keys=keys.ALPHA_VANTAGE_KEYS
        )

        # Update state based on presence of valid keys
        alpha_vantage_keys_list = [k.strip() for k in (keys.ALPHA_VANTAGE_KEYS or "").split(",") if k.strip()]
        has_keys = bool(keys.ALPHA_VANTAGE_KEY or alpha_vantage_keys_list)
        state.alpha_vantage_enabled = has_keys
        
        if has_keys:
            if alpha_vantage_keys_list:
                key_count = len(alpha_vantage_keys_list)
                logger.info(f"✅ Alpha Vantage keys saved: {key_count} keys configured")
            else:
                logger.info("✅ Alpha Vantage key saved (legacy single key)")
        else:
            logger.info("ℹ️ No valid Alpha Vantage keys provided; service disabled.")

        return {
            "message": "Settings saved successfully",
            "alpha_vantage_saved": bool(keys.ALPHA_VANTAGE_KEY or keys.ALPHA_VANTAGE_KEYS),
            "data_source": "yahoo_finance",
            "key_type": "multiple" if keys.ALPHA_VANTAGE_KEYS else "single" if keys.ALPHA_VANTAGE_KEY else None
        }
    except Exception as e:
        logger.error(f"Failed to save keys: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save keys: {str(e)}")


@router.delete("/keys")
async def reset_keys():
    """Clear all saved API keys."""
    try:
        clear_config()
        state.alpha_vantage_enabled = False
        return {"message": "Keys reset successfully"}
    except Exception as e:
        logger.error(f"Failed to reset keys: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset keys: {str(e)}")
