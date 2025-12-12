"""
Model Metadata Registry

Contains descriptions and reasoning for each model version,
explaining what was different about each training approach.
"""

from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"


@dataclass
class ModelInfo:
    """Information about a trained model."""
    version: str
    filename: str
    display_name: str
    description: str
    reasoning: str
    training_notes: str
    is_available: bool = False


# Model Registry with detailed reasoning
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    "v7": ModelInfo(
        version="v7",
        filename="lstm_model_v7.pth",
        display_name="V7 - Attention Model",
        description="BiLSTM + Multi-Head Attention with DirectionalLoss training",
        reasoning="""
        V7 introduces significant architectural improvements:
        - Bidirectional LSTM (captures forward and backward context)
        - Multi-Head Temporal Attention (learns which days matter most)
        - DirectionalLoss (penalizes wrong-sign predictions, not just magnitude)
        - Weighted sampling on high-volatility days
        - 41 features including lagged macro indicators
        
        Key characteristics:
        - More defensive/bearish (predicts "down" more often)
        - Better at vetoing bad trades
        - Down accuracy ~53% (higher than up accuracy)
        """,
        training_notes="Overall acc: 52.6%, Bias: -23.7% (bearish). Trained on 528 stocks, 1.7M samples."
    ),
    "v6": ModelInfo(
        version="v6",
        filename="lstm_model_v6.pth",
        display_name="V6 - Clean Data",
        description="Trained on deduplicated data with proper timestamp handling",
        reasoning="""
        Previous versions (v5 and earlier) had duplicate timestamps in the training data,
        which artificially inflated accuracy metrics. The model learned to predict 
        flat/unchanged prices because it saw the same data point multiple times.
        
        V6 fixes this by:
        - Removing all duplicate timestamps before training
        - Ensuring each data point is unique
        - This results in more REALISTIC accuracy (~54% for SPY) but genuine predictions
        """,
        training_notes="SPY accuracy: 54%, AAPL: 51%, AMD: 50%. Slight bullish bias (+8%)"
    ),
    "v5": ModelInfo(
        version="v5",
        filename="lstm_model_v5.pth",
        display_name="V5 - Full History (Dirty)",
        description="200+ stocks, streaming memmap, but contained duplicate data",
        reasoning="""
        V5 was designed to handle massive datasets using disk-based memmap.
        It successfully trained on ALL available stock history (1980-present).
        
        However, this version had an issue:
        - Training data contained duplicate timestamps
        - This caused artificially high accuracy (~68% for AAPL)
        - The model learned to predict 'flat' which often matched duplicated data
        - Strong bearish bias (-45%) due to data imbalance
        """,
        training_notes="SPY accuracy: 46%, AAPL: 68%* (*inflated), AMD: 69%* (*inflated). Heavy bearish bias."
    ),
    "v4": ModelInfo(
        version="v4",
        filename="lstm_model_v4.pth",
        display_name="V4 - RAM-Based",
        description="200 stocks loaded into RAM, 8GB limit",
        reasoning="""
        V4 attempted to scale training by loading more stocks into RAM.
        Limited to 200 stocks due to memory constraints (~8GB).
        
        Issues encountered:
        - Memory pressure caused OOM errors on smaller systems
        - Did not address the duplicate data problem
        - Similar bias issues to V5
        """,
        training_notes="May have OOM issues. Not recommended for production."
    ),
    "v3": ModelInfo(
        version="v3",
        filename="lstm_model_v3.pth",
        display_name="V3 - Legacy Fixed",
        description="Fixed 100 stock count, legacy architecture",
        reasoning="""
        V3 was an earlier iteration with a fixed training set of 100 stocks.
        Performed well on index ETFs (SPY ~54%) but poorly on individual stocks.
        
        Notes:
        - Good for index/ETF predictions
        - Poor for individual stock predictions (AAPL: 37%, AMD: 40%)
        - Simpler feature engineering
        """,
        training_notes="SPY: 54%, AAPL: 37%, AMD: 40%. Best for indices only."
    ),
    "v2": ModelInfo(
        version="v2",
        filename="lstm_model_v2.pth",
        display_name="V2 - Base Model",
        description="Initial production-ready model",
        reasoning="""
        V2 was the first stable, production-ready model.
        Used as the baseline for comparison with newer versions.
        
        Features:
        - Basic technical indicators
        - 60-day lookback window
        - Standard LSTM architecture
        """,
        training_notes="Baseline model. Stable but not optimized."
    ),
    "hybrid": ModelInfo(
        version="hybrid",
        filename="lstm_model_hybrid.pth",
        display_name="Hybrid - INCOMPATIBLE",
        description="⚠️ Expects 39 features (current: 37). Cannot be used.",
        reasoning="""
        The hybrid model was trained with 39 input features, but current
        feature engineering produces only 37 features.
        
        This model is INCOMPATIBLE and will be skipped during loading.
        
        Original purpose: Combined multiple data sources in training.
        """,
        training_notes="INCOMPATIBLE: Feature count mismatch (39 vs 37)."
    ),
    "legacy": ModelInfo(
        version="legacy",
        filename="lstm_model_legacy.pth",
        display_name="Legacy - INCOMPATIBLE",
        description="⚠️ Expects 39 features (current: 37). Cannot be used.",
        reasoning="""
        The legacy model was trained with 39 input features, but current
        feature engineering produces only 37 features.
        
        This model is INCOMPATIBLE and will be skipped during loading.
        
        Original purpose: Direct port from the original Keras/TensorFlow model.
        """,
        training_notes="INCOMPATIBLE: Feature count mismatch (39 vs 37)."
    ),
    "pre_stationarity": ModelInfo(
        version="pre_stationarity",
        filename="lstm_model_pre_stationarity.pth",
        display_name="Pre-Stationarity",
        description="Before log-return stationarity transforms were added",
        reasoning="""
        This checkpoint was saved before implementing stationarity transforms.
        Stock prices are non-stationary, which can confuse LSTM models.
        
        Later versions use log-returns which are more stationary.
        This model predicts raw prices instead of returns.
        """,
        training_notes="Predicts raw prices. May have scale issues."
    ),
    "backup": ModelInfo(
        version="backup",
        filename="lstm_model_backup.pth",
        display_name="Backup",
        description="Backup checkpoint before major changes",
        reasoning="""
        A backup checkpoint saved before a major architecture change.
        Kept for rollback purposes.
        """,
        training_notes="Backup only. Use other versions."
    ),
    "default": ModelInfo(
        version="default",
        filename="lstm_model.pth",
        display_name="Default",
        description="Default/fallback model",
        reasoning="""
        The default model used when no version is specified.
        Usually a copy of the current production model.
        """,
        training_notes="Fallback model."
    ),
}


def get_available_models() -> List[ModelInfo]:
    """
    Scan the models directory and return list of available models with metadata.
    """
    available = []
    
    # Models that are known to be incompatible with current feature engineering
    INCOMPATIBLE_VERSIONS = {"legacy", "hybrid"}
    
    for version, info in MODEL_REGISTRY.items():
        # Skip incompatible models
        if version in INCOMPATIBLE_VERSIONS:
            logger.debug(f"Skipping incompatible model: {version}")
            continue
            
        model_path = MODELS_DIR / info.filename
        scaler_path = MODELS_DIR / f"scaler_{version}.pkl"
        
        # Check if model file exists
        info.is_available = model_path.exists()
        
        if info.is_available:
            # Also check for matching scaler
            has_scaler = scaler_path.exists()
            if not has_scaler:
                # Try default scaler name
                has_scaler = (MODELS_DIR / "scaler.pkl").exists()
            
            logger.debug(f"Model {version}: available={info.is_available}, scaler={has_scaler}")
            available.append(info)
    
    # Sort by version (v6 first, then v5, etc.)
    def sort_key(m):
        if m.version.startswith("v"):
            try:
                return (0, -int(m.version[1:]))
            except ValueError:
                return (1, m.version)
        return (2, m.version)
    
    return sorted(available, key=sort_key)


def get_model_info(version: str) -> Optional[ModelInfo]:
    """Get metadata for a specific model version."""
    return MODEL_REGISTRY.get(version)


def get_model_path(version: str) -> Optional[Path]:
    """Get the file path for a model version."""
    info = MODEL_REGISTRY.get(version)
    if info:
        path = MODELS_DIR / info.filename
        if path.exists():
            return path
    return None


def get_scaler_path(version: str) -> Optional[Path]:
    """Get the scaler path for a model version."""
    # Try version-specific scaler first
    scaler_path = MODELS_DIR / f"scaler_{version}.pkl"
    if scaler_path.exists():
        return scaler_path
    
    # Fall back to default scaler
    default_scaler = MODELS_DIR / "scaler.pkl"
    if default_scaler.exists():
        return default_scaler
    
    # Try v2 scaler as last resort (most compatible)
    v2_scaler = MODELS_DIR / "scaler_v2.pkl"
    if v2_scaler.exists():
        return v2_scaler
    
    return None
