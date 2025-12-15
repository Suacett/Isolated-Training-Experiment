"""Custom exception classes for the Proxmox AI Stock Predictor application.

This module defines a hierarchy of exceptions to replace generic Exception catches
throughout the codebase, enabling more specific error handling and better debugging.
"""


# =============================================================================
# MODEL-RELATED EXCEPTIONS
# =============================================================================

class ModelError(Exception):
    """Base exception for model-related errors."""
    pass


class ModelLoadError(ModelError):
    """Error loading model weights or checkpoint file."""
    pass


class CheckpointCorruptedError(ModelLoadError):
    """Model checkpoint file is corrupted or has invalid structure."""
    pass


class FeatureDimensionMismatchError(ModelError):
    """Feature dimensions don't match model's expected input_dim."""
    pass


class InsufficientMemoryError(ModelError):
    """Not enough GPU memory available to load model."""
    pass


class PredictionError(ModelError):
    """Error occurred during model prediction/inference."""
    pass


# =============================================================================
# SCALER-RELATED EXCEPTIONS
# =============================================================================

class ScalerError(Exception):
    """Base exception for scaler-related errors."""
    pass


class ScalerMissingError(ScalerError):
    """Scaler file not found at specified path."""
    pass


class ScalerCorruptedError(ScalerError):
    """Scaler file is corrupted or has invalid format."""
    pass


# =============================================================================
# DATA INGESTION EXCEPTIONS
# =============================================================================

class DataIngestionError(Exception):
    """Base exception for data fetching/processing errors."""
    pass


class APIRateLimitError(DataIngestionError):
    """External API rate limit exceeded."""
    pass


class DataNotFoundError(DataIngestionError):
    """Requested ticker/data not found in external source."""
    pass


class DataQualityError(DataIngestionError):
    """Data received but quality is insufficient (missing fields, NaN values, etc.)."""
    pass


# =============================================================================
# RISK MANAGEMENT EXCEPTIONS
# =============================================================================

class RiskManagementError(Exception):
    """Base exception for risk management errors."""
    pass


class InsufficientDataError(RiskManagementError):
    """Not enough historical data for risk calculations."""
    pass


# =============================================================================
# CONFIGURATION EXCEPTIONS
# =============================================================================

class ConfigurationError(Exception):
    """Base exception for configuration-related errors."""
    pass


class MissingAPIKeyError(ConfigurationError):
    """Required API key is missing from configuration."""
    pass
