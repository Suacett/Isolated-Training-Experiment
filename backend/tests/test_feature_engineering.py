"""
Safety Net Tests - Feature Engineering Module

Tests for backend/services/feature_engineering.py to ensure refactoring
doesn't break core feature computation logic.

These tests verify:
- compute_technical_features() computes all 39 features correctly
- process_stock_data() handles edge cases (missing data, NaN values)
- Feature column lists match expected counts (37 for v6, 41 for v7)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# Import the module under test
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.feature_engineering import (
    compute_technical_features,
    merge_insider_data,
    merge_sentiment_data,
    create_target_variables,
    process_stock_data,
    get_feature_columns,
    get_model_input_features,
    get_model_input_features_v7,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Create realistic OHLCV data for testing (300 trading days for rolling windows)."""
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=300, freq="B")  # Business days
    
    # Generate realistic price data using random walk
    close_prices = 100 + np.cumsum(np.random.randn(300) * 2)  # Start at $100
    close_prices = np.maximum(close_prices, 10)  # Floor at $10
    
    # Generate OHLCV from close prices
    df = pd.DataFrame({
        "date": dates,
        "open": close_prices * (1 + np.random.uniform(-0.01, 0.01, 300)),
        "high": close_prices * (1 + np.random.uniform(0, 0.02, 300)),
        "low": close_prices * (1 + np.random.uniform(-0.02, 0, 300)),
        "close": close_prices,
        "volume": np.random.randint(1_000_000, 10_000_000, 300),
    })
    
    return df


@pytest.fixture
def minimal_ohlcv_data():
    """Minimal OHLCV data for edge case testing (50 days)."""
    np.random.seed(123)
    dates = pd.date_range(start="2023-01-01", periods=50, freq="B")
    close_prices = 100 + np.cumsum(np.random.randn(50) * 1)
    
    return pd.DataFrame({
        "date": dates,
        "open": close_prices * 0.99,
        "high": close_prices * 1.01,
        "low": close_prices * 0.98,
        "close": close_prices,
        "volume": [1_000_000] * 50,
    })


@pytest.fixture
def sample_insider_data():
    """Sample insider trading data."""
    dates = pd.date_range(start="2023-01-15", periods=5, freq="W")
    
    return pd.DataFrame({
        "date": dates,
        "shares": [1000, 5000, 2000, 0, 3000],
        "amount": [50000, 250000, 100000, 0, 150000],
        "buy_flag": [1, 1, 0, -1, 1],
    })


@pytest.fixture
def sample_sentiment_data():
    """Sample sentiment data."""
    dates = pd.date_range(start="2023-01-01", periods=20, freq="B")
    
    return pd.DataFrame({
        "date": dates,
        "sentiment": np.random.uniform(-1, 1, 20),
        "num_articles": np.random.randint(0, 50, 20),
    })


# =============================================================================
# TESTS: compute_technical_features()
# =============================================================================

def test_compute_technical_features_returns_dataframe(sample_ohlcv_data):
    """Test that compute_technical_features returns a DataFrame."""
    result = compute_technical_features(sample_ohlcv_data)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_compute_technical_features_expected_columns(sample_ohlcv_data):
    """Test that all expected technical feature columns are created."""
    result = compute_technical_features(sample_ohlcv_data)
    
    expected_columns = [
        # Log returns
        "YesterdayClose", "YesterdayOpenLogR", "YesterdayHighLogR",
        "YesterdayLowLogR", "YesterdayVolumeLogR", "YesterdayCloseLogR",
        # Moving averages
        "MA10", "MA20", "MA30", "EMA10", "EMA30",
        # Time features
        "DayOfWeek", "DayOfMonth", "MonthNumber",
        # Technical indicators
        "RSI", "MACD", "MACD_Signal", "BollingerUpper", "BollingerLower",
        # Volatility
        "Volatility_10", "Volatility_20", "Volatility_30",
        "volatility_5d", "volatility_20d",
        # Volume indicators
        "OBV", "abnormal_vol",
        # Price patterns
        "ZScore", "overnight_gap", "momentum_5d", "momentum_20d",
        "skew_5d", "intraday_range",
        # V7 features
        "high_52w_dist", "high_52w_dist_lag1", "market_momentum_lag1",
        "vix_proxy", "regime_volatility",
    ]
    
    for col in expected_columns:
        assert col in result.columns, f"Missing expected column: {col}"


def test_compute_technical_features_no_future_leakage(sample_ohlcv_data):
    """Test that lagged features don't use future data (data leakage check)."""
    result = compute_technical_features(sample_ohlcv_data)
    
    # The first row should have NaN for lagged features
    # (no "yesterday" data available for the first day)
    assert pd.isna(result.iloc[0]["YesterdayClose"])
    assert pd.isna(result.iloc[0]["YesterdayCloseLogR"])


def test_compute_technical_features_rsi_bounds(sample_ohlcv_data):
    """Test that RSI values are within expected bounds [0, 100]."""
    result = compute_technical_features(sample_ohlcv_data)
    
    # Drop NaN values before checking bounds
    rsi_valid = result["RSI"].dropna()
    
    assert (rsi_valid >= 0).all(), "RSI should be >= 0"
    assert (rsi_valid <= 100).all(), "RSI should be <= 100"


def test_compute_technical_features_handles_zeros_in_volume(sample_ohlcv_data):
    """Test that zero volumes don't cause division errors."""
    df = sample_ohlcv_data.copy()
    df.loc[5, "volume"] = 0  # Set one volume to zero
    
    # Should not raise an error
    result = compute_technical_features(df)
    
    assert isinstance(result, pd.DataFrame)


# =============================================================================
# TESTS: merge_insider_data()
# =============================================================================

def test_merge_insider_data_with_valid_data(sample_ohlcv_data, sample_insider_data):
    """Test merging insider data creates expected columns."""
    result = merge_insider_data(sample_ohlcv_data, sample_insider_data)
    
    assert "insider_shares" in result.columns
    assert "insider_amount" in result.columns
    assert "insider_buy_flag" in result.columns


def test_merge_insider_data_with_none(sample_ohlcv_data):
    """Test merging with None insider data creates zero-filled columns."""
    result = merge_insider_data(sample_ohlcv_data, None)
    
    assert "insider_shares" in result.columns
    assert (result["insider_shares"] == 0).all()
    assert (result["insider_buy_flag"] == -1).all()


def test_merge_insider_data_with_empty_df(sample_ohlcv_data):
    """Test merging with empty DataFrame creates zero-filled columns."""
    empty_insider = pd.DataFrame(columns=["date", "shares", "amount", "buy_flag"])
    result = merge_insider_data(sample_ohlcv_data, empty_insider)
    
    assert (result["insider_shares"] == 0).all()


# =============================================================================
# TESTS: merge_sentiment_data()
# =============================================================================

def test_merge_sentiment_data_with_valid_data(sample_ohlcv_data, sample_sentiment_data):
    """Test merging sentiment data creates expected columns."""
    result = merge_sentiment_data(sample_ohlcv_data, sample_sentiment_data)
    
    assert "sentiment" in result.columns
    assert "num_articles" in result.columns
    assert "sentiment_change" in result.columns


def test_merge_sentiment_data_with_none(sample_ohlcv_data):
    """Test merging with None sentiment data creates zero-filled columns."""
    result = merge_sentiment_data(sample_ohlcv_data, None)
    
    assert "sentiment" in result.columns
    assert (result["sentiment"] == 0).all()


# =============================================================================
# TESTS: create_target_variables()
# =============================================================================

def test_create_target_variables_expected_columns(sample_ohlcv_data):
    """Test that all target variable columns are created."""
    result = create_target_variables(sample_ohlcv_data.copy())
    
    expected_targets = ["Target_1d", "Target_1w", "Target_1m", "Target_6m"]
    
    for target in expected_targets:
        assert target in result.columns, f"Missing target: {target}"


def test_create_target_variables_are_log_returns(sample_ohlcv_data):
    """Test that targets are computed as log returns."""
    result = create_target_variables(sample_ohlcv_data.copy())
    
    # Manually verify 1-day target for row 0
    expected_1d = np.log(sample_ohlcv_data["close"].iloc[1] / sample_ohlcv_data["close"].iloc[0])
    actual_1d = result["Target_1d"].iloc[0]
    
    assert np.isclose(expected_1d, actual_1d, rtol=1e-10)


# =============================================================================
# TESTS: process_stock_data() - Full Pipeline
# =============================================================================

def test_process_stock_data_returns_complete_dataframe(sample_ohlcv_data):
    """Test that the full pipeline returns a complete DataFrame."""
    result = process_stock_data(sample_ohlcv_data.copy())
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    # NaN rows should be dropped
    assert not result.isnull().values.any()


def test_process_stock_data_without_targets(sample_ohlcv_data):
    """Test processing without creating target variables."""
    result = process_stock_data(sample_ohlcv_data.copy(), create_targets=False)
    
    assert "Target_1d" not in result.columns


def test_process_stock_data_string_dates(sample_ohlcv_data):
    """Test that string dates are converted to datetime."""
    df = sample_ohlcv_data.copy()
    df["date"] = df["date"].astype(str)  # Convert to string
    
    result = process_stock_data(df)
    
    assert pd.api.types.is_datetime64_any_dtype(result["date"])


# =============================================================================
# TESTS: Feature Column Lists
# =============================================================================

def test_get_feature_columns_count():
    """Test that get_feature_columns returns expected number of columns."""
    columns = get_feature_columns()
    
    # V7 has 43 total feature columns (including close, YesterdayClose)
    assert len(columns) >= 40, f"Expected ~43 columns, got {len(columns)}"


def test_get_model_input_features_count():
    """Test that v6 model uses 37 input features."""
    features = get_model_input_features()
    
    assert len(features) == 37, f"Expected 37 features for v6, got {len(features)}"


def test_get_model_input_features_v7_count():
    """Test that v7 model uses 41 input features."""
    features = get_model_input_features_v7()
    
    assert len(features) == 41, f"Expected 41 features for v7, got {len(features)}"


def test_v7_features_include_market_context():
    """Test that v7 features include market context features."""
    v7_features = get_model_input_features_v7()
    
    market_context_features = [
        "high_52w_dist_lag1", "market_momentum_lag1",
        "vix_proxy", "regime_volatility",
    ]
    
    for feature in market_context_features:
        assert feature in v7_features, f"V7 missing market context feature: {feature}"


def test_v6_features_exclude_v7_features():
    """Test that v6 features don't include v7-specific features."""
    v6_features = get_model_input_features()
    
    v7_only_features = [
        "high_52w_dist_lag1", "market_momentum_lag1",
        "vix_proxy", "regime_volatility",
    ]
    
    for feature in v7_only_features:
        assert feature not in v6_features, f"V6 should not have v7 feature: {feature}"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

def test_compute_technical_features_minimal_data(minimal_ohlcv_data):
    """Test behavior with minimal data (may have many NaN values)."""
    result = compute_technical_features(minimal_ohlcv_data)
    
    # Should not crash, even with minimal data
    assert isinstance(result, pd.DataFrame)


def test_process_stock_data_preserves_row_order(sample_ohlcv_data):
    """Test that processing preserves chronological order."""
    result = process_stock_data(sample_ohlcv_data.copy())
    
    # Dates should be in ascending order
    dates = result["date"].values
    assert all(dates[i] <= dates[i+1] for i in range(len(dates)-1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
