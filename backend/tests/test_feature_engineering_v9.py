"""
Safety Net Tests - V9 Feature Engineering

Tests for backend/services/feature_engineering_v9.py to ensure refactoring
doesn't break the V9 stationary feature computation.

These tests verify:
- All 12 V9 features are computed correctly
- Macro data (SPY, VIX) processing works
- Cross-sectional rank calculation
- Feature stationarity (no raw price leakage)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# Import the module under test
# Remove manual path manipulation

from services.feature_engineering_v9 import (
    V9_FEATURE_NAMES,
    N_FEATURES,
    compute_log_returns,
    compute_rsi,
    compute_macd_histogram_normalized,
    compute_volume_ratio,
    compute_distance_from_sma,
    compute_volatility,
    compute_spy_correlation,
    compute_relative_strength,
    process_spy_data,
    process_vix_data,
    compute_v9_features,
    compute_future_return,
    compute_cross_sectional_ranks,
    process_stock_for_v9,
    get_v9_feature_names,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Create realistic OHLCV data for testing (200 trading days)."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=200, freq="B")
    
    # Generate realistic price data using random walk
    close_prices = 100 + np.cumsum(np.random.randn(200) * 2)
    close_prices = np.maximum(close_prices, 10)
    
    df = pd.DataFrame({
        "date": dates,
        "open": close_prices * (1 + np.random.uniform(-0.01, 0.01, 200)),
        "high": close_prices * (1 + np.random.uniform(0, 0.02, 200)),
        "low": close_prices * (1 + np.random.uniform(-0.02, 0, 200)),
        "close": close_prices,
        "volume": np.random.randint(1_000_000, 10_000_000, 200),
    })
    
    return df


@pytest.fixture
def sample_spy_data():
    """Create sample SPY data for macro features."""
    np.random.seed(123)
    dates = pd.date_range(start="2023-01-01", periods=200, freq="B")
    
    close_prices = 400 + np.cumsum(np.random.randn(200) * 2)
    
    return pd.DataFrame({
        "date": dates,
        "close": close_prices,
        "high": close_prices * 1.01,
        "low": close_prices * 0.99,
        "volume": np.random.randint(50_000_000, 100_000_000, 200),
    })


@pytest.fixture
def sample_vix_data():
    """Create sample VIX data."""
    np.random.seed(456)
    dates = pd.date_range(start="2023-01-01", periods=200, freq="B")
    
    vix_levels = 20 + np.cumsum(np.random.randn(200) * 0.3)
    vix_levels = np.clip(vix_levels, 10, 50)
    
    return pd.DataFrame({
        "date": dates,
        "close": vix_levels,
    })


# =============================================================================
# TESTS: Constants
# =============================================================================

def test_n_features_equals_12():
    """Test that N_FEATURES is exactly 12."""
    assert N_FEATURES == 12


def test_v9_feature_names_count():
    """Test that V9_FEATURE_NAMES has 12 features."""
    assert len(V9_FEATURE_NAMES) == 12


def test_v9_feature_names_expected():
    """Test that V9 features include expected names."""
    expected = [
        'log_ret_1d', 'log_ret_5d', 'log_ret_20d',
        'rsi_14', 'macd_hist_norm', 'volume_ratio',
        'dist_sma20', 'volatility_20d',
        'spy_ret_5d', 'vix_level',
        'spy_corr_20d', 'rel_strength_20d',
    ]
    
    for feature in expected:
        assert feature in V9_FEATURE_NAMES, f"Missing feature: {feature}"


def test_get_v9_feature_names():
    """Test get_v9_feature_names returns same as V9_FEATURE_NAMES."""
    assert get_v9_feature_names() == V9_FEATURE_NAMES


# =============================================================================
# TESTS: Individual Feature Calculations
# =============================================================================

def test_compute_log_returns(sample_ohlcv_data):
    """Test log return computation."""
    result = compute_log_returns(sample_ohlcv_data.copy())
    
    assert "log_ret_1d" in result.columns
    assert "log_ret_5d" in result.columns
    assert "log_ret_20d" in result.columns


def test_compute_rsi(sample_ohlcv_data):
    """Test RSI computation."""
    result = compute_rsi(sample_ohlcv_data.copy())
    
    assert "rsi_14" in result.columns
    
    # RSI should be normalized to 0-1 range
    valid = result["rsi_14"].dropna()
    assert (valid >= 0).all() and (valid <= 1).all()


def test_compute_macd_histogram_normalized(sample_ohlcv_data):
    """Test normalized MACD histogram computation."""
    result = compute_macd_histogram_normalized(sample_ohlcv_data.copy())
    
    assert "macd_hist_norm" in result.columns


def test_compute_volume_ratio(sample_ohlcv_data):
    """Test volume ratio computation."""
    result = compute_volume_ratio(sample_ohlcv_data.copy())
    
    assert "volume_ratio" in result.columns
    
    # Volume ratio should be centered around 1.0
    valid = result["volume_ratio"].dropna()
    assert valid.mean() > 0.5 and valid.mean() < 2.0


def test_compute_distance_from_sma(sample_ohlcv_data):
    """Test distance from SMA computation."""
    result = compute_distance_from_sma(sample_ohlcv_data.copy())
    
    assert "dist_sma20" in result.columns


def test_compute_volatility(sample_ohlcv_data):
    """Test volatility computation."""
    result = compute_volatility(sample_ohlcv_data.copy())
    
    assert "volatility_20d" in result.columns
    
    # Volatility should be positive
    valid = result["volatility_20d"].dropna()
    assert (valid >= 0).all()


# =============================================================================
# TESTS: Macro Data Processing
# =============================================================================

def test_process_spy_data(sample_spy_data):
    """Test SPY data processing."""
    result = process_spy_data(sample_spy_data)
    
    assert "spy_ret_5d" in result
    assert "spy_ret_20d" in result
    assert "spy_daily_ret" in result


def test_process_vix_data(sample_vix_data):
    """Test VIX data processing."""
    result = process_vix_data(sample_vix_data)
    
    assert isinstance(result, pd.Series)
    assert result.name == "vix_level"
    
    # VIX should be normalized (divided by 100)
    assert result.max() < 1.0  # VIX=50 -> 0.50


# =============================================================================
# TESTS: compute_v9_features() - Main Function
# =============================================================================

def test_compute_v9_features_returns_dataframe(sample_ohlcv_data, sample_spy_data, sample_vix_data):
    """Test that compute_v9_features returns a DataFrame."""
    spy_data = process_spy_data(sample_spy_data)
    vix_data = process_vix_data(sample_vix_data)
    
    result = compute_v9_features(sample_ohlcv_data, spy_data, vix_data)
    
    assert isinstance(result, pd.DataFrame)


def test_compute_v9_features_all_columns_present(sample_ohlcv_data, sample_spy_data, sample_vix_data):
    """Test that all 12 V9 features are computed."""
    spy_data = process_spy_data(sample_spy_data)
    vix_data = process_vix_data(sample_vix_data)
    
    result = compute_v9_features(sample_ohlcv_data, spy_data, vix_data)
    
    for feature in V9_FEATURE_NAMES:
        assert feature in result.columns, f"Missing feature: {feature}"


def test_compute_v9_features_no_raw_prices(sample_ohlcv_data, sample_spy_data, sample_vix_data):
    """Test that V9 features don't include raw prices (stationarity)."""
    spy_data = process_spy_data(sample_spy_data)
    vix_data = process_vix_data(sample_vix_data)
    
    result = compute_v9_features(sample_ohlcv_data, spy_data, vix_data)
    
    # Should not include raw OHLCV
    for col in ["open", "high", "low", "close", "volume"]:
        assert col not in result.columns


def test_compute_v9_features_without_macro_data(sample_ohlcv_data):
    """Test V9 features computation without SPY/VIX data."""
    result = compute_v9_features(sample_ohlcv_data, None, None)
    
    assert isinstance(result, pd.DataFrame)
    
    # Macro features should be filled with defaults
    assert "spy_ret_5d" in result.columns
    assert "vix_level" in result.columns


# =============================================================================
# TESTS: Target Variable Computation
# =============================================================================

def test_compute_future_return(sample_ohlcv_data):
    """Test future return computation for ranking target."""
    result = compute_future_return(sample_ohlcv_data.copy(), horizon=5)
    
    assert "future_ret_5d" in result.columns


def test_compute_future_return_different_horizons(sample_ohlcv_data):
    """Test future return with different horizons."""
    for horizon in [1, 5, 10, 20]:
        result = compute_future_return(sample_ohlcv_data.copy(), horizon=horizon)
        
        assert f"future_ret_{horizon}d" in result.columns


# =============================================================================
# TESTS: Cross-Sectional Ranking
# =============================================================================

def test_compute_cross_sectional_ranks():
    """Test cross-sectional rank computation."""
    # Create sample data for multiple stocks
    dates = pd.date_range(start="2023-01-01", periods=10, freq="B")
    
    all_stocks = pd.DataFrame({
        "date": list(dates) * 3,
        "ticker": ["AAPL"] * 10 + ["MSFT"] * 10 + ["GOOGL"] * 10,
        "future_ret_5d": (
            [0.05] * 10 +  # AAPL: 5% return
            [0.10] * 10 +  # MSFT: 10% return (best)
            [0.02] * 10    # GOOGL: 2% return (worst)
        ),
    })
    
    result = compute_cross_sectional_ranks(all_stocks)
    
    assert "target_rank" in result.columns
    
    # Ranks should be between 0 and 1
    assert (result["target_rank"] >= 0).all()
    assert (result["target_rank"] <= 1).all()


def test_compute_cross_sectional_ranks_ordering():
    """Test that ranks are correctly ordered."""
    dates = pd.date_range(start="2023-01-01", periods=1, freq="B")
    
    all_stocks = pd.DataFrame({
        "date": list(dates) * 3,
        "ticker": ["AAPL", "MSFT", "GOOGL"],
        "future_ret_5d": [0.05, 0.10, 0.02],  # MSFT best, AAPL middle, GOOGL worst
    })
    
    result = compute_cross_sectional_ranks(all_stocks)
    
    msft_rank = result[result["ticker"] == "MSFT"]["target_rank"].iloc[0]
    aapl_rank = result[result["ticker"] == "AAPL"]["target_rank"].iloc[0]
    googl_rank = result[result["ticker"] == "GOOGL"]["target_rank"].iloc[0]
    
    # MSFT should have highest rank
    assert msft_rank > aapl_rank
    assert aapl_rank > googl_rank


# =============================================================================
# TESTS: Full Pipeline
# =============================================================================

def test_process_stock_for_v9(sample_ohlcv_data, sample_spy_data, sample_vix_data):
    """Test full V9 processing pipeline."""
    spy_data = process_spy_data(sample_spy_data)
    vix_data = process_vix_data(sample_vix_data)
    
    result = process_stock_for_v9(
        sample_ohlcv_data,
        ticker="AAPL",
        spy_data=spy_data,
        vix_data=vix_data
    )
    
    assert isinstance(result, pd.DataFrame)
    assert "ticker" in result.columns
    assert result["ticker"].iloc[0] == "AAPL"


def test_process_stock_for_v9_has_all_features(sample_ohlcv_data, sample_spy_data, sample_vix_data):
    """Test that pipeline produces all required features."""
    spy_data = process_spy_data(sample_spy_data)
    vix_data = process_vix_data(sample_vix_data)
    
    result = process_stock_for_v9(
        sample_ohlcv_data,
        ticker="AAPL",
        spy_data=spy_data,
        vix_data=vix_data
    )
    
    for feature in V9_FEATURE_NAMES:
        assert feature in result.columns, f"Missing feature: {feature}"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

def test_compute_v9_features_minimal_data():
    """Test with minimal data (edge case)."""
    dates = pd.date_range(start="2023-01-01", periods=30, freq="B")
    
    df = pd.DataFrame({
        "date": dates,
        "open": [100] * 30,
        "high": [101] * 30,
        "low": [99] * 30,
        "close": [100] * 30,
        "volume": [1000000] * 30,
    })
    
    # Should not crash with minimal data
    result = compute_v9_features(df, None, None)
    assert isinstance(result, pd.DataFrame)


def test_spy_correlation_with_missing_dates(sample_ohlcv_data):
    """Test SPY correlation when dates don't align perfectly."""
    # Create SPY data with different date range
    spy_dates = pd.date_range(start="2023-02-01", periods=100, freq="B")
    spy_df = pd.DataFrame({
        "date": spy_dates,
        "close": 400 + np.cumsum(np.random.randn(100) * 2),
    })
    
    spy_data = process_spy_data(spy_df)
    
    # Should handle misaligned dates gracefully
    result = compute_v9_features(sample_ohlcv_data, spy_data, None)
    assert isinstance(result, pd.DataFrame)


# Removed pytest.main()
