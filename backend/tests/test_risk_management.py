"""
Safety Net Tests - Risk Management Module

Tests for backend/services/risk_management.py to ensure refactoring
doesn't break risk management logic.

These tests verify:
- ATR calculation and trailing stops
- Drawdown-based exposure scaling
- Market regime detection (VIX + SPY)
- Correlation filtering for diversification
- Volatility weighting
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# Import the module under test
# Remove manual path manipulation for better test hygiene

from services.risk_management import (
    calculate_atr,
    get_atr_trailing_stop,
    get_drawdown_exposure,
    get_regime_exposure,
    PositionTracker,
    select_with_correlation_filter,
    calculate_volatility_weights,
    calculate_hybrid_weights,
    calculate_final_exposure,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_stock_df():
    """Create sample stock data for 100 trading days."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
    
    # Generate realistic price data
    close_prices = 100 + np.cumsum(np.random.randn(100) * 2)
    close_prices = np.maximum(close_prices, 10)
    
    df = pd.DataFrame({
        "date": dates,
        "open": close_prices * (1 + np.random.uniform(-0.01, 0.01, 100)),
        "high": close_prices * (1 + np.random.uniform(0, 0.03, 100)),
        "low": close_prices * (1 + np.random.uniform(-0.03, 0, 100)),
        "close": close_prices,
        "volume": np.random.randint(1_000_000, 10_000_000, 100),
    })
    
    return df


@pytest.fixture
def sample_spy_df():
    """Create sample SPY data for regime detection."""
    np.random.seed(123)
    dates = pd.date_range(start="2022-01-01", periods=300, freq="B")
    
    # SPY price with trending behavior
    close_prices = 400 + np.cumsum(np.random.randn(300) * 2)
    
    return pd.DataFrame({
        "date": dates,
        "close": close_prices,
        "high": close_prices * 1.01,
        "low": close_prices * 0.99,
    })


@pytest.fixture
def sample_vix_df():
    """Create sample VIX data."""
    np.random.seed(456)
    dates = pd.date_range(start="2022-01-01", periods=300, freq="B")
    
    # VIX ranging from 15 to 35
    vix_levels = 20 + np.cumsum(np.random.randn(300) * 0.5)
    vix_levels = np.clip(vix_levels, 12, 40)
    
    return pd.DataFrame({
        "date": dates,
        "close": vix_levels,
    })


@pytest.fixture
def sample_stock_data_dict(sample_stock_df):
    """Create a dictionary of stock data for multiple tickers."""
    np.random.seed(789)
    
    stock_data = {}
    for ticker in ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]:
        df = sample_stock_df.copy()
        # Add some variation to each stock
        df["close"] = df["close"] * np.random.uniform(0.8, 1.2)
        df["high"] = df["close"] * 1.02
        df["low"] = df["close"] * 0.98
        stock_data[ticker] = df
    
    return stock_data


# =============================================================================
# TESTS: calculate_atr()
# =============================================================================

def test_calculate_atr_returns_series(sample_stock_df):
    """Test that calculate_atr returns a Series."""
    result = calculate_atr(sample_stock_df)
    
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_stock_df)


def test_calculate_atr_positive_values(sample_stock_df):
    """Test that ATR values are positive (after warmup period)."""
    result = calculate_atr(sample_stock_df, period=14)
    
    # ATR should be positive after warmup period
    valid_atr = result.iloc[14:]
    assert (valid_atr.dropna() > 0).all()


def test_calculate_atr_period_affects_values(sample_stock_df):
    """Test that different periods produce different ATR values."""
    atr_14 = calculate_atr(sample_stock_df, period=14)
    atr_20 = calculate_atr(sample_stock_df, period=20)
    
    # Different periods should produce different values
    assert not (atr_14.iloc[25:] == atr_20.iloc[25:]).all()


# =============================================================================
# TESTS: get_atr_trailing_stop()
# =============================================================================

def test_get_atr_trailing_stop_basic(sample_stock_data_dict):
    """Test basic trailing stop calculation."""
    date = pd.Timestamp("2023-03-01")
    highest_price = 120.0
    
    stop_price = get_atr_trailing_stop(
        sample_stock_data_dict, 
        "AAPL", 
        date, 
        highest_price,
        multiplier=2.5
    )
    
    # Stop should be below highest price
    assert stop_price < highest_price
    # Stop should be at least 3% below (minimum floor)
    assert stop_price <= highest_price * 0.97


def test_get_atr_trailing_stop_missing_ticker(sample_stock_data_dict):
    """Test fallback when ticker is not in data."""
    date = pd.Timestamp("2023-03-01")
    highest_price = 100.0
    
    stop_price = get_atr_trailing_stop(
        sample_stock_data_dict, 
        "UNKNOWN", 
        date, 
        highest_price
    )
    
    # Should use 12% fallback
    assert np.isclose(stop_price, highest_price * 0.88)


def test_get_atr_trailing_stop_insufficient_data(sample_stock_df):
    """Test fallback when insufficient historical data."""
    # Create minimal data (not enough for ATR calculation)
    stock_data = {"AAPL": sample_stock_df.head(5)}
    date = pd.Timestamp("2023-01-05")
    highest_price = 100.0
    
    stop_price = get_atr_trailing_stop(
        stock_data, 
        "AAPL", 
        date, 
        highest_price
    )
    
    # Should use 12% fallback
    assert np.isclose(stop_price, highest_price * 0.88)


# =============================================================================
# TESTS: get_drawdown_exposure()
# =============================================================================

def test_get_drawdown_exposure_full_exposure_no_drawdown():
    """Test full exposure when no drawdown."""
    # Portfolio at all-time high
    portfolio_values = [100, 105, 110, 115, 120]
    
    exposure = get_drawdown_exposure(portfolio_values)
    
    assert exposure == 1.0


def test_get_drawdown_exposure_reduced_on_drawdown():
    """Test exposure reduction during drawdown."""
    # Portfolio with 20% drawdown
    portfolio_values = [100, 120, 110, 100, 96]  # 20% from peak of 120
    
    exposure = get_drawdown_exposure(portfolio_values)
    
    # Should be reduced (20% DD = 60% exposure with default thresholds)
    assert exposure < 1.0
    assert exposure == 0.6  # 20-25% DD bucket


def test_get_drawdown_exposure_defensive_mode():
    """Test defensive mode on severe drawdown."""
    # Portfolio with 35% drawdown
    portfolio_values = [100, 120, 90, 80, 78]  # 35% from peak of 120
    
    exposure = get_drawdown_exposure(portfolio_values)
    
    # Severe drawdown should trigger defensive mode (20%)
    assert exposure <= 0.2


def test_get_drawdown_exposure_minimal_history():
    """Test behavior with minimal portfolio history."""
    portfolio_values = [100]
    
    exposure = get_drawdown_exposure(portfolio_values)
    
    # Should return full exposure with no history
    assert exposure == 1.0


def test_get_drawdown_exposure_custom_thresholds():
    """Test with custom threshold configuration."""
    portfolio_values = [100, 110, 99]  # ~10% drawdown
    
    custom_thresholds = [
        (-0.05, 1.0),   # Under 5%: full
        (-0.10, 0.5),   # 5-10%: half
        (-1.0, 0.0),    # >10%: none
    ]
    
    exposure = get_drawdown_exposure(portfolio_values, thresholds=custom_thresholds)
    
    # 10% drawdown should give 50% exposure with custom thresholds
    assert exposure == 0.5


# =============================================================================
# TESTS: get_regime_exposure()
# =============================================================================

def test_get_regime_exposure_crisis_mode(sample_spy_df):
    """Test crisis mode when VIX > 30."""
    # Create VIX data with high values
    vix_df = pd.DataFrame({
        "date": sample_spy_df["date"],
        "close": [35] * len(sample_spy_df),  # VIX at 35 (crisis)
    })
    
    date = sample_spy_df["date"].iloc[-1]
    exposure, regime = get_regime_exposure(sample_spy_df, vix_df, date)
    
    assert exposure == 0.0
    assert regime == "crisis"


def test_get_regime_exposure_bull_calm(sample_spy_df):
    """Test bull market with low volatility."""
    # Create VIX data with low values
    vix_df = pd.DataFrame({
        "date": sample_spy_df["date"],
        "close": [15] * len(sample_spy_df),  # VIX at 15 (calm)
    })
    
    # Ensure SPY is above 200 SMA (bull market)
    spy_df = sample_spy_df.copy()
    spy_df["close"] = spy_df["close"].values + 50  # Push above SMA
    
    date = spy_df["date"].iloc[-1]
    exposure, regime = get_regime_exposure(spy_df, vix_df, date)
    
    # Should be bull_calm with full exposure
    assert exposure == 1.0
    assert regime == "bull_calm"


def test_get_regime_exposure_insufficient_data():
    """Test behavior with insufficient SPY history."""
    # Only 50 days of data (less than 200 SMA)
    dates = pd.date_range(start="2023-01-01", periods=50, freq="B")
    spy_df = pd.DataFrame({
        "date": dates,
        "close": [400] * 50,
    })
    
    date = dates[-1]
    exposure, regime = get_regime_exposure(spy_df, None, date)
    
    assert regime == "insufficient_data"


def test_get_regime_exposure_no_vix_data(sample_spy_df):
    """Test behavior when VIX data is None."""
    date = sample_spy_df["date"].iloc[-1]
    exposure, regime = get_regime_exposure(sample_spy_df, None, date)
    
    # Should use default VIX of 20 (assumes normal volatility)
    assert exposure in [0.3, 0.8, 1.0]  # Possible values based on SPY trend


# =============================================================================
# TESTS: PositionTracker
# =============================================================================

def test_position_tracker_add_and_get():
    """Test adding and retrieving positions."""
    tracker = PositionTracker()
    
    tracker.add_position("AAPL", 150.0, pd.Timestamp("2023-01-01"))
    
    assert tracker.get_entry_price("AAPL") == 150.0
    assert tracker.get_highest("AAPL") == 150.0


def test_position_tracker_update_highest():
    """Test updating highest price."""
    tracker = PositionTracker()
    
    tracker.add_position("AAPL", 150.0, pd.Timestamp("2023-01-01"))
    tracker.update_highest("AAPL", 160.0)
    tracker.update_highest("AAPL", 155.0)  # Lower, shouldn't update
    
    assert tracker.get_highest("AAPL") == 160.0


def test_position_tracker_remove():
    """Test removing positions."""
    tracker = PositionTracker()
    
    tracker.add_position("AAPL", 150.0, pd.Timestamp("2023-01-01"))
    tracker.remove_position("AAPL")
    
    assert tracker.get_entry_price("AAPL") is None
    assert tracker.get_highest("AAPL") is None


def test_position_tracker_unknown_ticker():
    """Test behavior with unknown tickers."""
    tracker = PositionTracker()
    
    assert tracker.get_entry_price("UNKNOWN") is None
    assert tracker.get_highest("UNKNOWN") is None


# =============================================================================
# TESTS: select_with_correlation_filter()
# =============================================================================

def test_select_with_correlation_filter_basic(sample_stock_data_dict):
    """Test basic correlation-filtered selection."""
    rankings = {"AAPL": 0.9, "MSFT": 0.8, "GOOGL": 0.7, "NVDA": 0.6, "TSLA": 0.5}
    date = pd.Timestamp("2023-03-01")
    
    selected = select_with_correlation_filter(
        sample_stock_data_dict,
        rankings,
        date,
        top_k=3,
        max_corr=0.70
    )
    
    # Should return list of tickers
    assert isinstance(selected, list)
    assert len(selected) <= 3


def test_select_with_correlation_filter_returns_top_ranked_first(sample_stock_data_dict):
    """Test that highest-ranked stock is always included."""
    rankings = {"AAPL": 0.95, "MSFT": 0.5, "GOOGL": 0.4, "NVDA": 0.3, "TSLA": 0.2}
    date = pd.Timestamp("2023-03-01")
    
    selected = select_with_correlation_filter(
        sample_stock_data_dict,
        rankings,
        date,
        top_k=3
    )
    
    # AAPL (highest ranked) should be first
    assert selected[0] == "AAPL"


def test_select_with_correlation_filter_empty_data():
    """Test behavior with empty stock data."""
    rankings = {"AAPL": 0.9, "MSFT": 0.8}
    date = pd.Timestamp("2023-03-01")
    
    selected = select_with_correlation_filter(
        {},  # Empty stock data
        rankings,
        date,
        top_k=2
    )
    
    # Should fall back to top-K without filtering
    assert len(selected) == 2


# =============================================================================
# TESTS: calculate_volatility_weights()
# =============================================================================

def test_calculate_volatility_weights_sums_to_one(sample_stock_data_dict):
    """Test that weights sum to 1.0."""
    holdings = ["AAPL", "MSFT", "GOOGL"]
    date = pd.Timestamp("2023-03-01")
    
    weights = calculate_volatility_weights(sample_stock_data_dict, holdings, date)
    
    total = sum(weights.values())
    assert np.isclose(total, 1.0)


def test_calculate_volatility_weights_all_positive(sample_stock_data_dict):
    """Test that all weights are positive."""
    holdings = ["AAPL", "MSFT", "GOOGL"]
    date = pd.Timestamp("2023-03-01")
    
    weights = calculate_volatility_weights(sample_stock_data_dict, holdings, date)
    
    assert all(w > 0 for w in weights.values())


def test_calculate_volatility_weights_missing_ticker(sample_stock_data_dict):
    """Test behavior with missing ticker."""
    holdings = ["AAPL", "MISSING"]
    date = pd.Timestamp("2023-03-01")
    
    weights = calculate_volatility_weights(sample_stock_data_dict, holdings, date)
    
    # Should still sum to 1 with default weight for missing
    assert np.isclose(sum(weights.values()), 1.0)


# =============================================================================
# TESTS: calculate_hybrid_weights()
# =============================================================================

def test_calculate_hybrid_weights_sums_to_one(sample_stock_data_dict):
    """Test that hybrid weights sum to 1.0."""
    rankings = {"AAPL": 0.9, "MSFT": 0.7, "GOOGL": 0.5}
    holdings = ["AAPL", "MSFT", "GOOGL"]
    date = pd.Timestamp("2023-03-01")
    
    weights = calculate_hybrid_weights(
        sample_stock_data_dict,
        rankings,
        holdings,
        date
    )
    
    assert np.isclose(sum(weights.values()), 1.0)


def test_calculate_hybrid_weights_respects_blend(sample_stock_data_dict):
    """Test that blend parameters affect output."""
    rankings = {"AAPL": 0.9, "MSFT": 0.1}
    holdings = ["AAPL", "MSFT"]
    date = pd.Timestamp("2023-03-01")
    
    # 100% volatility weighting
    vol_weights = calculate_hybrid_weights(
        sample_stock_data_dict, rankings, holdings, date,
        vol_weight=1.0, rank_weight=0.0
    )
    
    # 100% rank weighting
    rank_weights = calculate_hybrid_weights(
        sample_stock_data_dict, rankings, holdings, date,
        vol_weight=0.0, rank_weight=1.0
    )
    
    # Weights should be different
    assert vol_weights["AAPL"] != rank_weights["AAPL"]


# =============================================================================
# TESTS: calculate_final_exposure()
# =============================================================================

def test_calculate_final_exposure_combines_signals(sample_spy_df, sample_vix_df):
    """Test that final exposure combines multiple risk signals."""
    # Portfolio with some drawdown
    portfolio_values = [100, 110, 105]  # Minor drawdown
    date = sample_spy_df["date"].iloc[-1]
    
    exposure, details = calculate_final_exposure(
        portfolio_values,
        sample_spy_df,
        sample_vix_df,
        date
    )
    
    # Should return exposure between 0 and 1
    assert 0.0 <= exposure <= 1.0
    
    # Details should contain component exposures
    assert "drawdown_exposure" in details
    assert "regime_exposure" in details
    assert "final_exposure" in details


def test_calculate_final_exposure_disabled_signals(sample_spy_df, sample_vix_df):
    """Test behavior with signals disabled."""
    portfolio_values = [100, 80]  # 20% drawdown
    date = sample_spy_df["date"].iloc[-1]
    
    # Disable all risk signals
    exposure, details = calculate_final_exposure(
        portfolio_values,
        sample_spy_df,
        sample_vix_df,
        date,
        enable_drawdown=False,
        enable_regime=False
    )
    
    # With everything disabled, should be full exposure
    assert exposure == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
