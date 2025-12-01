import pytest
import pandas as pd
import numpy as np
from services.intrinsic import IntrinsicCalculator

# Mock Data
MOCK_PRICES_CSV = "../legacy/Intrinsic-Value-Monitor/HistoricalPrices.csv"

@pytest.fixture
def mock_price_data():
    # Load a snippet of real data for realism
    # Date, Open, High, Low, Close
    # We just need a Series of closes
    try:
        df = pd.read_csv(MOCK_PRICES_CSV)
        # Clean column names just in case
        df.columns = [c.strip() for c in df.columns]
        return df['Close'].iloc[:100][::-1] # Reverse to be chronological
    except FileNotFoundError:
        # Fallback if file not found (e.g. CI env)
        return pd.Series(np.linspace(100, 150, 100))

@pytest.fixture
def mock_eps_data():
    # Create a synthetic EPS series with steady growth
    # Start at 1.0, grow at 5% per quarter
    quarters = 20
    growth = 0.05
    eps = [1.0 * (1 + growth)**i for i in range(quarters)]
    dates = pd.date_range(start='2020-01-01', periods=quarters, freq='QE')
    return pd.Series(eps, index=dates)

def test_growth_rate_estimation(mock_eps_data):
    valuator = IntrinsicCalculator()
    growth = valuator.estimate_growth_rate(mock_eps_data)
    
    # We generated with 5% quarterly growth.
    # Annual growth should be (1.05)^4 - 1 ~= 21.55%
    expected_annual = (1.05)**4 - 1
    
    assert growth == pytest.approx(expected_annual, rel=0.1)

def test_graham_calculation():
    valuator = IntrinsicCalculator()
    
    eps = 5.0
    g = 0.10 # 10%
    y = 0.044 # 4.4% -> Factor should be 1
    
    # V = 5 * (8.5 + 20) * (4.4/4.4) = 5 * 28.5 * 1 = 142.5
    value = valuator.calculate_graham_value(eps, g, y)
    
    assert value == pytest.approx(142.5, rel=0.01)

def test_full_evaluation(mock_price_data, mock_eps_data):
    valuator = IntrinsicCalculator()
    bond_yield = 0.044 # 4.4%
    
    result = valuator.evaluate(
        ticker="TEST",
        price_series=mock_price_data,
        eps_quarterly=mock_eps_data,
        current_bond_yield=bond_yield
    )
    
    assert result["ticker"] == "TEST"
    assert "intrinsic_value" in result
    assert result["intrinsic_value"] > 0
    assert result["eps_ttm"] > 0
    assert result["growth_rate"] > 0
