import numpy as np
import pandas as pd
from typing import Dict, Any


class IntrinsicCalculator:
    """
    Calculates intrinsic value using methods ported from legacy/Intrinsic-Value-Monitor/1-produce_data.ipynb

    Supported methods:
    1. Graham Number (Benjamin Graham formula)
    2. Growth rate estimation from historical EPS

    The legacy formula:
        graham_intrinsic_value = eps_ttm * (8.5 + 2 * g * 100) * (bond_yield / 4.4)

    Where:
        - eps_ttm: Trailing twelve months earnings per share (sum of last 4 quarters)
        - g: Annual growth rate as decimal (e.g., 0.10 for 10%)
        - bond_yield: Current AAA corporate bond yield as decimal (e.g., 0.044 for 4.4%)

    Buy signal: Market Price < 0.5 * Intrinsic Value
    """

    # Default AAA corporate bond yield (legacy default was 4.4%)
    DEFAULT_BOND_YIELD = 0.044

    def calculate_graham(self, eps_ttm: float, growth_rate: float, bond_yield: float = None) -> float:
        """
        Calculates intrinsic value using the strict legacy Graham formula.

        Formula: eps_ttm * (8.5 + 2 * (growth_rate * 100)) * (bond_yield / 4.4)

        This is the EXACT formula from legacy/Intrinsic-Value-Monitor/1-produce_data.ipynb:
            g_for_formula = g * 100
            merged["graham_intrinsic_value"] = (
                merged["eps_ttm"] *
                (8.5 + 2 * g_for_formula) *
                (merged["bond_yield"] / 4.4)
            )

        Args:
            eps_ttm: Trailing twelve months EPS (sum of last 4 quarterly EPS)
            growth_rate: Annual growth rate as decimal (0.10 = 10%)
            bond_yield: AAA corporate bond yield as decimal (default 0.044 = 4.4%)

        Returns:
            Intrinsic value per share, or 0.0 if inputs invalid
        """
        if eps_ttm is None or growth_rate is None:
            return 0.0

        if bond_yield is None:
            bond_yield = self.DEFAULT_BOND_YIELD

        # Guard against invalid inputs
        if eps_ttm <= 0 or bond_yield <= 0:
            return 0.0

        # Legacy formula exactly as written
        g_for_formula = growth_rate * 100

        intrinsic_value = (
            eps_ttm *
            (8.5 + 2 * g_for_formula) *
            (bond_yield / 4.4)
        )

        return max(0.0, intrinsic_value)

    def calculate_graham_value(self, eps_ttm: float, growth_rate: float, bond_yield: float = None) -> float:
        """Alias for calculate_graham for backward compatibility with tests."""
        return self.calculate_graham(eps_ttm, growth_rate, bond_yield)

    def estimate_growth_rate(self, eps_quarterly: pd.Series, lookback_quarters: int = 30) -> float:
        """
        Estimate annual growth rate from historical quarterly EPS.

        This is the EXACT method from legacy/Intrinsic-Value-Monitor/1-produce_data.ipynb:
            def estimate_growth_rate(eps_quarter_series, lookback_quarters=30):
                eps = eps_quarter_series.dropna()
                eps = eps[eps > 0]
                if len(eps) < 6:
                    return np.nan
                eps_recent = eps.tail(lookback_quarters)
                y = np.log(eps_recent.values)
                x = np.arange(len(eps_recent))
                slope, intercept = np.polyfit(x, y, 1)
                quarterly_growth = np.exp(slope) - 1
                annual_growth = (1 + quarterly_growth)**4 - 1
                return annual_growth

        Args:
            eps_quarterly: Pandas Series of quarterly EPS values (index should be dates)
            lookback_quarters: Number of recent quarters to consider

        Returns:
            Estimated annual growth rate as decimal, or np.nan if insufficient data
        """
        # Clean data - remove NaN and non-positive values
        eps = eps_quarterly.dropna()
        eps = eps[eps > 0]

        # Need at least 6 quarters for meaningful regression
        if len(eps) < 6:
            return np.nan

        # Use most recent quarters
        eps_recent = eps.tail(lookback_quarters)

        # Log-linear regression to estimate exponential growth
        y = np.log(eps_recent.values)
        x = np.arange(len(eps_recent))

        try:
            slope, _ = np.polyfit(x, y, 1)
        except (np.linalg.LinAlgError, ValueError):
            return np.nan

        # Convert quarterly growth to annual
        quarterly_growth = np.exp(slope) - 1
        annual_growth = (1 + quarterly_growth) ** 4 - 1

        return annual_growth

    def calculate_eps_ttm(self, eps_quarterly: pd.Series) -> float:
        """
        Calculate trailing twelve months EPS from quarterly data.

        Args:
            eps_quarterly: Pandas Series of quarterly EPS values

        Returns:
            Sum of last 4 quarters EPS, or 0.0 if insufficient data
        """
        eps = eps_quarterly.dropna()

        if len(eps) < 4:
            return 0.0

        # Sum of last 4 quarters
        return eps.tail(4).sum()

    def evaluate(
        self,
        ticker: str,
        price_series: pd.Series,
        eps_quarterly: pd.Series,
        current_bond_yield: float = None
    ) -> Dict[str, Any]:
        """
        Full evaluation of a stock's intrinsic value.

        Args:
            ticker: Stock symbol
            price_series: Historical closing prices
            eps_quarterly: Quarterly EPS data
            current_bond_yield: Current AAA bond yield (default 0.044)

        Returns:
            Dict containing:
                - ticker: Stock symbol
                - current_price: Latest price
                - eps_ttm: Trailing twelve months EPS
                - growth_rate: Estimated annual growth rate
                - intrinsic_value: Graham intrinsic value
                - margin_of_safety: (intrinsic - price) / intrinsic
                - signal: "BUY" if price < 0.5 * intrinsic, else "HOLD"
        """
        if current_bond_yield is None:
            current_bond_yield = self.DEFAULT_BOND_YIELD

        # Get current price
        current_price = float(price_series.iloc[-1]) if len(price_series) > 0 else 0.0

        # Calculate EPS TTM
        eps_ttm = self.calculate_eps_ttm(eps_quarterly)

        # Estimate growth rate
        growth_rate = self.estimate_growth_rate(eps_quarterly)
        if np.isnan(growth_rate):
            growth_rate = 0.0  # Default to 0 growth if can't estimate

        # Calculate intrinsic value
        intrinsic_value = self.calculate_graham(eps_ttm, growth_rate, current_bond_yield)

        # Margin of safety
        if intrinsic_value > 0:
            margin_of_safety = (intrinsic_value - current_price) / intrinsic_value
        else:
            margin_of_safety = 0.0

        # Signal: BUY if price < 50% of intrinsic value (per legacy logic)
        signal = "HOLD"
        if intrinsic_value > 0 and current_price < 0.5 * intrinsic_value:
            signal = "BUY"

        return {
            "ticker": ticker,
            "current_price": current_price,
            "eps_ttm": eps_ttm,
            "growth_rate": growth_rate,
            "intrinsic_value": intrinsic_value,
            "margin_of_safety": margin_of_safety,
            "signal": signal
        }

    def calculate(self, ticker: str) -> float:  # noqa: ARG002
        """
        Placeholder for full calculation fetching data from DB.
        This will be implemented when we have EPS data in the database.

        For now, returns 0.0 to indicate no intrinsic value available.

        Args:
            ticker: Stock symbol (will be used for DB lookup when implemented)
        """
        # TODO: Fetch EPS data from database and calculate
        return 0.0
