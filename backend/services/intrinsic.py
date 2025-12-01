import numpy as np
import pandas as pd
from typing import Optional, Tuple

class IntrinsicCalculator:
    """
    Calculates the Intrinsic Value of a stock using the Benjamin Graham formula.
    
    Formula: V = EPS * (8.5 + 2g) * (4.4 / Y)
    
    Where:
    - EPS: Trailing Twelve Month Earnings Per Share
    - g: Expected Annual Growth Rate (%)
    - Y: Current Yield on AAA Corporate Bonds (%)
    - 4.4: The average yield of AAA Corporate Bonds when Graham wrote the formula
    """

    @staticmethod
    def estimate_growth_rate(eps_quarterly: pd.Series, lookback_quarters: int = 30) -> float:
        """
        Estimates annual growth rate using log-linear regression on quarterly EPS.
        """
        eps = eps_quarterly.dropna()
        eps = eps[eps > 0] # Log requires positive values
        
        if len(eps) < 6:
            return np.nan

        eps_recent = eps.tail(lookback_quarters)
        y = np.log(eps_recent.values)
        x = np.arange(len(eps_recent))

        # Linear regression: y = mx + c
        slope, _ = np.polyfit(x, y, 1)

        # Convert slope to quarterly growth rate, then annualized
        quarterly_growth = np.exp(slope) - 1
        annual_growth = (1 + quarterly_growth)**4 - 1
        
        return annual_growth

    @staticmethod
    def calculate_graham_value(
        eps_ttm: float,
        growth_rate: float,
        bond_yield: float
    ) -> float:
        """
        Applies the Graham Formula.
        
        Args:
            eps_ttm: Trailing 12-month EPS
            growth_rate: Annual growth rate (decimal, e.g., 0.05 for 5%)
            bond_yield: Current AAA Bond Yield (decimal, e.g., 0.04 for 4%)
        """
        if bond_yield <= 0:
            return 0.0
            
        g_percent = growth_rate * 100
        yield_percent = bond_yield * 100
        
        # Note: Legacy code had (yield / 4.4), which is incorrect.
        # Correct Graham formula uses (4.4 / yield).
        intrinsic_value = eps_ttm * (8.5 + 2 * g_percent) * (4.4 / yield_percent)
        
        return max(0.0, intrinsic_value)

    def evaluate(
        self,
        ticker: str,
        price_series: pd.Series,
        eps_quarterly: pd.Series,
        current_bond_yield: float
    ) -> dict:
        """
        Full evaluation pipeline for a single ticker.
        """
        # 1. Calculate TTM EPS
        # Assuming eps_quarterly is sorted by date
        if len(eps_quarterly) < 4:
            return {"error": "Insufficient EPS data"}
            
        eps_ttm = eps_quarterly.tail(4).sum()
        
        # 2. Estimate Growth
        growth_rate = self.estimate_growth_rate(eps_quarterly)
        
        if np.isnan(growth_rate):
            return {"error": "Could not estimate growth rate"}
            
        # 3. Calculate Intrinsic Value
        intrinsic_value = self.calculate_graham_value(eps_ttm, growth_rate, current_bond_yield)
        
        current_price = price_series.iloc[-1]
        
        return {
            "ticker": ticker,
            "current_price": current_price,
            "intrinsic_value": intrinsic_value,
            "eps_ttm": eps_ttm,
            "growth_rate": growth_rate,
            "bond_yield": current_bond_yield,
            "margin_of_safety": (intrinsic_value - current_price) / intrinsic_value if intrinsic_value > 0 else 0
        }
