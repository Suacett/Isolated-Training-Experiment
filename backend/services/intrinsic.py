class IntrinsicCalculator:
    def calculate_graham(self, eps_ttm: float, growth_rate: float, bond_yield: float) -> float:
        """
        Calculates intrinsic value using the strict legacy Graham formula found in 1-produce_data.ipynb.
        
        Formula: eps_ttm * (8.5 + 2 * (growth_rate * 100)) * (bond_yield / 4.4)
        
        Note: The legacy code uses `g * 100` for the growth multiplier.
        Also, it multiplies by (bond_yield / 4.4), which is non-standard (usually 4.4 / bond_yield).
        We strictly follow the legacy code here.
        """
        if eps_ttm is None or growth_rate is None or bond_yield is None:
            return 0.0
            
        g_for_formula = growth_rate * 100
        
        intrinsic_value = (
            eps_ttm * 
            (8.5 + 2 * g_for_formula) * 
            (bond_yield / 4.4)
        )
        
        return intrinsic_value

    def calculate(self, ticker: str) -> float:
        # Placeholder for full calculation fetching data from DB
        # For now, we return 0.0 or mock values as per current system state
        return 0.0
