import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging
import os
import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


# Common ticker typos and corrections
TICKER_CORRECTIONS = {
    "AMAZ": "AMZN",
    "GOOG": "GOOGL",  # Both valid, but GOOGL is more common
    "APPL": "AAPL",
    "TSLA": "TSLA",  # Correct
    "NVDA": "NVDA",  # Correct
}


def validate_ticker(ticker: str) -> tuple[str, Optional[str]]:
    """
    Validate and correct common ticker typos.

    Returns:
        (corrected_ticker, warning_message)
    """
    ticker = ticker.upper().strip()

    if ticker in TICKER_CORRECTIONS:
        correct = TICKER_CORRECTIONS[ticker]
        if correct != ticker:
            return correct, f"⚠️  Did you mean '{correct}'? (You typed '{ticker}')"

    # Check for suspicious patterns
    if len(ticker) > 5:
        return ticker, f"⚠️  Ticker '{ticker}' looks unusually long. Verify it's correct."

    if not ticker.isalpha():
        return ticker, f"⚠️  Ticker '{ticker}' contains non-alphabetic characters. Verify it's correct."

    return ticker, None


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

    def __init__(self):
        # Store last calculated values for breakdown display
        self.last_eps: Optional[float] = None
        self.last_growth_rate: Optional[float] = None
        self.last_bond_yield: Optional[float] = None
        self.is_estimated: bool = False  # Flag for fallback values

    def calculate_graham(self, eps_ttm: float, growth_rate: float, bond_yield: float = None) -> float:
        """
        Calculates intrinsic value using the strict legacy Graham formula.

        Formula: eps_ttm * (8.5 + 2 * (growth_rate * 100)) * ((bond_yield * 100) / 4.4)

        IMPORTANT: Both growth_rate and bond_yield are converted from decimal to percentage.
        - growth_rate: 0.10 → 10%
        - bond_yield: 0.044 → 4.4%

        This is the EXACT formula from legacy/Intrinsic-Value-Monitor/1-produce_data.ipynb:
            g_for_formula = g * 100
            bond_yield_pct = bond_yield * 100
            merged["graham_intrinsic_value"] = (
                merged["eps_ttm"] *
                (8.5 + 2 * g_for_formula) *
                (bond_yield_pct / 4.4)
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
        # Both growth_rate and bond_yield need to be converted to percentage form
        g_for_formula = growth_rate * 100
        bond_yield_pct = bond_yield * 100  # Convert decimal (0.0409) to percentage (4.09)

        intrinsic_value = (
            eps_ttm *
            (8.5 + 2 * g_for_formula) *
            (4.4 / bond_yield_pct)
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

    def fetch_eps_from_alpha_vantage(self, ticker: str, max_retries: int = 3) -> Optional[pd.Series]:
        """
        Fetch quarterly EPS data from Alpha Vantage API with retry logic.

        Args:
            ticker: Stock symbol
            max_retries: Maximum number of retry attempts

        Returns:
            Pandas Series of quarterly EPS indexed by date, or None if failed
        """
        # Validate ticker first
        corrected_ticker, warning = validate_ticker(ticker)
        if warning:
            logger.warning(warning)
            ticker = corrected_ticker

        # Use settings object which checks both secrets.json AND .env
        from utils.config_loader import settings
        api_key = settings.ALPHA_VANTAGE_KEY
        if not api_key or api_key == "your_alpha_vantage_key_here":
            logger.error("ALPHA_VANTAGE_KEY not set or still using placeholder value")
            return None

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "EARNINGS",
            "symbol": ticker,
            "apikey": api_key
        }

        # Retry with exponential backoff
        for attempt in range(max_retries):
            try:
                # Increase timeout and add retry logic
                session = requests.Session()
                retry_strategy = Retry(
                    total=2,
                    backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504],
                )
                adapter = HTTPAdapter(max_retries=retry_strategy)
                session.mount("https://", adapter)

                logger.info(f"Fetching EPS for {ticker} (attempt {attempt + 1}/{max_retries})")
                response = session.get(url, params=params, timeout=20)
                response.raise_for_status()
                data = response.json()

                # Check for API error messages
                if "Error Message" in data:
                    logger.error(f"Alpha Vantage error for {ticker}: {data['Error Message']}")
                    return None

                if "Note" in data:
                    logger.warning(f"Alpha Vantage rate limit for {ticker}: {data['Note']}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.info(f"Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    return None

                if "quarterlyEarnings" not in data:
                    logger.warning(f"No quarterly earnings data for {ticker}. Response keys: {list(data.keys())}")
                    return None

                # Parse quarterly earnings
                earnings = []
                for quarter in data["quarterlyEarnings"]:
                    try:
                        date = pd.to_datetime(quarter["fiscalDateEnding"])
                        eps = float(quarter["reportedEPS"])
                        earnings.append({"date": date, "eps": eps})
                    except (KeyError, ValueError) as e:
                        logger.debug(f"Skipping quarter data for {ticker}: {e}")
                        continue

                if not earnings:
                    logger.warning(f"No valid EPS entries parsed for {ticker}")
                    return None

                # Create Series indexed by date
                df = pd.DataFrame(earnings)
                df = df.sort_values("date")
                eps_series = pd.Series(df["eps"].values, index=df["date"])

                logger.info(f"✅ Fetched {len(eps_series)} quarters of EPS data for {ticker}")
                return eps_series

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout fetching EPS for {ticker} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                logger.error(f"All retry attempts exhausted for {ticker}")
                return None

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for {ticker}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None

            except Exception as e:
                logger.error(f"Unexpected error fetching EPS for {ticker}: {e}", exc_info=True)
                return None

        return None

    def fetch_current_bond_yield(self) -> float:
        """
        Fetch current AAA corporate bond yield from Treasury API.
        Falls back to default if fetch fails.

        Returns:
            Current bond yield as decimal (e.g., 0.044 for 4.4%)
        """
        try:
            # Try Federal Reserve Economic Data (FRED) for AAA Corporate Bond Yield
            # Note: This is a simplified example. You may need an API key for FRED.
            # Alternative: Use Alpha Vantage TREASURY_YIELD endpoint

            # Use settings object which checks both secrets.json AND .env
            from utils.config_loader import settings
            api_key = settings.ALPHA_VANTAGE_KEY
            if not api_key:
                logger.warning("Cannot fetch bond yield without API key, using default")
                return self.DEFAULT_BOND_YIELD

            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TREASURY_YIELD",
                "interval": "monthly",
                "maturity": "10year",  # 10-year treasury as proxy
                "apikey": api_key
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "data" in data and len(data["data"]) > 0:
                # Get most recent yield
                latest_yield = float(data["data"][0]["value"]) / 100  # Convert percentage to decimal
                logger.info(f"Fetched current bond yield: {latest_yield:.4f}")
                return latest_yield
            else:
                logger.warning("No bond yield data available, using default")
                return self.DEFAULT_BOND_YIELD

        except Exception as e:
            logger.warning(f"Failed to fetch bond yield: {e}, using default {self.DEFAULT_BOND_YIELD}")
            return self.DEFAULT_BOND_YIELD

    async def calculate(self, ticker: str, use_live_bond_yield: bool = True) -> Optional[float]:
        """
        Calculate intrinsic value with caching and fallback logic.
        
        1. Check cache (valid for 30 days)
        2. If cache miss/expired, try API
        3. If API fails, use expired cache
        4. If no cache, use fallback (Price / 20)
        """
        from services.db import get_cached_intrinsic_value, save_cached_intrinsic_value, get_latest_close
        from datetime import datetime, timedelta

        try:
            # 1. Check Cache
            cached = await get_cached_intrinsic_value(ticker)
            if cached:
                # Check if expired (30 days)
                if cached.last_updated > datetime.now() - timedelta(days=30):
                    logger.info(f"✅ Using cached intrinsic value for {ticker}: ${cached.value:.2f}")
                    # Populate last values for breakdown
                    self.last_eps = cached.eps
                    self.last_growth_rate = cached.growth_rate
                    self.last_bond_yield = cached.bond_yield
                    self.is_estimated = False
                    return cached.value
                else:
                    logger.info(f"⚠️ Cached value for {ticker} expired ({cached.last_updated}) - attempting refresh")

            # 2. Try API Calculation
            try:
                # Validate ticker
                corrected_ticker, warning = validate_ticker(ticker)
                if warning:
                    logger.warning(f"Ticker validation: {warning}")
                    ticker = corrected_ticker

                logger.info(f"🔍 Calculating intrinsic value for {ticker}...")

                # Fetch EPS data
                eps_series = self.fetch_eps_from_alpha_vantage(ticker)
                if eps_series is None:
                    raise ValueError("Failed to fetch EPS data")

                if len(eps_series) < 4:
                    raise ValueError(f"Insufficient EPS data: {len(eps_series)} quarters")

                # Calculate EPS TTM
                eps_ttm = self.calculate_eps_ttm(eps_series)
                if eps_ttm <= 0:
                    raise ValueError(f"Invalid EPS TTM: ${eps_ttm:.4f}")

                # Estimate growth rate
                growth_rate = self.estimate_growth_rate(eps_series)
                if np.isnan(growth_rate):
                    growth_rate = 0.0

                # Get bond yield
                if use_live_bond_yield:
                    bond_yield = self.fetch_current_bond_yield()
                else:
                    bond_yield = self.DEFAULT_BOND_YIELD

                # Calculate intrinsic value
                intrinsic_value = self.calculate_graham(eps_ttm, growth_rate, bond_yield)
                
                # Sanity check
                if intrinsic_value < 1.0:
                    raise ValueError(f"Suspiciously low intrinsic value: ${intrinsic_value:.2f}")

                # Save to cache
                await save_cached_intrinsic_value(
                    ticker=ticker,
                    value=intrinsic_value,
                    eps=eps_ttm,
                    growth_rate=growth_rate,
                    bond_yield=bond_yield * 100
                )
                
                # Store values for breakdown display
                self.last_eps = eps_ttm
                self.last_growth_rate = growth_rate
                self.last_bond_yield = bond_yield * 100
                self.is_estimated = False

                logger.info(f"✅ Intrinsic value for {ticker}: ${intrinsic_value:.2f} (Saved to cache)")
                return intrinsic_value

            except Exception as e:
                logger.warning(f"⚠️ API calculation failed for {ticker}: {e}")
                
                # 3. Fallback to expired cache if available
                if cached:
                    logger.info(f"⚠️ Using expired cached value for {ticker}: ${cached.value:.2f}")
                    self.last_eps = cached.eps
                    self.last_growth_rate = cached.growth_rate
                    self.last_bond_yield = cached.bond_yield
                    self.is_estimated = True  # Expired cache is considered estimated/stale
                    return cached.value
                
                # 4. Hard Fallback: Price / 20
                logger.warning(f"⚠️ No cache available. Using hard fallback (Price/20) for {ticker}")
                current_price = await get_latest_close(ticker)
                if current_price and current_price > 0:
                    fallback_value = current_price / 20.0  # As requested by user
                    
                    self.last_eps = 0.0
                    self.last_growth_rate = 0.0
                    self.last_bond_yield = 0.0
                    self.is_estimated = True
                    return fallback_value
                
                return None

        except Exception as e:
            logger.error(f"❌ Critical error in intrinsic calculation for {ticker}: {e}", exc_info=True)
            return None
