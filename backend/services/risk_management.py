#!/usr/bin/env python3
"""
Risk Management Module for V9 Momentum Strategy

Provides adaptive risk management tools that respect the momentum thesis:
- ATR-based trailing stops (volatility-adjusted)
- Drawdown-based exposure scaling
- Market regime detection (SPY trend + VIX)
"""

import os
import pandas as pd
from typing import List, Dict, Tuple, Optional
from config.constants import ATR_PARAMS, DRAWDOWN_THRESHOLDS, REGIME_PARAMS, VOLATILITY_PARAMS

# Constants
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.045"))  # Default to ~4.5% (current TBills)


# =============================================================================
# ATR (Average True Range) CALCULATIONS
# =============================================================================

def calculate_atr(df: pd.DataFrame, period: int = None) -> pd.Series:
    """
    Calculate Average True Range (ATR) for volatility measurement.
    """
    # 1. Input Validation
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input 'df' must be a non-empty pandas DataFrame.")
    
    required_cols = ['high', 'low', 'close']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for ATR: {missing}")

    if period is None:
        period = ATR_PARAMS.get("PERIOD", 14)
    
    if not isinstance(period, int) or period <= 0:
        raise ValueError(f"Invalid ATR period: {period}. Must be an integer > 0.")
        
    if len(df) < period:
        raise ValueError(f"DataFrame length ({len(df)}) is less than ATR period ({period}).")

    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    return atr


def get_atr_trailing_stop(
    stock_data: Dict[str, pd.DataFrame],
    ticker: str,
    date: pd.Timestamp,
    highest_price: float,
    multiplier: float = None,
    atr_period: int = None
) -> float:
    """
    Calculate ATR-based trailing stop price.

    Unlike fixed percentage stops, ATR stops adapt to each stock's volatility:
    - High-beta stocks (NVDA, TSLA) get wider stops (~12%)
    - Low-vol stocks (KO, PG) get tighter stops (~4%)

    Research shows 2.5x ATR is optimal for momentum strategies with
    weekly rebalancing (Perry Kaufman, "Trading Systems and Methods").

    Note: MIN_STOP (0.97 = 3% loss) is numerically greater than FALLBACK_STOP 
    (0.88 = 12% loss). This is correct as 0.97 represents a tighter exit.

    Args:
        stock_data: Dict mapping ticker -> DataFrame
        ticker: Stock symbol
        date: Current date
        highest_price: Highest price since entry (for trailing)
        multiplier: ATR multiplier (from ATR_PARAMS)
        atr_period: ATR lookback period (from ATR_PARAMS)

    Returns:
        Stop price (exit if current price falls below this)
    """
    if multiplier is None:
        multiplier = ATR_PARAMS["MULTIPLIER"]
    if atr_period is None:
        atr_period = ATR_PARAMS["PERIOD"]

    if ticker not in stock_data:
        # Fallback stop if no data
        return highest_price * ATR_PARAMS["FALLBACK_STOP"]

    df = stock_data[ticker]
    if 'date' not in df.columns:
        return highest_price * ATR_PARAMS["FALLBACK_STOP"]

    df = df[df['date'] <= date].copy()

    MIN_ATR_EXTRA_PERIODS = 5
    if len(df) < atr_period + MIN_ATR_EXTRA_PERIODS:
        return highest_price * ATR_PARAMS["FALLBACK_STOP"]

    # Calculate ATR
    atr_series = calculate_atr(df, period=atr_period)
    current_atr = atr_series.iloc[-1]

    if pd.isna(current_atr) or current_atr <= 0:
        return highest_price * ATR_PARAMS["FALLBACK_STOP"]

    # Trailing stop = highest price - (multiplier * ATR)
    stop_price = highest_price - (multiplier * current_atr)

    # Ensure stop is at least min threshold below highest (prevent too-tight stops)
    min_stop = highest_price * ATR_PARAMS["MIN_STOP"]
    stop_price = min(stop_price, min_stop)

    return stop_price


# =============================================================================
# DRAWDOWN-BASED EXPOSURE SCALING
# =============================================================================

def get_drawdown_exposure(
    portfolio_values: List[float],
    thresholds: Optional[List[Tuple[float, float]]] = None
) -> float:
    """
    Calculate portfolio exposure based on current drawdown level.

    Gradually reduces exposure as drawdown increases, cutting losses
    during sustained downtrends while avoiding premature exits.

    Default thresholds (tuned for HIGH-VOLATILITY momentum strategies):
    - Under 15% DD: Full exposure (1.0)
    - 15-20% DD: Reduce to 80%
    - 20-25% DD: Reduce to 60%
    - 25-30% DD: Reduce to 40%
    - Over 30% DD: Defensive mode (20%)

    Args:
        portfolio_values: List of historical portfolio values
        thresholds: List of (drawdown_level, exposure) tuples
                   Default: [(-0.15, 1.0), (-0.20, 0.8), (-0.25, 0.6), (-0.30, 0.4), (-1.0, 0.2)]

    Returns:
        Exposure multiplier (0.0 to 1.0)
    """
    if thresholds is None:
        # Default thresholds optimized for HIGH-VOLATILITY momentum
        thresholds = DRAWDOWN_THRESHOLDS

    if len(portfolio_values) < 2:
        return 1.0  # No history, full exposure

    # Calculate current drawdown
    peak = max(portfolio_values)
    current = portfolio_values[-1]
    drawdown = (current - peak) / peak

    # Find appropriate exposure level
    for dd_threshold, exposure in thresholds:
        if drawdown > dd_threshold:
            return exposure

    # Deep drawdown, minimal exposure
    return 0.0


# =============================================================================
# MARKET REGIME DETECTION
# =============================================================================

def get_regime_exposure(
    spy_df: pd.DataFrame,
    vix_df: Optional[pd.DataFrame],
    date: pd.Timestamp,
    sma_period: int = None
) -> Tuple[float, str]:
    """
    Determine market regime and appropriate exposure level.

    Uses two signals:
    1. SPY trend: Price above/below 200-day SMA
    2. VIX level: Market volatility/fear

    Regime classification:
    - Bull + Low Vol (VIX<20): 100% exposure (risk-on)
    - Bull + Moderate Vol (VIX 20-25): 80% exposure
    - Bull + Elevated Vol (VIX 25-30): 60% exposure
    - Bear Market (SPY < 200 SMA): 30% exposure (defensive)
    - Crisis (VIX > 30): 0% exposure (cash)

    Args:
        spy_df: DataFrame with SPY data ('date', 'close')
        vix_df: DataFrame with VIX data ('date', 'close'), optional
        date: Current date
        sma_period: Period for trend SMA (from REGIME_PARAMS)

    Returns:
        Tuple of (exposure_multiplier, regime_label)
    """
    if sma_period is None:
        sma_period = REGIME_PARAMS["SMA_PERIOD"]

    # Get SPY data up to date
    spy = spy_df[spy_df['date'] <= date].copy()

    if len(spy) < sma_period:
        # Not enough history, assume neutral
        return 0.8, "insufficient_data"

    spy_close = spy['close'].iloc[-1]
    spy_sma = spy['close'].rolling(sma_period).mean().iloc[-1]
    spy_above_trend = spy_close > spy_sma

    # Get VIX level
    vix_level = REGIME_PARAMS["VIX_LOW"]  # Default to normal if no data
    if vix_df is not None:
        vix = vix_df[vix_df['date'] <= date]
        if len(vix) > 0:
            vix_level = vix['close'].iloc[-1]

    # Regime classification
    if vix_level > REGIME_PARAMS["VIX_CRISIS"]:
        return 0.0, "crisis"
    elif not spy_above_trend:
        return 0.3, "bear_market"
    elif vix_level < REGIME_PARAMS["VIX_LOW"]:
        return 1.0, "bull_calm"
    elif vix_level < REGIME_PARAMS["VIX_MODERATE"]:
        return 0.8, "bull_moderate"
    else:  # moderate-crisis range
        return 0.6, "bull_elevated"


# =============================================================================
# POSITION TRACKING
# =============================================================================

class PositionTracker:
    """
    Track individual positions for trailing stops and risk management.
    """

    def __init__(self):
        self.positions: Dict[str, Dict] = {}

    def add_position(self, ticker: str, entry_price: float, date: pd.Timestamp):
        """Add a new position."""
        self.positions[ticker] = {
            'entry_price': entry_price,
            'highest_price': entry_price,
            'entry_date': date,
        }

    def update_highest(self, ticker: str, current_price: float):
        """Update highest price for trailing stop."""
        if ticker in self.positions:
            self.positions[ticker]['highest_price'] = max(
                self.positions[ticker]['highest_price'],
                current_price
            )

    def get_highest(self, ticker: str) -> Optional[float]:
        """Get highest price since entry."""
        if ticker in self.positions:
            return self.positions[ticker]['highest_price']
        return None

    def remove_position(self, ticker: str):
        """Remove a closed position."""
        if ticker in self.positions:
            del self.positions[ticker]

    def get_entry_price(self, ticker: str) -> Optional[float]:
        """Get entry price for a position."""
        if ticker in self.positions:
            return self.positions[ticker]['entry_price']
        return None


# =============================================================================
# POSITION-LEVEL DIVERSIFICATION
# =============================================================================

def select_with_correlation_filter(
    stock_data: Dict[str, pd.DataFrame],
    rankings: Dict[str, float],
    date: pd.Timestamp,
    top_k: int = 10,
    max_corr: float = 0.70,
    lookback: int = 60
) -> List[str]:
    """
    Select top stocks with correlation diversification constraint.

    Prevents holding highly correlated positions (e.g., 5 tech stocks
    that all crash together). Uses greedy selection: picks highest-ranked
    stocks that don't exceed correlation threshold with already-selected stocks.

    Args:
        stock_data: Dict mapping ticker -> DataFrame
        rankings: Dict of ticker -> predicted_rank (model output)
        date: Current date
        top_k: Target number of holdings
        max_corr: Maximum pairwise correlation allowed (0.70 recommended)
        lookback: Days for correlation calculation

    Returns:
        List of selected tickers (may be < top_k if constraints too tight)
    """
    # Sort candidates by rank (descending)
    sorted_candidates = sorted(rankings.items(), key=lambda x: x[1], reverse=True)

    # Build returns matrix for correlation calculation
    returns_dict = {}
    for ticker, rank in sorted_candidates[:top_k * 3]:  # Check 3x candidates
        if ticker not in stock_data:
            continue

        df = stock_data[ticker]
        recent = df[df['date'] <= date].tail(lookback)

        if len(recent) >= lookback // 2:  # Need at least half the lookback
            ret = recent['close'].pct_change().dropna()
            if len(ret) > 10:  # Need reasonable sample
                returns_dict[ticker] = ret.values

    if not returns_dict:
        # Fallback to top-K without filtering
        return [t for t, _ in sorted_candidates[:top_k]]

    # Create correlation matrix
    # Align all returns to same length (use shortest)
    min_len = min(len(v) for v in returns_dict.values())
    aligned_returns = {k: v[-min_len:] for k, v in returns_dict.items()}
    returns_df = pd.DataFrame(aligned_returns)
    corr_matrix = returns_df.corr()

    # Greedy selection with correlation constraint
    selected = []
    for ticker, rank in sorted_candidates:
        if ticker not in corr_matrix.columns:
            continue

        # Check correlation with already-selected stocks
        if selected:
            available_selected = [s for s in selected if s in corr_matrix.columns]
            if not available_selected:
                max_pair_corr = float("-inf")
            else:
                max_pair_corr = max(corr_matrix.loc[ticker, s] for s in available_selected)

            if max_pair_corr > max_corr:
                continue

        selected.append(ticker)

        if len(selected) >= top_k:
            break

    # If we didn't get enough stocks, fill with next-best (ignoring correlation)
    if len(selected) < top_k:
        for ticker, _ in sorted_candidates:
            if ticker not in selected:
                selected.append(ticker)
                if len(selected) >= top_k:
                    break

    return selected


def calculate_volatility_weights(
    stock_data: Dict[str, pd.DataFrame],
    holdings: List[str],
    date: pd.Timestamp,
    lookback: int = None
) -> Dict[str, float]:
    """
    Calculate inverse-volatility position weights.

    High-volatility stocks get smaller positions, low-volatility stocks
    get larger positions, so each contributes similar risk to the portfolio.

    Args:
        stock_data: Dict mapping ticker -> DataFrame
        holdings: List of tickers to weight
        date: Current date
        lookback: Days for volatility calculation (from VOLATILITY_PARAMS)

    Returns:
        Dict of ticker -> weight (sums to 1.0)
    """
    if lookback is None:
        lookback = VOLATILITY_PARAMS["LOOKBACK_PERIOD"]

    inv_vols = {}

    for ticker in holdings:
        if ticker not in stock_data:
            inv_vols[ticker] = 1.0  # Default weight
            continue

        df = stock_data[ticker]
        recent = df[df['date'] <= date].tail(lookback)

        if len(recent) < lookback // 2:
            inv_vols[ticker] = 1.0  # Not enough data
            continue

        # Calculate realized volatility (annualized)
        daily_ret = recent['close'].pct_change().dropna()
        realized_vol = daily_ret.std() * np.sqrt(VOLATILITY_PARAMS["TRADING_DAYS_PER_YEAR"])

        # Floor volatility at 10% (prevent extreme weights)
        realized_vol = max(realized_vol, 0.10)

        # Inverse volatility weight
        inv_vols[ticker] = 1.0 / realized_vol

    if not inv_vols:
        # Fallback to equal weight
        return {t: 1.0 / len(holdings) for t in holdings}

    # Normalize to sum to 1.0
    total = sum(inv_vols.values())
    if total == 0:
        return {t: 1.0 / len(holdings) for t in holdings}
        
    weights = {t: v / total for t, v in inv_vols.items()}

    return weights


def calculate_hybrid_weights(
    stock_data: Dict[str, pd.DataFrame],
    rankings: Dict[str, float],
    holdings: List[str],
    date: pd.Timestamp,
    vol_weight: float = 0.5,
    rank_weight: float = 0.5,
    lookback: int = 20
) -> Dict[str, float]:
    """
    Calculate hybrid position weights combining volatility and rank.

    Blends two signals:
    1. Inverse volatility (risk contribution)
    2. Model rank (conviction)

    Args:
        stock_data: Dict mapping ticker -> DataFrame
        rankings: Dict of ticker -> predicted_rank
        holdings: List of tickers to weight
        date: Current date
        vol_weight: Weight for volatility component (0.0-1.0)
        rank_weight: Weight for rank component (0.0-1.0)
        lookback: Days for volatility calculation

    Returns:
        Dict of ticker -> weight (sums to 1.0)
    """
    # Volatility weights
    vol_weights = calculate_volatility_weights(stock_data, holdings, date, lookback)

    # Rank weights (linear scaling)
    rank_values = {t: rankings.get(t, 0.5) for t in holdings}
    if not rank_values:
        return {t: 1.0 / len(holdings) for t in holdings} if holdings else {}

    min_rank = min(rank_values.values())
    max_rank = max(rank_values.values())
    rank_range = max_rank - min_rank

    if rank_range > 0:
        # Normalize ranks to 0.5-1.5 range
        normalized_ranks = {
            t: 0.5 + (r - min_rank) / rank_range
            for t, r in rank_values.items()
        }
    else:
        # All same rank
        normalized_ranks = {t: 1.0 for t in holdings}

    # Normalize rank weights
    total_rank = sum(normalized_ranks.values())
    rank_weights = {t: v / total_rank for t, v in normalized_ranks.items()}

    # Blend weights
    hybrid_weights = {}
    for ticker in holdings:
        hybrid = (
            vol_weight * vol_weights.get(ticker, 1.0 / len(holdings)) +
            rank_weight * rank_weights.get(ticker, 1.0 / len(holdings))
        )
        hybrid_weights[ticker] = hybrid

    # Normalize to sum to 1.0
    total = sum(hybrid_weights.values())
    if total == 0:
        return {t: 1.0 / len(holdings) for t in holdings}
        
    final_weights = {t: v / total for t, v in hybrid_weights.items()}

    return final_weights


# =============================================================================
# COMBINED RISK OVERLAY
# =============================================================================

def calculate_final_exposure(
    portfolio_values: List[float],
    spy_df: pd.DataFrame,
    vix_df: Optional[pd.DataFrame],
    date: pd.Timestamp,
    enable_drawdown: bool = True,
    enable_regime: bool = True
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate final portfolio exposure by combining multiple risk factors.

    Combines:
    1. Drawdown-based scaling (portfolio-level protection)
    2. Market regime filtering (macro risk)

    Final exposure = min(drawdown_exposure, regime_exposure)
    (Uses most conservative signal)

    Args:
        portfolio_values: Historical portfolio values
        spy_df: SPY data for regime detection
        vix_df: VIX data for regime detection
        date: Current date
        enable_drawdown: Enable drawdown-based scaling
        enable_regime: Enable regime filter

    Returns:
        Tuple of (final_exposure, details_dict)
    """
    details = {}

    # Drawdown exposure
    if enable_drawdown:
        dd_exposure = get_drawdown_exposure(portfolio_values)
        details['drawdown_exposure'] = dd_exposure
    else:
        dd_exposure = 1.0
        details['drawdown_exposure'] = 1.0

    # Regime exposure
    if enable_regime:
        regime_exposure, regime_label = get_regime_exposure(spy_df, vix_df, date)
        details['regime_exposure'] = regime_exposure
        details['regime_label'] = regime_label
    else:
        regime_exposure = 1.0
        details['regime_exposure'] = 1.0
        details['regime_label'] = 'disabled'

    # Take most conservative signal
    final_exposure = min(dd_exposure, regime_exposure)
    details['final_exposure'] = final_exposure

    return final_exposure, details
