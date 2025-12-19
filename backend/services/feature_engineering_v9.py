"""
V9 Feature Engineering - Stationary Features for Relative Strength Ranking

Key Design Principles:
1. STATIONARY ONLY - No raw prices (model must not know if stock is $10 or $1000)
2. MACRO CONTEXT - SPY returns & VIX for market regime awareness
3. CROSS-SECTIONAL RANKING - Target is relative rank among all stocks (0.0-1.0)

Features (12 inputs):
    - log_ret_1d, log_ret_5d, log_ret_20d: Log returns at different horizons
    - rsi_14: RSI normalized to 0-1
    - macd_hist_norm: MACD Histogram normalized by ATR
    - volume_ratio: Volume / MA(Volume, 20)
    - dist_sma20: (Close - SMA20) / SMA20 (percentage)
    - volatility_20d: 20-day realized volatility
    - spy_ret_5d: SPY 5-day return (macro context)
    - vix_level: VIX / 100 (normalized)
    - spy_corr_20d: Rolling correlation with SPY
    - rel_strength_20d: Stock return - SPY return (relative strength)

Target:
    - target_rank: Cross-sectional percentile rank of 5-day future return
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE NAMES (for model input ordering)
# =============================================================================

V9_FEATURE_NAMES = [
    'log_ret_1d',
    'log_ret_5d', 
    'log_ret_20d',
    'rsi_14',
    'macd_hist_norm',
    'volume_ratio',
    'dist_sma20',
    'volatility_20d',
    'spy_ret_5d',
    'vix_level',
    'spy_corr_20d',
    'rel_strength_20d',
]

N_FEATURES = len(V9_FEATURE_NAMES)  # 12 features


# =============================================================================
# INDIVIDUAL FEATURE CALCULATIONS
# =============================================================================

def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate log returns at multiple horizons."""
    df = df.copy()
    df['log_ret_1d'] = np.log(df['close'] / df['close'].shift(1))
    df['log_ret_5d'] = np.log(df['close'] / df['close'].shift(5))
    df['log_ret_20d'] = np.log(df['close'] / df['close'].shift(20))
    return df


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate RSI normalized to 0-1 range."""
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Normalize to 0-1
    df['rsi_14'] = rsi / 100.0
    return df


def compute_macd_histogram_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate MACD Histogram normalized by ATR."""
    df = df.copy()
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    
    # ATR for normalization
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_14 = true_range.rolling(14).mean()
    
    # Normalize: MACD_hist / ATR
    df['macd_hist_norm'] = macd_hist / (atr_14 + 1e-10)
    
    # Clip extreme values
    df['macd_hist_norm'] = df['macd_hist_norm'].clip(-5, 5)
    
    return df


def compute_volume_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume relative to 20-day moving average."""
    df = df.copy()
    vol_ma20 = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (vol_ma20 + 1e-10)
    
    # Log transform to handle extreme volume spikes
    df['volume_ratio'] = np.log1p(df['volume_ratio'])
    
    return df


def compute_distance_from_sma(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate percentage distance from SMA20."""
    df = df.copy()
    sma20 = df['close'].rolling(20).mean()
    df['dist_sma20'] = (df['close'] - sma20) / (sma20 + 1e-10)
    
    # Clip to reasonable range
    df['dist_sma20'] = df['dist_sma20'].clip(-0.5, 0.5)
    
    return df


def compute_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate 20-day realized volatility (annualized)."""
    df = df.copy()
    daily_ret = df['close'].pct_change()
    df['volatility_20d'] = daily_ret.rolling(20).std() * np.sqrt(252)
    return df


def compute_spy_correlation(df: pd.DataFrame, spy_returns: pd.Series) -> pd.DataFrame:
    """Calculate 20-day rolling correlation with SPY."""
    df = df.copy()
    stock_ret = df['close'].pct_change()
    
    # Align SPY returns with stock dates
    spy_aligned = spy_returns.reindex(df['date']).fillna(0)
    
    # Rolling correlation
    df['spy_corr_20d'] = stock_ret.rolling(20).corr(spy_aligned.reset_index(drop=True))
    df['spy_corr_20d'] = df['spy_corr_20d'].fillna(0)
    
    return df


def compute_relative_strength(df: pd.DataFrame, spy_ret_20d: pd.Series) -> pd.DataFrame:
    """Calculate relative strength vs SPY (20-day)."""
    df = df.copy()
    stock_ret_20d = np.log(df['close'] / df['close'].shift(20))
    
    # Merge SPY return
    spy_aligned = spy_ret_20d.reindex(df['date']).fillna(0)
    
    df['rel_strength_20d'] = stock_ret_20d - spy_aligned.reset_index(drop=True).values
    
    return df


# =============================================================================
# MACRO DATA PROCESSING
# =============================================================================

def process_spy_data(spy_df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Process SPY data to create macro context features.
    
    Returns dict with:
        - spy_ret_5d: 5-day SPY return indexed by date
        - spy_ret_20d: 20-day SPY return indexed by date  
        - spy_daily_ret: Daily SPY returns indexed by date
    """
    spy_df = spy_df.copy()
    spy_df = spy_df.sort_values('date').reset_index(drop=True)
    
    spy_df['spy_ret_5d'] = np.log(spy_df['close'] / spy_df['close'].shift(5))
    spy_df['spy_ret_20d'] = np.log(spy_df['close'] / spy_df['close'].shift(20))
    spy_df['spy_daily_ret'] = spy_df['close'].pct_change()
    
    return {
        'spy_ret_5d': spy_df.set_index('date')['spy_ret_5d'],
        'spy_ret_20d': spy_df.set_index('date')['spy_ret_20d'],
        'spy_daily_ret': spy_df.set_index('date')['spy_daily_ret'],
    }


def process_vix_data(vix_df: pd.DataFrame) -> pd.Series:
    """
    Process VIX data to create normalized VIX level.
    
    Returns:
        vix_level: VIX / 100 indexed by date (so VIX=20 -> 0.20)
    """
    vix_df = vix_df.copy()
    vix_df = vix_df.sort_values('date').reset_index(drop=True)
    
    # Use close as VIX level, normalize by dividing by 100
    vix_df['vix_level'] = vix_df['close'] / 100.0
    
    return vix_df.set_index('date')['vix_level']


# =============================================================================
# MAIN FEATURE ENGINEERING FUNCTION
# =============================================================================

def compute_v9_features(
    df: pd.DataFrame,
    spy_data: Optional[Dict[str, pd.Series]] = None,
    vix_data: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compute all V9 features for a single stock.
    
    Args:
        df: OHLCV DataFrame with columns [date, open, high, low, close, volume]
        spy_data: Dict with spy_ret_5d, spy_ret_20d, spy_daily_ret (from process_spy_data)
        vix_data: Series with VIX levels indexed by date (from process_vix_data)
    
    Returns:
        DataFrame with all 12 V9 features + date column
    """
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    df = df.sort_values('date').reset_index(drop=True)
    
    # Core stationary features
    df = compute_log_returns(df)
    df = compute_rsi(df)
    df = compute_macd_histogram_normalized(df)
    df = compute_volume_ratio(df)
    df = compute_distance_from_sma(df)
    df = compute_volatility(df)
    
    # Macro features (SPY context)
    if spy_data is not None:
        # SPY 5-day return
        df['spy_ret_5d'] = df['date'].map(spy_data['spy_ret_5d']).fillna(0)
        
        # SPY correlation (need daily returns)
        df = compute_spy_correlation(df, spy_data['spy_daily_ret'])
        
        # Relative strength
        df = compute_relative_strength(df, spy_data['spy_ret_20d'])
    else:
        df['spy_ret_5d'] = 0.0
        df['spy_corr_20d'] = 0.0
        df['rel_strength_20d'] = 0.0
    
    # VIX level
    if vix_data is not None:
        df['vix_level'] = df['date'].map(vix_data).fillna(0.2)  # Default VIX=20
    else:
        df['vix_level'] = 0.2  # Default normalized VIX
    
    # Replace inf/nan
    for col in V9_FEATURE_NAMES:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(0)
    
    return df


# =============================================================================
# TARGET VARIABLE: CROSS-SECTIONAL RANK
# =============================================================================

def compute_future_return(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Compute future return for ranking target.
    
    Args:
        df: DataFrame with close prices
        horizon: Number of days ahead for future return (default: 5 = 1 week)
    
    Returns:
        DataFrame with future_ret_5d column
    """
    df = df.copy()
    df['future_ret_5d'] = np.log(df['close'].shift(-horizon) / df['close'])
    return df


def compute_cross_sectional_ranks(
    all_stocks_df: pd.DataFrame,
    date_col: str = 'date',
    ticker_col: str = 'ticker',
    return_col: str = 'future_ret_5d',
) -> pd.DataFrame:
    """
    Compute cross-sectional percentile ranks for all stocks on each date.
    
    Args:
        all_stocks_df: Combined DataFrame with all stocks
        date_col: Column name for date
        ticker_col: Column name for ticker
        return_col: Column name for future return to rank
    
    Returns:
        DataFrame with target_rank column (0.0 = worst, 1.0 = best)
    """
    df = all_stocks_df.copy()
    
    # For each date, rank stocks by future return
    # pct=True gives percentile rank (0 to 1)
    df['target_rank'] = df.groupby(date_col)[return_col].rank(pct=True, method='average')
    
    # Handle NaN (stocks without future return data)
    df['target_rank'] = df['target_rank'].fillna(0.5)  # Neutral rank
    
    return df


# =============================================================================
# FULL PIPELINE FUNCTION
# =============================================================================

def process_stock_for_v9(
    stock_df: pd.DataFrame,
    ticker: str,
    spy_data: Optional[Dict[str, pd.Series]] = None,
    vix_data: Optional[pd.Series] = None,
    drop_na_target: bool = True,
) -> pd.DataFrame:
    """
    Full V9 processing pipeline for a single stock.
    
    Args:
        stock_df: OHLCV DataFrame
        ticker: Stock ticker symbol
        spy_data: Processed SPY data dict
        vix_data: Processed VIX data series
        drop_na_target: Whether to drop rows where future target is NaN (True for training, False for inference)
    
    Returns:
        DataFrame with all V9 features, future return, and metadata
    """
    df = stock_df.copy()
    
    # Ensure we have required columns
    required = ['date', 'open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Compute features
    df = compute_v9_features(df, spy_data, vix_data)
    
    # Compute future return for ranking
    df = compute_future_return(df, horizon=5)
    
    # Add ticker column
    df['ticker'] = ticker
    
    # Drop rows with NaN in features (warmup period)
    df = df.dropna(subset=V9_FEATURE_NAMES)
    
    # Drop rows with NaN in target ONLY if requested (usually for training)
    if drop_na_target:
        # Only drop columns that actually exist to avoid KeyError
        cols_to_drop = [c for c in ['target_5d', 'target_rank'] if c in df.columns]
        if cols_to_drop:
            df = df.dropna(subset=cols_to_drop)
    
    return df


def get_v9_feature_names() -> List[str]:
    """Return list of V9 feature names in order."""
    return V9_FEATURE_NAMES.copy()
