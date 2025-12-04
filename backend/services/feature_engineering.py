"""
Feature engineering module ported from legacy/LSTM_AI_Stock_Predictor/TrainingData/processor.py

This module processes raw OHLCV data into 39 features used by the LSTM model:
- Log returns (5)
- Moving averages (5)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Volatility measures (6)
- Volume indicators (OBV, abnormal_vol)
- Alternative data (insider trading, sentiment)
- Time features (3)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 39 technical features from OHLCV data.

    Args:
        df: DataFrame with columns [date, open, high, low, close, volume]

    Returns:
        DataFrame with all 39 features computed
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # === LOG RETURNS (5 features) ===
    df["YesterdayClose"] = df["close"].shift(1)

    # Handle divide by zero and invalid values in log returns
    with np.errstate(divide='ignore', invalid='ignore'):
        df["YesterdayOpenLogR"] = np.log(df["open"] / df["open"].shift(1))
        df["YesterdayHighLogR"] = np.log(df["high"] / df["high"].shift(1))
        df["YesterdayLowLogR"] = np.log(df["low"] / df["low"].shift(1))
        df["YesterdayVolumeLogR"] = np.log(df["volume"] / df["volume"].shift(1))
        df["YesterdayCloseLogR"] = np.log(df["close"] / df["YesterdayClose"])

    # Replace inf and -inf with NaN (will be dropped later)
    df["YesterdayOpenLogR"] = df["YesterdayOpenLogR"].replace([np.inf, -np.inf], np.nan)
    df["YesterdayHighLogR"] = df["YesterdayHighLogR"].replace([np.inf, -np.inf], np.nan)
    df["YesterdayLowLogR"] = df["YesterdayLowLogR"].replace([np.inf, -np.inf], np.nan)
    df["YesterdayVolumeLogR"] = df["YesterdayVolumeLogR"].replace([np.inf, -np.inf], np.nan)
    df["YesterdayCloseLogR"] = df["YesterdayCloseLogR"].replace([np.inf, -np.inf], np.nan)

    # === MOVING AVERAGES (5 features) ===
    df["MA10"] = df["close"].rolling(window=10).mean()
    df["MA20"] = df["close"].rolling(window=20).mean()
    df["MA30"] = df["close"].rolling(window=30).mean()
    df["EMA10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["EMA30"] = df["close"].ewm(span=30, adjust=False).mean()

    # === TIME FEATURES (3 features) ===
    df["DayOfWeek"] = df["date"].dt.weekday         # 0 = Monday, 6 = Sunday
    df["DayOfMonth"] = df["date"].dt.day            # 1 to 31
    df["MonthNumber"] = df["date"].dt.month         # 1 = January, 12 = December

    # === RSI (1 feature) ===
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # === MACD (2 features) ===
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # === BOLLINGER BANDS (2 features) ===
    ma20 = df["close"].rolling(window=20).mean()
    std20 = df["close"].rolling(window=20).std()
    df["BollingerUpper"] = ma20 + 2 * std20
    df["BollingerLower"] = ma20 - 2 * std20

    # === ROLLING VOLATILITY (3 features) ===
    df["Volatility_10"] = df["close"].pct_change().rolling(window=10).std()
    df["Volatility_20"] = df["close"].pct_change().rolling(window=20).std()
    df["Volatility_30"] = df["close"].pct_change().rolling(window=30).std()

    # === OBV (1 feature) ===
    df["OBV"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

    # === Z-SCORE (1 feature) ===
    mean = df["close"].rolling(window=20).mean()
    std = df["close"].rolling(window=20).std()
    df["ZScore"] = (df["close"] - mean) / std

    # === ADDITIONAL FEATURES (9 features) ===
    # Overnight gap %
    df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

    # Abnormal volume z-score
    rolling_vol = df['volume'].rolling(20)
    df['abnormal_vol'] = (df['volume'] - rolling_vol.mean()) / rolling_vol.std()

    # Short term realized volatility (annualized)
    df['volatility_5d'] = df['close'].pct_change().rolling(5).std() * np.sqrt(252)
    df['volatility_20d'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)

    # Momentum
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_20d'] = df['close'] / df['close'].shift(20) - 1

    # Skewness
    df['skew_5d'] = df['close'].pct_change().rolling(5).skew()

    # Intraday range
    df['intraday_range'] = (df['high'] - df['low']) / df['close']

    return df


def merge_insider_data(df: pd.DataFrame, insider_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge insider trading data (SEC Form 4) with price data.

    Args:
        df: DataFrame with price and technical features
        insider_df: DataFrame with columns [date, shares, amount, buy_flag]

    Returns:
        DataFrame with insider features added
    """
    if insider_df is not None and not insider_df.empty:
        insider_df = insider_df.rename(columns={
            "shares": "insider_shares",
            "amount": "insider_amount",
            "buy_flag": "insider_buy_flag"
        })
        df = df.merge(
            insider_df[["date", "insider_shares", "insider_amount", "insider_buy_flag"]],
            on="date", how="left"
        )
        df["insider_shares"] = df["insider_shares"].fillna(0)
        df["insider_amount"] = df["insider_amount"].fillna(0)
        df["insider_buy_flag"] = df["insider_buy_flag"].fillna(-1).astype(int)
    else:
        df["insider_shares"] = 0
        df["insider_amount"] = 0
        df["insider_buy_flag"] = -1

    return df


def merge_sentiment_data(df: pd.DataFrame, sentiment_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge sentiment data (news analysis) with price data.

    Args:
        df: DataFrame with price and technical features
        sentiment_df: DataFrame with columns [date, sentiment, num_articles]

    Returns:
        DataFrame with sentiment features added
    """
    if sentiment_df is not None and not sentiment_df.empty:
        try:
            df = df.merge(
                sentiment_df[["date", "sentiment", "num_articles"]],
                on="date", how="left"
            )
            df["sentiment"] = df["sentiment"].fillna(0)
            df["num_articles"] = df["num_articles"].fillna(0)
        except Exception as e:
            logger.warning(f"Failed to merge sentiment data: {e}")
            df["sentiment"] = 0
            df["num_articles"] = 0
    else:
        df["sentiment"] = 0
        df["num_articles"] = 0

    # Sentiment change
    df['sentiment_change'] = df['sentiment'] - df['sentiment'].shift(1)

    return df


def create_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target variables for multi-horizon forecasting.

    Targets are log returns at different horizons:
    - Target_1d: 1 day ahead
    - Target_1w: 5 days (1 week) ahead
    - Target_1m: 21 days (1 month) ahead
    - Target_6m: 126 days (6 months) ahead

    Args:
        df: DataFrame with close prices

    Returns:
        DataFrame with target columns added
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        df["Target_1d"] = np.log(df["close"].shift(-1) / df["close"])
        df["Target_1w"] = np.log(df["close"].shift(-5) / df["close"])
        df["Target_1m"] = np.log(df["close"].shift(-21) / df["close"])
        df["Target_6m"] = np.log(df["close"].shift(-126) / df["close"])

    # Replace inf/-inf with NaN
    df["Target_1d"] = df["Target_1d"].replace([np.inf, -np.inf], np.nan)
    df["Target_1w"] = df["Target_1w"].replace([np.inf, -np.inf], np.nan)
    df["Target_1m"] = df["Target_1m"].replace([np.inf, -np.inf], np.nan)
    df["Target_6m"] = df["Target_6m"].replace([np.inf, -np.inf], np.nan)

    return df


def process_stock_data(
    df: pd.DataFrame,
    insider_df: Optional[pd.DataFrame] = None,
    sentiment_df: Optional[pd.DataFrame] = None,
    create_targets: bool = True
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for stock data.

    Args:
        df: Raw OHLCV DataFrame with columns [date, open, high, low, close, volume]
        insider_df: Optional insider trading data
        sentiment_df: Optional sentiment data
        create_targets: Whether to create target variables (False for inference)

    Returns:
        Processed DataFrame with all 39 features
    """
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    # Compute technical features
    df = compute_technical_features(df)

    # Merge alternative data
    df = merge_insider_data(df, insider_df)
    df = merge_sentiment_data(df, sentiment_df)

    # Create targets if requested
    if create_targets:
        df = create_target_variables(df)

    # Drop rows with NaN values
    df = df.dropna()

    return df


def get_feature_columns() -> list:
    """
    Returns the list of 39 feature columns expected by the model.
    Order matters - this must match the training data.
    """
    return [
        # Price features (keep close for reference, but drop OHLV after processing)
        'close',
        'YesterdayClose',

        # Log returns (5)
        'YesterdayOpenLogR',
        'YesterdayHighLogR',
        'YesterdayLowLogR',
        'YesterdayVolumeLogR',
        'YesterdayCloseLogR',

        # Moving averages (5)
        'MA10', 'MA20', 'MA30', 'EMA10', 'EMA30',

        # Time features (3)
        'DayOfWeek', 'DayOfMonth', 'MonthNumber',

        # Technical indicators (5)
        'RSI', 'MACD', 'MACD_Signal',
        'BollingerUpper', 'BollingerLower',

        # Volatility (6)
        'Volatility_10', 'Volatility_20', 'Volatility_30',
        'volatility_5d', 'volatility_20d',

        # Volume indicators (2)
        'OBV', 'abnormal_vol',

        # Price patterns (5)
        'ZScore', 'overnight_gap', 'momentum_5d', 'momentum_20d',
        'skew_5d', 'intraday_range',

        # Alternative data (6)
        'insider_shares', 'insider_amount', 'insider_buy_flag',
        'sentiment', 'num_articles', 'sentiment_change'
    ]


def get_model_input_features() -> list:
    """
    Returns the 39 features used as model inputs (excludes date, close, targets).
    """
    all_features = get_feature_columns()
    # Remove close and YesterdayClose (used for reference only)
    return [f for f in all_features if f not in ['close', 'YesterdayClose']]
