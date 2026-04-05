"""
ingestion/harmoniser.py

Responsible for validating, normalising, and standardising raw data
received from the CoinGecko and CryptoPanic clients before it is
passed to the storage layer.

The harmoniser sits between ingestion and storage. It ensures every
DataFrame that reaches the database has the correct columns, clean
UTC-floored hourly timestamps, and only the 10 supported asset symbols.

No data is fetched here. No data is stored here. This module only
shapes and validates.
"""

import pandas as pd
from loguru import logger
from typing import List


# ── Constants ────────────────────────────────────────────────────────────────

VALID_ASSETS = [
    "BTC", "ETH", "BNB", "XRP", "SOL",
    "DOGE", "ADA", "TRX", "AVAX", "SHIB"
]

PRICE_REQUIRED_COLUMNS = ["asset", "timestamp", "close", "market_cap", "volume", "source"]
MARKET_REQUIRED_COLUMNS = ["timestamp", "btc_dominance", "total_market_cap_usd", "total_volume_usd", "source"]
NEWS_REQUIRED_COLUMNS = ["asset", "headline", "url", "timestamp", "votes_positive", "votes_negative", "source"]


# ── Private Helpers ──────────────────────────────────────────────────────────

def _validate_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Check that all required columns are present in the DataFrame.

    Raises ValueError immediately if any column is missing.
    Called before any other operation so bad data is rejected early.

    Args:
        df: Raw DataFrame from either client
        required_columns: List of column names that must be present

    Raises:
        ValueError: If one or more required columns are missing
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _floor_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Floor the timestamp column to the nearest hour in UTC.

    Ensures all records align to hourly boundaries regardless
    of the precision returned by the source API.

    Args:
        df: DataFrame containing a 'timestamp' column

    Returns:
        DataFrame with timestamp column floored to the hour
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("h")
    return df

def _validate_assets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to only include rows for supported assets.

    Removes any rows where the asset symbol is not in VALID_ASSETS.
    Logs a warning if rows are dropped so upstream issues are visible.

    Args:
        df: DataFrame containing an 'asset' column

    Returns:
        DataFrame containing only rows for valid assets
    """
    before_filter = len(df)
    filtered_df = df[df["asset"].isin(VALID_ASSETS)]
    after_filter = len(filtered_df)

    if before_filter != after_filter:
        logger.warning(
            f"_validate_assets dropped {before_filter - after_filter} rows "
            f"with unrecognised asset symbols"
        )

    return filtered_df

def harmonise_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate, filter, and normalise raw CoinGecko OHLCV price data.

    Ensures the DataFrame has the correct columns, contains only
    supported assets, and has timestamps floored to the nearest hour
    before being passed to the storage layer.

    Args:
        df: Raw DataFrame from fetch_ohlcv() or fetch_all_assets()

    Returns:
        Clean, normalised DataFrame ready for storage
    """
    _validate_columns(df, PRICE_REQUIRED_COLUMNS)
    df = _validate_assets(df)
    df = _floor_timestamps(df)

    logger.success(
        f"harmonise_price_data complete | {len(df)} rows ready for storage"
    )

    return df

def harmonise_market_data(global_dict: dict) -> pd.DataFrame:
    """
    Validate and normalise raw CoinGecko global market data.

    Converts the global market dictionary to a DataFrame and ensures
    correct columns and hourly-floored timestamps before storage.

    No asset filtering applied — global market data is market-wide,
    not asset-specific.

    Args:
        global_dict: Raw dictionary from fetch_global_market_data()

    Returns:
        Single-row DataFrame ready for storage
    """
    df = pd.DataFrame([global_dict])
    _validate_columns(df, MARKET_REQUIRED_COLUMNS)
    df = _floor_timestamps(df)

    logger.success(
        f"harmonise_market_data complete | timestamp: {df['timestamp'].iloc[0]}"
    )

    return df

def harmonise_news_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate, filter, and normalise raw CryptoPanic news data.

    Renames published_at to timestamp to match the unified schema,
    then validates columns, filters to supported assets, and floors
    timestamps to the nearest hour before storage.

    Args:
        df: Raw DataFrame from fetch_news_for_asset() or fetch_all_assets_news()

    Returns:
        Clean, normalised DataFrame ready for storage
    """
    df = df.rename(columns={"published_at": "timestamp"})
    _validate_columns(df, NEWS_REQUIRED_COLUMNS)
    df = _validate_assets(df)
    df = _floor_timestamps(df)

    logger.success(
        f"harmonise_news_data complete | {len(df)} headlines ready for storage"
    )

    return df