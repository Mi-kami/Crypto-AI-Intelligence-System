"""
storage/reader.py

Responsible for retrieving clean data from the SQLite database
for use in feature engineering and model training.

Three readers — one per table:
    - read_price_data      : hourly OHLCV rows per asset
    - read_market_signals  : global market data by time window
    - read_news_headlines  : news headlines per asset

Sits between storage and the feature engineering pipeline.
No data is written here. No data is fetched here. This module
only reads and returns DataFrames.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from loguru import logger
from storage.schema import DB_PATH
from datetime import datetime

def read_price_data(asset: str, start: datetime, end: datetime, db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Retrieve hourly OHLCV price rows for a single asset within a time window.

    Queries the price_ohlcv table filtered by asset symbol and timestamp
    range. Filter is applied in SQL — not in Python — so only matching
    rows are loaded into memory.

    Args:
        asset: Asset symbol e.g. "BTC", "ETH"
        start: Start of time window (UTC, inclusive)
        end:   End of time window (UTC, inclusive)

    Returns:
        DataFrame with columns: asset, timestamp, close, market_cap,
        volume, source. Timestamp column is UTC datetime.
        Returns empty DataFrame if no rows match.

    Raises:
        sqlite3.Error: If the database query fails
    """

    start_str = start.isoformat()
    end_str = end.isoformat()
    
    sql = """
        SELECT *
        FROM price_ohlcv
        WHERE asset = ?
        AND timestamp BETWEEN ? AND ?
    """

    conn = sqlite3.connect(db_path)
    try:
       df = pd.read_sql_query(sql, conn, params= (asset, start_str, end_str ))
    except sqlite3.Error as e:
        logger.error(f"read_price_data failed: {e}")
        raise
    finally:
        conn.close()

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    logger.success(
            f"read_price_data from {start} to {end} and {len(df)} row(s) read from price_ohlcv"
        )
    return df

def read_market_signals(start: datetime, end: datetime, db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Retrieve global market signal rows within a time window.

    Queries the market_signals table filtered by timestamp range.
    No asset filter — market signals are market-wide, not asset-specific.

    Args:
        start:   Start of time window (UTC, inclusive)
        end:     End of time window (UTC, inclusive)
        db_path: Path to the SQLite database file (default: DB_PATH)

    Returns:
        DataFrame with columns: timestamp, btc_dominance,
        total_market_cap_usd, total_volume_usd, source.
        Timestamp column is UTC datetime.
        Returns empty DataFrame if no rows match.

    Raises:
        sqlite3.Error: If the database query fails
    """
    start_str = start.isoformat()
    end_str = end.isoformat()

    sql = """
        SELECT *
        FROM market_signals
        WHERE timestamp BETWEEN ? AND ?
    """

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql, conn, params=(start_str, end_str))
    except sqlite3.Error as e:
        logger.error(f"read_market_signals failed: {e}")
        raise
    finally:
        conn.close()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    logger.success(
        f"read_market_signals | {len(df)} row(s) read from market_signals | "
        f"{start} → {end}"
    )
    return df


def read_news_headlines(asset: str, start: datetime, end: datetime, db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Retrieve news headline rows for a single asset within a time window.

    Queries the news_headlines table filtered by asset symbol and
    timestamp range. Filter is applied in SQL for efficiency.

    Args:
        asset:   Asset symbol e.g. "BTC", "ETH"
        start:   Start of time window (UTC, inclusive)
        end:     End of time window (UTC, inclusive)
        db_path: Path to the SQLite database file (default: DB_PATH)

    Returns:
        DataFrame with columns: id, asset, timestamp, headline,
        url, votes_positive, votes_negative, source.
        Timestamp column is UTC datetime.
        Returns empty DataFrame if no rows match.

    Raises:
        sqlite3.Error: If the database query fails
    """
    start_str = start.isoformat()
    end_str = end.isoformat()

    sql = """
        SELECT *
        FROM news_headlines
        WHERE asset = ?
        AND timestamp BETWEEN ? AND ?
    """

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql, conn, params=(asset, start_str, end_str))
    except sqlite3.Error as e:
        logger.error(f"read_news_headlines failed: {e}")
        raise
    finally:
        conn.close()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    logger.success(
        f"read_news_headlines | {len(df)} row(s) read from news_headlines | "
        f"{asset} | {start} → {end}"
    )
    return df