"""
storage/writer.py

Responsible for writing clean, harmonised DataFrames to the SQLite
database. Called by main.py after each harmoniser pass.

Three live writers:
    - write_ohlcv()          : upsert price data (INSERT OR REPLACE)
    - write_market_signals() : upsert market data (INSERT OR REPLACE)
    - write_headlines()      : insert news, skip duplicates (INSERT OR IGNORE)

One placeholder:
    - write_model_outputs()  : Phase 4 — not yet implemented
"""

import sqlite3
import pandas as pd
from pathlib import Path
from loguru import logger

from storage.schema import DB_PATH

def write_price_data(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    """
    Upsert hourly OHLCV data into price_ohlcv table.

    Uses INSERT OR REPLACE — if a row with the same (asset, timestamp)
    already exists, it is deleted and replaced with the new values.

    Args:
        df:      Harmonised DataFrame from harmonise_price_data()
        db_path: Path to the SQLite database file
    """
    if df.empty:
        logger.warning("write_price_data received empty DataFrame — nothing written")
        return

    records = [
        (
            row.asset,
            str(row.timestamp),
            row.close,
            row.source,
            row.market_cap,
            row.volume,
        )
        for row in df.itertuples(index=False)
    ]

    sql = """
        INSERT OR REPLACE INTO price_ohlcv
            (asset, timestamp, close, source, market_cap, volume)
        VALUES (?, ?, ?, ?, ?, ?)
    """

    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(sql, records)
        conn.commit()
        logger.success(
            f"write_price_data: {len(records)} row(s) written to price_ohlcv"
        )
    except sqlite3.Error as e:
        logger.error(f"write_price_data failed: {e}")
        raise
    finally:
        conn.close()

def write_market_signals(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    """
    Upsert hourly market data into market_signals table.

    Uses INSERT OR REPLACE — if a row with the same timestamp already exists,
    it is deleted and replaced with the new values.

    Args:
        df:      Harmonised DataFrame from harmonise_market_data()
        db_path: Path to the SQLite database file
    """
    if df.empty:
        logger.warning("write_market_signals received empty DataFrame — nothing written")
        return

    records = [
        (
            str(row.timestamp),
            row.btc_dominance,
            row.source,
            row.total_market_cap_usd,
            row.total_volume_usd,
        )
        for row in df.itertuples(index=False)
    ]

    sql = """
        INSERT OR REPLACE INTO market_signals
            (timestamp, btc_dominance, source, total_market_cap_usd, total_volume_usd)
        VALUES (?, ?, ?, ?, ?)
    """

    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(sql, records)
        conn.commit()
        logger.success(
            f"write_market_signals: {len(records)} row(s) written to market_signals"
        )
    except sqlite3.Error as e:
        logger.error(f"write_market_signals failed: {e}")
        raise
    finally:
        conn.close()

def write_news_headlines(df: pd.DataFrame, db_path: Path=DB_PATH) -> None:
    """
    Upsert hourly news data into news_headline table.

    Uses INSERT OR IGNORE — if a row with the same (asset,url) already exists,
      the incoming row is skipped and the existing record is retained.
This prevents duplicate headlines from being inserted across pipeline runs.
    Args:
        df:      Harmonised DataFrame from harmonise_news_data()
        db_path: Path to the SQLite database file
    """

    if df.empty:
        logger.warning("write_news_headlines received empty  DataFrame — nothing written")
        return
    
    records = [
        (
            row.asset          ,
            str(row.timestamp) ,
            row.headline       ,
            row.url            ,
            row.source         ,
            row.votes_positive ,
            row.votes_negative ,
        )
        for row in df.itertuples(index=False)
    ]

    sql = """
        INSERT OR IGNORE INTO news_headlines
            (asset, timestamp, headline, url, source, votes_positive, votes_negative)
        VALUES (?, ?, ?, ?, ?, ?, ?)

    """ 

    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(sql, records)
        conn.commit()
        logger.success(
            f"write_news_headlines: {len(records)} row(s) written to news_headlines"
        )
    except sqlite3.Error as e:
        logger.error(f"write_news_headlines failed: {e}")
        raise
    finally:
        conn.close()