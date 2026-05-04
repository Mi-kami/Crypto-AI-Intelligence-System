"""
storage/schema.py

Defines and creates the SQLite database schema for the Crypto AI
Intelligence System.

Three tables:
    - price_ohlcv      : hourly OHLCV data per asset from CoinGecko
    - market_signals   : global market data (BTC dominance, total cap/volume)
    - news_headlines   : news headlines per asset from CryptoCompare

Run this module directly to initialise the database:
    python -m storage.schema
"""

import sqlite3
from pathlib import Path
from loguru import logger


# ── Constants ────────────────────────────────────────────────────────────────

# Default path for the SQLite database file
# Path() handles Windows/Mac/Linux differences automatically
DB_PATH = Path("data/crypto_ai.db")


# ── SQL Definitions ──────────────────────────────────────────────────────────

CREATE_PRICE_OHLCV = """
CREATE TABLE IF NOT EXISTS price_ohlcv (
    asset       TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    close       REAL NOT NULL,
    source      TEXT NOT NULL,
    market_cap  REAL,
    volume      REAL,
    PRIMARY KEY (asset, timestamp)
)
"""

CREATE_MARKET_SIGNALS = """
CREATE TABLE IF NOT EXISTS market_signals (
    timestamp            TEXT NOT NULL,
    btc_dominance        REAL NOT NULL,
    source               TEXT NOT NULL,
    total_market_cap_usd REAL,
    total_volume_usd     REAL,
    PRIMARY KEY (timestamp)
)
"""

CREATE_NEWS_HEADLINES = """
CREATE TABLE IF NOT EXISTS news_headlines (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    asset          TEXT NOT NULL,
    timestamp      TEXT NOT NULL,
    headline       TEXT NOT NULL,
    url            TEXT NOT NULL,
    source         TEXT NOT NULL,
    votes_positive INTEGER,
    votes_negative INTEGER,
    UNIQUE (asset, url)
)
"""

# ── Index Definitions ────────────────────────────────────────────────────────

CREATE_INDEX_PRICE = """
CREATE INDEX IF NOT EXISTS idx_price_asset_timestamp
ON price_ohlcv (asset, timestamp)
"""

CREATE_INDEX_MARKET = """
CREATE INDEX IF NOT EXISTS idx_market_timestamp
ON market_signals (timestamp)
"""

CREATE_INDEX_NEWS = """
CREATE INDEX IF NOT EXISTS idx_news_asset_timestamp
ON news_headlines (asset, timestamp)
"""


# ── Core Function ────────────────────────────────────────────────────────────

def create_tables(db_path: Path = DB_PATH) -> None:
    """
    Create all three tables and their indexes in the SQLite database.

    Safe to call multiple times — CREATE TABLE IF NOT EXISTS and
    CREATE INDEX IF NOT EXISTS mean nothing is overwritten or destroyed
    if the tables already exist.

    Args:
        db_path: Path to the SQLite database file.
                 Created automatically if it does not exist.
    """
    # Ensure the parent directory exists before SQLite tries to create the file
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initialising database at {db_path}")

    conn = sqlite3.connect(db_path)

    try:
        cursor = conn.cursor()

        # Create tables
        cursor.execute(CREATE_PRICE_OHLCV)
        logger.info("Table ready: price_ohlcv")

        cursor.execute(CREATE_MARKET_SIGNALS)
        logger.info("Table ready: market_signals")

        cursor.execute(CREATE_NEWS_HEADLINES)
        logger.info("Table ready: news_headlines")

        # Create indexes
        cursor.execute(CREATE_INDEX_PRICE)
        cursor.execute(CREATE_INDEX_MARKET)
        cursor.execute(CREATE_INDEX_NEWS)
        logger.info("Indexes ready")

        conn.commit()
        logger.success(f"Database initialised successfully at {db_path}")

    except sqlite3.Error as e:
        logger.error(f"Database initialisation failed: {e}")
        raise

    finally:
        conn.close()


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    create_tables()