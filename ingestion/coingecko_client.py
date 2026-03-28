"""
ingestion/coingecko_client.py

Responsible for fetching price data and market signals from the CoinGecko API
for the top 10 cryptocurrency assets.

CoinGecko free tier limits:
- 30 requests per minute
- 10,000 requests per month
"""

import time
import requests
import pandas as pd
from datetime import datetime, timezone
from loguru import logger


# ── Constants ────────────────────────────────────────────────────────────────

# Root URL for all CoinGecko API calls
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

# Maps our internal symbol (BTC) to CoinGecko's coin ID (bitcoin)
ASSET_ID_MAP = {
    "BTC":  "bitcoin",
    "ETH":  "ethereum",
    "BNB":  "binancecoin",
    "XRP":  "ripple",
    "SOL":  "solana",
    "DOGE": "dogecoin",
    "ADA":  "cardano",
    "TRX":  "tron",
    "AVAX": "avalanche-2",
    "SHIB": "shiba-inu",
}

# Seconds to wait between API calls to respect the 30 requests/minute limit
# 60 seconds / 30 requests = 2 seconds minimum. We use 2.5 as a safety buffer.
RATE_LIMIT_DELAY = 2.5


# ── Helper Functions ─────────────────────────────────────────────────────────

def _unix_ms_to_utc(unix_ms: int) -> datetime:
    """
    Convert a Unix timestamp in milliseconds to a UTC datetime object.

    CoinGecko returns timestamps as milliseconds since Jan 1 1970.
    We convert to UTC datetime so all timestamps in our system are consistent.

    Args:
        unix_ms: Timestamp in milliseconds e.g. 1704067200000

    Returns:
        UTC datetime object e.g. 2024-01-01 00:00:00+00:00
    """
    return datetime.fromtimestamp(unix_ms / 1000, tz=timezone.utc)


def _floor_to_hour(dt: datetime) -> datetime:
    """
    Floor a datetime to the nearest hour by zeroing minutes and seconds.

    CoinGecko timestamps can drift slightly e.g. 14:01:23 instead of 14:00:00.
    Flooring ensures all records align cleanly to the top of each hour.

    Args:
        dt: Any datetime object

    Returns:
        Datetime with minutes, seconds, microseconds set to zero
    """
    return dt.replace(minute=0, second=0, microsecond=0)


# ── Core Fetch Functions ─────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, days: int = 1) -> pd.DataFrame:
    """
    Fetch hourly price, market cap, and volume data for a single asset.

    Uses CoinGecko /coins/{id}/market_chart endpoint.
    Returns hourly granularity when days is between 2 and 90.

    Args:
        symbol: Internal asset symbol e.g. "BTC", "ETH"
        days: Number of days of historical data to fetch (default 1)

    Returns:
        DataFrame with columns:
            asset, timestamp, close, market_cap, volume, source
    """
    if symbol not in ASSET_ID_MAP:
        raise ValueError(
            f"Symbol '{symbol}' not in ASSET_ID_MAP. "
            f"Valid symbols: {list(ASSET_ID_MAP.keys())}"
        )

    coin_id = ASSET_ID_MAP[symbol]
    url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "hourly",
    }

    logger.info(f"Fetching OHLCV for {symbol} ({coin_id}) | days={days}")

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching {symbol} from CoinGecko")
        raise

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching {symbol}: {e}")
        raise

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {symbol}: {e}")
        raise

    prices = data.get("prices", [])
    market_caps = data.get("market_caps", [])
    volumes = data.get("total_volumes", [])

    if not prices:
        logger.warning(f"No price data returned for {symbol}")
        return pd.DataFrame()

    records = []
    for i in range(len(prices)):
        timestamp_ms = prices[i][0]
        records.append({
            "asset":      symbol,
            "timestamp":  _floor_to_hour(_unix_ms_to_utc(timestamp_ms)),
            "close":      float(prices[i][1]),
            "market_cap": float(market_caps[i][1]) if i < len(market_caps) else None,
            "volume":     float(volumes[i][1]) if i < len(volumes) else None,
            "source":     "coingecko",
        })

    df = pd.DataFrame(records)

    logger.success(
        f"Fetched {len(df)} rows for {symbol} | "
        f"Range: {df['timestamp'].min()} to {df['timestamp'].max()}"
    )

    return df


def fetch_global_market_data() -> dict:
    """
    Fetch global cryptocurrency market data from CoinGecko.

    Returns market-wide signals including BTC dominance which is our
    primary regime detection input.

    Returns:
        Dictionary with keys:
            timestamp, btc_dominance, total_market_cap_usd,
            total_volume_usd, source
    """
    url = f"{COINGECKO_BASE_URL}/global"

    logger.info("Fetching global market data from CoinGecko")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json().get("data", {})

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch global market data: {e}")
        raise

    btc_dominance = data.get("market_cap_percentage", {}).get("btc", None)
    total_market_cap = data.get("total_market_cap", {}).get("usd", None)
    total_volume = data.get("total_volume", {}).get("usd", None)

    result = {
        "timestamp":            _floor_to_hour(datetime.now(tz=timezone.utc)),
        "btc_dominance":        btc_dominance,
        "total_market_cap_usd": total_market_cap,
        "total_volume_usd":     total_volume,
        "source":               "coingecko",
    }

    logger.success(
        f"Global data fetched | "
        f"BTC dominance: {btc_dominance:.2f}% | "
        f"Total market cap: ${total_market_cap:,.0f}"
    )

    return result


def fetch_all_assets(days: int = 1) -> pd.DataFrame:
    """
    Fetch OHLCV data for all 10 assets sequentially.

    Respects CoinGecko rate limits by sleeping between each request.
    If one asset fails, the pipeline continues with the remaining assets.

    Args:
        days: Number of days of history to fetch per asset (default 1)

    Returns:
        Combined DataFrame with all 10 assets stacked vertically.
    """
    all_frames = []

    for symbol in ASSET_ID_MAP:
        try:
            df = fetch_ohlcv(symbol=symbol, days=days)
            if not df.empty:
                all_frames.append(df)

        except Exception as e:
            logger.error(f"Failed to fetch {symbol} — skipping. Error: {e}")

        time.sleep(RATE_LIMIT_DELAY)

    if not all_frames:
        logger.error("No data fetched for any asset")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)

    logger.success(
        f"fetch_all_assets complete | "
        f"Total rows: {len(combined)} | "
        f"Assets: {combined['asset'].nunique()}"
    )

    return combined
