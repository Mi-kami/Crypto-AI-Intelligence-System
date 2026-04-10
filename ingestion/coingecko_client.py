"""
ingestion/coingecko_client.py

Responsible for fetching OHLCV-equivalent price data and market signals
from the CoinGecko API for the top 10 cryptocurrency assets.

CoinGecko free tier:
- 30 requests per minute
- 10,000 requests per month
- No authentication required for basic endpoints (API key recommended for stability)
"""

import time
import requests
import pandas as pd
from datetime import datetime, timezone
from loguru import logger
import random


# ── Constants ────────────────────────────────────────────────────────────────

# Base URL for all CoinGecko API calls
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

# Our top 10 assets — CoinGecko uses full coin IDs, not ticker symbols
# This maps our internal symbol (BTC) to CoinGecko's ID (bitcoin)
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

# How many seconds to wait between API calls to respect the 30/min rate limit
# 60 seconds / 30 requests = 6 seconds minimum between calls
# We use 2.5 to give ourselves a small safety buffer
RATE_LIMIT_DELAY = 15.0


# ── Helper Functions ─────────────────────────────────────────────────────────

def _unix_ms_to_utc(unix_ms: int) -> datetime:
    """
    Convert a Unix timestamp in milliseconds to a UTC datetime object.

    CoinGecko returns timestamps as milliseconds since epoch (Jan 1 1970).
    We convert to UTC datetime so all timestamps in our system are consistent
    and human-readable.

    Args:
        unix_ms: Timestamp in milliseconds (e.g. 1704067200000)

    Returns:
        UTC datetime object (e.g. 2024-01-01 00:00:00+00:00)
    """
    return datetime.fromtimestamp(unix_ms / 1000, tz=timezone.utc)


def _floor_to_hour(dt: datetime) -> datetime:
    """
    Floor a datetime to the nearest hour by zeroing out minutes and seconds.

    We do this because CoinGecko's hourly data can have slight timestamp
    drift (e.g. 14:01:23 instead of 14:00:00). Flooring ensures all our
    records align cleanly to the top of each hour for consistent storage
    and querying.

    Args:
        dt: Any datetime object

    Returns:
        Datetime with minutes, seconds, microseconds set to zero
    """
    return dt.replace(minute=0, second=0, microsecond=0)

def _request_with_backoff(url: str, params: dict = None, max_retries: int = 3) -> requests.Response:
    """
    Execute a GET request with exponential backoff and jitter on 429 responses.

    Retries up to max_retries times when rate limited. Each retry waits
    exponentially longer plus random jitter to avoid thundering herd.

    Args:
        url: Full endpoint URL
        params: Query parameters to send with the request
        max_retries: Maximum number of retry attempts (default 3)

    Returns:
        requests.Response object on success

    Raises:
        requests.HTTPError: If max retries exhausted or non-429 HTTP error
        requests.RequestException: On connection or timeout failure
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params or {}, timeout=30)
            response.raise_for_status()
            return response

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    f"Rate limited (429) on attempt {attempt + 1}/{max_retries} — "
                    f"retrying in {wait:.1f}s"
                )
                time.sleep(wait)
                last_exception = e
            else:
                # Non-429 HTTP errors are not retryable — fail immediately
                raise

        except requests.exceptions.RequestException as e:
            last_exception = e
            logger.error(f"Request failed on attempt {attempt + 1}/{max_retries}: {e}")

    logger.error(f"All {max_retries} retry attempts exhausted for {url}")
    raise last_exception

# ── Core Fetch Functions ─────────────────────────────────────────────────────

def fetch_ohlcv(
    symbol: str,
    days: int = 1
) -> pd.DataFrame:
    """
    Fetch hourly price, market cap, and volume data for a single asset.

    Uses CoinGecko's /coins/{id}/market_chart endpoint which returns
    time series data for price, market cap, and volume.

    Note: CoinGecko automatically returns hourly granularity when
    days parameter is between 2 and 90. For days=1 it returns
    5-minute intervals, so we default to days=2 for hourly data
    and filter to the last 24 hours after fetching.

    Args:
        symbol: Internal asset symbol e.g. "BTC", "ETH"
        days: Number of days of historical data to fetch (default 1)

    Returns:
        DataFrame with columns:
            - asset: str (e.g. "BTC")
            - timestamp: datetime (UTC, floored to hour)
            - close: float (price in USD)
            - market_cap: float (total market cap in USD)
            - volume: float (24h trading volume in USD)
            - source: str (always "coingecko")

    Raises:
        ValueError: If symbol is not in ASSET_ID_MAP
        requests.HTTPError: If CoinGecko API returns an error status
    """
    # Validate the symbol exists in our map
    if symbol not in ASSET_ID_MAP:
        raise ValueError(
            f"Symbol '{symbol}' not found in ASSET_ID_MAP. "
            f"Valid symbols: {list(ASSET_ID_MAP.keys())}"
        )

    coin_id = ASSET_ID_MAP[symbol]

    # Build the API URL
    # /market_chart returns prices, market_caps, total_volumes as time series
    url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart"

    # Parameters sent with the request
    params = {
        "vs_currency": "usd",   # We want prices in USD
        "days": days,            # How far back to fetch
        "interval": "hourly",    # Force hourly granularity
    }

    logger.info(f"Fetching OHLCV for {symbol} ({coin_id}) | days={days}")

    try:
        response = _request_with_backoff(url=url, params=params)
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

    # ── Parse the response into a DataFrame ──────────────────────────────────

    # Each of these is a list of [timestamp_ms, value] pairs
    prices      = data.get("prices", [])
    market_caps = data.get("market_caps", [])
    volumes     = data.get("total_volumes", [])

    # If any of the lists are empty, log a warning and return empty DataFrame
    if not prices:
        logger.warning(f"No price data returned for {symbol}")
        return pd.DataFrame()

    # Build the DataFrame row by row
    records = []
    for i in range(len(prices)):
        timestamp_ms = prices[i][0]
        records.append({
            "asset":      symbol,
            "timestamp":  _floor_to_hour(_unix_ms_to_utc(timestamp_ms)),
            "close":      float(prices[i][1]),
            "market_cap": float(market_caps[i][1]) if i < len(market_caps) else None,
            "volume":     float(volumes[i][1])     if i < len(volumes)     else None,
            "source":     "coingecko",
        })

    df = pd.DataFrame(records)

    logger.success(
        f"Fetched {len(df)} rows for {symbol} | "
        f"Range: {df['timestamp'].min()} → {df['timestamp'].max()}"
    )

    return df


def fetch_global_market_data() -> dict:
    """
    Fetch global cryptocurrency market data from CoinGecko.

    This endpoint returns market-wide signals that are the same for all
    assets — most importantly BTC dominance, which is our primary
    regime detection input.

    Returns:
        Dictionary containing:
            - timestamp: datetime (UTC, floored to hour)
            - btc_dominance: float (BTC % of total crypto market cap)
            - total_market_cap_usd: float
            - total_volume_usd: float
            - source: str (always "coingecko")
    """
    url = f"{COINGECKO_BASE_URL}/global"

    logger.info("Fetching global market data from CoinGecko")

    try:
        response = _request_with_backoff(url=url)
        data = response.json().get("data", {})

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch global market data: {e}")
        raise

    # Extract BTC dominance — this is the key regime signal
    # CoinGecko returns it as a percentage e.g. 56.3 means BTC = 56.3% of market
    btc_dominance = data.get(
        "market_cap_percentage", {}
    ).get("btc", None)

    total_market_cap = data.get(
        "total_market_cap", {}
    ).get("usd", None)

    total_volume = data.get(
        "total_volume", {}
    ).get("usd", None)

    result = {
        "timestamp":           _floor_to_hour(datetime.now(tz=timezone.utc)),
        "btc_dominance":       btc_dominance,
        "total_market_cap_usd": total_market_cap,
        "total_volume_usd":    total_volume,
        "source":              "coingecko",
    }

    logger.success(
        f"Global data fetched | BTC dominance: {btc_dominance:.2f}% | "
        f"Total market cap: ${total_market_cap:,.0f}"
    )

    return result


def fetch_all_assets(days: int = 1) -> pd.DataFrame:
    """
    Fetch OHLCV data for all 10 assets sequentially.

    Iterates through every asset in ASSET_ID_MAP, fetches data for each,
    and concatenates into one unified DataFrame. Respects CoinGecko's
    rate limit by sleeping between each request.

    Args:
        days: Number of days of history to fetch per asset (default 1)

    Returns:
        Combined DataFrame with all 10 assets stacked vertically.
        Same schema as fetch_ohlcv() output.
    """
    all_frames = []

    for symbol in ASSET_ID_MAP:
        try:
            df = fetch_ohlcv(symbol=symbol, days=days)

            if not df.empty:
                all_frames.append(df)

        except Exception as e:
            # If one asset fails, log the error but continue with the others
            # We never want one bad asset to break the entire pipeline
            logger.error(f"Failed to fetch {symbol} — skipping. Error: {e}")

        # Wait between requests to respect the 30 requests/minute rate limit
        # This is critical — without this sleep, CoinGecko will return 429
        # (Too Many Requests) and block us temporarily
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