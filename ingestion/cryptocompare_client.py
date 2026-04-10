"""
ingestion/cryptocompare_client.py

Responsible for fetching cryptocurrency news headlines from the CoinDesk
(CryptoCompare) News API. Headlines are used as input to the FinBERT
sentiment analysis model.

CoinDesk / CryptoCompare free tier:
- 11,000 calls per month
- No community voting system (votes_positive and votes_negative default to 0)
- Supports filtering by asset category (BTC, ETH, etc.)
- Authentication via Authorization header (not query param)

Replaces: ingestion/cryptopanic_client.py
Reason: CryptoPanic discontinued free developer API access from April 1 2026.
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from loguru import logger
from dotenv import load_dotenv

# Load environment variables from .env file
# This is how we access the API key without hardcoding it
load_dotenv()


# ── Constants ────────────────────────────────────────────────────────────────

# Base URL for all CoinDesk / CryptoCompare API calls
CRYPTOCOMPARE_BASE_URL = "https://data-api.cryptocompare.com"

# Our API authentication token loaded from .env
# Never hardcode this — always load from environment
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")

# Asset symbols we track — used as category filters in the API request
# CryptoCompare accepts ticker symbols directly: BTC, ETH, etc.
ASSET_SYMBOLS = [
    "BTC", "ETH", "BNB", "XRP", "SOL",
    "DOGE", "ADA", "TRX", "AVAX", "SHIB"
]

# Seconds to wait between API calls to avoid hitting rate limits
RATE_LIMIT_DELAY = 2.0

# Maximum number of articles to fetch per asset per call
# CryptoCompare returns up to 20 per page on free tier
MAX_ARTICLES_PER_ASSET = 20


# ── Helper Functions ─────────────────────────────────────────────────────────

def _parse_timestamp(unix_timestamp: int) -> datetime:
    """
    Convert a CryptoCompare Unix timestamp integer to a UTC datetime object.

    CryptoCompare returns timestamps as Unix integers (seconds since epoch),
    e.g. 1713654000. We convert to UTC datetime so all timestamps in our
    system are consistent and human-readable.

    This differs from the old CryptoPanic client which returned ISO 8601
    strings. CryptoCompare always returns integers.

    Args:
        unix_timestamp: Unix timestamp in seconds e.g. 1713654000

    Returns:
        UTC datetime object e.g. 2024-04-20 21:00:00+00:00
    """
    return datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)


def _extract_vote_counts(votes: dict) -> tuple:
    """
    Extract positive and negative vote counts from a votes dictionary.

    Retained for forward compatibility — CryptoCompare does not provide
    community voting. If a paid API with voting data is used in future,
    this function handles the parsing without any upstream changes needed.

    Args:
        votes: Dictionary of vote counts

    Returns:
        Tuple of (positive_votes, negative_votes) as integers
    """
    if not votes or not isinstance(votes, dict):
        return 0, 0

    positive = votes.get("UPVOTES", 0) or 0
    negative = votes.get("DOWNVOTES", 0) or 0

    return int(positive), int(negative)


# ── Core Fetch Functions ─────────────────────────────────────────────────────

def fetch_news_for_asset(symbol: str) -> pd.DataFrame:
    """
    Fetch recent news headlines for a single cryptocurrency asset.

    Calls the CoinDesk /news/v1/article/list endpoint filtered by asset
    category. Returns headlines ready for FinBERT sentiment inference.

    Note: CryptoCompare does not provide community vote scores. Both
    votes_positive and votes_negative are set to 0. This is consistent
    with the final state of the CryptoPanic v2 client.

    Args:
        symbol: Asset symbol e.g. "BTC", "ETH"

    Returns:
        DataFrame with columns:
            asset          - str: which asset this news is about
            headline       - str: the news headline text
            url            - str: link to the full article (used for deduplication)
            published_at   - datetime: when the article was published (UTC)
            votes_positive - int: always 0 (not available on this API)
            votes_negative - int: always 0 (not available on this API)
            source         - str: always "cryptocompare"

        Returns empty DataFrame if no news found or API call fails.
    """
    if not CRYPTOCOMPARE_API_KEY:
        raise EnvironmentError(
            "CRYPTOCOMPARE_API_KEY not found in environment variables. "
            "Make sure it is set in your .env file."
        )

    endpoint = f"{CRYPTOCOMPARE_BASE_URL}/news/v1/article/list"

    # CoinDesk API authenticates via Authorization header, not query param
    headers = {
        "Authorization": f"Apikey {CRYPTOCOMPARE_API_KEY}",
    }

    # Parameters for the API request
    params = {
        "categories": symbol,           # Filter articles to this asset only
        "limit":      MAX_ARTICLES_PER_ASSET,
    }

    logger.info(f"Fetching news for {symbol} from CryptoCompare")

    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching news for {symbol}")
        return pd.DataFrame()

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching news for {symbol}: {e}")
        return pd.DataFrame()

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {symbol}: {e}")
        return pd.DataFrame()

    # CryptoCompare wraps results in a "Data" key (not "results" like CryptoPanic)
    articles = data.get("Data", [])

    if not articles:
        logger.warning(f"No news found for {symbol}")
        return pd.DataFrame()

    # ── Parse each article into a structured record ───────────────────────

    records = []
    for article in articles:
        # Extract the headline
        headline = article.get("TITLE", "")

        # Skip articles with no headline — useless for sentiment
        if not headline:
            continue

        # CryptoCompare returns a proper url field (unlike CryptoPanic v2
        # which switched to domain — we are back to full URLs here)
        article_url = article.get("URL", "")

        # CryptoCompare returns timestamps as Unix integers (seconds),
        # not ISO strings — _parse_timestamp handles the conversion
        published_unix = article.get("PUBLISHED_ON", 0)
        try:
            published_at = _parse_timestamp(published_unix)
        except (ValueError, TypeError, OSError):
            # If timestamp parsing fails, fall back to current time
            published_at = datetime.now(tz=timezone.utc)

        records.append({
            "asset":           symbol,
            "headline":        headline,
            "url":             article_url,
            "published_at":    published_at,
            "votes_positive":  0,
            "votes_negative":  0,
            "source":          "cryptocompare",
        })

    if not records:
        logger.warning(f"No valid articles parsed for {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    logger.success(
        f"Fetched {len(df)} headlines for {symbol} | "
        f"Latest: {df['published_at'].max()}"
    )

    return df


def fetch_all_assets_news() -> pd.DataFrame:
    """
    Fetch news headlines for all 10 assets sequentially.

    Iterates through every asset, fetches news, and combines into one
    unified DataFrame. Respects rate limits by sleeping between requests.

    If one asset fails, the pipeline continues with the remaining assets.

    Returns:
        Combined DataFrame with news for all assets stacked vertically.
        Duplicates are removed based on URL and asset — the same article
        tagged for multiple assets appears once per asset it is relevant to.
    """
    all_frames = []

    for symbol in ASSET_SYMBOLS:
        try:
            df = fetch_news_for_asset(symbol=symbol)

            if not df.empty:
                all_frames.append(df)

        except Exception as e:
            # One failed asset must never stop the rest of the pipeline
            logger.error(f"Failed to fetch news for {symbol} — skipping. Error: {e}")

        # Wait between requests to respect rate limits
        time.sleep(RATE_LIMIT_DELAY)

    if not all_frames:
        logger.error("No news fetched for any asset")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)

    # Remove duplicate headlines — same URL appearing for the same asset twice
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["asset", "url"])
    after_dedup = len(combined)

    if before_dedup != after_dedup:
        logger.info(
            f"Removed {before_dedup - after_dedup} duplicate headlines"
        )

    logger.success(
        f"fetch_all_assets_news complete | "
        f"Total headlines: {len(combined)} | "
        f"Assets covered: {combined['asset'].nunique()}"
    )

    return combined