"""
ingestion/cryptopanic_client.py

Responsible for fetching cryptocurrency news headlines from the CryptoPanic API.
Headlines are used as input to the FinBERT sentiment analysis model.

CryptoPanic free tier:
- Limited requests per day
- Returns news with community sentiment votes (bullish/bearish)
- Supports filtering by cryptocurrency
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

# Root URL for all CryptoPanic API calls
CRYPTOPANIC_BASE_URL = "https://cryptopanic.com/api/developer/v2"

# Our API authentication token loaded from .env
# Never hardcode this — always load from environment
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY")

# Maps our internal symbol to CryptoPanic's currency filter format
# CryptoPanic uses ticker symbols directly — BTC, ETH etc.
# This matches our internal naming so no translation needed
ASSET_SYMBOLS = [
    "BTC", "ETH", "BNB", "XRP", "SOL",
    "DOGE", "ADA", "TRX", "AVAX", "SHIB"
]

# Seconds to wait between API calls to avoid hitting rate limits
RATE_LIMIT_DELAY = 2.0

# Maximum number of news posts to fetch per asset per call
# CryptoPanic returns up to 20 per page on free tier
MAX_POSTS_PER_ASSET = 20


# ── Helper Functions ─────────────────────────────────────────────────────────

def _parse_timestamp(timestamp_str: str) -> datetime:
    """
    Parse a CryptoPanic ISO 8601 timestamp string into a UTC datetime object.

    CryptoPanic returns timestamps as strings like: "2024-01-15T14:32:00Z"
    The Z at the end means UTC. We parse this into a proper datetime object
    so it is consistent with the rest of our system.

    Args:
        timestamp_str: ISO 8601 timestamp string e.g. "2024-01-15T14:32:00Z"

    Returns:
        UTC datetime object
    """
    # Remove the Z suffix and parse — then attach UTC timezone
    clean = timestamp_str.replace("Z", "+00:00")
    return datetime.fromisoformat(clean).astimezone(timezone.utc)


def _extract_vote_counts(votes: dict) -> tuple:
    """
    Extract positive and negative vote counts from CryptoPanic votes dictionary.

    CryptoPanic returns votes as a dictionary with keys like:
    positive, negative, important, liked, disliked, lol, toxic, saved, comments

    We only care about positive and negative for our sentiment weighting.

    Args:
        votes: Dictionary of vote counts from CryptoPanic response

    Returns:
        Tuple of (positive_votes, negative_votes) as integers
    """
    if not votes or not isinstance(votes, dict):
        return 0, 0

    positive = votes.get("positive", 0) or 0
    negative = votes.get("negative", 0) or 0

    return int(positive), int(negative)


# ── Core Fetch Functions ─────────────────────────────────────────────────────

def fetch_news_for_asset(symbol: str) -> pd.DataFrame:
    """
    Fetch recent news headlines for a single cryptocurrency asset.

    Calls CryptoPanic /posts/ endpoint filtered by currency symbol.
    Returns headlines with community vote counts for sentiment weighting.

    Args:
        symbol: Asset symbol e.g. "BTC", "ETH"

    Returns:
        DataFrame with columns:
            asset       - str: which asset this news is about
            headline    - str: the news headline text
            url         - str: link to the full article (used for deduplication)
            published_at - datetime: when the article was published (UTC)
            votes_positive - int: number of bullish community votes
            votes_negative - int: number of bearish community votes
            source      - str: always "cryptopanic"

        Returns empty DataFrame if no news found or API call fails.
    """
    if not CRYPTOPANIC_API_KEY:
        raise EnvironmentError(
            "CRYPTOPANIC_API_KEY not found in environment variables. "
            "Make sure it is set in your .env file."
        )

    url = f"{CRYPTOPANIC_BASE_URL}/posts/"

    # Parameters for the API request
    params = {
        "auth_token": CRYPTOPANIC_API_KEY,  # Authentication
        "currencies": symbol,               # Filter news to this asset only
        "public": "true",                   # Only return public posts
    }

    logger.info(f"Fetching news for {symbol} from CryptoPanic")

    try:
        response = requests.get(url, params=params, timeout=30)
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

    # Extract the list of news posts from the response
    # CryptoPanic wraps results in a "results" key
    posts = data.get("results", [])

    if not posts:
        logger.warning(f"No news found for {symbol}")
        return pd.DataFrame()

    # ── Parse each post into a structured record ──────────────────────────

    records = []
    for post in posts:
        # Extract the headline title
        headline = post.get("title", "")

        # Skip posts with no headline — they are useless for sentiment
        if not headline:
            continue

        # v2 API uses 'domain' instead of 'url'
        # We construct a unique identifier from title for deduplication
        domain = post.get("domain", "")

        # Extract and parse the publication timestamp
        published_raw = post.get("published_at", "")
        try:
            published_at = _parse_timestamp(published_raw)
        except (ValueError, AttributeError):
            # If timestamp parsing fails, use current time as fallback
            published_at = datetime.now(tz=timezone.utc)

        records.append({
            "asset":           symbol,
            "headline":        headline,
            "url":             domain,
            "published_at":    published_at,
            "votes_positive":  0,
            "votes_negative":  0,
            "source":          "cryptopanic",
        })

    if not records:
        logger.warning(f"No valid posts parsed for {symbol}")
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
        Duplicates are removed based on URL — same story tagged for
        multiple assets only appears once per asset.
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
