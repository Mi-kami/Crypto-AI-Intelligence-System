"""
tests/conftest.py

Shared pytest fixtures for the ingestion test suite.

Fixtures defined here are automatically available to every test file
in the tests/ directory without any import needed. Think of this file
as the prep bowl — everything is chopped and ready before cooking starts.

Three fixture groups:
    1. Fake API response payloads  — what a real API would return as JSON
    2. Fake parsed DataFrames      — what the client functions return after parsing
    3. Fake global market dict     — what fetch_global_market_data() returns
"""

import pytest
import pandas as pd
from datetime import datetime, timezone


# ── Group 1: Fake Raw API Payloads ────────────────────────────────────────────
# These mimic the raw JSON responses from the external APIs.
# Used when we mock requests.get() to return controlled data.

@pytest.fixture
def fake_coingecko_ohlcv_payload() -> dict:
    """
    Mimics the JSON body returned by CoinGecko /coins/{id}/market_chart.

    Structure: three parallel lists of [timestamp_ms, value] pairs.
    Timestamp is milliseconds since epoch (not seconds).
    """
    return {
        "prices":        [[1713654000000, 65000.0]],
        "market_caps":   [[1713654000000, 1250000000000.0]],
        "total_volumes": [[1713654000000, 45000000000.0]],
    }


@pytest.fixture
def fake_coingecko_global_payload() -> dict:
    """
    Mimics the JSON body returned by CoinGecko /global.

    Structure: nested under a "data" key. BTC dominance is the
    primary regime detection signal we extract from this endpoint.
    """
    return {
        "data": {
            "market_cap_percentage": {"btc": 56.3},
            "total_market_cap":      {"usd": 2_500_000_000_000.0},
            "total_volume":          {"usd": 95_000_000_000.0},
        }
    }


@pytest.fixture
def fake_cryptocompare_payload() -> dict:
    """
    Mimics the JSON body returned by CryptoCompare /news/v1/article/list.

    Structure: articles wrapped in a "Data" key (uppercase D).
    All field names uppercase: TITLE, URL, PUBLISHED_ON.
    """
    return {
        "Data": [
            {
                "TITLE":        "Bitcoin surges past 100k",
                "URL":          "https://example.com/btc-article",
                "PUBLISHED_ON": 1713654000,
            }
        ]
    }


@pytest.fixture
def fake_cryptocompare_empty_payload() -> dict:
    """Mimics a valid response with no articles — e.g. no news for SHIB today."""
    return {"Data": []}


# ── Group 2: Fake Parsed DataFrames ──────────────────────────────────────────
# These mimic what the client functions return AFTER parsing the API response.
# Used to test harmoniser functions without running the full ingestion chain.

@pytest.fixture
def fake_price_df() -> pd.DataFrame:
    """
    Mimics the output of fetch_ohlcv() or fetch_all_assets().
    Matches PRICE_REQUIRED_COLUMNS exactly.
    """
    return pd.DataFrame([{
        "asset":      "BTC",
        "timestamp":  datetime(2024, 4, 21, 0, 0, 0, tzinfo=timezone.utc),
        "close":      65000.0,
        "market_cap": 1_250_000_000_000.0,
        "volume":     45_000_000_000.0,
        "source":     "coingecko",
    }])


@pytest.fixture
def fake_market_dict() -> dict:
    """
    Mimics the output of fetch_global_market_data().
    Matches MARKET_REQUIRED_COLUMNS exactly.
    """
    return {
        "timestamp":            datetime(2024, 4, 21, 0, 0, 0, tzinfo=timezone.utc),
        "btc_dominance":        56.3,
        "total_market_cap_usd": 2_500_000_000_000.0,
        "total_volume_usd":     95_000_000_000.0,
        "source":               "coingecko",
    }


@pytest.fixture
def fake_news_df() -> pd.DataFrame:
    """
    Mimics the output of fetch_news_for_asset() or fetch_all_assets_news().
    Uses published_at (pre-rename) — as it comes out of the client.
    Matches what harmonise_news_data() expects BEFORE the rename step.
    """
    return pd.DataFrame([{
        "asset":          "BTC",
        "headline":       "Bitcoin surges past 100k",
        "url":            "https://example.com/btc-article",
        "published_at":   datetime(2024, 4, 21, 0, 5, 0, tzinfo=timezone.utc),
        "votes_positive": 0,
        "votes_negative": 0,
        "source":         "cryptocompare",
    }])


# ── Group 3: Edge Case DataFrames ─────────────────────────────────────────────
# Used to test that harmoniser functions reject bad input correctly.

@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Completely empty DataFrame — tests early-exit paths in harmonisers."""
    return pd.DataFrame()


@pytest.fixture
def price_df_missing_column() -> pd.DataFrame:
    """
    Price DataFrame with the 'close' column removed.
    Used to verify harmonise_price_data raises ValueError on missing columns.
    """
    return pd.DataFrame([{
        "asset":      "BTC",
        "timestamp":  datetime(2024, 4, 21, 0, 0, 0, tzinfo=timezone.utc),
        # 'close' deliberately omitted
        "market_cap": 1_250_000_000_000.0,
        "volume":     45_000_000_000.0,
        "source":     "coingecko",
    }])


@pytest.fixture
def news_df_missing_published_at() -> pd.DataFrame:
    """
    News DataFrame where published_at column is missing entirely.
    After the rename step, timestamp won't exist either → ValueError.
    """
    return pd.DataFrame([{
        "asset":          "BTC",
        "headline":       "Some headline",
        "url":            "https://example.com/article",
        # 'published_at' deliberately omitted
        "votes_positive": 0,
        "votes_negative": 0,
        "source":         "cryptocompare",
    }])