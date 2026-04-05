"""
smoke_test_harmoniser.py

Manual smoke test for ingestion/harmoniser.py

Verifies all three public functions behave correctly against:
  - Happy path: valid data passes through clean
  - Missing column: ValueError raised immediately
  - Invalid asset: row filtered out with warning logged
  - Timestamp flooring: messy timestamps floored to the hour

Run from project root:
    python smoke_test_harmoniser.py
"""

import pandas as pd
from datetime import datetime, timezone
from loguru import logger

from ingestion.harmoniser import (
    harmonise_price_data,
    harmonise_market_data,
    harmonise_news_data,
)


# ── Shared Helpers ────────────────────────────────────────────────────────────

def section(title: str) -> None:
    """Print a section divider to make test output readable."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def check(label: str, condition: bool) -> None:
    """Print a PASS or FAIL line for a single assertion."""
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"  {status} — {label}")


# ── Mock Data Builders ────────────────────────────────────────────────────────

def build_price_df() -> pd.DataFrame:
    """
    21 rows of mock OHLCV price data.

    Includes:
    - Messy timestamps (not on the hour) — should be floored
    - 1 invalid asset symbol (LUNA) — should be filtered out
    - All 6 required columns present
    """
    records = [
        # BTC — messy timestamps
        {"asset": "BTC",  "timestamp": "2024-01-01 02:05:00+00:00", "close": 42000.0, "market_cap": 820e9, "volume": 18e9, "source": "coingecko"},
        {"asset": "BTC",  "timestamp": "2024-01-01 02:10:00+00:00", "close": 42100.0, "market_cap": 821e9, "volume": 18e9, "source": "coingecko"},
        {"asset": "BTC",  "timestamp": "2024-01-01 02:15:00+00:00", "close": 42200.0, "market_cap": 822e9, "volume": 18e9, "source": "coingecko"},
        {"asset": "BTC",  "timestamp": "2024-01-01 02:17:00+00:00", "close": 42150.0, "market_cap": 821e9, "volume": 18e9, "source": "coingecko"},
        {"asset": "BTC",  "timestamp": "2024-01-01 02:19:00+00:00", "close": 42300.0, "market_cap": 823e9, "volume": 18e9, "source": "coingecko"},
        # ETH — more messy timestamps
        {"asset": "ETH",  "timestamp": "2024-01-01 05:05:00+00:00", "close": 2200.0,  "market_cap": 265e9, "volume": 9e9,  "source": "coingecko"},
        {"asset": "ETH",  "timestamp": "2024-01-01 05:16:00+00:00", "close": 2210.0,  "market_cap": 266e9, "volume": 9e9,  "source": "coingecko"},
        {"asset": "ETH",  "timestamp": "2024-01-01 05:18:00+00:00", "close": 2205.0,  "market_cap": 265e9, "volume": 9e9,  "source": "coingecko"},
        # SOL
        {"asset": "SOL",  "timestamp": "2024-01-01 08:03:00+00:00", "close": 98.5,    "market_cap": 42e9,  "volume": 2e9,  "source": "coingecko"},
        {"asset": "SOL",  "timestamp": "2024-01-01 08:11:00+00:00", "close": 99.0,    "market_cap": 42e9,  "volume": 2e9,  "source": "coingecko"},
        # XRP
        {"asset": "XRP",  "timestamp": "2024-01-01 09:07:00+00:00", "close": 0.62,    "market_cap": 34e9,  "volume": 1e9,  "source": "coingecko"},
        {"asset": "XRP",  "timestamp": "2024-01-01 09:22:00+00:00", "close": 0.63,    "market_cap": 34e9,  "volume": 1e9,  "source": "coingecko"},
        # BNB
        {"asset": "BNB",  "timestamp": "2024-01-01 10:14:00+00:00", "close": 310.0,   "market_cap": 47e9,  "volume": 1.5e9,"source": "coingecko"},
        # DOGE
        {"asset": "DOGE", "timestamp": "2024-01-01 11:08:00+00:00", "close": 0.088,   "market_cap": 12e9,  "volume": 0.5e9,"source": "coingecko"},
        # ADA
        {"asset": "ADA",  "timestamp": "2024-01-01 12:33:00+00:00", "close": 0.59,    "market_cap": 21e9,  "volume": 0.6e9,"source": "coingecko"},
        # TRX
        {"asset": "TRX",  "timestamp": "2024-01-01 13:45:00+00:00", "close": 0.11,    "market_cap": 9e9,   "volume": 0.4e9,"source": "coingecko"},
        # AVAX
        {"asset": "AVAX", "timestamp": "2024-01-01 14:02:00+00:00", "close": 37.5,    "market_cap": 15e9,  "volume": 0.7e9,"source": "coingecko"},
        # SHIB
        {"asset": "SHIB", "timestamp": "2024-01-01 15:29:00+00:00", "close": 0.000009,"market_cap": 5e9,   "volume": 0.3e9,"source": "coingecko"},
        # Two more valid rows so we hit 20 valid rows
        {"asset": "BTC",  "timestamp": "2024-01-01 16:44:00+00:00", "close": 42500.0, "market_cap": 825e9, "volume": 18e9, "source": "coingecko"},
        {"asset": "ETH",  "timestamp": "2024-01-01 17:55:00+00:00", "close": 2250.0,  "market_cap": 270e9, "volume": 9e9,  "source": "coingecko"},
        # Row 21 — INVALID ASSET — should be filtered out
        {"asset": "LUNA", "timestamp": "2024-01-01 18:00:00+00:00", "close": 0.0001,  "market_cap": 0.0,   "volume": 0.0,  "source": "coingecko"},
    ]
    return pd.DataFrame(records)


def build_market_dict() -> dict:
    """
    Single mock global market dictionary.
    Simulates what fetch_global_market_data() returns.
    """
    return {
        "timestamp":            "2024-01-01 03:47:00+00:00",  # messy — should be floored
        "btc_dominance":        52.3,
        "total_market_cap_usd": 1.65e12,
        "total_volume_usd":     87e9,
        "source":               "coingecko",
    }


def build_news_df() -> pd.DataFrame:
    """
    Mock news DataFrame with published_at column (not yet renamed).
    Includes 1 invalid asset (PEPE) and messy timestamps.
    """
    records = [
        {"asset": "BTC",  "headline": "Bitcoin breaks $42k resistance",       "url": "domain1.com", "published_at": "2024-01-01 02:07:00+00:00", "votes_positive": 12, "votes_negative": 2,  "source": "cryptopanic"},
        {"asset": "ETH",  "headline": "Ethereum gas fees hit monthly low",     "url": "domain2.com", "published_at": "2024-01-01 05:13:00+00:00", "votes_positive": 8,  "votes_negative": 1,  "source": "cryptopanic"},
        {"asset": "SOL",  "headline": "Solana network upgrade scheduled",      "url": "domain3.com", "published_at": "2024-01-01 08:44:00+00:00", "votes_positive": 5,  "votes_negative": 0,  "source": "cryptopanic"},
        {"asset": "XRP",  "headline": "Ripple lawsuit update expected soon",   "url": "domain4.com", "published_at": "2024-01-01 09:31:00+00:00", "votes_positive": 20, "votes_negative": 5,  "source": "cryptopanic"},
        {"asset": "BNB",  "headline": "Binance announces new product feature", "url": "domain5.com", "published_at": "2024-01-01 10:52:00+00:00", "votes_positive": 3,  "votes_negative": 3,  "source": "cryptopanic"},
        # Invalid asset — should be filtered out
        {"asset": "PEPE", "headline": "PEPE meme coin surges 40% overnight",  "url": "domain6.com", "published_at": "2024-01-01 11:19:00+00:00", "votes_positive": 30, "votes_negative": 0,  "source": "cryptopanic"},
    ]
    return pd.DataFrame(records)


# ── Test Functions ────────────────────────────────────────────────────────────

def test_harmonise_price_data() -> None:
    section("TEST 1 — harmonise_price_data()")

    # ── Happy path ────────────────────────────────────────────────────────
    print("\n[Happy Path]")
    df = build_price_df()
    result = harmonise_price_data(df)

    check("Returns a DataFrame",             isinstance(result, pd.DataFrame))
    check("LUNA row filtered out (20 rows)", len(result) == 20)
    check("All assets valid",                result["asset"].isin(["BTC","ETH","BNB","XRP","SOL","DOGE","ADA","TRX","AVAX","SHIB"]).all())
    check("All timestamps on the hour",      (result["timestamp"].dt.minute == 0).all())
    check("All timestamps on the hour (seconds also zero)", (result["timestamp"].dt.second == 0).all())

    # ── Missing column ────────────────────────────────────────────────────
    print("\n[Missing Column — should raise ValueError]")
    df_missing = build_price_df().drop(columns=["volume"])
    try:
        harmonise_price_data(df_missing)
        check("ValueError raised for missing 'volume'", False)  # should never reach here
    except ValueError as e:
        check("ValueError raised for missing 'volume'", True)
        print(f"    Error message: {e}")


def test_harmonise_market_data() -> None:
    section("TEST 2 — harmonise_market_data()")

    # ── Happy path ────────────────────────────────────────────────────────
    print("\n[Happy Path]")
    global_dict = build_market_dict()
    result = harmonise_market_data(global_dict)

    check("Returns a DataFrame",             isinstance(result, pd.DataFrame))
    check("Exactly 1 row",                   len(result) == 1)
    check("btc_dominance present",           "btc_dominance" in result.columns)
    check("Timestamp floored to the hour",   result["timestamp"].iloc[0].minute == 0)

    # ── Missing column ────────────────────────────────────────────────────
    print("\n[Missing Column — should raise ValueError]")
    bad_dict = {k: v for k, v in global_dict.items() if k != "btc_dominance"}
    try:
        harmonise_market_data(bad_dict)
        check("ValueError raised for missing 'btc_dominance'", False)
    except ValueError as e:
        check("ValueError raised for missing 'btc_dominance'", True)
        print(f"    Error message: {e}")


def test_harmonise_news_data() -> None:
    section("TEST 3 — harmonise_news_data()")

    # ── Happy path ────────────────────────────────────────────────────────
    print("\n[Happy Path]")
    df = build_news_df()
    result = harmonise_news_data(df)

    check("Returns a DataFrame",              isinstance(result, pd.DataFrame))
    check("PEPE row filtered out (5 rows)",   len(result) == 5)
    check("published_at renamed to timestamp","timestamp" in result.columns and "published_at" not in result.columns)
    check("All timestamps on the hour",       (result["timestamp"].dt.minute == 0).all())

    # ── Missing column ────────────────────────────────────────────────────
    print("\n[Missing Column — should raise ValueError]")
    df_missing = build_news_df().drop(columns=["headline"])
    try:
        harmonise_news_data(df_missing)
        check("ValueError raised for missing 'headline'", False)
    except ValueError as e:
        check("ValueError raised for missing 'headline'", True)
        print(f"    Error message: {e}")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🧪 HARMONISER SMOKE TEST")
    print("Running all tests...\n")

    test_harmonise_price_data()
    test_harmonise_market_data()
    test_harmonise_news_data()

    print(f"\n{'─' * 60}")
    print("  Smoke test complete. Review any ❌ FAIL lines above.")
    print(f"{'─' * 60}\n")
