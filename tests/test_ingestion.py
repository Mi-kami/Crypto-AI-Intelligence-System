"""
tests/test_ingestion.py

Pytest suite for Phase 1 ingestion pipeline.

Covers three modules:
    - ingestion/coingecko_client.py
    - ingestion/cryptocompare_client.py
    - ingestion/harmoniser.py

No real API calls are made. All HTTP responses are mocked using
unittest.mock.patch so tests run offline, instantly, and deterministically.

Run from project root:
    pytest tests/test_ingestion.py -v
"""

import pytest
import requests as req
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from ingestion.coingecko_client import fetch_ohlcv, fetch_global_market_data
from ingestion.cryptocompare_client import fetch_news_for_asset
from ingestion.harmoniser import (
    harmonise_price_data,
    harmonise_market_data,
    harmonise_news_data,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CoinGecko Client Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFetchOHLCV:
    """Tests for fetch_ohlcv() — single asset price fetcher."""

    def test_happy_path_returns_correct_schema(self, fake_coingecko_ohlcv_payload):
        """Valid response → DataFrame with all required columns and correct values."""
        mock_response = MagicMock()
        mock_response.json.return_value = fake_coingecko_ohlcv_payload

        with patch("ingestion.coingecko_client._request_with_backoff", return_value=mock_response):
            df = fetch_ohlcv("BTC", days=1)

        assert not df.empty
        assert set(df.columns) == {"asset", "timestamp", "close", "market_cap", "volume", "source"}
        assert df["asset"].iloc[0] == "BTC"
        assert df["close"].iloc[0] == 65000.0
        assert df["market_cap"].iloc[0] == 1_250_000_000_000.0
        assert df["volume"].iloc[0] == 45_000_000_000.0
        assert df["source"].iloc[0] == "coingecko"

    def test_timestamp_is_utc_datetime(self, fake_coingecko_ohlcv_payload):
        """Timestamp column must be a UTC-aware datetime object, not a raw int."""
        mock_response = MagicMock()
        mock_response.json.return_value = fake_coingecko_ohlcv_payload

        with patch("ingestion.coingecko_client._request_with_backoff", return_value=mock_response):
            df = fetch_ohlcv("BTC", days=1)

        ts = df["timestamp"].iloc[0]
        assert isinstance(ts, datetime)
        assert ts.tzinfo is not None

    def test_invalid_symbol_raises_value_error(self):
        """Unknown symbol must raise ValueError immediately — no API call made."""
        with pytest.raises(ValueError, match="not found in ASSET_ID_MAP"):
            fetch_ohlcv("INVALID_COIN")

    def test_empty_prices_returns_empty_dataframe(self):
        """API returns empty prices list → function returns empty DataFrame."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "prices": [], "market_caps": [], "total_volumes": []
        }

        with patch("ingestion.coingecko_client._request_with_backoff", return_value=mock_response):
            df = fetch_ohlcv("ETH", days=1)

        assert df.empty

    def test_http_error_propagates(self):
        """Non-429 HTTP error must propagate — not be swallowed silently."""
        with patch(
            "ingestion.coingecko_client._request_with_backoff",
            side_effect=req.exceptions.HTTPError("403 Forbidden")
        ):
            with pytest.raises(req.exceptions.HTTPError):
                fetch_ohlcv("BTC", days=1)


class TestFetchGlobalMarketData:
    """Tests for fetch_global_market_data() — market-wide signals."""

    def test_happy_path_returns_correct_fields(self, fake_coingecko_global_payload):
        """Valid response → dict with all required keys and correct values."""
        mock_response = MagicMock()
        mock_response.json.return_value = fake_coingecko_global_payload

        with patch("ingestion.coingecko_client._request_with_backoff", return_value=mock_response):
            result = fetch_global_market_data()

        assert result["btc_dominance"] == 56.3
        assert result["total_market_cap_usd"] == 2_500_000_000_000.0
        assert result["total_volume_usd"] == 95_000_000_000.0
        assert result["source"] == "coingecko"

    def test_timestamp_is_utc_datetime(self, fake_coingecko_global_payload):
        """Timestamp must be a UTC-aware datetime — floored to the hour."""
        mock_response = MagicMock()
        mock_response.json.return_value = fake_coingecko_global_payload

        with patch("ingestion.coingecko_client._request_with_backoff", return_value=mock_response):
            result = fetch_global_market_data()

        assert isinstance(result["timestamp"], datetime)
        assert result["timestamp"].tzinfo is not None
        assert result["timestamp"].minute == 0
        assert result["timestamp"].second == 0


# ═══════════════════════════════════════════════════════════════════════════════
# CryptoCompare Client Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFetchNewsForAsset:
    """Tests for fetch_news_for_asset() — single asset news fetcher."""

    def test_happy_path_returns_correct_schema(self, fake_cryptocompare_payload):
        """Valid response → DataFrame with all required columns and correct values."""
        mock_response = MagicMock()
        mock_response.json.return_value = fake_cryptocompare_payload
        mock_response.raise_for_status.return_value = None

        with patch("ingestion.cryptocompare_client.requests.get", return_value=mock_response):
            df = fetch_news_for_asset("BTC")

        assert not df.empty
        assert df["headline"].iloc[0] == "Bitcoin surges past 100k"
        assert df["url"].iloc[0] == "https://example.com/btc-article"
        assert df["asset"].iloc[0] == "BTC"
        assert df["source"].iloc[0] == "cryptocompare"

    def test_votes_always_zero(self, fake_cryptocompare_payload):
        """CryptoCompare free tier has no voting — both vote fields must be 0."""
        mock_response = MagicMock()
        mock_response.json.return_value = fake_cryptocompare_payload
        mock_response.raise_for_status.return_value = None

        with patch("ingestion.cryptocompare_client.requests.get", return_value=mock_response):
            df = fetch_news_for_asset("BTC")

        assert df["votes_positive"].iloc[0] == 0
        assert df["votes_negative"].iloc[0] == 0

    def test_missing_api_key_raises_environment_error(self, monkeypatch):
        """No API key → EnvironmentError before any HTTP call is made."""
        monkeypatch.setattr("ingestion.cryptocompare_client.CRYPTOCOMPARE_API_KEY", None)

        with pytest.raises(EnvironmentError):
            fetch_news_for_asset("BTC")

    def test_empty_data_key_returns_empty_dataframe(self, fake_cryptocompare_empty_payload):
        """API returns Data: [] → empty DataFrame, no crash."""
        mock_response = MagicMock()
        mock_response.json.return_value = fake_cryptocompare_empty_payload
        mock_response.raise_for_status.return_value = None

        with patch("ingestion.cryptocompare_client.requests.get", return_value=mock_response):
            df = fetch_news_for_asset("ETH")

        assert df.empty

    def test_http_error_returns_empty_dataframe(self):
        """500 or 429 → function returns empty DataFrame, does not raise."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = req.exceptions.HTTPError("500 Server Error")

        with patch("ingestion.cryptocompare_client.requests.get", return_value=mock_response):
            df = fetch_news_for_asset("SOL")

        assert df.empty

    def test_missing_title_article_is_skipped(self):
        """Article with empty TITLE must be skipped — useless for FinBERT."""
        payload = {
            "Data": [
                {"TITLE": "", "URL": "https://example.com/no-title", "PUBLISHED_ON": 1713654000}
            ]
        }
        mock_response = MagicMock()
        mock_response.json.return_value = payload
        mock_response.raise_for_status.return_value = None

        with patch("ingestion.cryptocompare_client.requests.get", return_value=mock_response):
            df = fetch_news_for_asset("BTC")

        assert df.empty

    def test_published_at_is_utc_datetime(self, fake_cryptocompare_payload):
        """published_at must be a UTC-aware datetime — not a raw Unix int."""
        mock_response = MagicMock()
        mock_response.json.return_value = fake_cryptocompare_payload
        mock_response.raise_for_status.return_value = None

        with patch("ingestion.cryptocompare_client.requests.get", return_value=mock_response):
            df = fetch_news_for_asset("BTC")

        ts = df["published_at"].iloc[0]
        assert isinstance(ts, datetime)
        assert ts.tzinfo is not None

    def test_timeout_returns_empty_dataframe(self):
        """Network timeout → function returns empty DataFrame, does not raise."""
        with patch(
            "ingestion.cryptocompare_client.requests.get",
            side_effect=req.exceptions.Timeout()
        ):
            df = fetch_news_for_asset("ADA")

        assert df.empty


# ═══════════════════════════════════════════════════════════════════════════════
# Harmoniser Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestHarmonisePriceData:
    """Tests for harmonise_price_data() — validates and normalises OHLCV data."""

    def test_happy_path_passes_through_correctly(self, fake_price_df):
        """Valid DataFrame → returns clean DataFrame with same row count."""
        result = harmonise_price_data(fake_price_df)
        assert not result.empty
        assert len(result) == len(fake_price_df)

    def test_missing_column_raises_value_error(self, price_df_missing_column):
        """DataFrame missing 'close' → ValueError with informative message."""
        with pytest.raises(ValueError, match="Missing required columns"):
            harmonise_price_data(price_df_missing_column)

    def test_invalid_asset_is_filtered_out(self, fake_price_df):
        """Asset not in VALID_ASSETS → row dropped silently."""
        df = fake_price_df.copy()
        df["asset"] = "SHITCOIN"
        result = harmonise_price_data(df)
        assert result.empty

    def test_timestamp_floored_to_hour(self, fake_price_df):
        """Timestamp with minutes/seconds → floored to :00:00."""
        df = fake_price_df.copy()
        df["timestamp"] = datetime(2024, 4, 21, 14, 37, 52, tzinfo=timezone.utc)
        result = harmonise_price_data(df)
        ts = result["timestamp"].iloc[0]
        assert ts.minute == 0
        assert ts.second == 0

    def test_empty_dataframe_raises_value_error(self, empty_df):
        """Completely empty DataFrame → ValueError because all columns missing."""
        with pytest.raises(ValueError, match="Missing required columns"):
            harmonise_price_data(empty_df)


class TestHarmoniseMarketData:
    """Tests for harmonise_market_data() — validates global market signals dict."""

    def test_happy_path_returns_single_row_dataframe(self, fake_market_dict):
        """Valid dict → single-row DataFrame with all required columns."""
        result = harmonise_market_data(fake_market_dict)
        assert not result.empty
        assert len(result) == 1
        assert result["btc_dominance"].iloc[0] == 56.3
        assert result["source"].iloc[0] == "coingecko"

    def test_timestamp_floored_to_hour(self, fake_market_dict):
        """Timestamp must be floored to nearest hour."""
        result = harmonise_market_data(fake_market_dict)
        ts = result["timestamp"].iloc[0]
        assert ts.minute == 0
        assert ts.second == 0

    def test_missing_key_raises_value_error(self):
        """Dict missing btc_dominance → ValueError because column absent."""
        bad_dict = {
            "timestamp":            datetime(2024, 4, 21, tzinfo=timezone.utc),
            # btc_dominance deliberately omitted
            "total_market_cap_usd": 2_500_000_000_000.0,
            "total_volume_usd":     95_000_000_000.0,
            "source":               "coingecko",
        }
        with pytest.raises(ValueError, match="Missing required columns"):
            harmonise_market_data(bad_dict)

    def test_no_asset_filter_applied(self, fake_market_dict):
        """Global market data has no asset column — no filtering should be applied."""
        result = harmonise_market_data(fake_market_dict)
        # If asset filtering were wrongly applied, this would raise or return empty
        assert not result.empty
        assert "asset" not in result.columns


class TestHarmoniseNewsData:
    """Tests for harmonise_news_data() — validates and normalises news headlines."""

    def test_happy_path_passes_through_correctly(self, fake_news_df):
        """Valid DataFrame → returns clean DataFrame with renamed timestamp column."""
        result = harmonise_news_data(fake_news_df)
        assert not result.empty
        assert len(result) == len(fake_news_df)

    def test_published_at_renamed_to_timestamp(self, fake_news_df):
        """published_at must be renamed to timestamp — published_at must not remain."""
        result = harmonise_news_data(fake_news_df)
        assert "timestamp" in result.columns
        assert "published_at" not in result.columns

    def test_timestamp_floored_to_hour(self, fake_news_df):
        """published_at at 00:05:00 → floored to 00:00:00 after rename."""
        result = harmonise_news_data(fake_news_df)
        ts = result["timestamp"].iloc[0]
        assert ts.minute == 0
        assert ts.second == 0

    def test_missing_published_at_raises_value_error(self, news_df_missing_published_at):
        """
        DataFrame without published_at → rename does nothing → timestamp absent
        → _validate_columns raises ValueError.
        """
        with pytest.raises(ValueError, match="Missing required columns"):
            harmonise_news_data(news_df_missing_published_at)

    def test_invalid_asset_is_filtered_out(self, fake_news_df):
        """Asset not in VALID_ASSETS → row dropped."""
        df = fake_news_df.copy()
        df["asset"] = "RUGPULL"
        result = harmonise_news_data(df)
        assert result.empty

    def test_empty_dataframe_raises_value_error(self, empty_df):
        """Empty DataFrame → rename does nothing → all columns missing → ValueError."""
        with pytest.raises(ValueError, match="Missing required columns"):
            harmonise_news_data(empty_df)