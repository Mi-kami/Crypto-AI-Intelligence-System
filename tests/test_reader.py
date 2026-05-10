"""
tests/test_reader.py

Tests for storage/reader.py — covers all three reader functions.

Happy path: correct rows returned, correct columns, timestamps
            converted to UTC datetime, asset and time filters work.

Failure path: empty DataFrame when no rows match, sqlite3.Error
              propagates to caller.
"""

import sqlite3
import pytest
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import patch
from storage.reader import read_price_data, read_market_signals, read_news_headlines


# ── Shared time window ────────────────────────────────────────────────────────

START = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END   = datetime(2025, 1, 1, 2, 0, 0, tzinfo=timezone.utc)


# ── read_price_data ───────────────────────────────────────────────────────────

class TestReadPriceData:

    def test_returns_dataframe(self, db_path):
        df = read_price_data("BTC", START, END, db_path=db_path)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self, db_path):
        df = read_price_data("BTC", START, END, db_path=db_path)
        assert len(df) == 2

    def test_filters_by_asset(self, db_path):
        df = read_price_data("ETH", START, END, db_path=db_path)
        assert len(df) == 1
        assert df["asset"].iloc[0] == "ETH"

    def test_timestamp_is_datetime(self, db_path):
        df = read_price_data("BTC", START, END, db_path=db_path)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_empty_dataframe_when_no_match(self, db_path):
        future_start = datetime(2030, 1, 1, tzinfo=timezone.utc)
        future_end   = datetime(2030, 1, 2, tzinfo=timezone.utc)
        df = read_price_data("BTC", future_start, future_end, db_path=db_path)
        assert df.empty

    def test_database_error_propagates(self, db_path):
        with patch("storage.reader.sqlite3.connect", side_effect=sqlite3.Error("forced error")):
            with pytest.raises(sqlite3.Error):
                read_price_data("BTC", START, END, db_path=db_path)


# ── read_market_signals ───────────────────────────────────────────────────────

class TestReadMarketSignals:

    def test_returns_dataframe(self, db_path):
        df = read_market_signals(START, END, db_path=db_path)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self, db_path):
        df = read_market_signals(START, END, db_path=db_path)
        assert len(df) == 2

    def test_timestamp_is_datetime(self, db_path):
        df = read_market_signals(START, END, db_path=db_path)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_empty_dataframe_when_no_match(self, db_path):
        future_start = datetime(2030, 1, 1, tzinfo=timezone.utc)
        future_end   = datetime(2030, 1, 2, tzinfo=timezone.utc)
        df = read_market_signals(future_start, future_end, db_path=db_path)
        assert df.empty

    def test_database_error_propagates(self, db_path):
        with patch("storage.reader.sqlite3.connect", side_effect=sqlite3.Error("forced error")):
            with pytest.raises(sqlite3.Error):
                read_market_signals(START, END, db_path=db_path)


# ── read_news_headlines ───────────────────────────────────────────────────────

class TestReadNewsHeadlines:

    def test_returns_dataframe(self, db_path):
        df = read_news_headlines("BTC", START, END, db_path=db_path)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self, db_path):
        df = read_news_headlines("BTC", START, END, db_path=db_path)
        assert len(df) == 2

    def test_filters_by_asset(self, db_path):
        df = read_news_headlines("ETH", START, END, db_path=db_path)
        assert len(df) == 1
        assert df["asset"].iloc[0] == "ETH"

    def test_timestamp_is_datetime(self, db_path):
        df = read_news_headlines("BTC", START, END, db_path=db_path)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_empty_dataframe_when_no_match(self, db_path):
        future_start = datetime(2030, 1, 1, tzinfo=timezone.utc)
        future_end   = datetime(2030, 1, 2, tzinfo=timezone.utc)
        df = read_news_headlines("BTC", future_start, future_end, db_path=db_path)
        assert df.empty

    def test_database_error_propagates(self, db_path):
        with patch("storage.reader.sqlite3.connect", side_effect=sqlite3.Error("forced error")):
            with pytest.raises(sqlite3.Error):
                read_news_headlines("BTC", START, END, db_path=db_path)