"""
tests/test_writer.py

Tests for storage/writer.py — validates all three write functions
across happy path, empty DataFrame, and upsert/ignore behaviour.
"""

import sqlite3
import pandas as pd
import pytest
from pathlib import Path

from storage.schema import create_tables
from storage.writer import write_price_data, write_market_signals, write_news_headlines


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Fresh temporary database with all tables initialised."""
    path = tmp_path / "test_crypto_ai.db"
    create_tables(db_path=path)
    return path


@pytest.fixture
def sample_price_df() -> pd.DataFrame:
    ts = pd.Timestamp("2024-01-01 10:00:00", tz="UTC")
    return pd.DataFrame([
        {
            "asset":      "BTC",
            "timestamp":  ts,
            "close":      42000.0,
            "source":     "coingecko",
            "market_cap": 800_000_000_000.0,
            "volume":     20_000_000_000.0,
        },
        {
            "asset":      "ETH",
            "timestamp":  ts,
            "close":      2200.0,
            "source":     "coingecko",
            "market_cap": 260_000_000_000.0,
            "volume":     10_000_000_000.0,
        },
    ])


@pytest.fixture
def sample_market_df() -> pd.DataFrame:
    ts = pd.Timestamp("2024-01-01 10:00:00", tz="UTC")
    return pd.DataFrame([
        {
            "timestamp":            ts,
            "btc_dominance":        52.5,
            "source":               "coingecko",
            "total_market_cap_usd": 1_600_000_000_000.0,
            "total_volume_usd":     80_000_000_000.0,
        }
    ])


@pytest.fixture
def sample_news_df() -> pd.DataFrame:
    ts = pd.Timestamp("2024-01-01 10:00:00", tz="UTC")
    return pd.DataFrame([
        {
            "asset":           "BTC",
            "timestamp":       ts,
            "headline":        "Bitcoin reaches new high",
            "url":             "https://example.com/btc-high",
            "source":          "cryptocompare",
            "votes_positive":  0,
            "votes_negative":  0,
        },
        {
            "asset":           "ETH",
            "timestamp":       ts,
            "headline":        "Ethereum upgrade complete",
            "url":             "https://example.com/eth-upgrade",
            "source":          "cryptocompare",
            "votes_positive":  0,
            "votes_negative":  0,
        },
    ])


# ── write_price_data ──────────────────────────────────────────────────────────

class TestWritePriceData:

    def test_happy_path_writes_correct_row_count(self, db_path, sample_price_df):
        write_price_data(sample_price_df, db_path=db_path)
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM price_ohlcv").fetchone()[0]
        conn.close()
        assert count == 2

    def test_values_persisted_correctly(self, db_path, sample_price_df):
        write_price_data(sample_price_df, db_path=db_path)
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT asset, close, source FROM price_ohlcv WHERE asset = 'BTC'"
        ).fetchone()
        conn.close()
        assert row == ("BTC", 42000.0, "coingecko")

    def test_empty_dataframe_writes_nothing(self, db_path):
        write_price_data(pd.DataFrame(), db_path=db_path)
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM price_ohlcv").fetchone()[0]
        conn.close()
        assert count == 0

    def test_upsert_replaces_existing_row(self, db_path, sample_price_df):
        write_price_data(sample_price_df, db_path=db_path)

        updated = sample_price_df.copy()
        updated.loc[updated["asset"] == "BTC", "close"] = 99999.0
        write_price_data(updated, db_path=db_path)

        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM price_ohlcv").fetchone()[0]
        close = conn.execute(
            "SELECT close FROM price_ohlcv WHERE asset = 'BTC'"
        ).fetchone()[0]
        conn.close()

        assert count == 2        # no duplicate rows created
        assert close == 99999.0  # new value replaced old


# ── write_market_signals ──────────────────────────────────────────────────────

class TestWriteMarketSignals:

    def test_happy_path_writes_correct_row_count(self, db_path, sample_market_df):
        write_market_signals(sample_market_df, db_path=db_path)
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM market_signals").fetchone()[0]
        conn.close()
        assert count == 1

    def test_values_persisted_correctly(self, db_path, sample_market_df):
        write_market_signals(sample_market_df, db_path=db_path)
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT btc_dominance, source FROM market_signals"
        ).fetchone()
        conn.close()
        assert row == (52.5, "coingecko")

    def test_empty_dataframe_writes_nothing(self, db_path):
        write_market_signals(pd.DataFrame(), db_path=db_path)
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM market_signals").fetchone()[0]
        conn.close()
        assert count == 0

    def test_upsert_replaces_existing_row(self, db_path, sample_market_df):
        write_market_signals(sample_market_df, db_path=db_path)

        updated = sample_market_df.copy()
        updated.loc[0, "btc_dominance"] = 60.0
        write_market_signals(updated, db_path=db_path)

        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM market_signals").fetchone()[0]
        dominance = conn.execute(
            "SELECT btc_dominance FROM market_signals"
        ).fetchone()[0]
        conn.close()

        assert count == 1       # no duplicate rows
        assert dominance == 60.0  # new value replaced old


# ── write_news_headlines ──────────────────────────────────────────────────────

class TestWriteNewsHeadlines:

    def test_happy_path_writes_correct_row_count(self, db_path, sample_news_df):
        write_news_headlines(sample_news_df, db_path=db_path)
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM news_headlines").fetchone()[0]
        conn.close()
        assert count == 2

    def test_values_persisted_correctly(self, db_path, sample_news_df):
        write_news_headlines(sample_news_df, db_path=db_path)
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT asset, headline, source FROM news_headlines WHERE asset = 'BTC'"
        ).fetchone()
        conn.close()
        assert row == ("BTC", "Bitcoin reaches new high", "cryptocompare")

    def test_empty_dataframe_writes_nothing(self, db_path):
        write_news_headlines(pd.DataFrame(), db_path=db_path)
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM news_headlines").fetchone()[0]
        conn.close()
        assert count == 0

    def test_ignore_keeps_original_on_duplicate_url(self, db_path, sample_news_df):
        write_news_headlines(sample_news_df, db_path=db_path)

        duplicate = sample_news_df.copy()
        duplicate.loc[duplicate["asset"] == "BTC", "headline"] = "Different headline"
        write_news_headlines(duplicate, db_path=db_path)

        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM news_headlines").fetchone()[0]
        headline = conn.execute(
            "SELECT headline FROM news_headlines WHERE asset = 'BTC'"
        ).fetchone()[0]
        conn.close()

        assert count == 2  # no duplicate rows
        assert headline == "Bitcoin reaches new high"  # original retained

# ── Failure Path Tests ────────────────────────────────────────────────────────

class TestWritePriceDataFailures:

    def test_raises_when_table_does_not_exist(self, tmp_path, sample_price_df):
        """Writer must raise OperationalError when price_ohlcv table is absent."""
        bad_path = tmp_path / "empty.db"
        sqlite3.connect(bad_path).close()  # file exists, but no tables
        with pytest.raises(sqlite3.OperationalError):
            write_price_data(sample_price_df, db_path=bad_path)

    def test_exception_propagates_to_caller(self, tmp_path, sample_price_df, monkeypatch):
        """DB connection failure must surface to caller — never silently swallowed."""
        def mock_connect(_: Path) -> None:
            raise sqlite3.OperationalError("simulated connection failure")

        monkeypatch.setattr(sqlite3, "connect", mock_connect)

        with pytest.raises(sqlite3.OperationalError):
            write_price_data(sample_price_df, db_path=tmp_path / "any.db")


class TestWriteMarketSignalsFailures:

    def test_raises_when_table_does_not_exist(self, tmp_path, sample_market_df):
        """Writer must raise OperationalError when market_signals table is absent."""
        bad_path = tmp_path / "empty.db"
        sqlite3.connect(bad_path).close()
        with pytest.raises(sqlite3.OperationalError):
            write_market_signals(sample_market_df, db_path=bad_path)

    def test_exception_propagates_to_caller(self, tmp_path, sample_market_df, monkeypatch):
        """DB connection failure must surface to caller — never silently swallowed."""
        def mock_connect(_: Path) -> None:
            raise sqlite3.OperationalError("simulated connection failure")

        monkeypatch.setattr(sqlite3, "connect", mock_connect)

        with pytest.raises(sqlite3.OperationalError):
            write_market_signals(sample_market_df, db_path=tmp_path / "any.db")


class TestWriteNewsHeadlinesFailures:

    def test_raises_when_table_does_not_exist(self, tmp_path, sample_news_df):
        """Writer must raise OperationalError when news_headlines table is absent."""
        bad_path = tmp_path / "empty.db"
        sqlite3.connect(bad_path).close()
        with pytest.raises(sqlite3.OperationalError):
            write_news_headlines(sample_news_df, db_path=bad_path)

    def test_exception_propagates_to_caller(self, tmp_path, sample_news_df, monkeypatch):
        """DB connection failure must surface to caller — never silently swallowed."""
        def mock_connect(_: Path) -> None:
            raise sqlite3.OperationalError("simulated connection failure")

        monkeypatch.setattr(sqlite3, "connect", mock_connect)

        with pytest.raises(sqlite3.OperationalError):
            write_news_headlines(sample_news_df, db_path=tmp_path / "any.db")