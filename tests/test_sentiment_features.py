"""
tests/test_sentiment_features.py

Unit tests for features/sentiment_features.py.

All tests mock MLflow to prevent writes to mlflow.db.
VADER runs for real — it requires no external API calls.
"""

# Standard library
from datetime import datetime, timezone, timedelta

# Third party
import pandas as pd
import pytest
from unittest.mock import patch

# Local
from features.sentiment_features import build_sentiment_features


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_mlflow():
    """Patch MLflow for every test — no writes to mlflow.db."""
    with patch("features.sentiment_features.log_feature_params"), \
         patch("features.sentiment_features.log_feature_metrics"):
        yield


@pytest.fixture
def sample_news_df():
    """
    Build a valid news DataFrame with 3 assets, 2 hours, 3 headlines each.
    Total input: 18 rows.
    Expected output: 6 rows (3 assets x 2 hours).
    Headlines are deliberately positive, negative, and neutral so all
    three sentiment labels appear in the output.
    """
    base = datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc)
    assets = ["BTC", "ETH", "SOL"]
    headlines = [
        "Bitcoin surges to record high amid massive institutional buying",
        "Exchange collapses amid fraud allegations wiping billions off market",
        "Crypto market remains stable as traders await regulatory news",
    ]

    records = []
    for hour_offset in range(2):
        for asset in assets:
            for headline in headlines:
                records.append({
                    "asset":     asset,
                    "timestamp": base + timedelta(hours=hour_offset),
                    "headline":  headline,
                    "url":       f"https://example.com/{asset}/{hour_offset}/{headline[:10]}",
                    "source":    "cryptocompare",
                })
    return pd.DataFrame(records)


# ── Happy Path Tests ──────────────────────────────────────────────────────────

def test_returns_dataframe(sample_news_df):
    result = build_sentiment_features(sample_news_df)
    assert isinstance(result, pd.DataFrame)


def test_output_columns(sample_news_df):
    result = build_sentiment_features(sample_news_df)
    expected = {
        "asset", "timestamp", "headline_count",
        "sentiment_score", "sentiment_label",
        "feature_version", "created_at",
    }
    assert set(result.columns) == expected


def test_one_row_per_asset_per_hour(sample_news_df):
    """3 assets x 2 hours = 6 output rows."""
    result = build_sentiment_features(sample_news_df)
    assert len(result) == 6


def test_headline_count_correct(sample_news_df):
    """Each (asset, hour) group has 3 headlines in our fixture."""
    result = build_sentiment_features(sample_news_df)
    assert (result["headline_count"] == 3).all()


def test_sentiment_score_in_valid_range(sample_news_df):
    """Compound scores must sit between -1.0 and +1.0."""
    result = build_sentiment_features(sample_news_df)
    assert result["sentiment_score"].between(-1.0, 1.0).all()


def test_sentiment_label_valid_values(sample_news_df):
    result = build_sentiment_features(sample_news_df)
    assert set(result["sentiment_label"].unique()).issubset(
        {"positive", "negative", "neutral"}
    )


def test_no_null_sentiment_scores(sample_news_df):
    result = build_sentiment_features(sample_news_df)
    assert result["sentiment_score"].notna().all()


def test_no_null_sentiment_labels(sample_news_df):
    result = build_sentiment_features(sample_news_df)
    assert result["sentiment_label"].notna().all()


def test_feature_version_is_v1(sample_news_df):
    result = build_sentiment_features(sample_news_df)
    assert (result["feature_version"] == "v1").all()


def test_asset_column_present(sample_news_df):
    """Unlike market features, sentiment features are per-asset."""
    result = build_sentiment_features(sample_news_df)
    assert "asset" in result.columns


def test_all_three_assets_in_output(sample_news_df):
    result = build_sentiment_features(sample_news_df)
    assert set(result["asset"].unique()) == {"BTC", "ETH", "SOL"}


def test_empty_headline_does_not_crash(sample_news_df):
    """Empty headline returns 0.0 compound score — neutral, not a crash."""
    sample_news_df.loc[0, "headline"] = ""
    result = build_sentiment_features(sample_news_df)
    assert isinstance(result, pd.DataFrame)


def test_source_column_not_in_output(sample_news_df):
    result = build_sentiment_features(sample_news_df)
    assert "source" not in result.columns


# ── Failure Path Tests ────────────────────────────────────────────────────────

def test_raises_on_empty_dataframe():
    with pytest.raises(ValueError, match="empty or None"):
        build_sentiment_features(pd.DataFrame())


def test_raises_on_none():
    with pytest.raises(ValueError, match="empty or None"):
        build_sentiment_features(None)