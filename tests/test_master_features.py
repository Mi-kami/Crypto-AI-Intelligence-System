"""
tests/test_master_features.py

Unit tests for features/master_features.py.

All tests mock MLflow to prevent writes to mlflow.db.
Input DataFrames are built in-memory — no DB touched.
"""

# Standard library
from datetime import datetime, timezone, timedelta

# Third party
import pandas as pd
import pytest
from unittest.mock import patch

# Local
from features.master_features import build_master_features


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_mlflow():
    """Patch MLflow for every test — no writes to mlflow.db."""
    with patch("features.master_features.log_feature_params"), \
         patch("features.master_features.log_feature_metrics"):
        yield


@pytest.fixture
def base_time():
    return datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_price_df(base_time):
    """
    3 assets x 2 hours = 6 rows.
    Matches the structure of build_price_features() output.
    """
    assets = ["BTC", "ETH", "SOL"]
    records = []
    for hour in range(2):
        for asset in assets:
            records.append({
                "asset":                  asset,
                "timestamp":              base_time + timedelta(hours=hour),
                "log_return":             0.01 * hour,
                "rolling_volatility_24h": 0.02,
                "momentum_24h":           0.005,
                "volume_ratio":           1.1,
                "volume_momentum":        0.003,
                "feature_version":        "v1",
                "created_at":             base_time.isoformat(),
            })
    return pd.DataFrame(records)


@pytest.fixture
def sample_market_df(base_time):
    """
    2 hours = 2 rows.
    Matches the structure of build_market_features() output.
    """
    records = []
    for hour in range(2):
        records.append({
            "timestamp":            base_time + timedelta(hours=hour),
            "btc_dominance_change": 0.5 * hour,
            "market_cap_return":    0.01 * hour,
            "volume_change":        1_000_000 * hour,
            "dominance_regime":     "risk_off",
            "feature_version":      "v1",
            "created_at":           base_time.isoformat(),
        })
    return pd.DataFrame(records)


@pytest.fixture
def sample_sentiment_df(base_time):
    """
    3 assets x 2 hours = 6 rows.
    Matches the structure of build_sentiment_features() output.
    """
    assets = ["BTC", "ETH", "SOL"]
    records = []
    for hour in range(2):
        for asset in assets:
            records.append({
                "asset":           asset,
                "timestamp":       base_time + timedelta(hours=hour),
                "headline_count":  3,
                "sentiment_score": 0.25,
                "sentiment_label": "positive",
                "feature_version": "v1",
                "created_at":      base_time.isoformat(),
            })
    return pd.DataFrame(records)


# ── Happy Path Tests ──────────────────────────────────────────────────────────

def test_returns_dataframe(sample_price_df, sample_market_df, sample_sentiment_df):
    result = build_master_features(sample_price_df, sample_market_df, sample_sentiment_df)
    assert isinstance(result, pd.DataFrame)


def test_row_count_equals_price_rows(sample_price_df, sample_market_df, sample_sentiment_df):
    """Output must never have fewer rows than price_df."""
    result = build_master_features(sample_price_df, sample_market_df, sample_sentiment_df)
    assert len(result) == len(sample_price_df)


def test_price_columns_present(sample_price_df, sample_market_df, sample_sentiment_df):
    result = build_master_features(sample_price_df, sample_market_df, sample_sentiment_df)
    for col in ["log_return", "rolling_volatility_24h", "momentum_24h",
                "volume_ratio", "volume_momentum"]:
        assert col in result.columns


def test_market_columns_present(sample_price_df, sample_market_df, sample_sentiment_df):
    result = build_master_features(sample_price_df, sample_market_df, sample_sentiment_df)
    for col in ["btc_dominance_change", "market_cap_return",
                "volume_change", "dominance_regime"]:
        assert col in result.columns


def test_sentiment_columns_present(sample_price_df, sample_market_df, sample_sentiment_df):
    result = build_master_features(sample_price_df, sample_market_df, sample_sentiment_df)
    for col in ["headline_count", "sentiment_score", "sentiment_label"]:
        assert col in result.columns


def test_feature_version_is_v1(sample_price_df, sample_market_df, sample_sentiment_df):
    result = build_master_features(sample_price_df, sample_market_df, sample_sentiment_df)
    assert (result["feature_version"] == "v1").all()


def test_sorted_by_asset_and_timestamp(sample_price_df, sample_market_df, sample_sentiment_df):
    result = build_master_features(sample_price_df, sample_market_df, sample_sentiment_df)
    expected = result.sort_values(["asset", "timestamp"])
    pd.testing.assert_frame_equal(result.reset_index(drop=True),
                                  expected.reset_index(drop=True))


def test_all_assets_present(sample_price_df, sample_market_df, sample_sentiment_df):
    result = build_master_features(sample_price_df, sample_market_df, sample_sentiment_df)
    assert set(result["asset"].unique()) == {"BTC", "ETH", "SOL"}


def test_no_duplicate_feature_version_columns(sample_price_df, sample_market_df, sample_sentiment_df):
    """Joining must not produce feature_version_x or feature_version_y."""
    result = build_master_features(sample_price_df, sample_market_df, sample_sentiment_df)
    assert "feature_version_x" not in result.columns
    assert "feature_version_y" not in result.columns


def test_sentiment_nan_when_no_sentiment(sample_price_df, sample_market_df):
    """When sentiment_df is empty, sentiment columns must be NaN — not missing rows."""
    result = build_master_features(sample_price_df, sample_market_df, pd.DataFrame())
    assert len(result) == len(sample_price_df)
    assert result["sentiment_score"].isna().all()


def test_market_context_same_for_all_assets_same_hour(
    sample_price_df, sample_market_df, sample_sentiment_df
):
    """All assets at the same hour must share identical market column values."""
    result = build_master_features(sample_price_df, sample_market_df, sample_sentiment_df)
    hour_0 = result[result["timestamp"] == result["timestamp"].min()]
    assert hour_0["btc_dominance_change"].nunique() == 1
    assert hour_0["dominance_regime"].nunique() == 1


# ── Failure Path Tests ────────────────────────────────────────────────────────

def test_raises_on_empty_price_df(sample_market_df, sample_sentiment_df):
    with pytest.raises(ValueError, match="price_df"):
        build_master_features(pd.DataFrame(), sample_market_df, sample_sentiment_df)


def test_raises_on_none_price_df(sample_market_df, sample_sentiment_df):
    with pytest.raises(ValueError, match="price_df"):
        build_master_features(None, sample_market_df, sample_sentiment_df)


def test_raises_on_empty_market_df(sample_price_df, sample_sentiment_df):
    with pytest.raises(ValueError, match="market_df"):
        build_master_features(sample_price_df, pd.DataFrame(), sample_sentiment_df)


def test_raises_on_none_market_df(sample_price_df, sample_sentiment_df):
    with pytest.raises(ValueError, match="market_df"):
        build_master_features(sample_price_df, None, sample_sentiment_df)