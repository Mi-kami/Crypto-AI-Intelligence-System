"""
tests/test_price_features.py

Tests for features/price_features.py — specifically build_price_features().

Covers: output schema, row count, feature correctness, NaN behaviour
for rolling windows, empty DataFrame handling, and metadata columns.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from features.price_features import build_price_features


VALID_ASSETS = [
    "BTC", "ETH", "BNB", "XRP", "SOL",
    "DOGE", "ADA", "TRX", "AVAX", "SHIB",
]

EXPECTED_COLUMNS = [
    "asset",
    "timestamp",
    "log_return",
    "rolling_volatility_24h",
    "momentum_24h",
    "volume_ratio",
    "volume_momentum",
    "feature_version",
    "created_at",
]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_mlflow():
    """
    Patch MLflow logging calls for all tests in this file.
    Tests should not depend on an active MLflow run being present.
    """
    with patch("features.price_features.log_feature_params"), \
         patch("features.price_features.log_feature_metrics"):
        yield


@pytest.fixture
def price_df() -> pd.DataFrame:
    """
    30 clean rows per asset — all 10 assets, positive close and volume.
    Baseline fixture that all failure tests mutate.
    """
    records = []
    for asset in VALID_ASSETS:
        for i in range(50):
            records.append({
                "asset":     asset,
                "timestamp": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=i),
                "close":     100.0 + i,
                "volume":    1_000_000 + (i * 1000),
            })
    return pd.DataFrame(records)


# ── Pass cases ────────────────────────────────────────────────────────────────

def test_returns_dataframe(price_df):
    result = build_price_features(price_df)
    assert isinstance(result, pd.DataFrame)


def test_output_not_empty(price_df):
    result = build_price_features(price_df)
    assert not result.empty


def test_output_has_correct_columns(price_df):
    result = build_price_features(price_df)
    assert set(EXPECTED_COLUMNS).issubset(result.columns)


def test_output_row_count_matches_input(price_df):
    result = build_price_features(price_df)
    assert len(result) == len(price_df)


def test_output_asset_count(price_df):
    result = build_price_features(price_df)
    assert result["asset"].nunique() == 10


def test_feature_version_is_v1(price_df):
    result = build_price_features(price_df)
    assert (result["feature_version"] == "v1").all()


def test_created_at_is_present(price_df):
    result = build_price_features(price_df)
    assert result["created_at"].notna().all()


def test_output_index_is_clean(price_df):
    result = build_price_features(price_df)
    assert list(result.index) == list(range(len(result)))


# ── Feature correctness ───────────────────────────────────────────────────────

def test_first_row_per_asset_log_return_is_nan(price_df):
    result = build_price_features(price_df)
    first_rows = result.groupby("asset").nth(0)
    assert first_rows["log_return"].isna().all()


def test_log_return_has_no_infinities(price_df):
    result = build_price_features(price_df)
    assert not np.isinf(result["log_return"].dropna()).any()


def test_rolling_features_nan_for_first_23_rows_per_asset(price_df):
    result = build_price_features(price_df)
    for asset in VALID_ASSETS:
        asset_rows = result[result["asset"] == asset].reset_index(drop=True)
        assert asset_rows.loc[:22, "rolling_volatility_24h"].isna().all()
        assert asset_rows.loc[:22, "momentum_24h"].isna().all()


def test_rolling_features_not_nan_after_window(price_df):
    result = build_price_features(price_df)
    for asset in VALID_ASSETS:
        asset_rows = result[result["asset"] == asset].reset_index(drop=True)
        assert asset_rows.loc[24:, "rolling_volatility_24h"].notna().all()
        assert asset_rows.loc[24:, "momentum_24h"].notna().all()


def test_volume_ratio_positive_for_positive_volume(price_df):
    result = build_price_features(price_df)
    valid_ratios = result["volume_ratio"].dropna()
    assert (valid_ratios > 0).all()


# ── Failure cases ─────────────────────────────────────────────────────────────

def test_empty_df_returns_empty_df():
    result = build_price_features(pd.DataFrame())
    assert result.empty


def test_zero_close_raises_on_infinity():
    """
    A zero close price produces log(0) = -inf.
    Validator catches it and raises ValueError before the function returns.
    """
    records = []
    for asset in VALID_ASSETS:
        for i in range(50):
            records.append({
                "asset":     asset,
                "timestamp": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=i),
                "close":     0.0 if i == 1 else 100.0 + i,
                "volume":    1_000_000,
            })
    df = pd.DataFrame(records)
    with pytest.raises(ValueError):
        build_price_features(df)