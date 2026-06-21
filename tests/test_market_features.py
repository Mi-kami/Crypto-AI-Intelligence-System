"""
tests/test_market_features.py

Unit tests for features/market_features.py.

All tests use mock_mlflow to prevent any writes to mlflow.db.
No real database is touched — inputs are in-memory DataFrames.
"""

# Standard library
from datetime import datetime, timezone, timedelta

# Third party
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

# Local
from features.market_features import build_market_features


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_mlflow():
    """
    Patch MLflow logging functions for every test in this file.

    autouse=True means this fixture runs automatically for every test
    without needing to be declared as a parameter. No test in this file
    should ever write to mlflow.db.
    """
    with patch("features.market_features.log_feature_params"), \
         patch("features.market_features.log_feature_metrics"):
        yield


@pytest.fixture
def sample_market_df():
    """
    Build a valid 30-row market signals DataFrame.

    BTC dominance alternates between 48 and 53 so both regime
    labels (risk_on and risk_off) appear in the output.
    Market cap and volume increase slightly each hour so log
    returns and diffs are computable and non-zero.
    """
    base = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    records = []
    for i in range(30):
        records.append({
            "timestamp":            base + timedelta(hours=i),
            "btc_dominance":        48.0 + (i % 2) * 5,       # alternates 48, 53
            "total_market_cap_usd": 2_000_000_000_000 + i * 1_000_000,
            "total_volume_usd":     80_000_000_000  + i * 500_000,
            "source":               "coingecko",
        })
    return pd.DataFrame(records)


# ── Happy Path Tests ──────────────────────────────────────────────────────────

def test_returns_dataframe(sample_market_df):
    result = build_market_features(sample_market_df)
    assert isinstance(result, pd.DataFrame)


def test_output_columns(sample_market_df):
    result = build_market_features(sample_market_df)
    expected = {
        "timestamp",
        "btc_dominance_change",
        "market_cap_return",
        "volume_change",
        "dominance_regime",
        "feature_version",
        "created_at",
    }
    assert set(result.columns) == expected


def test_row_count_preserved(sample_market_df):
    result = build_market_features(sample_market_df)
    assert len(result) == len(sample_market_df)


def test_feature_version_is_v1(sample_market_df):
    result = build_market_features(sample_market_df)
    assert (result["feature_version"] == "v1").all()


def test_first_row_nan_for_diff_features(sample_market_df):
    """Row 0 must be NaN for all diff-based features."""
    result = build_market_features(sample_market_df)
    assert pd.isna(result["btc_dominance_change"].iloc[0])
    assert pd.isna(result["market_cap_return"].iloc[0])
    assert pd.isna(result["volume_change"].iloc[0])


def test_non_null_after_first_row(sample_market_df):
    """Rows 1 onward must be non-null for all diff-based features."""
    result = build_market_features(sample_market_df)
    assert result["btc_dominance_change"].iloc[1:].notna().all()
    assert result["market_cap_return"].iloc[1:].notna().all()
    assert result["volume_change"].iloc[1:].notna().all()


def test_dominance_regime_no_nulls(sample_market_df):
    """dominance_regime must never be null — every row gets a label."""
    result = build_market_features(sample_market_df)
    assert result["dominance_regime"].notna().all()


def test_dominance_regime_valid_labels_only(sample_market_df):
    """Only risk_off and risk_on are valid labels. Nothing else."""
    result = build_market_features(sample_market_df)
    assert set(result["dominance_regime"].unique()).issubset({"risk_off", "risk_on"})


def test_risk_off_when_dominance_above_50(sample_market_df):
    """Rows where btc_dominance > 50 must produce risk_off."""
    result = build_market_features(sample_market_df)
    merged = sample_market_df.merge(result, on="timestamp")
    high_dominance = merged[merged["btc_dominance"] > 50]
    assert (high_dominance["dominance_regime"] == "risk_off").all()


def test_risk_on_when_dominance_at_or_below_50(sample_market_df):
    """Rows where btc_dominance <= 50 must produce risk_on."""
    result = build_market_features(sample_market_df)
    merged = sample_market_df.merge(result, on="timestamp")
    low_dominance = merged[merged["btc_dominance"] <= 50]
    assert (low_dominance["dominance_regime"] == "risk_on").all()


def test_output_sorted_by_timestamp(sample_market_df):
    """Output must be sorted by timestamp even if input is shuffled."""
    shuffled = sample_market_df.sample(frac=1, random_state=42)
    result = build_market_features(shuffled)
    assert result["timestamp"].is_monotonic_increasing


def test_market_cap_return_is_log_return(sample_market_df):
    """Manually verify row 1 log return matches expected formula."""
    result = build_market_features(sample_market_df)
    caps = sample_market_df.sort_values("timestamp")["total_market_cap_usd"].values
    expected_row1 = np.log(caps[1] / caps[0])
    assert abs(result["market_cap_return"].iloc[1] - expected_row1) < 1e-10


def test_no_asset_column_in_output(sample_market_df):
    """Market features are market-wide. No asset column should exist."""
    result = build_market_features(sample_market_df)
    assert "asset" not in result.columns


def test_source_column_dropped(sample_market_df):
    """Source column from raw data must not appear in feature output."""
    result = build_market_features(sample_market_df)
    assert "source" not in result.columns


# ── Failure Path Tests ────────────────────────────────────────────────────────

def test_raises_on_empty_dataframe():
    with pytest.raises(ValueError, match="empty or None"):
        build_market_features(pd.DataFrame())


def test_raises_on_none():
    with pytest.raises(ValueError, match="empty or None"):
        build_market_features(None)