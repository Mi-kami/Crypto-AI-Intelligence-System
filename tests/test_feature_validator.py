"""
tests/test_feature_validator.py

Validates the lightweight feature validation module.
Tests cover: pass cases, infinity detection, empty DataFrame,
null rate thresholds, and asset coverage gaps.
"""

import numpy as np
import pandas as pd
import pytest

from validation.feature_validator import (
    VALID_ASSETS,
    validate_feature_df,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_df() -> pd.DataFrame:
    """
    30 clean rows per asset — all 10 assets present, no nulls, no infinities.
    This is the baseline that all failure tests mutate.
    """
    records = []
    for asset in VALID_ASSETS:
        for i in range(30):
            records.append({
                "asset":                  asset,
                "timestamp":              pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=i),
                "log_return":             0.001 * i,
                "rolling_volatility_24h": 0.02,
                "momentum_24h":           0.001,
                "volume_ratio":           1.0,
                "volume_momentum":        0.0,
            })
    return pd.DataFrame(records)


# ── Pass cases ────────────────────────────────────────────────────────────────

def test_valid_df_passes(valid_df):
    result = validate_feature_df(valid_df, context="test")
    assert result["passed"] is True
    assert result["errors"] == []


def test_returns_correct_row_count(valid_df):
    result = validate_feature_df(valid_df, context="test")
    assert result["row_count"] == 300  # 10 assets × 30 rows


def test_returns_correct_asset_count(valid_df):
    result = validate_feature_df(valid_df, context="test")
    assert result["asset_count"] == 10


def test_no_missing_assets_on_valid_df(valid_df):
    result = validate_feature_df(valid_df, context="test")
    assert result["missing_assets"] == []


def test_result_contains_expected_keys(valid_df):
    result = validate_feature_df(valid_df, context="test")
    expected_keys = {
        "passed", "context", "row_count", "asset_count",
        "null_rates", "has_infinities", "missing_assets",
        "rows_per_asset", "warnings", "errors",
    }
    assert expected_keys.issubset(result.keys())


# ── Empty DataFrame ───────────────────────────────────────────────────────────

def test_empty_df_fails():
    result = validate_feature_df(pd.DataFrame(), context="test", raise_on_error=False)
    assert result["passed"] is False
    assert any("empty" in e.lower() for e in result["errors"])


def test_empty_df_raises_when_configured():
    with pytest.raises(ValueError, match="Validation failed"):
        validate_feature_df(pd.DataFrame(), context="test", raise_on_error=True)


def test_empty_df_returns_all_missing_assets():
    result = validate_feature_df(pd.DataFrame(), context="test", raise_on_error=False)
    assert result["missing_assets"] == VALID_ASSETS


# ── Infinity detection ────────────────────────────────────────────────────────

def test_detects_positive_infinity(valid_df):
    valid_df.loc[0, "log_return"] = float("inf")
    result = validate_feature_df(valid_df, context="test", raise_on_error=False)
    assert result["has_infinities"] is True
    assert result["passed"] is False


def test_detects_negative_infinity(valid_df):
    valid_df.loc[0, "log_return"] = float("-inf")
    result = validate_feature_df(valid_df, context="test", raise_on_error=False)
    assert result["has_infinities"] is True
    assert result["passed"] is False


def test_raises_on_infinity_when_configured(valid_df):
    valid_df.loc[0, "log_return"] = float("inf")
    with pytest.raises(ValueError, match="Validation failed"):
        validate_feature_df(valid_df, context="test", raise_on_error=True)


def test_no_infinity_on_valid_df(valid_df):
    result = validate_feature_df(valid_df, context="test")
    assert result["has_infinities"] is False


# ── Null rate thresholds ──────────────────────────────────────────────────────

def test_high_null_rate_is_hard_failure(valid_df):
    # 85% nulls — exceeds 80% error threshold
    n = len(valid_df)
    null_count = int(n * 0.85)
    valid_df.loc[:null_count - 1, "log_return"] = float("nan")
    result = validate_feature_df(valid_df, context="test", raise_on_error=False)
    assert result["passed"] is False
    assert any("log_return" in e for e in result["errors"])


def test_moderate_null_rate_is_warning_not_failure(valid_df):
    # 40% nulls — between 30% warning and 80% error thresholds
    n = len(valid_df)
    null_count = int(n * 0.40)
    valid_df.loc[:null_count - 1, "log_return"] = float("nan")
    result = validate_feature_df(valid_df, context="test", raise_on_error=False)
    assert result["passed"] is True
    assert any("log_return" in w for w in result["warnings"])


def test_low_null_rate_triggers_nothing(valid_df):
    # 10% nulls — below 30% warning threshold
    n = len(valid_df)
    null_count = int(n * 0.10)
    valid_df.loc[:null_count - 1, "log_return"] = float("nan")
    result = validate_feature_df(valid_df, context="test", raise_on_error=False)
    assert result["passed"] is True
    assert not any("log_return" in w for w in result["warnings"])


# ── Asset coverage ────────────────────────────────────────────────────────────

def test_missing_asset_appears_in_result(valid_df):
    df_missing = valid_df[valid_df["asset"] != "SHIB"].copy()
    result = validate_feature_df(df_missing, context="test", raise_on_error=False)
    assert "SHIB" in result["missing_assets"]


def test_missing_asset_triggers_warning(valid_df):
    df_missing = valid_df[valid_df["asset"] != "SHIB"].copy()
    result = validate_feature_df(df_missing, context="test", raise_on_error=False)
    assert any("SHIB" in w for w in result["warnings"])


def test_missing_asset_does_not_cause_hard_failure(valid_df):
    # Asset dropout is serious but pipeline should continue with a warning
    df_missing = valid_df[valid_df["asset"] != "SHIB"].copy()
    result = validate_feature_df(df_missing, context="test", raise_on_error=False)
    assert result["passed"] is True


# ── Thin row counts ───────────────────────────────────────────────────────────

def test_thin_asset_triggers_warning(valid_df):
    # BTC gets only 5 rows — below MIN_ROWS_PER_ASSET (24)
    btc_slim  = valid_df[valid_df["asset"] == "BTC"].head(5)
    non_btc   = valid_df[valid_df["asset"] != "BTC"]
    df_thin   = pd.concat([btc_slim, non_btc], ignore_index=True)
    result    = validate_feature_df(df_thin, context="test", raise_on_error=False)
    assert any("BTC" in w for w in result["warnings"])


def test_thin_asset_does_not_cause_hard_failure(valid_df):
    btc_slim  = valid_df[valid_df["asset"] == "BTC"].head(5)
    non_btc   = valid_df[valid_df["asset"] != "BTC"]
    df_thin   = pd.concat([btc_slim, non_btc], ignore_index=True)
    result    = validate_feature_df(df_thin, context="test", raise_on_error=False)
    assert result["passed"] is True