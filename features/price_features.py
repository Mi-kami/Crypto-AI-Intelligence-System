"""
features/price_features.py

Responsible for engineering price-based features from raw OHLCV data
stored in the price_ohlcv table.

Sits between the storage layer and the model inference layer. Reads
nothing directly — receives a harmonised DataFrame from
run_feature_pipeline() and returns a feature DataFrame ready for
validation, MLflow logging, and writing to the asset_features table.

Features computed:
    - log_return             : log(close_t / close_{t-1})
    - rolling_volatility_24h : std of log returns over 24h window
    - momentum_24h           : rolling mean of log returns over 24h window
    - volume_ratio           : volume / rolling mean volume over 24h window
    - volume_momentum        : period-over-period change in volume ratio
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from loguru import logger

from tracking.mlflow_tracker import log_feature_metrics, log_feature_params
from validation.feature_validator import validate_feature_df


# ── Private Helpers ───────────────────────────────────────────────────────────

def _compute_log_return(df: pd.DataFrame) -> pd.Series:
    """Compute log return per asset: log(close_t / close_{t-1})."""
    return (
        df.sort_values(["asset", "timestamp"])
          .groupby("asset")["close"]
          .transform(lambda x: np.log(x / x.shift(1)))
    )


def _compute_rolling_volatility(df: pd.DataFrame) -> pd.Series:
    """Compute 24h rolling std of log returns per asset."""
    return (
        df.sort_values(["asset", "timestamp"])
          .groupby("asset")["log_return"]
          .transform(lambda x: x.rolling(24).std())
    )


def _compute_momentum_24h(df: pd.DataFrame) -> pd.Series:
    """Compute 24h rolling mean of log returns per asset."""
    return (
        df.sort_values(["asset", "timestamp"])
          .groupby("asset")["log_return"]
          .transform(lambda x: x.rolling(24).mean())
    )


def _compute_volume_ratio(df: pd.DataFrame) -> pd.Series:
    """Compute volume ratio: volume / 24h rolling mean volume per asset."""
    return (
        df.sort_values(["asset", "timestamp"])
          .groupby("asset")["volume"]
          .transform(lambda x: x / x.rolling(24).mean())
    )


def _compute_volume_momentum(df: pd.DataFrame) -> pd.Series:
    """Compute volume momentum: period-over-period change in volume ratio per asset."""
    return (
        df.sort_values(["asset", "timestamp"])
          .groupby("asset")["volume_ratio"]
          .transform(lambda x: x.diff(1))
    )


# ── Public Function ───────────────────────────────────────────────────────────

def build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute price-based features from raw OHLCV data for all assets.

    Takes a harmonised price DataFrame from the price_ohlcv table and
    engineers five features used as inputs to the volatility and regime
    detection models.

    Args:
        df: Harmonised price DataFrame from read_price_data().
            Required columns: asset, timestamp, close, volume.

    Returns:
        DataFrame with columns: asset, timestamp, log_return,
        rolling_volatility_24h, momentum_24h, volume_ratio,
        volume_momentum, feature_version, created_at.
        Same number of rows as input. First 23 rows per asset will
        have NaN in rolling features — expected, not a bug.
    """
    if df.empty:
        logger.warning("build_price_features received empty DataFrame — returning empty")
        return pd.DataFrame()

    logger.info(
        f"Building price features | "
        f"{df['asset'].nunique()} assets | {len(df)} rows"
    )

    # ── Compute features sequentially ────────────────────────────────────
    working = df.sort_values(["asset", "timestamp"]).copy()

    # log_return first — volatility and momentum depend on it
    working["log_return"] = _compute_log_return(working)

    # Rolling features depend on log_return being present
    working["rolling_volatility_24h"] = _compute_rolling_volatility(working)
    working["momentum_24h"]           = _compute_momentum_24h(working)

    # Volume ratio before volume momentum
    working["volume_ratio"]    = _compute_volume_ratio(working)
    working["volume_momentum"] = _compute_volume_momentum(working)

    # ── Add tracking metadata ─────────────────────────────────────────────
    working["feature_version"] = "v1"
    working["created_at"]      = datetime.now(timezone.utc).isoformat()

    # ── Select and order output columns ───────────────────────────────────
    output_cols = [
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
    feature_df = working[output_cols].reset_index(drop=True)

    # ── Validate ──────────────────────────────────────────────────────────
    validate_feature_df(feature_df, context="price_features")

    # ── MLflow logging ────────────────────────────────────────────────────
    log_feature_params({
        "feature_version": "v1",
        "rolling_window":  24,
        "asset_count":     df["asset"].nunique(),
        "row_count":       len(df),
    })
    log_feature_metrics({
        "price_row_count":        len(feature_df),
        "price_asset_count":      feature_df["asset"].nunique(),
        "log_return_null_rate":   round(float(feature_df["log_return"].isna().mean()), 4),
        "volatility_null_rate":   round(float(feature_df["rolling_volatility_24h"].isna().mean()), 4),
        "momentum_null_rate":     round(float(feature_df["momentum_24h"].isna().mean()), 4),
        "volume_ratio_null_rate": round(float(feature_df["volume_ratio"].isna().mean()), 4),
    })

    logger.success(
        f"build_price_features complete | "
        f"{len(feature_df)} rows | {feature_df['asset'].nunique()} assets"
    )

    return feature_df