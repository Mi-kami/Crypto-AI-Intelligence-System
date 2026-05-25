"""
validation/feature_validator.py

Lightweight validation for feature DataFrames before they are written
to the asset_features table or passed to model inference.

Not great_expectations — intentionally minimal.

Three failure modes we protect against:
    1. Infinities   — log(0) = -inf. GARCH and LSTM cannot handle them.
    2. Null rates   — rolling windows produce NaNs at the head of each
                      series. Too many means something upstream broke.
    3. Asset gaps   — silent asset dropout means missing intelligence outputs.
"""

import numpy as np
import pandas as pd
from loguru import logger


# ── Thresholds ────────────────────────────────────────────────────────────────

VALID_ASSETS = [
    "BTC", "ETH", "BNB", "XRP", "SOL",
    "DOGE", "ADA", "TRX", "AVAX", "SHIB",
]

NULL_RATE_WARNING_THRESHOLD = 0.30   # 30%+ nulls → warning
NULL_RATE_ERROR_THRESHOLD   = 0.80   # 80%+ nulls → hard failure
MIN_ROWS_PER_ASSET          = 24     # Fewer than 24 rows → rolling windows useless


# ── Private Helpers ───────────────────────────────────────────────────────────

def _check_nulls(df: pd.DataFrame) -> dict[str, float]:
    """
    Compute null rate (0.0–1.0) for every numeric column in df.

    Returns:
        Dict mapping column name to null rate e.g. {"log_return": 0.12}
    """
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    return {
        col: round(float(df[col].isna().mean()), 4)
        for col in numeric_cols
    }


def _check_infinities(df: pd.DataFrame) -> bool:
    """
    Return True if any numeric column contains inf or -inf.

    This is the most critical check — log(0) produces -inf and will
    silently corrupt GARCH and LSTM inputs without raising an error.
    """
    numeric_df = df.select_dtypes(include=[float, int])
    return bool(np.isinf(numeric_df.values).any())


def _check_asset_coverage(df: pd.DataFrame) -> list[str]:
    """
    Return list of expected assets absent from df.

    A missing asset means the ingestion or feature pipeline dropped
    it silently — no row will exist in the intelligence output.

    Returns:
        List of VALID_ASSETS not present in the 'asset' column.
        Empty list if all assets are present.
    """
    if "asset" not in df.columns:
        return VALID_ASSETS.copy()
    present = set(df["asset"].unique())
    return [a for a in VALID_ASSETS if a not in present]


def _check_row_counts(df: pd.DataFrame) -> dict[str, int]:
    """
    Return row count per asset.

    Used to flag assets with so few rows that rolling windows
    will produce mostly NaNs regardless of threshold checks.

    Returns:
        Dict mapping asset symbol to row count.
    """
    if "asset" not in df.columns:
        return {}
    return df.groupby("asset").size().to_dict()


def _build_result(
    context:        str,
    passed:         bool,
    row_count:      int,
    asset_count:    int,
    null_rates:     dict[str, float],
    has_infinities: bool,
    missing_assets: list[str],
    rows_per_asset: dict[str, int],
    warnings:       list[str],
    errors:         list[str],
) -> dict:
    """Assemble the standardised validation result dictionary."""
    return {
        "passed":          passed,
        "context":         context,
        "row_count":       row_count,
        "asset_count":     asset_count,
        "null_rates":      null_rates,
        "has_infinities":  has_infinities,
        "missing_assets":  missing_assets,
        "rows_per_asset":  rows_per_asset,
        "warnings":        warnings,
        "errors":          errors,
    }


# ── Public Validator ──────────────────────────────────────────────────────────

def validate_feature_df(
    df:             pd.DataFrame,
    context:        str,
    raise_on_error: bool = True,
) -> dict:
    """
    Run lightweight validation checks on a feature DataFrame.

    Called after each feature builder (price, market, sentiment)
    and before writing to asset_features. Catches infinities, excessive
    nulls, and silent asset dropout before they reach model inference.

    Args:
        df:             Feature DataFrame to validate.
        context:        Label for log messages e.g. "price_features".
                        Used to identify which stage failed.
        raise_on_error: If True, raises ValueError on hard failures.
                        Set False in tests or graceful degradation paths.

    Returns:
        Validation result dict with keys:
            passed          bool        True if no hard errors
            context         str         Label passed in
            row_count       int         Total rows in df
            asset_count     int         Distinct assets present
            null_rates      dict        Null rate per numeric column
            has_infinities  bool        True if any inf/-inf found
            missing_assets  list[str]   VALID_ASSETS absent from df
            rows_per_asset  dict        Row count per asset
            warnings        list[str]   Soft issues — pipeline continues
            errors          list[str]   Hard failures — pipeline should stop

    Raises:
        ValueError: If raise_on_error=True and hard errors are present.
    """
    warnings: list[str] = []
    errors:   list[str] = []

    # ── Guard: empty DataFrame ────────────────────────────────────────────
    if df.empty:
        errors.append("DataFrame is empty — no features to validate")
        result = _build_result(
            context=context,
            passed=False,
            row_count=0,
            asset_count=0,
            null_rates={},
            has_infinities=False,
            missing_assets=VALID_ASSETS.copy(),
            rows_per_asset={},
            warnings=warnings,
            errors=errors,
        )
        logger.error(f"[{context}] Validation FAILED: {errors}")
        if raise_on_error:
            raise ValueError(f"[{context}] Validation failed: {errors}")
        return result

    # ── Check 1: Infinities ───────────────────────────────────────────────
    has_inf = _check_infinities(df)
    if has_inf:
        errors.append(
            "Infinite values detected — likely log(0) in return calculation. "
            "Check for zero or negative close prices upstream."
        )

    # ── Check 2: Null rates ───────────────────────────────────────────────
    null_rates = _check_nulls(df)
    for col, rate in null_rates.items():
        if rate >= NULL_RATE_ERROR_THRESHOLD:
            errors.append(
                f"Column '{col}' null rate {rate:.1%} exceeds "
                f"{NULL_RATE_ERROR_THRESHOLD:.0%} threshold — feature is unusable"
            )
        elif rate >= NULL_RATE_WARNING_THRESHOLD:
            warnings.append(
                f"Column '{col}' null rate {rate:.1%} — "
                f"verify rolling window size is appropriate"
            )

    # ── Check 3: Asset coverage ───────────────────────────────────────────
    missing_assets = _check_asset_coverage(df)
    if missing_assets:
        warnings.append(
            f"Assets absent from DataFrame: {missing_assets} — "
            f"these will produce no intelligence output"
        )

    # ── Check 4: Row counts per asset ─────────────────────────────────────
    rows_per_asset = _check_row_counts(df)
    thin_assets = [
        asset for asset, count in rows_per_asset.items()
        if count < MIN_ROWS_PER_ASSET
    ]
    if thin_assets:
        warnings.append(
            f"Assets with fewer than {MIN_ROWS_PER_ASSET} rows: {thin_assets} — "
            f"rolling windows will produce mostly nulls"
        )

    # ── Result ────────────────────────────────────────────────────────────
    passed = len(errors) == 0

    result = _build_result(
        context=context,
        passed=passed,
        row_count=len(df),
        asset_count=df["asset"].nunique() if "asset" in df.columns else 0,
        null_rates=null_rates,
        has_infinities=has_inf,
        missing_assets=missing_assets,
        rows_per_asset=rows_per_asset,
        warnings=warnings,
        errors=errors,
    )

    if passed:
        logger.success(
            f"[{context}] Validation passed | "
            f"{result['row_count']} rows | {result['asset_count']} assets | "
            f"{len(warnings)} warning(s)"
        )
    else:
        logger.error(f"[{context}] Validation FAILED: {errors}")
        if raise_on_error:
            raise ValueError(f"[{context}] Validation failed: {errors}")

    for w in warnings:
        logger.warning(f"[{context}] {w}")

    return result