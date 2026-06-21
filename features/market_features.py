"""
features/market_features.py

Responsible for computing market-wide features from the global
market signals stored in the market_signals table.

Unlike price features, market features are NOT per-asset.
There is one row per timestamp — a single snapshot of the entire
cryptocurrency market at that hour. These features capture the
macro market environment that all 10 assets operate within.

Market features computed:
    btc_dominance_change  — how fast BTC's share of the market is
                            growing or shrinking (diff of dominance %)
    market_cap_return     — log return of total crypto market cap
    volume_change         — period-over-period USD volume shift
    dominance_regime      — categorical: "risk_off" if BTC dominance
                            > 50%, else "risk_on"

Output feeds into master_features.py, where it is joined to
per-asset price features on the timestamp column.
"""

# ── Imports ───────────────────────────────────────────────────────────────────

# Standard library
from datetime import datetime, timezone

# Third party
import numpy as np
import pandas as pd
from loguru import logger

# Local
from tracking.mlflow_tracker import log_feature_params, log_feature_metrics
from validation.feature_validator import validate_feature_df


# ── Constants ─────────────────────────────────────────────────────────────────

FEATURE_VERSION = "v1"

# The threshold above which BTC dominance signals a risk-off market.
# When BTC holds more than 50% of total crypto market cap, capital
# is flowing INTO BTC and OUT of altcoins — defensive positioning.
BTC_DOMINANCE_RISK_OFF_THRESHOLD = 50.0


# ── Private Helpers ───────────────────────────────────────────────────────────

def _compute_btc_dominance_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute period-over-period change in BTC dominance percentage.

    diff(1) subtracts the previous row's value from the current row's value.
    A positive result means BTC's share of the market grew this hour.
    A negative result means altcoins gained ground relative to BTC.
    Row 0 will always be NaN — there is no prior row to subtract from.

    Args:
        df: DataFrame sorted by timestamp, containing btc_dominance column.

    Returns:
        DataFrame with btc_dominance_change column added.
    """
    df["btc_dominance_change"] = df["btc_dominance"].diff(1)
    return df


def _compute_market_cap_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the log return of total cryptocurrency market capitalisation.

    log(current_market_cap / previous_market_cap) gives a normalised,
    additive measure of how much the total market grew or shrank this hour.
    We use log return — not simple percentage change — for the same reason
    we use it on price: it is scale-normalised and approximately normally
    distributed, which GARCH requires.

    Row 0 will always be NaN because shift(1) has no prior value to divide by.

    Args:
        df: DataFrame sorted by timestamp, containing total_market_cap_usd.

    Returns:
        DataFrame with market_cap_return column added.
    """
    df["market_cap_return"] = np.log(
        df["total_market_cap_usd"] / df["total_market_cap_usd"].shift(1)
    )
    return df


def _compute_volume_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute period-over-period change in total crypto trading volume (USD).

    diff(1) on total_volume_usd tells us whether more or less money
    moved through the market this hour compared to last hour. Rising volume
    is a momentum signal. Falling volume signals hesitation or consolidation.

    Row 0 will always be NaN.

    Args:
        df: DataFrame sorted by timestamp, containing total_volume_usd.

    Returns:
        DataFrame with volume_change column added.
    """
    df["volume_change"] = df["total_volume_usd"].diff(1)
    return df


def _compute_dominance_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each hour as risk_off or risk_on based on BTC dominance.

    When BTC holds more than 50% of total market cap, investors are
    concentrating in the "safer" crypto asset. This is risk-off behaviour —
    not safe like cash, but defensive within the crypto universe.

    When BTC dominance is below 50%, altcoins are absorbing capital.
    Investors are comfortable taking on more risk for higher returns.
    This is risk-on behaviour.

    This column produces no NaN values — every row gets a label.

    Args:
        df: DataFrame containing btc_dominance column.

    Returns:
        DataFrame with dominance_regime column added.
    """
    df["dominance_regime"] = np.where(
        df["btc_dominance"] > BTC_DOMINANCE_RISK_OFF_THRESHOLD,
        "risk_off",
        "risk_on",
    )
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def build_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all market-wide features from raw market signals data.

    Accepts the DataFrame returned by read_market_signals() and produces
    a clean, validated feature DataFrame ready for master_features.py.

    This function does NOT use groupby() — market data has no asset
    dimension. One row per timestamp. All computations are straight
    column operations on a sorted DataFrame.

    Processing steps:
        1. Guard clause — reject None or empty input immediately
        2. Sort by timestamp — diff/shift correctness depends on row order
        3. Compute btc_dominance_change
        4. Compute market_cap_return
        5. Compute volume_change
        6. Compute dominance_regime
        7. Add feature_version and created_at metadata
        8. Select only output columns
        9. Validate output with feature_validator
        10. Log parameters and metrics to MLflow
        11. Return clean DataFrame

    Args:
        df: Raw DataFrame from read_market_signals() with columns:
            timestamp, btc_dominance, total_market_cap_usd,
            total_volume_usd, source.

    Returns:
        DataFrame with columns:
            timestamp, btc_dominance_change, market_cap_return,
            volume_change, dominance_regime, feature_version, created_at.

    Raises:
        ValueError: If df is None or empty.
    """
    # ── 1. Guard clause ───────────────────────────────────────────────────
    # Fail loudly and immediately if the input is bad.
    # Never let bad data silently produce an empty output.
    if df is None or df.empty:
        raise ValueError(
            "build_market_features received empty or None DataFrame. "
            "Check that read_market_signals() returned data."
        )

    logger.info(f"build_market_features started | input rows: {len(df)}")

    # ── 2. Sort by timestamp ──────────────────────────────────────────────
    # diff() and shift() work on row position, not on the timestamp value.
    # If the rows are out of order, row[n] - row[n-1] gives a nonsense result.
    # reset_index(drop=True) cleans up any leftover index from the original df.
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── 3–6. Compute features ─────────────────────────────────────────────
    # Each helper adds one column and returns the df.
    # dominance_regime must come after btc_dominance is still present in df.
    # The others are independent of each other.
    df = _compute_btc_dominance_change(df)
    df = _compute_market_cap_return(df)
    df = _compute_volume_change(df)
    df = _compute_dominance_regime(df)

    # ── 7. Add metadata columns ───────────────────────────────────────────
    # feature_version lets us track which version of the feature logic
    # produced this row — important when we upgrade the formula in v2.
    # created_at is an audit trail — when was this feature computed?
    df["feature_version"] = FEATURE_VERSION
    df["created_at"]      = datetime.now(tz=timezone.utc).isoformat()

    # ── 8. Select only output columns ─────────────────────────────────────
    # The input df still has columns like btc_dominance, total_market_cap_usd,
    # total_volume_usd, source. We do not want those in the feature output.
    # Only the computed features + metadata go forward.
    output_columns = [
        "timestamp",
        "btc_dominance_change",
        "market_cap_return",
        "volume_change",
        "dominance_regime",
        "feature_version",
        "created_at",
    ]
    df = df[output_columns]

    # ── 9. Validate ───────────────────────────────────────────────────────
    # The validator checks null rates, infinities, and row counts.
    # raise_on_error=True means a bad output crashes the pipeline
    # rather than silently passing garbage to the models.
    validate_feature_df(df, context="market_features", raise_on_error=True)

    # ── 10. Log to MLflow ─────────────────────────────────────────────────
    # Parameters = inputs (what configuration produced this output)
    # Metrics = outputs (what does the result look like numerically)
    null_counts = df[
        ["btc_dominance_change", "market_cap_return", "volume_change"]
    ].isnull().sum()

    log_feature_params({
        "market_feature_version":           FEATURE_VERSION,
        "btc_dominance_risk_off_threshold": BTC_DOMINANCE_RISK_OFF_THRESHOLD,
    })

    log_feature_metrics({
        "market_features_row_count":              int(len(df)),
        "btc_dominance_change_null_count":        int(null_counts["btc_dominance_change"]),
        "market_cap_return_null_count":           int(null_counts["market_cap_return"]),
        "volume_change_null_count":               int(null_counts["volume_change"]),
        "risk_off_hours":                         int((df["dominance_regime"] == "risk_off").sum()),
        "risk_on_hours":                          int((df["dominance_regime"] == "risk_on").sum()),
    })

    logger.success(
        f"build_market_features complete | {len(df)} rows | "
        f"risk_off={(df['dominance_regime'] == 'risk_off').sum()} | "
        f"risk_on={(df['dominance_regime'] == 'risk_on').sum()}"
    )

    return df