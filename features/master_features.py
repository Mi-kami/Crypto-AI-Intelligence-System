"""
features/master_features.py

Responsible for joining price, market, and sentiment features
into one unified DataFrame per asset per hour.

This is the final step in the feature engineering pipeline.
The output of this module is what the intelligence models consume.

Join architecture:
    Step 1 — price_features LEFT JOIN market_features ON timestamp
             Market data is market-wide. Every asset at a given hour
             receives the same market context for that hour.

    Step 2 — result LEFT JOIN sentiment_features ON (asset, timestamp)
             Sentiment is per-asset. Each asset gets its own sentiment
             context matched by both coin and hour.

    Left joins are used throughout. Price data is the source of truth.
    A missing market or sentiment row produces NaN — it never drops
    a price row from the output.
"""

# Standard library
from datetime import datetime, timezone

# Third party
import pandas as pd
from loguru import logger

# Local
from tracking.mlflow_tracker import log_feature_params, log_feature_metrics
from validation.feature_validator import validate_feature_df


# ── Constants ─────────────────────────────────────────────────────────────────

FEATURE_VERSION = "v1"


# ── Private Helpers ───────────────────────────────────────────────────────────

def _join_price_and_market(
    price_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left join price features to market features on timestamp.

    Every asset row in price_df receives the market context for
    that hour. If no market row exists for a given timestamp,
    market columns are NaN for that hour — the price row is kept.

    Args:
        price_df:  Output of build_price_features(). One row per
                   (asset, timestamp).
        market_df: Output of build_market_features(). One row per
                   timestamp.

    Returns:
        DataFrame with all price columns plus market columns.
        Row count equals price_df row count — never fewer.
    """
    # Drop metadata columns from market_df before joining —
    # we do not want duplicate feature_version or created_at columns
    market_cols_to_join = [
        "timestamp",
        "btc_dominance_change",
        "market_cap_return",
        "volume_change",
        "dominance_regime",
    ]

    merged = price_df.merge(
        market_df[market_cols_to_join],
        on="timestamp",
        how="left",
    )

    logger.info(
        f"_join_price_and_market complete | "
        f"price rows: {len(price_df)} | "
        f"output rows: {len(merged)}"
    )

    return merged


def _join_sentiment(
    df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left join sentiment features onto the price+market DataFrame
    on both asset and timestamp.

    Each asset's row receives only that asset's sentiment score for
    that hour. Joining on both columns prevents cross-asset mismatch —
    BTC's price row cannot accidentally receive ETH's sentiment score.

    If no sentiment exists for an (asset, timestamp) pair, sentiment
    columns are NaN — the row is never dropped.

    Args:
        df:           Price+market DataFrame from _join_price_and_market().
        sentiment_df: Output of build_sentiment_features(). One row per
                      (asset, timestamp).

    Returns:
        DataFrame with all price, market, and sentiment columns.
        Row count equals df row count — never fewer.
    """
    sentiment_cols_to_join = [
        "asset",
        "timestamp",
        "headline_count",
        "sentiment_score",
        "sentiment_label",
    ]

    merged = df.merge(
        sentiment_df[sentiment_cols_to_join],
        on=["asset", "timestamp"],
        how="left",
    )

    logger.info(
        f"_join_sentiment complete | "
        f"input rows: {len(df)} | "
        f"output rows: {len(merged)}"
    )

    return merged


# ── Public API ────────────────────────────────────────────────────────────────

def build_master_features(
    price_df: pd.DataFrame,
    market_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join price, market, and sentiment features into one unified DataFrame.

    This is the final feature engineering step. The output is consumed
    directly by the intelligence models in Phase 4.

    Processing steps:
        1. Guard clauses — price and market are mandatory.
                           sentiment absence is tolerated.
        2. Join price to market on timestamp
        3. Join sentiment on (asset, timestamp) if available
        4. Add feature_version and created_at metadata
        5. Sort by (asset, timestamp) for clean model input
        6. Validate output
        7. Log to MLflow
        8. Return unified DataFrame

    Args:
        price_df:     Output of build_price_features().
        market_df:    Output of build_market_features().
        sentiment_df: Output of build_sentiment_features().
                      May be empty — sentiment failure is tolerated.

    Returns:
        Unified DataFrame with all computed features per asset per hour.

    Raises:
        ValueError: If price_df or market_df is None or empty.
    """
    # ── 1. Guard clauses ──────────────────────────────────────────────────
    # Price and market are mandatory — without them the models have nothing.
    # Sentiment is optional — an hour without news still has a valid row.
    if price_df is None or price_df.empty:
        raise ValueError(
            "build_master_features requires price_df. "
            "Check that build_price_features() returned data."
        )

    if market_df is None or market_df.empty:
        raise ValueError(
            "build_master_features requires market_df. "
            "Check that build_market_features() returned data."
        )

    if sentiment_df is None or sentiment_df.empty:
        logger.warning(
            "build_master_features received empty sentiment_df — "
            "sentiment columns will be NaN. Pipeline continues."
        )
        sentiment_df = pd.DataFrame(
            columns=["asset", "timestamp", "headline_count",
                     "sentiment_score", "sentiment_label"]
        )

    logger.info(
        f"build_master_features started | "
        f"price rows: {len(price_df)} | "
        f"market rows: {len(market_df)} | "
        f"sentiment rows: {len(sentiment_df)}"
    )

    # ── 2. Join price to market ───────────────────────────────────────────
    result = _join_price_and_market(price_df, market_df)

    # ── 3. Join sentiment ─────────────────────────────────────────────────
    result = _join_sentiment(result, sentiment_df)

    # ── 4. Add metadata ───────────────────────────────────────────────────
    result["feature_version"] = FEATURE_VERSION
    result["created_at"]      = datetime.now(tz=timezone.utc).isoformat()

    # ── 5. Sort by (asset, timestamp) ─────────────────────────────────────
    # Clean, consistent ordering for model input.
    result = result.sort_values(
        ["asset", "timestamp"]
    ).reset_index(drop=True)

    # ── 6. Validate ───────────────────────────────────────────────────────
    validate_feature_df(result, context="master_features", raise_on_error=True)

    # ── 7. Log to MLflow ──────────────────────────────────────────────────
    sentiment_coverage = (
        result["sentiment_score"].notna().sum() / len(result)
    ) * 100

    log_feature_params({
        "master_feature_version": FEATURE_VERSION,
    })

    log_feature_metrics({
        "master_row_count":          int(len(result)),
        "master_asset_count":        int(result["asset"].nunique()),
        "sentiment_coverage_pct":    round(float(sentiment_coverage), 2),
    })

    logger.success(
        f"build_master_features complete | "
        f"{len(result)} rows | "
        f"{result['asset'].nunique()} assets | "
        f"sentiment coverage: {sentiment_coverage:.1f}%"
    )

    return result