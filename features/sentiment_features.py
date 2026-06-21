"""
features/sentiment_features.py

Responsible for computing per-asset hourly sentiment features
from raw news headlines stored in the news_headlines table.

For each asset, for each hour, this module:
    1. Groups all headlines published in that hour
    2. Scores each headline using VADER (Valence Aware Dictionary
       and sEntiment Reasoner) — a lexicon-based sentiment tool
       that requires no model download or GPU
    3. Aggregates scores across headlines into a single row

Features produced per (asset, hour):
    headline_count    — how many headlines were available
    sentiment_score   — mean VADER compound score (-1.0 to +1.0)
    sentiment_label   — "positive", "negative", or "neutral"

Phase 4 note:
    FinBERT will replace VADER as the scoring engine. The aggregation
    structure — one row per (asset, hour) — does not change.
"""

# Standard library
from datetime import datetime, timezone

# Third party
import pandas as pd
from loguru import logger
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Local
from tracking.mlflow_tracker import log_feature_params, log_feature_metrics
from validation.feature_validator import validate_feature_df


# ── Constants ─────────────────────────────────────────────────────────────────

FEATURE_VERSION = "v1"

# Standard VADER thresholds used in research literature
POSITIVE_THRESHOLD =  0.05
NEGATIVE_THRESHOLD = -0.05


# ── Private Helpers ───────────────────────────────────────────────────────────

def _score_headline(analyzer: SentimentIntensityAnalyzer, headline: str) -> float:
    """
    Score a single headline using VADER and return the compound score.

    Compound score ranges from -1.0 (most negative) to +1.0 (most positive).
    Returns 0.0 if the headline is empty or not a string — treated as neutral
    rather than an error so one bad headline never crashes the pipeline.

    Args:
        analyzer: Initialised VADER SentimentIntensityAnalyzer instance.
                  Passed in rather than created here so it is built once
                  and reused across all headlines — not rebuilt per call.
        headline: Raw headline text string.

    Returns:
        Float compound score between -1.0 and +1.0.
    """
    if not headline or not isinstance(headline, str):
        return 0.0
    return analyzer.polarity_scores(headline)["compound"]


def _aggregate_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-headline sentiment scores into one row per
    (asset, timestamp) combination.

    For each asset-hour pair, computes:
        headline_count  — number of headlines scored in that hour
        sentiment_score — mean compound score across all headlines
        sentiment_label — "positive", "negative", or "neutral"
                          based on where the mean score lands

    Args:
        df: DataFrame with columns: asset, timestamp, compound_score.

    Returns:
        DataFrame with one row per (asset, timestamp).
    """
    aggregated = (
        df.groupby(["asset", "timestamp"])["compound_score"]
        .agg(
            headline_count="count",
            sentiment_score="mean",
        )
        .reset_index()
    )

    aggregated["sentiment_label"] = aggregated["sentiment_score"].apply(
        lambda score: "positive" if score >= POSITIVE_THRESHOLD
                      else "negative" if score <= NEGATIVE_THRESHOLD
                      else "neutral"
    )

    return aggregated


# ── Public API ────────────────────────────────────────────────────────────────

def build_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-asset hourly sentiment features from raw news headlines.

    Scores each headline with VADER, aggregates scores per (asset, hour),
    and returns a validated feature DataFrame ready for master_features.py.

    Processing steps:
        1. Guard clause — reject None or empty input immediately
        2. Defensive copy — never mutate the caller's DataFrame
        3. Initialise VADER analyzer once — reused across all headlines
        4. Score each headline — compound score added as new column
        5. Floor timestamps to hour — align with price and market features
        6. Aggregate per (asset, hour) — one row per combination
        7. Add feature_version and created_at metadata
        8. Select only output columns — drop raw inputs
        9. Validate output with feature_validator
        10. Log parameters and metrics to MLflow
        11. Return clean DataFrame

    Args:
        df: Raw DataFrame from read_news_headlines() with columns:
            asset, timestamp, headline, url, source.

    Returns:
        DataFrame with columns:
            asset, timestamp, headline_count, sentiment_score,
            sentiment_label, feature_version, created_at.

    Raises:
        ValueError: If df is None or empty.
    """
    # ── 1. Guard clause ───────────────────────────────────────────────────
    if df is None or df.empty:
        raise ValueError(
            "build_sentiment_features received empty or None DataFrame. "
            "Check that read_news_headlines() returned data."
        )

    logger.info(f"build_sentiment_features started | input rows: {len(df)}")

    # ── 2. Defensive copy ─────────────────────────────────────────────────
    # We are about to add columns to df. Without .copy(), we risk mutating
    # the original DataFrame that was passed in by the caller.
    df = df.copy()

    # ── 3. Initialise VADER once ──────────────────────────────────────────
    # Built here, outside _score_headline, so the dictionary loads once
    # and is reused across every headline rather than rebuilt per call.
    analyzer = SentimentIntensityAnalyzer()

    # ── 4. Score each headline ────────────────────────────────────────────
    df["compound_score"] = df["headline"].apply(
        lambda headline: _score_headline(analyzer, headline)
    )

    # ── 5. Floor timestamps to hour ───────────────────────────────────────
    # Ensures alignment with price_features and market_features which also
    # floor to the hour. Without this, the master join on timestamp fails.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("h")

    # ── 6. Aggregate per (asset, hour) ────────────────────────────────────
    result = _aggregate_sentiment(df)

    # ── 7. Add metadata columns ───────────────────────────────────────────
    result["feature_version"] = FEATURE_VERSION
    result["created_at"]      = datetime.now(tz=timezone.utc).isoformat()

    # ── 8. Select output columns ──────────────────────────────────────────
    output_columns = [
        "asset",
        "timestamp",
        "headline_count",
        "sentiment_score",
        "sentiment_label",
        "feature_version",
        "created_at",
    ]
    result = result[output_columns]

    # ── 9. Validate ───────────────────────────────────────────────────────
    validate_feature_df(result, context="sentiment_features", raise_on_error=True)

    # ── 10. Log to MLflow ─────────────────────────────────────────────────
    label_counts = result["sentiment_label"].value_counts()

    log_feature_params({
        "sentiment_feature_version": FEATURE_VERSION,
        "positive_threshold":        POSITIVE_THRESHOLD,
        "negative_threshold":        NEGATIVE_THRESHOLD,
        "scoring_engine":            "vader",
    })

    log_feature_metrics({
        "sentiment_row_count":  int(len(result)),
        "positive_hours":       int(label_counts.get("positive", 0)),
        "negative_hours":       int(label_counts.get("negative", 0)),
        "neutral_hours":        int(label_counts.get("neutral",  0)),
        "mean_headline_count":  float(result["headline_count"].mean()),
    })

    logger.success(
        f"build_sentiment_features complete | {len(result)} rows | "
        f"positive={label_counts.get('positive', 0)} | "
        f"negative={label_counts.get('negative', 0)} | "
        f"neutral={label_counts.get('neutral', 0)}"
    )

    return result