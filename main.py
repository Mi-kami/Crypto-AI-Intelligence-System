"""
main.py

Full pipeline orchestrator for the Crypto AI Intelligence System.

Coordinates all pipeline stages sequentially:
    1. Ingestion    — fetch, harmonise, write raw data to storage
    2. Feature      — read from storage, compute features
    3. Model        — run intelligence model inference

Each stage gates the next. Ingestion failure aborts feature engineering.
Feature engineering failure aborts model inference.

Run manually:
    python main.py
"""

import pandas as pd
from loguru import logger

from ingestion.coingecko_client import fetch_all_assets, fetch_global_market_data
from ingestion.cryptocompare_client import fetch_all_assets_news
from ingestion.harmoniser import (
    harmonise_price_data,
    harmonise_market_data,
    harmonise_news_data,
)
from storage.writer import (
    write_price_data,
    write_market_signals,
    write_news_headlines,
)
from storage.reader import (
    read_price_data,
    read_market_signals,
    read_news_headlines,
)


# ── Ingestion Pipeline ────────────────────────────────────────────────────────

def run_ingestion_pipeline(days: int = 1) -> dict:
    """
    Run the full data ingestion pipeline for all three data sources.

    Fetches and writes price, market, and news data sequentially into the database.
    Each source is wrapped in its own try/except block so a failure in one source
    never prevents the other two from completing.

    Args:
        days: Number of days of OHLCV history to fetch per asset (default 1)

    Returns:
        Dictionary tracking success/failure of each ingestion source.
    """
    logger.info("─" * 60)
    logger.info("Ingestion pipeline started")
    logger.info("─" * 60)

    results = {
        "price":  False,
        "market": False,
        "news":   False,
    }

    # ── Price Data (CoinGecko OHLCV) ─────────────────────────────────────
    try:
        logger.info("Starting price data ingestion...")
        raw_price_df = fetch_all_assets(days=days)
        clean_price_df = harmonise_price_data(raw_price_df)
        write_price_data(clean_price_df)
        results["price"] = True

    except Exception as e:
        logger.error(f"Price ingestion failed — skipping. Error: {e}")

    # ── Market Data (CoinGecko Global) ───────────────────────────────────
    try:
        logger.info("Starting market data ingestion...")
        raw_market_dict = fetch_global_market_data()
        clean_market_df = harmonise_market_data(raw_market_dict)
        write_market_signals(clean_market_df)
        results["market"] = True

    except Exception as e:
        logger.error(f"Market ingestion failed — skipping. Error: {e}")

    # ── News Data (CryptoCompare) ─────────────────────────────────────────
    try:
        logger.info("Starting news data ingestion...")
        raw_news_df = fetch_all_assets_news()
        clean_news_df = harmonise_news_data(raw_news_df)
        write_news_headlines(clean_news_df)
        results["news"] = True

    except Exception as e:
        logger.error(f"News ingestion failed — skipping. Error: {e}")

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("─" * 60)
    succeeded = [k for k, v in results.items() if v]
    failed    = [k for k, v in results.items() if not v]

    if succeeded:
        logger.success(f"Ingestion complete — succeeded: {succeeded}")
    if failed:
        logger.warning(f"Ingestion complete — failed: {failed}")

    logger.info("─" * 60)

    return results


# ── Feature Pipeline ──────────────────────────────────────────────────────────
def run_feature_pipeline() -> pd.DataFrame:
    """
    Read raw data from storage, compute price, market, sentiment,
    and master features, and return a single unified DataFrame
    ready for model inference.

    Returns:
        DataFrame containing all computed features across all assets.
        Returns empty DataFrame if any critical feature stage fails.
    """
    logger.info("─" * 60)
    logger.info("Feature pipeline started")
    logger.info("─" * 60)

    results = {
        "price_features":     False,
        "market_features":    False,
        "sentiment_features": False,
        "master_features":    False,
    }

    # ── Price Features ────────────────────────────────────────────────────
    try:
        logger.info("Computing price features...")
        # Phase 3: call build_price_features() here
        results["price_features"] = True

    except Exception as e:
        logger.error(f"Price features failed: {e}")

    # ── Market Features ───────────────────────────────────────────────────
    try:
        logger.info("Computing market features...")
        # Phase 3: call build_market_features() here
        results["market_features"] = True

    except Exception as e:
        logger.error(f"Market features failed: {e}")

    # ── Sentiment Features ────────────────────────────────────────────────
    try:
        logger.info("Computing sentiment features...")
        # Phase 3: call build_sentiment_features() here
        results["sentiment_features"] = True

    except Exception as e:
        logger.error(f"Sentiment features failed: {e}")

    # ── Master Features (conditional execution gate) ───────────────────────
    # Master depends on price and market. Sentiment failure is tolerated.
    if not results["price_features"]:
        logger.warning("Master features aborted — price features failed")
    elif not results["market_features"]:
        logger.warning("Master features aborted — market features failed")
    else:
        try:
            logger.info("Computing master features...")
            # Phase 3: call build_master_features() here
            results["master_features"] = True

        except Exception as e:
            logger.error(f"Master features failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("─" * 60)
    succeeded = [k for k, v in results.items() if v]
    failed    = [k for k, v in results.items() if not v]

    if succeeded:
        logger.success(f"Feature pipeline complete — succeeded: {succeeded}")
    if failed:
        logger.warning(f"Feature pipeline complete — failed: {failed}")

    logger.info("─" * 60)

    return pd.DataFrame()


def run_model_pipeline(features: pd.DataFrame) -> dict:
    """
    Run inference across all four intelligence models — volatility
    (GARCH+LSTM), regime (HMM), sentiment (FinBERT), and composite
    risk scorer — using the unified feature DataFrame as input.

    Args:
        features: Unified feature DataFrame from run_feature_pipeline()

    Returns:
        Dictionary tracking success/failure of each model stage.
    """
    logger.info("─" * 60)
    logger.info("Model pipeline started")
    logger.info("─" * 60)

    results = {
        "volatility": False,
        "regime":     False,
        "sentiment":  False,
        "risk":       False,
    }

    # ── Volatility (GARCH + LSTM) ─────────────────────────────────────────
    try:
        logger.info("Running volatility model...")
        # Phase 4: call run_volatility_model(features) here
        results["volatility"] = True

    except Exception as e:
        logger.error(f"Volatility model failed: {e}")

    # ── Regime Detection (HMM) ────────────────────────────────────────────
    if not results["volatility"]:
        logger.warning("Regime detection aborted — volatility model failed")
    else:
        try:
            logger.info("Running regime detection...")
            # Phase 4: call run_regime_model(features) here
            results["regime"] = True

        except Exception as e:
            logger.error(f"Regime detection failed: {e}")

    # ── Sentiment Analysis (FinBERT) ──────────────────────────────────────
    if not results["regime"]:
        logger.warning("Sentiment model aborted — regime detection failed")
    else:
        try:
            logger.info("Running sentiment analysis...")
            # Phase 4: call run_sentiment_model(features) here
            results["sentiment"] = True

        except Exception as e:
            logger.error(f"Sentiment model failed: {e}")

    # ── Risk Scorer (composite) ───────────────────────────────────────────
    # Risk requires volatility and regime. Sentiment failure is tolerated.
    if not results["volatility"]:
        logger.warning("Risk scorer aborted — volatility model failed")
    elif not results["regime"]:
        logger.warning("Risk scorer aborted — regime detection failed")
    else:
        try:
            logger.info("Running risk scorer...")
            # Phase 4: call run_risk_scorer(features) here
            results["risk"] = True

        except Exception as e:
            logger.error(f"Risk scorer failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("─" * 60)
    succeeded = [k for k, v in results.items() if v]
    failed    = [k for k, v in results.items() if not v]

    if succeeded:
        logger.success(f"Model pipeline complete — succeeded: {succeeded}")
    if failed:
        logger.warning(f"Model pipeline complete — failed: {failed}")

    logger.info("─" * 60)

    return results


def run_pipeline(days: int = 1) -> dict:
    """
    Top-level orchestrator for the full Crypto AI Intelligence pipeline.

    Calls run_ingestion_pipeline(), run_feature_pipeline(), and
    run_model_pipeline() sequentially. Each stage gates the next —
    if ingestion fails, feature engineering does not run.
    If feature engineering fails, model inference does not run.

    Args:
        days: Number of days of OHLCV history to fetch (default 1)

    Returns:
        Dictionary tracking success/failure of all pipeline stages.
    """
    logger.info("═" * 60)
    logger.info("Full pipeline started")
    logger.info("═" * 60)

    # ── Stage 1: Ingestion ────────────────────────────────────────────────
    ingestion_results = run_ingestion_pipeline(days=days)

    # Gate: both price and market must succeed for features to be meaningful
    if not ingestion_results["price"] or not ingestion_results["market"]:
        logger.error(
            "Pipeline aborted — critical ingestion sources failed. "
            "Feature engineering will not run."
        )
        return {**ingestion_results}

    # ── Stage 2: Feature Engineering ─────────────────────────────────────
    feature_df = run_feature_pipeline()

    feature_results = {
        "price_features":     not feature_df.empty,
        "market_features":    not feature_df.empty,
        "sentiment_features": not feature_df.empty,
        "master_features":    not feature_df.empty,
    }

    # Gate: master features must exist for models to run
    if feature_df.empty:
        logger.error(
            "Pipeline aborted — feature engineering produced no output. "
            "Model inference will not run."
        )
        return {**ingestion_results, **feature_results}

    # ── Stage 3: Model Inference ──────────────────────────────────────────
    model_results = run_model_pipeline(features=feature_df)

    # ── Final Summary ─────────────────────────────────────────────────────
    all_results = {**ingestion_results, **feature_results, **model_results}

    logger.info("═" * 60)
    succeeded = [k for k, v in all_results.items() if v]
    failed    = [k for k, v in all_results.items() if not v]

    if succeeded:
        logger.success(f"Full pipeline complete — succeeded: {succeeded}")
    if failed:
        logger.warning(f"Full pipeline complete — failed: {failed}")

    logger.info("═" * 60)

    return all_results

# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()