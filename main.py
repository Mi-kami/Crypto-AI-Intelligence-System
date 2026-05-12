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
    pass


# ── Model Pipeline ────────────────────────────────────────────────────────────

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
    pass


# ── Top-Level Orchestrator ────────────────────────────────────────────────────

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
    pass


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()