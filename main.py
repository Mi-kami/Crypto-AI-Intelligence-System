"""
main.py

Ingestion pipeline orchestrator for the Crypto AI Intelligence System.

Coordinates all three data sources — price, market, and news — by calling
the ingestion clients, passing raw data through the harmoniser, and writing
clean DataFrames to storage.

Each data source runs independently. A failure in one source must never
prevent the other two from completing.

Phase 1: Storage writes are placeholders — replaced in Phase 2.

Run manually:
    python main.py
"""

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
    write_news_headlines
)


# ── Pipeline Orchestrator ─────────────────────────────────────────────────────

def run_ingestion_pipeline(days: int = 1) -> None:
    """
    Run the full data ingestion pipeline for all three data sources.

    Fetches price, market, and news data sequentially. Each source is
    wrapped in its own try/except block so a failure in one source
    never prevents the other two from completing.

    Args:
        days: Number of days of OHLCV history to fetch per asset (default 1)
    """
    logger.info("─" * 60)
    logger.info("Ingestion pipeline started")
    logger.info("─" * 60)

    # Track which sources succeeded and which failed
    results = {"price": False, "market": False, "news": False}

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

    # ── News Data (Cryptocomparec) ──────────────────────────────────────────
    try:
        logger.info("Starting news data ingestion...")
        raw_news_df = fetch_all_assets_news()
        clean_news_df = harmonise_news_data(raw_news_df)
        write_news_headlines(clean_news_df)
        results["news"] = True

    except Exception as e:
        logger.error(f"News ingestion failed — skipping. Error: {e}")

    # ── Pipeline Summary ─────────────────────────────────────────────────
    logger.info("─" * 60)
    succeeded = [source for source, ok in results.items() if ok]
    failed = [source for source, ok in results.items() if not ok]

    if succeeded:
        logger.success(f"Pipeline complete — succeeded: {succeeded}")
    if failed:
        logger.warning(f"Pipeline complete — failed sources: {failed}")
    if not failed:
        logger.success("All three sources ingested successfully")

    logger.info("─" * 60)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_ingestion_pipeline()