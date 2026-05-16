# Crypto AI Intelligence System

> AI-powered intelligence system for cryptocurrency markets. Building in public.

## What This Will Be

An end-to-end AI system covering:

- **Volatility Prediction** : forecasting price volatility across crypto assets
- **Market Regime Detection** : identifying bull, bear and sideways market states
- **Sentiment Analysis** : processing live data sources for market sentiment signals
- **Risk Signal Generation** : combining signals into actionable risk indicators

## Stack
Python · Time-Series Forecasting · NLP · Scikit-learn · Streamlit · Docker · MLflow · SQLite · CI/CD · Railway

## Status
🔨 Currently building — active development.

## Current Progress
 **Phase 1 — Ingestion Pipeline (Complete)**
 - ingestion/coingecko_client.py — fetches hourly OHLCV price data and global market signals for 10 assets (BTC, ETH, BNB, XRP, SOL, DOGE, ADA, TRX, AVAX, SHIB) from the CoinGecko free tier API with exponential backoff      and jitter on rate limit responses
 - ingestion/cryptocompare_client.py — fetches cryptocurrency news headlines per asset from the CryptoCompare News API; replaced CryptoPanic which discontinued free developer access April 2026
 - ingestion/harmoniser.py — validates, normalises, and standardises raw data from both clients before storage; enforces column contracts, filters to supported assets, floors all timestamps to hourly UTC boundaries;  stateless functions, not classes
 - Full rate limit handling, retry logic, and per-asset failure isolation — one failed asset never breaks the pipeline

 **Phase 2 — Storage Layer (Nearly Complete)**
 - storage/schema.py — defines and initialises a SQLite database with three tables: price_ohlcv, market_signals, news_headlines; composite indexes on (asset, timestamp) for query performance
 - storage/writer.py — three writers: write_price_data (INSERT OR REPLACE), write_market_signals (INSERT OR REPLACE), write_news_headlines (INSERT OR IGNORE); all use executemany for efficiency, full try/except/finally on  every database connection
 - storage/reader.py — three parameterised readers returning UTC-aware DataFrames by asset and time window; SQL filtering applied at query level, not in Python
 - main.py — full pipeline orchestrator with three staged functions: run_ingestion_pipeline(), run_feature_pipeline(), run_model_pipeline(); explicit gate logic between stages; Phase 3 and Phase 4 placeholders in place
 65 passing tests across test_ingestion.py, test_writer.py, test_reader.py using pytest and unittest.mock

## Author
Deborah Olofin — Data Scientist & ML Engineer  
[Portfolio](https://Mi-kami.github.io) · [LinkedIn](https://linkedin.com/in/deborah-olofin)
