"""Run the anomaly engine over stored prices and save results to the database.

Usage:
    python scripts/run_analysis.py                  # all stored tickers
    python scripts/run_analysis.py --tickers AAPL NVDA
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from fmip.analysis.anomalies import detect_all
from fmip.config import load_settings
from fmip.data.repository import MarketRepository, create_db_engine
from fmip.exceptions import InsufficientDataError
from fmip.logging_config import get_logger, setup_logging
from fmip.universe import BENCHMARK_INDEX, STOCK_UNIVERSE

log = get_logger("scripts.run_analysis")


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect anomalies and store them in PostgreSQL.")
    parser.add_argument("--tickers", nargs="*", default=None)
    args = parser.parse_args()

    settings = load_settings()
    setup_logging(settings.log_level)
    if not settings.database_configured:
        log.error("DATABASE_URL is not set.")
        return 1

    repository = MarketRepository(create_db_engine(settings.database_url))
    tickers = args.tickers or [*STOCK_UNIVERSE.keys(), BENCHMARK_INDEX]
    frames = []
    for ticker in tickers:
        prices = repository.load_prices([ticker])
        if prices.empty:
            log.warning("No stored prices for %s; run scripts/ingest.py first", ticker)
            continue
        ohlcv = prices.set_index("trade_date")[["open", "high", "low", "close", "adj_close", "volume"]]
        ohlcv.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        try:
            frames.append(detect_all(ticker, ohlcv, window=settings.rolling_window,
                                     threshold=settings.zscore_threshold,
                                     contamination=settings.isolation_contamination))
        except InsufficientDataError as exc:
            log.warning("Skipping %s: %s", ticker, exc)

    if not frames:
        log.error("No anomaly results produced")
        return 1
    combined = pd.concat(frames, ignore_index=True)
    stored = repository.replace_anomalies(combined)
    log.info("Stored %d anomaly records across %d tickers", stored, combined["ticker"].nunique())
    return 0


if __name__ == "__main__":
    sys.exit(main())
