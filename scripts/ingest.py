"""Download and store daily OHLCV data incrementally.

Examples:
    python scripts/ingest.py                              # default universe, DEFAULT_START_DATE onward
    python scripts/ingest.py --tickers AAPL MSFT --start 2015-01-01
"""

from __future__ import annotations

import argparse
import sys

from fmip.config import load_settings
from fmip.data.repository import MarketRepository, create_db_engine
from fmip.data.service import ingest_tickers
from fmip.logging_config import get_logger, setup_logging
from fmip.universe import BENCHMARK_INDEX, STOCK_UNIVERSE

log = get_logger("scripts.ingest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest historical OHLCV data into PostgreSQL.")
    parser.add_argument("--tickers", nargs="*", default=None,
                        help="Tickers to load. Defaults to the built-in universe plus the benchmark index.")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (default from DEFAULT_START_DATE).")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD, exclusive (default: today).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings()
    setup_logging(settings.log_level)
    if not settings.database_configured:
        log.error("DATABASE_URL is not set. Copy .env.example to .env and fill it in.")
        return 1

    tickers = args.tickers or [*STOCK_UNIVERSE.keys(), BENCHMARK_INDEX]
    repository = MarketRepository(create_db_engine(settings.database_url))
    results = ingest_tickers(repository, tickers, args.start or settings.default_start_date, args.end or None)

    failed = [r for r in results if r.status == "failed"]
    for result in results:
        log.info("%-8s %-8s %s", result.ticker, result.status, result.message)
    log.info("Done: %d ok, %d skipped, %d failed",
             sum(r.status == "ok" for r in results),
             sum(r.status == "skipped" for r in results),
             len(failed))
    return 1 if failed and len(failed) == len(results) else 0


if __name__ == "__main__":
    sys.exit(main())
