"""Historical OHLCV fetcher backed by yfinance, with retries and clear errors."""

from __future__ import annotations

import time
from datetime import date

import pandas as pd

from fmip.exceptions import DataFetchError, UnknownTickerError
from fmip.logging_config import get_logger

log = get_logger(__name__)

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 2.0


def _to_date(value: date | str | None) -> str | None:
    if value is None or value == "":
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def fetch_history(
    ticker: str,
    start: date | str,
    end: date | str | None = None,
    *,
    max_attempts: int = MAX_ATTEMPTS,
    backoff_seconds: float = BACKOFF_SECONDS,
) -> pd.DataFrame:
    """Download daily OHLCV history for one ticker.

    `end` is exclusive, matching yfinance. Returns the raw frame as provided by yfinance, with a
    timezone-aware index in the exchange's local time and `Dividends` / `Stock Splits` columns
    (used to detect corporate actions that require a full re-download).
    `Close` is split-adjusted; `Adj Close` is split- and dividend-adjusted.
    Raises UnknownTickerError if the provider returns no rows, and DataFetchError if
    every attempt fails with a transport or API error.
    """
    import yfinance as yf  # imported lazily so unit tests do not need network libraries

    start_str = _to_date(start)
    end_str = _to_date(end)
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            history = yf.Ticker(ticker).history(
                start=start_str, end=end_str, interval="1d", auto_adjust=False, actions=True
            )
        except Exception as exc:  # yfinance raises a variety of network/parse errors
            last_error = exc
            log.warning("Fetch attempt %d/%d failed for %s: %s", attempt, max_attempts, ticker, exc)
            if attempt < max_attempts:
                time.sleep(backoff_seconds * attempt)
            continue

        if history is None or history.empty:
            raise UnknownTickerError(
                f"No price data returned for '{ticker}' between {start_str} and {end_str}. "
                "Check the symbol or the date range."
            )
        log.info("Fetched %d rows for %s (%s to %s)", len(history), ticker, start_str, end_str)
        return history

    raise DataFetchError(f"Failed to fetch {ticker} after {max_attempts} attempts: {last_error}")
