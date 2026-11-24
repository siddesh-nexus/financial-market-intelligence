"""Incremental ingestion: fetch only what the database is missing, then serve price panels.

Caching strategy:
- The database is the cache. A ticker is re-downloaded only from its latest stored date
  (minus a short overlap so revised bars are picked up), never from scratch.
- Backfill requests are remembered per ticker (`stocks.history_requested_from`), so a listing
  that starts later than the requested date is not requested again on every run.
- Failures are isolated per ticker, so one bad symbol does not stop the batch.

Data-integrity rules:
- **Corporate actions.** Yahoo back-adjusts the whole history when a split or dividend occurs.
  Each incremental download is compared with the stored overlap rows. If stored `close` or
  `adj_close` no longer match, the full history is re-downloaded so every stored bar uses
  the same adjustment basis. Otherwise a split would appear as an artificial one-day crash.
- **Incomplete sessions.** A bar for the current session is dropped unless that exchange has
  closed (in its own time zone), so intraday partial bars are never stored as daily closes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

import numpy as np
import pandas as pd

from fmip.data.fetcher import fetch_history
from fmip.data.repository import MarketRepository
from fmip.data.validation import clean_ohlcv
from fmip.exceptions import DataFetchError, DataValidationError, UnknownTickerError
from fmip.logging_config import get_logger
from fmip.universe import exchange_of, known_names

log = get_logger(__name__)

OVERLAP_DAYS = 5
BACKFILL_TOLERANCE_DAYS = 7
ADJUSTMENT_TOLERANCE = 1e-4          # relative difference that counts as a changed adjustment
SESSION_SETTLE_MINUTES = 30          # wait this long after the close before trusting a daily bar
Fetcher = Callable[..., pd.DataFrame]

# Regular-session closing times by exchange time zone (local time).
EXCHANGE_CLOSE: dict[str, time] = {
    "America/New_York": time(16, 0),
    "America/Toronto": time(16, 0),
    "America/Chicago": time(15, 0),
    "Asia/Kolkata": time(15, 30),
    "Asia/Calcutta": time(15, 30),
    "Europe/London": time(16, 30),
    "Europe/Berlin": time(17, 30),
    "Europe/Paris": time(17, 30),
    "Europe/Amsterdam": time(17, 30),
    "Europe/Zurich": time(17, 30),
    "Asia/Tokyo": time(15, 30),
    "Asia/Hong_Kong": time(16, 10),
    "Asia/Shanghai": time(15, 0),
    "Asia/Seoul": time(15, 30),
    "Australia/Sydney": time(16, 10),
}
DEFAULT_CLOSE = time(17, 0)


@dataclass(frozen=True)
class IngestResult:
    ticker: str
    status: str  # "ok", "skipped" or "failed"
    rows_written: int
    message: str
    full_refresh: bool = False


def plan_fetch_windows(
    bounds: tuple[date | None, date | None],
    requested_start: date,
    requested_end: date,
    requested_from: date | None = None,
    incremental: bool = True,
) -> list[tuple[date, date, str]]:
    """Decide which (start, end, reason) windows must be downloaded, given what is stored.

    - Nothing stored: download the whole requested range.
    - Stored history starts after the requested start, and no earlier request covered it:
      backfill the missing earlier range.
    - Stored history ends before the requested end (and `incremental` is on): download forward
      from the latest date, with a short overlap so revised bars are refreshed.
    `end` is exclusive, matching yfinance.
    """
    earliest, latest = bounds
    if earliest is None or latest is None:
        return [(requested_start, requested_end, "initial load")]

    windows: list[tuple[date, date, str]] = []
    known_from = min(earliest, requested_from) if requested_from else earliest
    # A small tolerance avoids re-requesting weekends/holidays before the first stored bar.
    if known_from > requested_start + timedelta(days=BACKFILL_TOLERANCE_DAYS):
        windows.append((requested_start, earliest, "backfill"))
    if incremental and latest < requested_end - timedelta(days=1):
        forward_start = max(requested_start, latest - timedelta(days=OVERLAP_DAYS))
        windows.append((forward_start, requested_end, "incremental"))
    return windows


def drop_incomplete_session(raw: pd.DataFrame, now: datetime | None = None) -> tuple[pd.DataFrame, int]:
    """Remove bars for sessions that have not closed yet in the exchange's own time zone.

    `raw` is a provider frame. With a timezone-aware index, the exchange close time is looked
    up from EXCHANGE_CLOSE (default 17:00 local). With a naive index, only future-dated bars
    are removed. Returns (frame, number of rows dropped).
    """
    if raw is None or raw.empty:
        return raw, 0
    index = pd.DatetimeIndex(raw.index)
    tz = index.tz
    if tz is not None:
        reference = now if now is not None else datetime.now(timezone.utc)
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)
        local_now = pd.Timestamp(reference).tz_convert(tz)
        close = EXCHANGE_CLOSE.get(str(tz), DEFAULT_CLOSE)
        settle = (datetime.combine(local_now.date(), close) + timedelta(minutes=SESSION_SETTLE_MINUTES)).time()
        today = local_now.date()
        session_done = local_now.time() >= settle
        dates = np.array([ts.date() for ts in index])
        incomplete = (dates > today) | ((dates == today) & (not session_done))
    else:
        today = (now or datetime.now()).date()
        dates = np.array([ts.date() for ts in index])
        incomplete = dates > today
    dropped = int(incomplete.sum())
    return raw[~incomplete], dropped


def adjustment_mismatch(stored: pd.DataFrame, fetched: pd.DataFrame,
                        tolerance: float = ADJUSTMENT_TOLERANCE) -> str | None:
    """Compare stored bars with a fresh download on their common dates.

    `stored` is long format (trade_date, close, adj_close); `fetched` is a cleaned frame indexed
    by date (Close, Adj Close). Returns a description of the first mismatch, or None.
    """
    if stored.empty or fetched.empty:
        return None
    old = stored.set_index(pd.to_datetime(stored["trade_date"]))[["close", "adj_close"]]
    new = fetched[["Close", "Adj Close"]].rename(columns={"Close": "close", "Adj Close": "adj_close"})
    common = old.index.intersection(new.index)
    if common.empty:
        return None
    for column in ("close", "adj_close"):
        ratio = (new.loc[common, column] / old.loc[common, column] - 1.0).abs()
        if (ratio > tolerance).any():
            worst = ratio.idxmax()
            return (f"{column} changed by {ratio.max():.4%} on {worst.date()} "
                    "(provider re-adjusted history for a split or dividend)")
    return None


def corporate_actions(raw: pd.DataFrame, after: date | None = None) -> list[str]:
    """Splits and dividends present in a provider frame, optionally only those after `after`."""
    notes: list[str] = []
    if raw is None or raw.empty:
        return notes
    dates = [ts.date() for ts in pd.DatetimeIndex(raw.index)]
    for column, label in (("Stock Splits", "split"), ("Dividends", "dividend")):
        if column not in raw.columns:
            continue
        values = pd.to_numeric(raw[column], errors="coerce").fillna(0.0).to_numpy()
        for day, value in zip(dates, values):
            if value > 0 and (after is None or day > after):
                notes.append(f"{label} {value:g} on {day}")
    return notes


def _metadata_frame(tickers: list[str], metadata: pd.DataFrame | None) -> pd.DataFrame:
    names = known_names()
    rows = []
    for ticker in tickers:
        name, sector = names.get(ticker, (None, None))
        rows.append({"ticker": ticker, "company_name": name,
                     "sector": None if sector == "Index" else sector,
                     "exchange": exchange_of(ticker),
                     "quote_type": "INDEX" if ticker.startswith("^") else None})
    frame = pd.DataFrame(rows)
    if metadata is not None and not metadata.empty:
        extra = metadata.set_index("ticker")
        frame = frame.set_index("ticker")
        for column in extra.columns:
            if column in frame.columns:
                frame[column] = extra[column].reindex(frame.index).combine_first(frame[column])
            else:
                frame[column] = extra[column].reindex(frame.index)
        frame = frame.reset_index()
    return frame


def ingest_tickers(
    repository: MarketRepository,
    tickers: list[str],
    start: date | str,
    end: date | str | None = None,
    fetcher: Fetcher = fetch_history,
    *,
    incremental: bool = True,
    metadata: pd.DataFrame | None = None,
    now: datetime | None = None,
) -> list[IngestResult]:
    """Download and store daily prices for `tickers`. Returns one result per ticker.

    `incremental=False` only loads tickers or date ranges that are missing entirely (no
    forward update of already-stored tickers). The dashboard uses it to avoid API calls on
    every interaction. `metadata` optionally supplies company_name/sector/exchange/currency.
    """
    requested_start = pd.Timestamp(start).date()
    requested_end = pd.Timestamp(end).date() if end else date.today() + timedelta(days=1)
    unique = list(dict.fromkeys(tickers))  # de-duplicate, keep order
    meta_frame = _metadata_frame(unique, metadata).set_index("ticker")
    # Metadata is written for symbols that are already stored now, and for new symbols only once
    # prices were stored, so mistyped tickers do not leave empty rows behind.
    stored_now = [t for t in unique if repository.price_date_bounds(t)[0] is not None]
    if stored_now:
        repository.upsert_stocks(meta_frame.loc[stored_now].reset_index())

    results: list[IngestResult] = []
    for ticker in unique:
        bounds = repository.price_date_bounds(ticker)
        windows = plan_fetch_windows(bounds, requested_start, requested_end,
                                     repository.history_requested_from(ticker), incremental)
        if not windows:
            results.append(IngestResult(ticker, "skipped", 0, "already up to date"))
            continue

        written = 0
        notes: list[str] = []
        full_refresh = False
        try:
            for window_start, window_end, reason in windows:
                try:
                    raw = fetcher(ticker, window_start, window_end)
                except UnknownTickerError:
                    if reason in ("backfill", "incremental"):
                        # Listing may start later than requested, or no new sessions exist yet.
                        notes.append(f"no rows returned ({reason})")
                        if reason == "backfill":
                            repository.mark_history_requested(ticker, requested_start)
                        continue
                    raise
                raw, partial = drop_incomplete_session(raw, now)
                if partial:
                    notes.append(f"skipped {partial} unfinished session bar(s)")
                if raw.empty:
                    notes.append(f"no completed sessions ({reason})")
                    continue
                cleaned, report = clean_ohlcv(raw, ticker)

                if reason == "incremental":
                    stored = repository.load_prices([ticker], window_start, bounds[1])
                    mismatch = adjustment_mismatch(stored, cleaned)
                    if mismatch is None and stored.empty and corporate_actions(raw, after=bounds[1]):
                        mismatch = "corporate action with no overlap rows to compare"
                    if mismatch:
                        full_refresh = True
                        history_start = min(bounds[0], requested_start) if bounds[0] else requested_start
                        log.warning("%s: %s; re-downloading full history from %s", ticker, mismatch, history_start)
                        notes.append(f"full refresh: {mismatch}")
                        refreshed_raw, _ = drop_incomplete_session(fetcher(ticker, history_start, window_end), now)
                        cleaned, report = clean_ohlcv(refreshed_raw, ticker)

                written += repository.upsert_prices(ticker, cleaned)
                if reason == "initial load":
                    repository.upsert_stocks(meta_frame.loc[[ticker]].reset_index())
                if reason in ("initial load", "backfill"):
                    repository.mark_history_requested(ticker, requested_start)
                notes.append(f"{reason}: {report.rows_out} rows, dropped {report.invalid_rows_dropped}, "
                             f"filled {report.prices_forward_filled} prices")
            results.append(IngestResult(ticker, "ok", written, "; ".join(notes), full_refresh))
        except (DataFetchError, DataValidationError) as exc:
            log.error("Ingestion failed for %s: %s", ticker, exc)
            results.append(IngestResult(ticker, "failed", written, str(exc), full_refresh))
    return results


def load_price_panel(
    repository: MarketRepository,
    tickers: list[str],
    start: date | str | None = None,
    end: date | str | None = None,
    column: str = "adj_close",
) -> pd.DataFrame:
    """Wide panel: index = trade_date, one column per ticker, values = `column`.

    Defaults to adjusted close so returns reflect dividends and splits. Dates are the union of
    all tickers' sessions; a NaN means that ticker's exchange did not trade that day.
    """
    if column not in {"open", "high", "low", "close", "adj_close", "volume"}:
        raise ValueError(f"Unknown price column: {column}")
    long = repository.load_prices(tickers, start, end)
    if long.empty:
        return pd.DataFrame(columns=tickers, index=pd.DatetimeIndex([], name="trade_date"))
    panel = long.pivot(index="trade_date", columns="ticker", values=column).sort_index()
    return panel.reindex(columns=[t for t in tickers if t in panel.columns])
