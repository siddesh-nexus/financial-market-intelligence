"""Shared dashboard plumbing: database connection, cached loaders and on-demand ingestion.

Caching layers:
1. The database is the long-term cache of provider data (incremental, see fmip.data.service).
2. `st.cache_data` caches query results and computations per argument set, so moving between
   pages or toggling a chart option does not re-query or re-compute.
3. Provider calls only happen when a selected symbol or date range is missing from the
   database, or when the user presses "Refresh prices".
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from fmip.analysis.anomalies import annotate, detect_all
from fmip.config import load_settings
from fmip.data.quality import check_bars, check_coverage, issues_frame
from fmip.data.repository import MarketRepository, create_db_engine, create_schema
from fmip.data.service import IngestResult, ingest_tickers, load_price_panel
from fmip.data.symbols import search_symbols
from fmip.exceptions import InsufficientDataError
from fmip.logging_config import get_logger
from fmip.universe import BENCHMARKS, STOCK_UNIVERSE, known_names

log = get_logger("app.common")

DISCLAIMER = (
    "Historical analysis only, from Yahoo Finance end-of-day data. Anomalies, correlations, clusters "
    "and regimes are statistical descriptions. They are not evidence of manipulation, of causation, "
    "or of future performance. Not investment advice."
)
PRICE_COLUMNS = {"price": "close", "total": "adj_close"}


# ---- database ---------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_repository() -> MarketRepository | None:
    settings = load_settings()
    if not settings.database_configured:
        return None
    engine = create_db_engine(settings.database_url)
    create_schema(engine)  # idempotent: creates missing tables, adds v0.2 columns
    return MarketRepository(engine)


def require_database() -> MarketRepository:
    """Stop the page with a helpful message if the database is not configured."""
    repository = get_repository()
    if repository is None:
        st.error("DATABASE_URL is not configured. Copy `.env.example` to `.env` and set DATABASE_URL "
                 "(or, for a quick start, `$env:DATABASE_URL = \"sqlite:///data/fmip_dev.db\"`).")
        st.stop()
    return repository


# ---- cached reads -----------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def cached_coverage(tickers: tuple[str, ...]) -> pd.DataFrame:
    repository = get_repository()
    if repository is None or not tickers:
        return pd.DataFrame(columns=["first_date", "last_date", "sessions", "last_ingested"])
    return repository.coverage(list(tickers))


@st.cache_data(ttl=600, show_spinner=False)
def cached_stored_tickers() -> list[str]:
    repository = get_repository()
    return [] if repository is None else list(repository.coverage().index)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_panel(tickers: tuple[str, ...], start: date | None, end: date | None, column: str) -> pd.DataFrame:
    repository = get_repository()
    if repository is None or not tickers:
        return pd.DataFrame()
    return load_price_panel(repository, list(tickers), start, end, column=column)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_bars(ticker: str, start: date | None, end: date | None) -> pd.DataFrame:
    """Lower-case OHLCV bars for one ticker, indexed by trade_date."""
    repository = get_repository()
    if repository is None:
        return pd.DataFrame()
    long = repository.load_prices([ticker], start, end)
    if long.empty:
        return pd.DataFrame()
    return long.set_index("trade_date")[["open", "high", "low", "close", "adj_close", "volume"]]


def cached_ohlcv(ticker: str, start: date | None, end: date | None) -> pd.DataFrame:
    """Title-case OHLCV frame as used by the analysis modules (Open, High, ..., Adj Close, Volume)."""
    bars = cached_bars(ticker, start, end)
    if bars.empty:
        return bars
    frame = bars.copy()
    frame.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    return frame


@st.cache_data(ttl=3600, show_spinner=False)
def cached_stock_meta(tickers: tuple[str, ...]) -> pd.DataFrame:
    repository = get_repository()
    if repository is None or not tickers:
        return pd.DataFrame(columns=["ticker", "company_name", "sector", "exchange", "currency",
                                     "quote_type", "history_requested_from"])
    return repository.load_stocks(list(tickers))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_search(query: str) -> pd.DataFrame:
    return search_symbols(query, limit=10)


def display_names(tickers: list[str]) -> dict[str, str]:
    """Company name per ticker: stored metadata first, then search results, built-in lists, ticker."""
    names = {t: v[0] for t, v in known_names().items()}
    names.update({t: m.get("company_name") or t for t, m in st.session_state.get("symbol_meta", {}).items()})
    meta = cached_stock_meta(tuple(sorted(set(tickers))))
    for row in meta.itertuples(index=False):
        if row.company_name:
            names[row.ticker] = row.company_name
    return {t: names.get(t, t) for t in tickers}


def sector_map(tickers: list[str]) -> dict[str, str]:
    """Sector per ticker from stored metadata, falling back to the built-in lists."""
    mapping = {t: s for t, (_, s) in known_names().items() if s != "Index"}
    mapping.update({t: s for t, (_, s) in STOCK_UNIVERSE.items()})
    meta = cached_stock_meta(tuple(sorted(set(tickers))))
    for row in meta.itertuples(index=False):
        if row.sector:
            mapping[row.ticker] = row.sector
    return {t: mapping[t] for t in tickers if t in mapping}


def currencies(tickers: list[str]) -> dict[str, str]:
    """Trading currency per ticker where known (from search metadata), else inferred from suffix."""
    meta = cached_stock_meta(tuple(sorted(set(tickers))))
    known = {r.ticker: r.currency for r in meta.itertuples(index=False) if r.currency}
    inferred = {}
    for t in tickers:
        if t in known:
            inferred[t] = known[t]
        elif t.startswith("^"):
            continue
        elif t.upper().endswith((".NS", ".BO")):
            inferred[t] = "INR"
        elif "." not in t:
            inferred[t] = "USD"
    return inferred


def benchmark_label(symbol: str) -> str:
    return BENCHMARKS.get(symbol, display_names([symbol]).get(symbol, symbol))


# ---- provider access --------------------------------------------------------------------------
def clear_data_caches() -> None:
    for fn in (cached_coverage, cached_stored_tickers, cached_panel, cached_bars, cached_stock_meta, live_anomalies, quality_report):
        fn.clear()


def missing_history(tickers: list[str], start: date) -> list[str]:
    """Tickers with nothing stored, or whose stored history starts well after `start` and for
    which that earlier range has never been requested."""
    if get_repository() is None or not tickers:
        return []
    key = tuple(sorted(set(tickers)))
    coverage = cached_coverage(key)
    meta = cached_stock_meta(key).set_index("ticker")
    needed = []
    for ticker in tickers:
        if ticker not in coverage.index:
            needed.append(ticker)
            continue
        known_from = coverage.loc[ticker, "first_date"].date()
        requested = meta["history_requested_from"].get(ticker) if "history_requested_from" in meta.columns else None
        if requested is not None and not pd.isna(requested):
            known_from = min(known_from, pd.Timestamp(requested).date())
        if known_from > start + timedelta(days=7):
            needed.append(ticker)
    return needed


def ingest(tickers: list[str], start: date, incremental: bool) -> list[IngestResult]:
    """Download prices for `tickers` (missing ranges only unless `incremental`)."""
    repository = require_database()
    meta_rows = [m for t, m in st.session_state.get("symbol_meta", {}).items() if t in tickers]
    metadata = pd.DataFrame(meta_rows) if meta_rows else None
    results = ingest_tickers(repository, tickers, start, None, incremental=incremental, metadata=metadata)
    clear_data_caches()
    return results


# ---- computations -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=512)
def live_anomalies(ticker: str, last_date: date, window: int, threshold: float,
                   contamination: float) -> pd.DataFrame:
    """Anomalies over a ticker's full stored history (cache key includes the last stored date,
    so new data invalidates it). Annotated with severity and explanation."""
    ohlcv = cached_ohlcv(ticker, None, None)
    if ohlcv.empty:
        return pd.DataFrame()
    try:
        frame = detect_all(ticker, ohlcv, window=window, threshold=threshold, contamination=contamination)
    except InsufficientDataError as exc:
        log.info("Anomaly detection skipped for %s: %s", ticker, exc)
        return pd.DataFrame()
    frame["detection_date"] = pd.to_datetime(frame["detection_date"])
    return annotate(frame)


def anomalies_for(tickers: list[str], start: date, end: date, window: int | None = None,
                  threshold: float | None = None, contamination: float | None = None) -> pd.DataFrame:
    """Live anomalies for several tickers, restricted to [start, end]."""
    settings = load_settings()
    window = window or settings.rolling_window
    threshold = threshold or settings.zscore_threshold
    contamination = contamination or settings.isolation_contamination
    coverage = cached_coverage(tuple(sorted(set(tickers))))
    frames = []
    for ticker in tickers:
        if ticker not in coverage.index:
            continue
        last = coverage.loc[ticker, "last_date"].date()
        result = live_anomalies(ticker, last, int(window), float(threshold), float(contamination))
        if not result.empty:
            frames.append(result)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    dates = combined["detection_date"]
    return combined[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))].reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def quality_report(tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    """Data-quality issues for the selection over the period (see fmip.data.quality)."""
    coverage = cached_coverage(tuple(sorted(set(tickers))))
    issues = []
    for ticker in tickers:
        if ticker not in coverage.index:
            issues.extend(check_coverage(ticker, None, None, 0, start, end))
            continue
        bars = cached_bars(ticker, start, end)
        issues.extend(check_bars(ticker, bars, is_index=ticker.startswith("^")))
        sessions = len(bars)
        first = bars.index[0] if sessions else coverage.loc[ticker, "first_date"]
        issues.extend(check_coverage(ticker, first, coverage.loc[ticker, "last_date"], sessions, start, end))
    return issues_frame(issues)
