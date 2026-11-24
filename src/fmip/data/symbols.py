"""Symbol search by ticker or company name.

Uses Yahoo Finance's search endpoint through yfinance. Results are normalised to a small
frame. The network call is isolated here so the dashboard can cache it and tests can replace it.
"""

from __future__ import annotations

import re

import pandas as pd

from fmip.logging_config import get_logger
from fmip.universe import exchange_of, known_names, normalise_sector

log = get_logger(__name__)

SEARCH_COLUMNS = ["ticker", "name", "exchange", "quote_type", "sector", "industry"]
SUPPORTED_QUOTE_TYPES = {"EQUITY", "ETF", "INDEX"}
_TICKER_PATTERN = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=&]{0,19}$")


def looks_like_ticker(text: str) -> bool:
    """Loose check that a user-typed string is a plausible Yahoo symbol (e.g. AAPL, TCS.NS, ^NSEI)."""
    return bool(_TICKER_PATTERN.match(text.strip().upper()))


def local_matches(query: str, limit: int = 10) -> pd.DataFrame:
    """Search the built-in reference lists (no network)."""
    q = query.strip().lower()
    rows = []
    if q:
        for ticker, (name, sector) in known_names().items():
            if q in ticker.lower() or q in name.lower():
                rows.append({"ticker": ticker, "name": name, "exchange": exchange_of(ticker),
                             "quote_type": "INDEX" if sector == "Index" else "EQUITY",
                             "sector": None if sector == "Index" else sector, "industry": None})
    return pd.DataFrame(rows, columns=SEARCH_COLUMNS).head(limit)


def search_symbols(query: str, limit: int = 10) -> pd.DataFrame:
    """Search Yahoo Finance for equities, ETFs and indices matching `query`.

    Falls back to the built-in lists if the provider is unreachable. Never raises for network
    problems; returns an empty frame instead.
    """
    query = query.strip()
    if len(query) < 1:
        return pd.DataFrame(columns=SEARCH_COLUMNS)
    try:
        import yfinance as yf  # lazy import keeps tests offline

        quotes = yf.Search(query, max_results=limit, news_count=0, lists_count=0).quotes or []
    except Exception as exc:  # network errors, provider changes
        log.warning("Symbol search failed for %r: %s", query, exc)
        return local_matches(query, limit)

    rows = []
    for quote in quotes:
        quote_type = str(quote.get("quoteType", "")).upper()
        symbol = quote.get("symbol")
        if not symbol or quote_type not in SUPPORTED_QUOTE_TYPES:
            continue
        rows.append({
            "ticker": symbol,
            "name": quote.get("longname") or quote.get("shortname") or symbol,
            "exchange": quote.get("exchDisp") or quote.get("exchange") or exchange_of(symbol),
            "quote_type": quote_type,
            "sector": normalise_sector(quote.get("sectorDisp") or quote.get("sector")),
            "industry": quote.get("industryDisp") or quote.get("industry"),
        })
    frame = pd.DataFrame(rows, columns=SEARCH_COLUMNS).drop_duplicates("ticker")
    if frame.empty:
        return local_matches(query, limit)
    return frame.head(limit).reset_index(drop=True)
