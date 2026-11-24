"""Global selection state, sidebar controls and the context header.

All user choices live in `st.session_state` under stable keys. The sidebar is rendered once
per run by the entry script, before the page, so selections persist across pages and a change
in one control never resets the others.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from common import (
    PRICE_COLUMNS,
    benchmark_label,
    cached_coverage,
    cached_search,
    cached_stored_tickers,
    display_names,
    get_repository,
    ingest,
    missing_history,
    quality_report,
)
from fmip.data.symbols import looks_like_ticker
from fmip.universe import BENCHMARKS, DEFAULT_MARKET, MARKETS, in_market, known_names

PERIODS: dict[str, int | None] = {"1M": 31, "3M": 92, "6M": 183, "1Y": 366, "3Y": 3 * 365 + 1,
                                  "5Y": 5 * 365 + 1, "Max": None, "Custom": None}
BASIS_LABELS = {"price": "Price return", "total": "Total return"}
MARKET_SHORT = {"United States": "US", "India": "India", "All markets": "All"}
NEW_SYMBOL_HISTORY_DAYS = 5 * 365 + 2


@dataclass
class Selection:
    tickers: list[str]                 # stocks the user selected and that have stored data
    benchmark: str
    start: date
    end: date
    period: str
    basis: str                         # "price" or "total"
    market: str
    unavailable: list[str] = field(default_factory=list)   # selected but no data available
    as_of: date | None = None          # latest stored session across the selection

    @property
    def price_column(self) -> str:
        return PRICE_COLUMNS[self.basis]

    @property
    def basis_label(self) -> str:
        return BASIS_LABELS[self.basis]

    @property
    def benchmark_name(self) -> str:
        return benchmark_label(self.benchmark)

    @property
    def period_label(self) -> str:
        return f"{self.start:%d %b %Y} – {self.end:%d %b %Y}"


# ---- state helpers --------------------------------------------------------------------------
def init_state() -> None:
    market = MARKETS[DEFAULT_MARKET]
    defaults = {
        "market": DEFAULT_MARKET,
        "tickers": list(market.defaults),
        "benchmark": market.default_benchmark,
        "period": "1Y",
        "custom_range": (date.today() - timedelta(days=365 * 2), date.today()),
        "basis": "price",
        "symbol_meta": {},        # ticker -> metadata dict from search results
        "flash": [],              # messages shown once after a rerun
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def flash(message: str, kind: str = "info") -> None:
    st.session_state["flash"].append((kind, message))


def _set_tickers(tickers: list[str]) -> None:
    st.session_state["tickers"] = list(dict.fromkeys(tickers))


def _normalise_typed_tickers() -> None:
    """on_change for the stock picker: upper-case typed symbols and reject implausible ones."""
    cleaned, rejected = [], []
    for value in st.session_state["tickers"]:
        symbol = str(value).strip().upper()
        if looks_like_ticker(symbol):
            cleaned.append(symbol)
        else:
            rejected.append(str(value))
    st.session_state["tickers"] = list(dict.fromkeys(cleaned))
    if rejected:
        flash(f"Ignored '{', '.join(rejected)}': not a valid ticker format. Use search to find symbols by name.",
              "warning")


def _add_symbol(row: dict) -> None:
    ticker = row["ticker"]
    st.session_state["symbol_meta"][ticker] = {
        "ticker": ticker, "company_name": row.get("name"), "sector": row.get("sector"),
        "exchange": row.get("exchange"), "quote_type": row.get("quote_type"),
    }
    if ticker not in st.session_state["tickers"]:
        _set_tickers([*st.session_state["tickers"], ticker])
        flash(f"Added {ticker}.", "success")


def _use_market_defaults() -> None:
    market = MARKETS[st.session_state["market"]]
    _set_tickers(list(market.defaults))
    st.session_state["benchmark"] = market.default_benchmark


def _on_market_change() -> None:
    # Only the benchmark follows the market; the user's stock selection is kept.
    st.session_state["benchmark"] = MARKETS[st.session_state["market"]].default_benchmark


def _clear_tickers() -> None:
    _set_tickers([])


def _load_watchlist(name: str) -> None:
    repository = get_repository()
    lists = repository.list_watchlists() if repository else pd.DataFrame()
    row = lists[lists["name"] == name]
    if row.empty:
        flash(f"Watchlist '{name}' not found.", "warning")
        return
    _set_tickers(list(row.iloc[0]["tickers"]))
    if row.iloc[0]["benchmark"]:
        st.session_state["benchmark"] = row.iloc[0]["benchmark"]
    flash(f"Loaded watchlist '{name}'.", "success")


def _save_watchlist() -> None:
    repository = get_repository()
    name = st.session_state.get("watchlist_new_name", "").strip()
    try:
        repository.save_watchlist(name, st.session_state["tickers"], st.session_state["benchmark"])
    except ValueError as exc:
        flash(str(exc), "warning")
        return
    st.session_state["watchlist_new_name"] = ""
    flash(f"Saved watchlist '{name}'.", "success")


def _delete_watchlist(name: str) -> None:
    get_repository().delete_watchlist(name)
    flash(f"Deleted watchlist '{name}'.", "info")


# ---- sidebar --------------------------------------------------------------------------------
def _catalog(market: str) -> list[str]:
    """Picker options: stored symbols plus built-in examples, filtered by market."""
    stored = [t for t in cached_stored_tickers() if not t.startswith("^")]
    examples = [t for t, (_, s) in known_names().items() if s != "Index"]
    searched = list(st.session_state["symbol_meta"].keys())
    pool = list(dict.fromkeys([*stored, *examples, *searched]))
    return sorted(t for t in pool if in_market(t, market))


def _resolve_dates(coverage: pd.DataFrame, tickers: list[str]) -> tuple[date, date]:
    period = st.session_state["period"]
    last_dates = [coverage.loc[t, "last_date"].date() for t in tickers if t in coverage.index]
    end = max(last_dates) if last_dates else date.today()
    if period == "Custom":
        picked = st.session_state.get("custom_range") or ()
        if isinstance(picked, (tuple, list)) and len(picked) == 2:
            return picked[0], picked[1]
        start = picked[0] if picked else end - timedelta(days=365)
        return start, end
    if period == "Max":
        firsts = [coverage.loc[t, "first_date"].date() for t in tickers if t in coverage.index]
        return (min(firsts) if firsts else end - timedelta(days=365 * 5)), end
    return end - timedelta(days=PERIODS[period]), end


def render_sidebar() -> Selection:
    """Draw the global controls and return the resolved selection."""
    init_state()
    names = display_names(st.session_state["tickers"])
    sb = st.sidebar

    sb.markdown('<div class="fm-section">Universe</div>', unsafe_allow_html=True)
    sb.segmented_control("Market", list(MARKETS), key="market", format_func=lambda m: MARKET_SHORT[m],
                         on_change=_on_market_change, label_visibility="collapsed", required=True,
                         width="stretch")
    options = list(dict.fromkeys([*st.session_state["tickers"], *_catalog(st.session_state["market"])]))
    sb.multiselect(
        f"Stocks ({len(st.session_state['tickers'])})", options, key="tickers", accept_new_options=True,
        on_change=_normalise_typed_tickers, placeholder="Add a ticker, e.g. TSLA or TCS.NS",
        help="Chips show tickers to save space. Pick from the list or type any Yahoo Finance symbol and press "
             "Enter. Use 'Find by company name' to search names across exchanges.",
    )
    with sb.popover("Find by company name", icon=":material/search:", width="stretch"):
        query = st.text_input("Company or ticker", key="search_query", placeholder="e.g. Reliance, Tata, Nvidia")
        if query.strip():
            results = cached_search(query.strip())
            if results.empty:
                st.caption("No matches. Check the spelling or try the ticker symbol.")
            for i, row in enumerate(results.to_dict(orient="records")):
                label = f"**{row['ticker']}** · {row['name']}  \n{row['exchange']} · {row['quote_type'].title()}" \
                        + (f" · {row['sector']}" if row.get("sector") else "")
                st.button(label, key=f"add_{i}_{row['ticker']}", on_click=_add_symbol, args=(row,),
                          width="stretch", disabled=row["ticker"] in st.session_state["tickers"])
            st.caption("Results from Yahoo Finance. Availability of history differs by exchange.")
    c2, c3 = sb.columns(2, gap="small")
    c2.button("Defaults", on_click=_use_market_defaults, width="stretch",
              help="Replace the selection with the example set for this market")
    c3.button("Clear all", on_click=_clear_tickers, width="stretch")

    sb.markdown('<div class="fm-section">Benchmark & period</div>', unsafe_allow_html=True)
    bench_options = list(dict.fromkeys([st.session_state["benchmark"], *BENCHMARKS]))
    sb.selectbox("Benchmark", bench_options, key="benchmark", accept_new_options=True,
                 format_func=lambda s: f"{BENCHMARKS[s]} ({s})" if s in BENCHMARKS else s,
                 label_visibility="collapsed",
                 help="Applied on every page. You can type another index or ETF symbol, e.g. ^HSI or SPY.")
    sb.segmented_control("Period", list(PERIODS), key="period", required=True,
                         label_visibility="collapsed", width="stretch")
    if st.session_state["period"] == "Custom":
        sb.date_input("Custom range", key="custom_range", min_value=date(1990, 1, 1), max_value=date.today(),
                      format="DD/MM/YYYY", label_visibility="collapsed")
    sb.segmented_control("Return basis", list(BASIS_LABELS), key="basis", format_func=BASIS_LABELS.get,
                         required=True, width="stretch",
                         help="Price return uses split-adjusted closes (comparable with price indices such as "
                              "^GSPC). Total return uses dividend-adjusted closes (dividends reinvested). "
                              "Benchmark indices are price indices in both modes.")

    _watchlist_controls(sb)

    # Resolve symbols and dates, downloading anything missing.
    tickers = list(st.session_state["tickers"])
    benchmark = str(st.session_state["benchmark"]).strip().upper()
    wanted = list(dict.fromkeys([*tickers, benchmark]))
    coverage = cached_coverage(tuple(sorted(wanted)))
    start, end = _resolve_dates(coverage, tickers or [benchmark])
    needed = missing_history(wanted, start)
    attempted = st.session_state.setdefault("_attempted", set())
    to_fetch = [t for t in needed if (t, start) not in attempted]
    if to_fetch:
        # New symbols get at least five years (and 400 days before the period start) so moving
        # averages, rolling windows and regimes have warm-up history. One request per symbol.
        fetch_start = min(start - timedelta(days=400), date.today() - timedelta(days=NEW_SYMBOL_HISTORY_DAYS))
        with st.spinner(f"Downloading {len(to_fetch)} symbol(s) from Yahoo Finance: {', '.join(to_fetch)}"):
            results = ingest(to_fetch, fetch_start, incremental=False)
        attempted.update((t, start) for t in to_fetch)
        failed = [r for r in results if r.status == "failed"]
        for r in failed:
            flash(f"No data for {r.ticker}: {r.message}", "warning")
        coverage = cached_coverage(tuple(sorted(wanted)))
        start, end = _resolve_dates(coverage, tickers or [benchmark])

    available = [t for t in tickers if t in coverage.index]
    unavailable = [t for t in tickers if t not in coverage.index]
    as_of_dates = [coverage.loc[t, "last_date"].date() for t in [*available, benchmark] if t in coverage.index]

    sb.markdown('<div class="fm-section">Data</div>', unsafe_allow_html=True)
    if sb.button("Refresh prices", icon=":material/refresh:", width="stretch",
                 help="Fetch the latest completed sessions for the selection and benchmark from Yahoo Finance."):
        with st.spinner("Updating prices…"):
            results = ingest(wanted, start, incremental=True)
        refreshed = [r.ticker for r in results if r.full_refresh]
        failed = [r.ticker for r in results if r.status == "failed"]
        msg = f"Updated {sum(r.status == 'ok' for r in results)} symbol(s)"
        if refreshed:
            msg += f"; re-downloaded full history for {', '.join(refreshed)} (corporate action)"
        if failed:
            msg += f"; failed: {', '.join(failed)}"
        flash(msg + ".", "success" if not failed else "warning")
        st.rerun()
    sb.caption("Source: Yahoo Finance via yfinance · daily, end of day. Unofficial; verify key figures.")

    return Selection(tickers=available, benchmark=benchmark, start=start, end=end,
                     period=st.session_state["period"], basis=st.session_state["basis"],
                     market=st.session_state["market"], unavailable=unavailable,
                     as_of=max(as_of_dates) if as_of_dates else None)


def _watchlist_controls(sb) -> None:
    repository = get_repository()
    if repository is None:
        return
    lists = repository.list_watchlists()
    with sb.expander("Watchlists", icon=":material/bookmarks:"):
        if lists.empty:
            st.caption("No saved watchlists yet.")
        else:
            choice = st.selectbox("Saved", list(lists["name"]), key="watchlist_choice",
                                  format_func=lambda n: f"{n} ({len(lists.loc[lists['name'] == n, 'tickers'].iloc[0])})")
            l1, l2 = st.columns(2)
            l1.button("Load", on_click=_load_watchlist, args=(choice,), width="stretch")
            l2.button("Delete", on_click=_delete_watchlist, args=(choice,), width="stretch")
        st.text_input("Save current selection as", key="watchlist_new_name", placeholder="Name, e.g. Megacap tech")
        st.button("Save watchlist", on_click=_save_watchlist, width="stretch",
                  disabled=not st.session_state["tickers"])


def current() -> Selection:
    return st.session_state["_selection"]


# ---- header ---------------------------------------------------------------------------------
def _freshness(as_of: date | None) -> tuple[str, str]:
    if as_of is None:
        return "fm-bad", "No data"
    lag = int(np.busday_count(as_of, date.today()))
    css = "fm-good" if lag <= 1 else "fm-warn" if lag <= 3 else "fm-bad"
    when = "latest session" if lag <= 1 else f"{lag} business days old"
    return css, f"Data as of <b>{as_of:%d %b %Y}</b> · {when}"


def render_header(title: str, selection: Selection) -> None:
    css, freshness = _freshness(selection.as_of)
    quality = quality_report(tuple([*selection.tickers, selection.benchmark]), selection.start, selection.end)
    errors = int((quality["severity"] == "error").sum()) if not quality.empty else 0
    warnings = int((quality["severity"] == "warning").sum()) if not quality.empty else 0
    q_css = "fm-bad" if errors else "fm-warn" if warnings else "fm-good"
    q_text = (f"{errors} data error(s)" if errors else f"{warnings} data note(s)" if warnings else "Data checks passed")
    chips = [
        f'<span class="fm-chip">Benchmark <b>{html.escape(selection.benchmark_name)}</b></span>',
        f'<span class="fm-chip"><b>{html.escape(selection.period)}</b> {selection.period_label}</span>',
        f'<span class="fm-chip"><b>{len(selection.tickers)}</b> stocks · {html.escape(selection.basis_label)}</span>',
        f'<span class="fm-chip"><span class="fm-dot {css}"></span>{freshness}</span>',
        f'<span class="fm-chip" title="See Market Overview → Data quality"><span class="fm-dot {q_css}"></span>{q_text}</span>',
        '<span class="fm-chip">Yahoo Finance · EOD</span>',
    ]
    st.html(f'<div class="fm-header"><div class="fm-brand"><span class="fm-mark">FMIP</span>'
            f'<span class="fm-title">{html.escape(title)}</span></div>'
            f'<div class="fm-chips">{"".join(chips)}</div></div>')
    for kind, message in st.session_state.get("flash", []):
        {"success": st.success, "warning": st.warning}.get(kind, st.info)(message)
    st.session_state["flash"] = []
    if selection.unavailable:
        st.warning(f"No price data available for {', '.join(selection.unavailable)}. They are excluded from the "
                   "analysis. Check the symbol (Yahoo format, e.g. RELIANCE.NS) or remove it from the selection.")
