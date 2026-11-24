"""Reusable page building blocks: section labels, KPI tiles, ranked lists, warnings, formatting."""

from __future__ import annotations

import html
from datetime import timedelta

import pandas as pd
import streamlit as st

from common import DISCLAIMER, cached_panel, currencies
from fmip.analysis.returns import rebase
from fmip.data.quality import calendar_overlap
from state import Selection
from theme import tokens


# ---- formatting --------------------------------------------------------------------------------
def pct(value: float | None, signed: bool = True, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:+.{digits}%}" if signed else f"{value:.{digits}%}"


def num(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def polarity_style(value: float) -> str:
    """Pandas Styler cell style: up/down colour for signed numbers (sign is shown too)."""
    if value is None or pd.isna(value) or value == 0:
        return ""
    t = tokens()
    return f"color: {t.up}" if value > 0 else f"color: {t.down}"


def section(label: str) -> None:
    st.markdown(f'<div class="fm-section">{html.escape(label)}</div>', unsafe_allow_html=True)


def footer() -> None:
    st.caption(DISCLAIMER)


def empty_state(message: str, icon: str = ":material/info:") -> None:
    st.info(message, icon=icon)


def kpi(container, label: str, value: str, delta: str | None = None, help: str | None = None,
        delta_color: str = "normal", chart: list[float] | None = None, description: str | None = None) -> None:
    kwargs = {}
    if description:
        kwargs["delta_description"] = description
    if delta_color == "off":
        kwargs["delta_arrow"] = "off"   # neutral context, not a direction
    container.metric(label, value, delta=delta, help=help, delta_color=delta_color, border=True,
                     chart_data=chart, chart_type="area" if chart else "line", **kwargs)


def ranked_list(container, title: str, values: pd.Series, names: dict[str, str], *, signed: bool = True,
                fmt: str = "pct", limit: int = 5, ascending: bool = False) -> None:
    """Compact ranked list in a bordered card. Polarity colour always comes with a sign."""
    data = values.dropna().sort_values(ascending=ascending).head(limit)
    items = []
    for ticker, value in data.items():
        if fmt == "pct":
            text = pct(value, signed=signed)
        elif fmt == "ratio":
            text = f"{value:.2f}×"
        else:
            text = f"{value:,.2f}"
        css = ("fm-up" if value >= 0 else "fm-down") if signed else "fm-num"
        name = html.escape(names.get(ticker, ticker))
        items.append(f'<li><span><span class="t">{html.escape(ticker)}</span> '
                     f'<span class="fm-note">{name[:26]}</span></span><span class="{css}">{text}</span></li>')
    with container.container(border=True):
        st.markdown(f"**{html.escape(title)}**")
        if items:
            st.markdown(f'<ul class="fm-list">{"".join(items)}</ul>', unsafe_allow_html=True)
        else:
            st.caption("Not enough data.")


# ---- data for the current selection ------------------------------------------------------------
def period_prices(sel: Selection, include_benchmark: bool = True, lookback_days: int = 0,
                  column: str | None = None) -> pd.DataFrame:
    """Wide panel for the selection (and benchmark) over the period, optionally with extra
    history before the start (for moving averages that need warm-up)."""
    tickers = list(sel.tickers)
    if include_benchmark and sel.benchmark not in tickers:
        tickers.append(sel.benchmark)
    start = sel.start - timedelta(days=lookback_days) if lookback_days else sel.start
    return cached_panel(tuple(tickers), start, sel.end, column or sel.price_column)


def require_stocks(sel: Selection, minimum: int = 1, reason: str = "") -> bool:
    if len(sel.tickers) >= minimum:
        return True
    noun = "stock" if minimum == 1 else "stocks"
    empty_state(f"Select at least {minimum} {noun} in the sidebar{reason}.")
    return False


def context_notes(sel: Selection, panel: pd.DataFrame) -> None:
    """Compact, specific warnings about comparability for the current selection."""
    notes = []
    stocks = panel[[c for c in panel.columns if c in sel.tickers]]
    if not stocks.empty:
        info = rebase(stocks)
        late = {t: d for t, d in info.late_starters.items() if d > pd.Timestamp(sel.start) + pd.Timedelta(days=7)}
        if late:
            listing = ", ".join(f"{t} (from {d:%d %b %Y})" for t, d in late.items())
            notes.append(f"History starts after the period start for {listing}. Comparisons use the common period "
                         f"from {info.common_start:%d %b %Y}.")
    ccys = sorted(set(currencies(sel.tickers).values()))
    if len(ccys) > 1:
        notes.append(f"The selection trades in several currencies ({', '.join(ccys)}). Returns are in each stock's "
                     "local currency, with no FX conversion.")
    overlap = calendar_overlap(panel)
    if overlap["share_common"] < 0.9 and panel.shape[1] > 1:
        notes.append(f"Exchange calendars differ: only {overlap['share_common']:.0%} of sessions are common to all "
                     "symbols. Returns are computed on each symbol's own calendar. Daily correlations across "
                     "time zones are understated because the markets close at different times.")
    for note in notes:
        st.caption(f":material/info: {note}")
