"""Relative Strength: stock performance against the selected benchmark."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import charts
from common import display_names, require_database
from components import context_notes, footer, period_prices, polarity_style, require_stocks, section
from fmip.analysis.relative import (
    excess_returns,
    relative_strength_line,
    relative_strength_summary,
    rolling_beta,
    rolling_relative_return,
)
from fmip.analysis.returns import TRAILING_PERIODS
from fmip.exceptions import InsufficientDataError
from state import current
from theme import entity_colours

LOOKBACK_DAYS = 400  # load enough history for 1Y trailing figures whatever the period


def render() -> None:
    require_database()
    sel = current()
    if not require_stocks(sel):
        return
    names = display_names([*sel.tickers, sel.benchmark])
    colours = entity_colours(sel.tickers)
    panel = period_prices(sel)
    if sel.benchmark not in panel.columns:
        st.warning(f"No data for the benchmark {sel.benchmark}.")
        return
    bench = panel[sel.benchmark]
    stocks = panel[[t for t in sel.tickers if t in panel.columns]]

    # ---- summary -----------------------------------------------------------------------------
    summary = relative_strength_summary(stocks, bench)
    out = int((summary["rs_ratio"] >= 1).sum()) if not summary.empty else 0
    section(f"Relative to {sel.benchmark_name} · {sel.period} · {sel.basis_label.lower()}")
    k = st.columns(4)
    k[0].metric("Outperforming", f"{out} of {len(summary)}", border=True,
                help="Stocks whose relative-strength ratio is above 1 at the end of the period.")
    improving = int(summary["rs_above_trend"].eq(True).sum()) if not summary.empty else 0
    k[1].metric("RS line above its 63-day average", f"{improving}", border=True,
                help="Relative strength currently rising: the RS line is above its own 63-session average.")
    if not summary.empty:
        best = summary["geometric_excess"].idxmax()
        worst = summary["geometric_excess"].idxmin()
        k[2].metric("Strongest vs benchmark", best, f"{summary.loc[best, 'geometric_excess']:+.1%}", border=True)
        k[3].metric("Weakest vs benchmark", worst, f"{summary.loc[worst, 'geometric_excess']:+.1%}", border=True)

    # ---- RS lines ----------------------------------------------------------------------------
    with st.container(border=True):
        lines = {}
        for ticker in stocks.columns:
            try:
                lines[ticker] = relative_strength_line(stocks[ticker], bench)
            except InsufficientDataError:
                continue
        frame = pd.DataFrame(lines)
        charts.show(charts.lines(frame, f"Relative strength line (stock ÷ {sel.benchmark}, both rebased to 1)",
                                 "RS ratio", colours=colours, ref_line=1.0, height=360, value_format="%{y:.3f}"))
        st.caption("Above 1.0: the stock has outperformed the benchmark since the period start. "
                   "A rising line means it is currently outperforming; a falling line means it is lagging.")
        context_notes(sel, panel)

    # ---- excess returns table ----------------------------------------------------------------
    section("Excess return over trailing periods (stock − benchmark)")
    longer = period_prices(sel, lookback_days=LOOKBACK_DAYS)
    table = excess_returns(longer[[t for t in sel.tickers if t in longer.columns]], longer[sel.benchmark])
    if not summary.empty:
        table = table.join(summary[["status"]])
    table.insert(0, "name", [names.get(t, t) for t in table.index])
    left, right = st.columns([3, 2])
    with left:
        config = {p: st.column_config.NumberColumn(p, format="percent") for p in TRAILING_PERIODS}
        config.update({"name": st.column_config.TextColumn("Company"), "status": "Trend vs benchmark"})
        st.dataframe(table.style.map(polarity_style, subset=list(TRAILING_PERIODS)), width="stretch",
                     column_config=config)
        st.caption("Trailing windows are counted in trading sessions (1W = 5, 1M = 21, 3M = 63, 6M = 126, 1Y = 252), "
                   "ending at the latest stored session, on dates where both the stock and benchmark traded.")
    with right:
        period_choice = st.segmented_control("Window", list(TRAILING_PERIODS), default="3M", key="rs_window")
        charts.show(charts.ranked_bars(table[period_choice or "3M"], f"{period_choice or '3M'} excess return",
                                       names=names))

    # ---- rolling analytics -------------------------------------------------------------------
    section("Rolling relative performance and beta")
    c = st.columns([1, 1])
    window = c[0].select_slider("Rolling window (sessions)", options=[21, 42, 63, 126, 252], value=63, key="rs_roll")
    focus = c[1].multiselect("Stocks", list(stocks.columns), default=list(stocks.columns)[:4], key="rs_focus")
    if focus:
        a, b = st.columns(2)
        rel = pd.DataFrame({t: rolling_relative_return(longer[t], longer[sel.benchmark], window) for t in focus})
        rel = rel.loc[pd.Timestamp(sel.start):]
        with a.container(border=True):
            charts.show(charts.lines(rel, f"{window}-session return minus benchmark", "Excess return (%)",
                                     colours=colours, percent=True, zero_line=True))
        betas = pd.DataFrame({t: rolling_beta(longer[t].dropna().pct_change(), longer[sel.benchmark].dropna().pct_change(),
                                              max(window, 63)) for t in focus}).loc[pd.Timestamp(sel.start):]
        with b.container(border=True):
            charts.show(charts.lines(betas, f"Rolling beta ({max(window, 63)} sessions)", "Beta", colours=colours,
                                     ref_line=1.0, value_format="%{y:.2f}"))
        st.caption("Beta above 1: the stock has tended to move more than the benchmark on the same days; "
                   "below 1: less. Beta is estimated from past returns and changes over time.")
    footer()
