"""Correlation Analysis: matrices, strongest pairs, rolling correlation, sector co-movement."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import charts
from common import display_names, require_database, sector_map
from components import context_notes, footer, kpi, num, period_prices, require_stocks, section
from fmip.analysis.correlation import (
    average_pairwise_correlation,
    cluster_order,
    correlation_matrix,
    interpret_correlation,
    returns_panel,
    rolling_correlation,
    sector_returns,
    strongest_pairs,
)
from fmip.exceptions import InsufficientDataError
from state import current
from theme import entity_colours

MIN_OVERLAP = 60
MIN_SHORT_OVERLAP = 15


def render() -> None:
    require_database()
    sel = current()
    if not require_stocks(sel, 2, " to compare co-movement"):
        return
    names = display_names([*sel.tickers, sel.benchmark])
    panel = period_prices(sel)
    returns = returns_panel(panel)
    stock_cols = [t for t in sel.tickers if t in returns.columns]

    c = st.columns([2, 2, 3])
    method = c[0].segmented_control("Method", ["pearson", "spearman"], default="pearson", key="co_method",
                                    format_func=str.title, label_visibility="collapsed",
                                    help="Pearson: linear co-movement of returns. Spearman: rank-based, less "
                                         "sensitive to extreme days.") or "pearson"
    order_mode = c[1].segmented_control("Order", ["Clustered", "Selection"], default="Clustered", key="co_order",
                                        label_visibility="collapsed") or "Clustered"
    include_bench = c[2].toggle(f"Include benchmark ({sel.benchmark})", value=True, key="co_bench")
    cols = stock_cols + ([sel.benchmark] if include_bench and sel.benchmark in returns.columns else [])
    available = int(returns[cols].dropna(how="all").shape[0])
    min_overlap = MIN_OVERLAP if available >= MIN_OVERLAP else MIN_SHORT_OVERLAP
    try:
        corr = correlation_matrix(returns[cols], method=method, min_periods=min_overlap)
    except InsufficientDataError:
        st.info(f"Need at least {MIN_SHORT_OVERLAP} overlapping daily returns. Choose a longer period.")
        return
    if min_overlap < MIN_OVERLAP:
        st.warning(f"Short sample: only {available} daily returns in this period. Correlations from fewer than "
                   f"{MIN_OVERLAP} observations are noisy; a ±0.3 difference may not be meaningful.",
                   icon=":material/warning:")

    stock_corr = corr.loc[stock_cols, stock_cols]
    pairs = strongest_pairs(stock_corr, positive_threshold=-1.01, negative_threshold=-2)
    k = st.columns(4)
    kpi(k[0], "Average pairwise correlation", num(average_pairwise_correlation(stock_corr)),
        help="Mean of all stock-pair correlations. Lower means more diversification across the selection.")
    if not pairs.empty:
        hi, lo = pairs.iloc[0], pairs.iloc[-1]
        kpi(k[1], "Most correlated pair", f"{hi.asset_a} · {hi.asset_b}", delta=f"ρ {hi.correlation:.2f}", delta_color="off")
        kpi(k[2], "Least correlated pair", f"{lo.asset_a} · {lo.asset_b}", delta=f"ρ {lo.correlation:.2f}", delta_color="off")
    if sel.benchmark in corr.columns:
        avg_b = corr.loc[stock_cols, sel.benchmark].mean()
        kpi(k[3], f"Avg correlation with {sel.benchmark}", num(avg_b),
            help="How closely the selected stocks have moved with the benchmark on average.")

    with st.container(border=True):
        order = cluster_order(corr) if order_mode == "Clustered" else cols
        matrix = corr.loc[order, order]
        charts.show(charts.heatmap(matrix, f"{method.title()} correlation of daily returns · {sel.period}"))
        st.caption("Blue = move together, red = move in opposite directions, grey = little relationship. "
                   "Clustered order places similar stocks next to each other. "
                   "Guide: |ρ| ≥ 0.8 very strong · 0.6 strong · 0.4 moderate · 0.2 weak.")
        context_notes(sel, panel)

    left, right = st.columns([2, 3])
    with left:
        section("Pairs")
        view = st.segmented_control("Pairs", ["Highest", "Lowest"], default="Highest", key="co_pairs",
                                    label_visibility="collapsed") or "Highest"
        shown = pairs.head(10) if view == "Highest" else pairs.tail(10).iloc[::-1]
        shown = shown.assign(interpretation=shown["correlation"].map(interpret_correlation))
        st.dataframe(shown[["asset_a", "asset_b", "correlation", "interpretation"]], hide_index=True, width="stretch",
                     column_config={"asset_a": "Stock A", "asset_b": "Stock B",
                                    "correlation": st.column_config.ProgressColumn("ρ", min_value=-1, max_value=1,
                                                                                   format="%.2f"),
                                    "interpretation": "Interpretation"})
    with right:
        section("Rolling correlation")
        r = st.columns([1, 1, 1])
        options = cols
        first = r[0].selectbox("Stock A", options, index=0, key="co_a")
        default_b = options.index(sel.benchmark) if sel.benchmark in options else min(1, len(options) - 1)
        second = r[1].selectbox("Stock B", options, index=default_b, key="co_b")
        window = r[2].select_slider("Window", [21, 63, 126, 252], value=63, key="co_win")
        if first == second:
            st.info("Choose two different series.")
        else:
            rolling = rolling_correlation(returns, first, second, window).dropna()
            if rolling.empty:
                st.info("Not enough overlapping history for this window.")
            else:
                frame = rolling.rename(f"{first} vs {second}").to_frame()
                charts.show(charts.lines(frame, f"{window}-session rolling correlation", "ρ",
                                         colours={frame.columns[0]: entity_colours(sel.tickers).get(first)},
                                         zero_line=True, height=300, value_format="%{y:.2f}"))
                st.caption(f"Latest {rolling.iloc[-1]:.2f} ({interpret_correlation(rolling.iloc[-1])}); "
                           f"range {rolling.min():.2f} to {rolling.max():.2f}. Correlations often rise in sell-offs.")

    section("Sector co-movement")
    mapping = sector_map(stock_cols)
    sec = sector_returns(returns[stock_cols], mapping)
    if sec.shape[1] < 2:
        st.caption("Select stocks from at least two sectors to compare sector co-movement.")
    else:
        try:
            sec_corr = correlation_matrix(sec, method=method, min_periods=MIN_OVERLAP)
            with st.container(border=True):
                charts.show(charts.heatmap(sec_corr.loc[cluster_order(sec_corr), cluster_order(sec_corr)],
                                           "Correlation of equal-weighted sector returns (selected stocks)"))
                counts = pd.Series(mapping).value_counts()
                st.caption("Members per sector: " + ", ".join(f"{s} {n}" for s, n in counts.items()))
        except InsufficientDataError:
            st.caption("Not enough overlapping sector history.")
    footer()
