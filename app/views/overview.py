"""Market Overview: summary KPIs, performance vs benchmark, leaders/laggards, breadth, sectors."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import charts
from common import anomalies_for, cached_coverage, display_names, quality_report, require_database, sector_map
from components import context_notes, footer, kpi, pct, period_prices, ranked_list, require_stocks, section
from fmip.analysis.breadth import breadth_snapshot, share_above_ma
from fmip.analysis.correlation import sector_vs_benchmark
from fmip.analysis.eda import annualised_volatility, drawdown_series
from fmip.analysis.returns import asset_returns, rebase
from state import current
from theme import entity_colours

VOLUME_WINDOW = 20


def _volume_change(volumes: pd.DataFrame) -> pd.Series:
    """Average volume of the last 20 sessions relative to the average of the rest of the period."""
    out = {}
    for ticker in volumes.columns:
        series = volumes[ticker].dropna()
        if len(series) < VOLUME_WINDOW * 2:
            continue
        recent = series.iloc[-VOLUME_WINDOW:].mean()
        base = series.iloc[:-VOLUME_WINDOW].mean()
        if base > 0:
            out[ticker] = recent / base - 1.0
    return pd.Series(out, dtype=float)


def render() -> None:
    require_database()
    sel = current()
    if not require_stocks(sel):
        return
    panel = period_prices(sel)
    if panel.empty or sel.benchmark not in panel.columns:
        st.warning(f"No stored data for the benchmark {sel.benchmark} in this period.")
        return
    stocks = panel[[t for t in sel.tickers if t in panel.columns]]
    bench = panel[sel.benchmark].dropna()
    names = display_names([*sel.tickers, sel.benchmark])
    colours = entity_colours(sel.tickers)
    info = rebase(stocks)
    totals = (info.values.ffill().iloc[-1] - 1.0) if not info.values.empty else pd.Series(dtype=float)
    bench_window = bench.loc[info.common_start:] if info.common_start is not None else bench
    bench_return = float(bench_window.iloc[-1] / bench_window.iloc[0] - 1.0) if len(bench_window) > 1 else float("nan")
    bench_vol = annualised_volatility(bench.pct_change().dropna()) if len(bench) > 2 else float("nan")
    anomalies = anomalies_for(sel.tickers, sel.start, sel.end)

    # ---- KPI row ---------------------------------------------------------------------------
    section(f"Summary · {sel.period} · {sel.basis_label.lower()}")
    cols = st.columns(6)
    spark = (bench / bench.iloc[0]).resample("W").last().dropna().round(4).tolist() if len(bench) > 10 else None
    kpi(cols[0], f"{sel.benchmark_name} return", pct(bench_return), chart=spark,
        help=f"{sel.benchmark} price return over the common period of the selection "
             f"({sel.period_label}). Index levels exclude dividends. Sparkline: weekly closes.")
    kpi(cols[1], "Benchmark volatility", pct(bench_vol, signed=False),
        help="Annualised standard deviation of daily benchmark returns in the period (×√252).")
    beat = int((totals > bench_return).sum()) if not totals.empty else 0
    kpi(cols[2], "Stocks selected", f"{len(stocks.columns)}",
        delta=f"{beat} beat benchmark", delta_color="off",
        help="Stocks with stored data in the period. Returns are measured over the common period.")
    if not totals.empty:
        best, worst = totals.idxmax(), totals.idxmin()
        kpi(cols[3], "Best performer", best, delta=pct(totals[best]), help=names.get(best, best))
        kpi(cols[4], "Worst performer", worst, delta=pct(totals[worst]), help=names.get(worst, worst))
    recent_cut = pd.Timestamp(sel.end) - pd.Timedelta(days=30)
    recent = int((anomalies["detection_date"] >= recent_cut).sum()) if not anomalies.empty else 0
    kpi(cols[5], "Anomalies in period", f"{len(anomalies)}", delta=f"{recent} in last 30 days", delta_color="off",
        help="Flagged stock-days (rolling z-score and Isolation Forest) for the selected stocks in the period. "
             "Statistical outliers, not evidence of wrongdoing.")

    # ---- performance -------------------------------------------------------------------------
    section("Performance vs benchmark")
    with st.container(border=True):
        top = st.columns([2, 2, 3])
        view = top[0].segmented_control("View", ["Change %", "Index = 100"], default="Change %",
                                        key="ov_view", label_visibility="collapsed")
        align = top[1].segmented_control("Alignment", ["Common start", "Own start"], default="Common start",
                                         key="ov_align", label_visibility="collapsed",
                                         help="Common start compares every series over the same dates.")
        top[2].caption(f"{sel.basis_label} for stocks; {sel.benchmark} is a price index. "
                       "Click a legend item to hide it, double-click to isolate.")
        rebased = rebase(panel, "own" if align == "Own start" else "common")
        if rebased.values.empty:
            st.info("Not enough overlapping data to compare.")
        else:
            charts.show(charts.performance_lines(rebased.values, colours, sel.benchmark, names,
                                                 as_index=view == "Index = 100"))
        context_notes(sel, panel)

    # ---- leaders and laggards ----------------------------------------------------------------
    section("Leaders and laggards")
    stock_returns = asset_returns(stocks)
    vols = stock_returns.std(ddof=1) * (252 ** 0.5)
    drawdowns = pd.Series({t: drawdown_series(stocks[t].dropna()).min() for t in stocks.columns
                           if stocks[t].notna().sum() > 1})
    volumes = period_prices(sel, include_benchmark=False, column="volume")
    vol_change = _volume_change(volumes)
    c = st.columns(5)
    ranked_list(c[0], "Top performers", totals, names)
    ranked_list(c[1], "Bottom performers", totals, names, ascending=True)
    ranked_list(c[2], "Highest volatility", vols, names, signed=False)
    ranked_list(c[3], "Deepest drawdown", drawdowns, names, ascending=True)
    ranked_list(c[4], f"Volume change ({VOLUME_WINDOW}d)", vol_change, names)

    # ---- breadth and sectors -----------------------------------------------------------------
    section(f"Breadth and sectors · from your {len(stocks.columns)} selected stocks, not the whole market")
    left, right = st.columns([1, 1])
    with left.container(border=True):
        history = period_prices(sel, include_benchmark=False, lookback_days=320)
        snapshot = breadth_snapshot(history[[t for t in sel.tickers if t in history.columns]])
        b = st.columns(3)
        if "advancers" in snapshot:
            b[0].metric("Advancing / declining", f"{snapshot['advancers']} / {snapshot['decliners']}",
                        help=f"Last session ({snapshot['as_of']:%d %b %Y}). Unchanged: {snapshot['unchanged']}.")
        b[1].metric("Above 50-day MA", pct(snapshot.get("pct_above_ma50"), signed=False, digits=0),
                    help="Share of selected stocks closing above their own 50-session average.")
        b[2].metric("Above 200-day MA", pct(snapshot.get("pct_above_ma200"), signed=False, digits=0),
                    help="Share of selected stocks closing above their own 200-session average.")
        shares = pd.DataFrame({"% above 50-day MA": share_above_ma(history, 50),
                               "% above 200-day MA": share_above_ma(history, 200)}).loc[pd.Timestamp(sel.start):]
        if shares.dropna(how="all").empty:
            st.caption("Not enough history for moving-average breadth.")
        else:
            charts.show(charts.lines(shares, None, "Share of stocks (%)", percent=True, height=240,
                                     value_format="%{y:.0f}%"))
    with right.container(border=True):
        mapping = sector_map(list(stocks.columns))
        sectors = sector_vs_benchmark(stocks, mapping, bench)
        if sectors.empty:
            st.caption("Sector data is not available for the selected stocks.")
        else:
            st.markdown(f"**Sector return vs {sel.benchmark_name}** (equal-weighted, selected stocks)")
            labels = {s: f"{s} ({int(n)})" for s, n in sectors["members"].items()}
            rel = sectors["relative_return"].rename(index=labels)
            charts.show(charts.ranked_bars(rel, None, names={v: f"{k}: total {pct(sectors.loc[k, 'total_return'])}"
                                                             for k, v in labels.items()}))
            st.caption("Bars show sector return minus benchmark return over the same sessions. "
                       "Numbers in brackets are member counts.")

    # ---- anomalies ---------------------------------------------------------------------------
    section("Most significant recent anomalies")
    with st.container(border=True):
        if anomalies.empty:
            st.caption("No anomalies flagged for the selection in this period.")
        else:
            order = {"Extreme": 0, "High": 1, "Moderate": 2}
            top_rows = (anomalies.assign(_o=anomalies["severity"].map(order))
                        .sort_values(["detection_date", "_o"], ascending=[False, True]).head(8))
            st.dataframe(
                top_rows[["detection_date", "ticker", "signal", "severity", "explanation"]],
                hide_index=True, width="stretch",
                column_config={"detection_date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                               "ticker": "Ticker", "signal": "Signal", "severity": "Severity",
                               "explanation": st.column_config.TextColumn("Why it was flagged", width="large")})
            st.caption("Open Anomaly Detection for the full timeline and to tune thresholds.")

    # ---- data quality ------------------------------------------------------------------------
    quality = quality_report(tuple([*sel.tickers, sel.benchmark]), sel.start, sel.end)
    label = f"Data quality and coverage · {len(quality)} note(s)"
    with st.expander(label, icon=":material/fact_check:"):
        coverage = cached_coverage(tuple(sorted([*sel.tickers, sel.benchmark])))
        st.dataframe(coverage, width="stretch",
                     column_config={"first_date": st.column_config.DateColumn("First stored"),
                                    "last_date": st.column_config.DateColumn("Last stored"),
                                    "sessions": "Sessions", "last_ingested": st.column_config.DatetimeColumn("Ingested (UTC)")})
        if quality.empty:
            st.success("No data-quality issues detected for the selection in this period.")
        else:
            st.dataframe(quality, hide_index=True, width="stretch",
                         column_config={"detail": st.column_config.TextColumn("Detail", width="large"),
                                        "first_date": st.column_config.DateColumn("First date")})
        st.caption("Checks: duplicate dates, OHLC consistency, unadjusted-split signatures, adjustment-factor jumps, "
                   "moves of 25% or more, gaps over 7 days, stale price runs, zero volume, late starts and staleness. "
                   "Flags are for review; data is never altered to hide them.")
    footer()
