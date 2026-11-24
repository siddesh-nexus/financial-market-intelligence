"""Anomaly Detection: live statistical and Isolation Forest flags with severity and explanations."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import charts
from common import anomalies_for, cached_ohlcv, display_names, require_database
from components import footer, kpi, require_stocks, section
from fmip.config import load_settings
from state import current

SEVERITY_ORDER = {"Extreme": 0, "High": 1, "Moderate": 2}


def render() -> None:
    require_database()
    sel = current()
    stocks = [t for t in sel.tickers]
    if not require_stocks(sel):
        return
    settings = load_settings()
    names = display_names(stocks)

    with st.container(border=True):
        c = st.columns([1.2, 1.2, 1.2, 2.4])
        threshold = c[0].select_slider("z-score threshold", [2.5, 3.0, 3.5, 4.0, 5.0],
                                       value=float(settings.zscore_threshold) if settings.zscore_threshold in
                                       (2.5, 3.0, 3.5, 4.0, 5.0) else 3.0, key="an_thr",
                                       help="A day is flagged when |z| is at or above this value.")
        window = c[1].select_slider("Baseline window", [20, 40, 60, 90, 120], value=int(settings.rolling_window)
                                    if settings.rolling_window in (20, 40, 60, 90, 120) else 60, key="an_win",
                                    help="Sessions used for the baseline mean and standard deviation (previous days only).")
        contamination = c[2].select_slider("Isolation Forest share", [0.005, 0.01, 0.02, 0.05],
                                           value=settings.isolation_contamination if settings.isolation_contamination
                                           in (0.005, 0.01, 0.02, 0.05) else 0.01, key="an_cont",
                                           format_func=lambda v: f"{v:.1%}",
                                           help="Expected share of days the model flags across each stock's full history.")
        signals = c[3].pills("Signals", ["Price move", "Volume surge", "Intraday range", "Volatility spike",
                                         "Multivariate (Isolation Forest)"], selection_mode="multi", key="an_sig",
                             default=["Price move", "Volume surge", "Intraday range", "Volatility spike",
                                      "Multivariate (Isolation Forest)"])
        st.caption("Detection runs on each stock's full stored history (so baselines are warm at the period start) "
                   "and is shown for the selected period. A flag means the day was unusual compared with recent "
                   "history. It is not evidence of manipulation, insider activity or a future move.")

    with st.spinner("Scoring anomalies…"):
        found = anomalies_for(stocks, sel.start, sel.end, window, threshold, contamination)
    if found.empty:
        st.info("No anomalies flagged for the selection in this period with these settings.")
        footer()
        return
    found = found[found["signal"].isin(signals or [])]
    severities = st.segmented_control("Severity", ["Extreme", "High", "Moderate"], selection_mode="multi",
                                      default=["Extreme", "High", "Moderate"], key="an_sev",
                                      label_visibility="collapsed")
    found = found[found["severity"].isin(severities or [])]

    k = st.columns(5)
    kpi(k[0], "Flagged stock-days", f"{len(found)}", help="One row per stock, day, signal and method.")
    kpi(k[1], "Distinct days", f"{found['detection_date'].nunique()}")
    kpi(k[2], "Extreme", f"{int((found['severity'] == 'Extreme').sum())}",
        help="|z| ≥ 6, or Isolation Forest score > 0.10.")
    top_ticker = found["ticker"].value_counts()
    kpi(k[3], "Most flagged", top_ticker.index[0] if not top_ticker.empty else "n/a",
        delta=f"{int(top_ticker.iloc[0])} flags" if not top_ticker.empty else None, delta_color="off")
    both = (found.groupby(["ticker", "detection_date"])["method"].nunique() > 1).sum()
    kpi(k[4], "Confirmed by both methods", f"{int(both)}",
        help="Stock-days flagged by both the z-score engine and the Isolation Forest.")

    section("Timeline")
    with st.container(border=True):
        charts.show(charts.anomaly_timeline(found))
        st.caption("Colour and size show severity (Moderate, High, Extreme); marker shape shows the signal. "
                   "Hover for the reason.")

    left, right = st.columns([1, 1])
    with left:
        section("Flags by stock and signal")
        matrix = found.pivot_table(index="ticker", columns="signal", values="anomaly_score", aggfunc="size",
                                   fill_value=0)
        st.dataframe(matrix, width="stretch")
    with right:
        section("Inspect a stock")
        focus = st.selectbox("Stock", sorted(found["ticker"].unique()), key="an_focus",
                             format_func=lambda t: f"{t} · {names.get(t, t)}")
        ohlcv = cached_ohlcv(focus, sel.start, sel.end)
        if not ohlcv.empty:
            charts.show(charts.candlestick(ohlcv, focus, None, found[found["ticker"] == focus], height=330))

    section("Details")
    table = (found.assign(_o=found["severity"].map(SEVERITY_ORDER))
             .sort_values(["detection_date", "_o"], ascending=[False, True]))
    event = st.dataframe(
        table[["detection_date", "ticker", "signal", "method", "severity", "anomaly_score", "explanation"]],
        hide_index=True, width="stretch", height=360, on_select="rerun", selection_mode="single-row", key="an_table",
        column_config={
            "detection_date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
            "ticker": "Ticker", "signal": "Signal", "severity": "Severity",
            "method": st.column_config.TextColumn("Method"),
            "anomaly_score": st.column_config.NumberColumn("Score", format="%.2f",
                                                           help="|z| for z-score signals; Isolation Forest anomaly score otherwise."),
            "explanation": st.column_config.TextColumn("Why it was flagged", width="large"),
        })
    rows = event.selection.rows if event and hasattr(event, "selection") else []
    if rows:
        row = table.iloc[rows[0]]
        with st.container(border=True):
            st.markdown(f"**{row['ticker']} · {row['detection_date']:%d %b %Y} · {row['signal']} ({row['severity']})**")
            st.write(row["explanation"])
            try:
                st.json(json.loads(row["details"]), expanded=False)
            except (TypeError, ValueError):
                pass
    else:
        st.caption("Select a row to see the full detection parameters.")
    st.download_button("Download flags (CSV)", table.drop(columns=["_o"]).to_csv(index=False).encode(),
                       file_name=f"anomalies_{sel.start}_{sel.end}.csv", mime="text/csv",
                       icon=":material/download:")

    with st.expander("How detection works"):
        st.markdown(
            f"- **Rolling z-score.** For each signal, the baseline mean and standard deviation come from the previous "
            f"**{window}** sessions only (shifted by one day), so a day never sets its own baseline. Signals are the "
            "daily return, log volume, the high-low range relative to the close, and 20-day volatility.\n"
            f"- **Isolation Forest.** Five standardised daily features; about **{contamination:.1%}** of each stock's "
            "days are flagged by construction, so the count depends on this setting.\n"
            "- **Severity.** z-score: |z| 3–4 Moderate, 4–6 High, 6+ Extreme. Isolation Forest score: "
            "<0.05 Moderate, 0.05–0.10 High, >0.10 Extreme. These bands are heuristics.\n"
            "- **Limits.** Heavy tails make large |z| values more common than a normal distribution suggests. "
            "Clustered volatility (e.g. March 2020) raises the baseline, so later days look less unusual.")
    footer()
