"""Market Regimes: rule-based trend/volatility regimes and a Gaussian Mixture view."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import streamlit as st

import charts
from common import cached_panel, display_names, require_database
from components import footer, kpi, pct, section
from fmip.analysis.regimes import (
    GMM_STATE_NAMES,
    current_regime,
    gmm_regimes,
    regime_summary,
    rule_based_regimes,
)
from fmip.exceptions import InsufficientDataError
from state import current

WARMUP_DAYS = 3 * 365   # regimes need a 200-day MA and a year of volatility history before the period


def render() -> None:
    require_database()
    sel = current()
    options = [sel.benchmark, *sel.tickers]
    names = display_names(options)
    c = st.columns([2, 2, 3])
    symbol = c[0].selectbox("Series", options, key="rg_symbol", format_func=lambda t: f"{t} · {names.get(t, t)}",
                            help="Regimes are usually studied on a broad index; any selected stock also works.")
    method = c[1].segmented_control("Method", ["Rule-based", "Gaussian Mixture"], default="Rule-based", key="rg_method")
    c[2].caption("Regimes describe past conditions. They are not a forecast of future market behaviour.")

    prices = cached_panel((symbol,), sel.start - timedelta(days=WARMUP_DAYS), sel.end, sel.price_column)
    if prices.empty or symbol not in prices:
        st.warning(f"No stored data for {symbol}.")
        return
    series = prices[symbol].dropna()
    in_period = series.loc[pd.Timestamp(sel.start):]

    if method == "Gaussian Mixture":
        _gmm_view(symbol, series, in_period, sel)
    else:
        _rule_view(symbol, series, in_period, sel)
    footer()


def _rule_view(symbol: str, series: pd.Series, in_period: pd.Series, sel) -> None:
    try:
        rules = rule_based_regimes(series)
    except InsufficientDataError as exc:
        st.info(f"{exc}. Choose a longer period or a series with more history.")
        return
    window = rules.loc[in_period.index[0]:]
    now = current_regime(window)
    trend_summary = regime_summary(window["trend_regime"], window["return"])
    vol_summary = regime_summary(window["vol_regime"], window["return"])
    k = st.columns(4)
    kpi(k[0], "Current trend regime", now["trend"] or "n/a", help="Bull: close > MA200 and MA50 > MA200. "
        "Bear: close < MA200 and MA50 < MA200. Otherwise Sideways / transition.")
    kpi(k[1], "Current volatility regime", now["volatility"] or "n/a",
        help="21-day volatility vs the 20th/80th percentiles of its own past values.")
    if "Bull" in trend_summary.index:
        kpi(k[2], "Time in Bull regime", pct(trend_summary.loc["Bull", "share_of_time"], signed=False, digits=0))
    if "High" in vol_summary.index:
        kpi(k[3], "Time in high volatility", pct(vol_summary.loc["High", "share_of_time"], signed=False, digits=0))

    with st.container(border=True):
        charts.show(charts.regime_bands(window, f"{symbol}: trend regimes"))
        charts.show(charts.volatility_regime(window, "Volatility regime thresholds (past data only)"))

    a, b = st.columns(2)
    config = {
        "share_of_time": st.column_config.NumberColumn("Time", format="percent", width="small"),
        "sessions": st.column_config.NumberColumn("Sessions", width="small"),
        "annualised_mean_return": st.column_config.NumberColumn("Ann. return", format="percent", width="small"),
        "annualised_volatility": st.column_config.NumberColumn("Vol.", format="percent", width="small"),
        "episodes": st.column_config.NumberColumn("Episodes", width="small"),
        "avg_episode_sessions": st.column_config.NumberColumn("Avg length", format="%.0f", width="small",
                                                              help="Average episode length in sessions"),
    }
    with a:
        section("Trend regimes")
        st.dataframe(trend_summary, width="stretch", column_config=config)
    with b:
        section("Volatility regimes")
        st.dataframe(vol_summary, width="stretch", column_config=config)
    st.caption("Mean return and volatility use the same sessions as the label, so they describe each regime and are "
               "not forward-looking. Every input is computed only from data available on that date.")


def _gmm_view(symbol: str, series: pd.Series, in_period: pd.Series, sel) -> None:
    n_states = st.segmented_control("States", list(GMM_STATE_NAMES), default=3, key="rg_states") or 3
    window_series = series.loc[in_period.index[0] - pd.Timedelta(days=60):]
    try:
        frame, params = gmm_regimes(window_series, n_states=n_states)
    except InsufficientDataError as exc:
        st.info(f"{exc}. Choose a longer period.")
        return
    frame = frame.loc[in_period.index[0]:]
    order = [n for n in GMM_STATE_NAMES[n_states] if n in params.index]
    last = frame.iloc[-1]
    k = st.columns(3)
    kpi(k[0], "Latest state", last["state"], delta=f"p = {last['probability']:.0%}", delta_color="off",
        help="Most likely mixture component for the latest session, with its posterior probability.")
    kpi(k[1], "States fitted", str(n_states))
    kpi(k[2], "Sessions classified", f"{len(frame):,}")
    with st.container(border=True):
        charts.show(charts.gmm_states(frame, order, f"{symbol}: Gaussian Mixture states"))
        st.caption("Features: 21-session return and 21-session annualised volatility, standardised. The model is "
                   "fitted on the whole window shown (in-sample), so states can change when the period changes. "
                   "States are named by average volatility.")
    summary = regime_summary(frame["state"], frame["return"]).reindex(order)
    summary = summary.join(params[["mean_21d_return", "mean_volatility"]])
    st.dataframe(summary, width="stretch", column_config={
        "share_of_time": st.column_config.NumberColumn("Share of time", format="percent"),
        "annualised_mean_return": st.column_config.NumberColumn("Mean return (ann.)", format="percent"),
        "annualised_volatility": st.column_config.NumberColumn("Realised volatility", format="percent"),
        "avg_episode_sessions": st.column_config.NumberColumn("Avg length (sessions)", format="%.0f"),
        "mean_21d_return": st.column_config.NumberColumn("Mean 21d return", format="percent"),
        "mean_volatility": st.column_config.NumberColumn("Mean 21d vol", format="percent"),
    })
    st.caption("A Gaussian Mixture is used rather than K-means because it allows elliptical clusters and gives a "
               "probability for each label. Short episodes are common at the boundaries between states.")
