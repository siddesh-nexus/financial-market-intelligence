"""Stock Explorer: candlesticks, moving averages, volume, returns, volatility, drawdowns, comparison."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import streamlit as st

import charts
from common import anomalies_for, cached_ohlcv, display_names, require_database
from components import footer, kpi, num, pct, period_prices, require_stocks, section
from fmip.analysis.eda import (
    DEFAULT_MA_WINDOWS,
    distribution_stats,
    drawdown_episodes,
    drawdown_series,
    max_drawdown,
    moving_averages,
    rolling_returns,
    rolling_volatility,
)
from fmip.analysis.relative import beta
from fmip.analysis.returns import rebase
from fmip.analysis.risk import conditional_var, value_at_risk
from fmip.exceptions import InsufficientDataError
from state import current
from theme import entity_colours

MA_LOOKBACK_DAYS = 320   # calendar days loaded before the period so a 200-day MA exists at the start


def render() -> None:
    require_database()
    sel = current()
    if not require_stocks(sel):
        return
    names = display_names([*sel.tickers, sel.benchmark])
    options = [*sel.tickers, sel.benchmark]
    if st.session_state.get("ex_ticker") not in options:
        st.session_state["ex_ticker"] = sel.tickers[0]

    head = st.columns([2.2, 3, 1.4])
    ticker = head[0].selectbox("Symbol", options, key="ex_ticker", label_visibility="collapsed",
                               format_func=lambda t: f"{t} · {names.get(t, t)}")
    ma_choices = head[1].pills("Moving averages", [f"MA {w}" for w in DEFAULT_MA_WINDOWS], selection_mode="multi",
                               default=["MA 20", "MA 50", "MA 200"], key="ex_mas", label_visibility="collapsed")
    show_anomalies = head[2].toggle("Anomaly markers", value=True, key="ex_anom")

    full = cached_ohlcv(ticker, sel.start - timedelta(days=MA_LOOKBACK_DAYS), sel.end)
    if full.empty:
        st.warning(f"No stored prices for {ticker}.")
        return
    ohlcv = full.loc[pd.Timestamp(sel.start):]
    if len(ohlcv) < 2:
        st.warning(f"Not enough sessions for {ticker} in this period.")
        return
    basis_col = "Close" if sel.basis == "price" else "Adj Close"
    series = ohlcv[basis_col]
    returns = series.pct_change().dropna()

    # ---- KPIs --------------------------------------------------------------------------------
    last, prev = ohlcv["Close"].iloc[-1], ohlcv["Close"].iloc[-2]
    try:
        stats = distribution_stats(returns)
        dd = max_drawdown(series)
        var95 = value_at_risk(returns, 0.95)
        cvar95 = conditional_var(returns, 0.95)
    except InsufficientDataError:
        st.info("Not enough data in this period for return statistics (at least 20 sessions).")
        return
    bench = period_prices(sel, include_benchmark=True)[sel.benchmark].dropna()
    try:
        b = beta(returns, bench.pct_change().dropna()) if ticker != sel.benchmark else 1.0
    except InsufficientDataError:
        b = float("nan")
    hi52 = full["High"].loc[full.index >= full.index[-1] - pd.Timedelta(days=365)].max()
    lo52 = full["Low"].loc[full.index >= full.index[-1] - pd.Timedelta(days=365)].min()
    k = st.columns(6)
    kpi(k[0], f"Last close · {ohlcv.index[-1]:%d %b}", f"{last:,.2f}", delta=pct(last / prev - 1, digits=2),
        help="Split-adjusted close of the latest stored session; delta is the one-day change.")
    kpi(k[1], f"Return · {sel.period}", pct(series.iloc[-1] / series.iloc[0] - 1),
        help=f"{sel.basis_label} from {ohlcv.index[0]:%d %b %Y} to {ohlcv.index[-1]:%d %b %Y}.")
    kpi(k[2], "Volatility (ann.)", pct(stats["std_daily_return"] * 252 ** 0.5, signed=False),
        help="Standard deviation of daily returns × √252.")
    kpi(k[3], "Max drawdown", pct(dd["max_drawdown"]),
        help=f"Peak {dd['peak_date']:%d %b %Y} → trough {dd['trough_date']:%d %b %Y}"
             + (f", recovered {dd['recovery_date']:%d %b %Y}." if dd["recovery_date"] is not None else ", not recovered."))
    kpi(k[4], f"Beta vs {sel.benchmark}", num(b), help="Covariance with benchmark daily returns / benchmark variance.")
    kpi(k[5], "52-week range", f"{lo52:,.0f} – {hi52:,.0f}",
        delta=f"{(last - lo52) / (hi52 - lo52):.0%} of range" if hi52 > lo52 else None, delta_color="off",
        help="Lowest low and highest high over the last 365 calendar days of stored data.")

    # ---- price chart -------------------------------------------------------------------------
    windows = sorted(int(m.split()[1]) for m in (ma_choices or []))
    mas = moving_averages(full["Close"], windows).loc[ohlcv.index] if windows else None
    marks = None
    if show_anomalies and not ticker.startswith("^"):
        marks = anomalies_for([ticker], sel.start, sel.end)
    with st.container(border=True):
        charts.show(charts.candlestick(ohlcv, ticker, mas, marks))
        st.caption("Candles and moving averages use split-adjusted prices (dividends not subtracted). "
                   "Moving averages are warmed up with data from before the period, so they are defined from day one. "
                   "▼ marks a flagged anomaly day.")

    # ---- analysis tabs -----------------------------------------------------------------------
    t1, t2, t3, t4, t5 = st.tabs(["Returns & volatility", "Drawdowns", "Rolling returns", "Compare", "OHLCV data"])
    with t1:
        a, b_ = st.columns([1, 1])
        with a:
            charts.show(charts.histogram(returns, "Daily return distribution", var95, cvar95))
            st.caption(f"Skewness {stats['skewness']:.2f} · excess kurtosis {stats['excess_kurtosis']:.2f} "
                       f"(normal = 0) · 1-day 95% VaR {var95:.2%}, CVaR {cvar95:.2%} (historical).")
        with b_:
            full_returns = full[basis_col].pct_change()
            vol = pd.DataFrame({"21-day": rolling_volatility(full_returns, 21),
                                "63-day": rolling_volatility(full_returns, 63)}).loc[ohlcv.index]
            charts.show(charts.lines(vol, "Rolling annualised volatility", "Volatility (%)", percent=True,
                                     value_format="%{y:.1f}%"))
    with t2:
        a, b_ = st.columns([3, 2])
        with a:
            charts.show(charts.drawdown_lines(drawdown_series(series).to_frame(ticker), entity_colours([ticker])))
        with b_:
            episodes = drawdown_episodes(series, top=5)
            st.markdown("**Deepest drawdown episodes in the period**")
            st.dataframe(episodes, hide_index=True, width="stretch", column_config={
                "peak_date": st.column_config.DateColumn("Peak", format="DD MMM YY"),
                "trough_date": st.column_config.DateColumn("Trough", format="DD MMM YY"),
                "recovery_date": st.column_config.DateColumn("Recovered", format="DD MMM YY"),
                "depth": st.column_config.NumberColumn("Depth", format="percent"),
                "decline_sessions": st.column_config.NumberColumn("Fall (sessions)"),
                "recovery_sessions": st.column_config.NumberColumn("Recovery (sessions)"),
            })
            st.caption("Blank recovery means the previous peak had not been regained by the end of the period.")
    with t3:
        rolling = {f"{w}-session": rolling_returns(full[basis_col], w) for w in (21, 63, 252)}
        frame = pd.DataFrame(rolling).loc[ohlcv.index].dropna(how="all")
        charts.show(charts.lines(frame, "Trailing compounded returns", "Return (%)", percent=True, zero_line=True))
    with t4:
        peers = period_prices(sel)
        default = [t for t in sel.tickers if t != ticker][:4]
        picks = st.multiselect("Compare with", [t for t in peers.columns if t != ticker], default=default,
                               key="ex_compare", format_func=lambda t: f"{t} · {names.get(t, t)}")
        cols = list(dict.fromkeys([ticker, *picks, sel.benchmark]))
        info = rebase(peers[[c for c in cols if c in peers.columns]])
        charts.show(charts.performance_lines(info.values, entity_colours(sel.tickers), sel.benchmark, names,
                                             title=f"{sel.basis_label}, common start"))
    with t5:
        table = ohlcv.sort_index(ascending=False).copy()
        table["Change %"] = ohlcv["Close"].pct_change().reindex(table.index)
        st.dataframe(table, width="stretch", height=380, column_config={
            "Open": st.column_config.NumberColumn(format="%.2f"), "High": st.column_config.NumberColumn(format="%.2f"),
            "Low": st.column_config.NumberColumn(format="%.2f"), "Close": st.column_config.NumberColumn(format="%.2f"),
            "Adj Close": st.column_config.NumberColumn(format="%.2f"),
            "Volume": st.column_config.NumberColumn(format="compact"),
            "Change %": st.column_config.NumberColumn(format="percent"),
        })
        st.download_button("Download CSV", table.to_csv().encode(), file_name=f"{ticker}_{sel.start}_{sel.end}.csv",
                           mime="text/csv", icon=":material/download:")
    section("Notes")
    st.caption("Close = split-adjusted; Adj Close = split- and dividend-adjusted (Yahoo Finance). "
               f"Statistics on this page use the {sel.basis_label.lower()} series.")
    footer()
