"""Risk Analytics: volatility, Sharpe/Sortino, drawdowns, VaR/CVaR, beta and rolling risk."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import charts
from common import display_names, require_database
from components import context_notes, footer, period_prices, require_stocks, section
from fmip.analysis.eda import drawdown_series, rolling_volatility
from fmip.analysis.relative import beta
from fmip.analysis.returns import asset_returns
from fmip.analysis.risk import risk_comparison, rolling_sharpe, rolling_var
from fmip.config import load_settings
from fmip.exceptions import InsufficientDataError
from state import current
from theme import entity_colours

LOOKBACK_DAYS = 380  # warm-up history for rolling windows


def render() -> None:
    require_database()
    sel = current()
    if not require_stocks(sel):
        return
    settings = load_settings()
    names = display_names([*sel.tickers, sel.benchmark])
    colours = entity_colours(sel.tickers)

    c = st.columns([1, 1, 3])
    rf_percent = c[0].number_input("Risk-free rate (annual %)", min_value=0.0, max_value=20.0, step=0.25,
                                   value=float(settings.risk_free_rate * 100), key="rk_rf")
    confidence = c[1].select_slider("VaR / CVaR confidence", options=[0.90, 0.95, 0.99], value=0.95,
                                    format_func=lambda v: f"{v:.0%}", key="rk_conf")
    c[2].caption(f"Simple daily {sel.basis_label.lower()}s, annualised with 252 sessions. Risk-free rate converted "
                 "to daily as rf / 252. VaR and CVaR are one-day losses shown as positive numbers. "
                 "Past risk does not guarantee future risk.")

    panel = period_prices(sel)
    if panel.empty:
        st.warning("No stored data for this selection.")
        return
    table = risk_comparison(panel, risk_free_rate=rf_percent / 100, confidence=confidence)
    if table.empty:
        st.info("Not enough data (at least 20 returns per series) for risk metrics.")
        return
    returns = asset_returns(panel)
    bench_returns = returns[sel.benchmark] if sel.benchmark in returns else None
    betas = {}
    for ticker in table.index:
        try:
            betas[ticker] = 1.0 if ticker == sel.benchmark else beta(returns[ticker], bench_returns)
        except (InsufficientDataError, TypeError):
            betas[ticker] = float("nan")
    table["beta"] = pd.Series(betas)
    table.insert(0, "name", [names.get(t, t) for t in table.index])

    section(f"Risk comparison · {sel.period} · benchmark {sel.benchmark} shown for reference")
    st.dataframe(table.drop(columns=["risk_free_rate_annual", "confidence_level"]), width="stretch",
                 column_config={
                     "name": "Company",
                     "annualised_volatility": st.column_config.NumberColumn("Volatility", format="percent"),
                     "annualised_return": st.column_config.NumberColumn("Ann. return", format="percent"),
                     "sharpe_ratio": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                     "sortino_ratio": st.column_config.NumberColumn("Sortino", format="%.2f"),
                     "max_drawdown": st.column_config.NumberColumn("Max drawdown", format="percent"),
                     "var_historical_1d": st.column_config.NumberColumn(f"VaR {confidence:.0%} (hist.)", format="percent"),
                     "var_parametric_1d": st.column_config.NumberColumn(f"VaR {confidence:.0%} (normal)", format="percent"),
                     "cvar_1d": st.column_config.NumberColumn(f"CVaR {confidence:.0%}", format="percent"),
                     "beta": st.column_config.NumberColumn(f"Beta vs {sel.benchmark}", format="%.2f"),
                     "observations": st.column_config.NumberColumn("Returns", format="%d"),
                 })
    context_notes(sel, panel)

    a, b = st.columns([1, 1])
    with a.container(border=True):
        charts.show(charts.scatter_labeled(table, "annualised_volatility", "annualised_return",
                                           "Risk vs return", "Volatility", "Annualised return",
                                           colours={**colours, sel.benchmark: "#898781"},
                                           x_percent=True, y_percent=True, height=360))
        st.caption("Up and to the left is a better trade-off. Lines of equal Sharpe ratio run through the origin.")
    with b.container(border=True):
        tail = table[[f"var_historical_1d", "cvar_1d"]].rename(columns={"var_historical_1d": f"VaR {confidence:.0%}",
                                                                        "cvar_1d": f"CVaR {confidence:.0%}"})
        charts.show(charts.grouped_bars(tail, "One-day tail loss (historical)", percent=True, height=360,
                                        y_title="Loss (%)"))
        st.caption("CVaR is the average loss on days at or beyond the VaR cut-off, so it is always at least as large as VaR.")

    section("Drawdowns")
    with st.container(border=True):
        drawdowns = pd.DataFrame({t: drawdown_series(panel[t].dropna()) for t in panel.columns
                                  if panel[t].notna().sum() > 1})
        charts.show(charts.drawdown_lines(drawdowns, colours, sel.benchmark, None, height=320))

    section("Rolling risk")
    r = st.columns([1, 1, 2])
    metric = r[0].segmented_control("Metric", ["Volatility", "Sharpe", "VaR"], default="Volatility", key="rk_metric")
    window = r[1].select_slider("Window (sessions)", [21, 63, 126, 252], value=63, key="rk_window")
    focus = r[2].multiselect("Series", list(panel.columns), default=list(panel.columns)[:5], key="rk_focus")
    longer = period_prices(sel, lookback_days=LOOKBACK_DAYS)
    longer_returns = asset_returns(longer)
    frame = {}
    for ticker in focus:
        series = longer_returns[ticker].dropna()
        if metric == "Sharpe":
            frame[ticker] = rolling_sharpe(series, window, rf_percent / 100)
        elif metric == "VaR":
            frame[ticker] = rolling_var(series, max(window, 63), confidence)
        else:
            frame[ticker] = rolling_volatility(series, window)
    if frame:
        data = pd.DataFrame(frame).loc[pd.Timestamp(sel.start):]
        y_title = {"Volatility": "Annualised volatility (%)", "Sharpe": "Sharpe ratio", "VaR": "1-day VaR (%)"}[metric or "Volatility"]
        with st.container(border=True):
            charts.show(charts.lines(data, f"Rolling {metric.lower() if metric else 'volatility'} ({window if metric != 'VaR' else max(window, 63)} sessions)",
                                     y_title, colours=colours, percent=metric != "Sharpe", benchmark=sel.benchmark,
                                     zero_line=metric == "Sharpe",
                                     value_format="%{y:.2f}" if metric == "Sharpe" else "%{y:.1f}%"))
    with st.expander("Assumptions and definitions"):
        st.markdown(
            f"- **Volatility** = std(daily returns, ddof=1) × √252.\n"
            f"- **Sharpe** = mean(daily return − rf/252) / std × √252, with rf = **{rf_percent:.2f}%**.\n"
            "- **Sortino** uses downside deviation over all observations: √mean(min(excess, 0)²).\n"
            f"- **VaR {confidence:.0%}**: historical uses the empirical quantile; normal assumes Gaussian returns "
            "(it usually understates tail losses).\n"
            "- **CVaR**: average loss on days at or beyond the historical VaR cut-off.\n"
            f"- **Beta**: covariance with {sel.benchmark} daily returns / benchmark variance, on common dates.\n"
            "- **Max drawdown**: largest peak-to-trough fall within the period.")
    footer()
