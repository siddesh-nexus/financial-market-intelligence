"""Portfolio Analytics: hypothetical long-only portfolios from the selected stocks."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import charts
from common import display_names, require_database
from components import footer, kpi, num, pct, period_prices, require_stocks, section
from fmip.analysis.eda import drawdown_series
from fmip.analysis.portfolio import (
    benchmark_comparison,
    diversification,
    equal_weights,
    normalise_weights,
    performance,
    simulate,
)
from fmip.config import load_settings
from fmip.exceptions import InsufficientDataError
from state import current
from theme import SERIES, mode

REBALANCE_LABELS = {"monthly": "Monthly", "daily": "Daily (constant weights)", "none": "Buy and hold"}


def _weight_table(tickers: list[str], names: dict[str, str]) -> pd.DataFrame:
    """Editable weights, remembered per ticker across reruns and selection changes."""
    stored: dict[str, float] = st.session_state.setdefault("pf_weight_map", {})
    default = round(100.0 / len(tickers), 2)
    return pd.DataFrame({"ticker": tickers, "name": [names.get(t, t) for t in tickers],
                         "weight": [stored.get(t, default) for t in tickers]})


def _reset_weights() -> None:
    st.session_state["pf_weight_map"] = {}
    st.session_state.pop("pf_editor", None)


def render() -> None:
    require_database()
    sel = current()
    if not require_stocks(sel, 2, " to build a portfolio"):
        return
    settings = load_settings()
    names = display_names([*sel.tickers, sel.benchmark])
    st.caption(":material/warning: Hypothetical, back-calculated performance on historical prices. It ignores "
               "transaction costs, taxes, spreads, FX conversion and the hindsight in choosing these stocks. "
               "It is not an investable or executable return, and not investment advice.")

    left, right = st.columns([2, 3])
    with left.container(border=True):
        st.markdown("**Holdings and target weights (%)**")
        editor = st.data_editor(
            _weight_table(sel.tickers, names), key="pf_editor", hide_index=True, width="stretch",
            disabled=["ticker", "name"],
            column_config={"ticker": "Ticker", "name": "Company",
                           "weight": st.column_config.NumberColumn("Weight %", min_value=0.0, max_value=100.0,
                                                                   step=0.5, format="%.1f")})
        st.session_state["pf_weight_map"].update(dict(zip(editor["ticker"], editor["weight"].fillna(0.0))))
        total = float(editor["weight"].fillna(0).sum())
        st.caption(f"Weights sum to {total:.1f}%. They are scaled to 100% for the calculation. "
                   "Long-only: negative weights are not allowed.")
        st.button("Reset to equal weights", on_click=_reset_weights, icon=":material/restart_alt:")
    with right.container(border=True):
        c = st.columns(2)
        rebalance = c[0].selectbox("Rebalancing", list(REBALANCE_LABELS), format_func=REBALANCE_LABELS.get,
                                   key="pf_rebalance")
        rf = c[1].number_input("Risk-free rate (annual %)", 0.0, 20.0, float(settings.risk_free_rate * 100), 0.25,
                               key="pf_rf") / 100
        compare_equal = st.toggle("Compare with an equal-weighted portfolio", value=True, key="pf_equal")
        st.caption(f"{sel.basis_label} basis. Daily rebalancing keeps the target weights every day. Monthly resets "
                   "them on the first session of each month. Buy and hold lets weights drift with prices.")

    try:
        weights = normalise_weights(dict(zip(editor["ticker"], editor["weight"].fillna(0.0))))
    except ValueError as exc:
        st.warning(str(exc))
        return
    panel = period_prices(sel)
    stocks = panel[[t for t in weights.index if t in panel.columns]]
    is_equal = bool((weights - 1.0 / len(weights)).abs().max() < 1e-6) and len(weights) == len(sel.tickers)
    if compare_equal and is_equal:
        st.caption(":material/info: The custom weights are already equal, so the equal-weight comparison is "
                   "identical. Edit the weights to compare.")
        compare_equal = False
    try:
        custom = simulate(stocks, weights, rebalance)
        custom_perf = performance(custom, rf)
        equal = simulate(stocks, equal_weights(list(stocks.columns)), rebalance) if compare_equal else None
        equal_perf = performance(equal, rf) if equal else None
    except InsufficientDataError as exc:
        st.info(f"{exc}. Choose a longer period or different stocks.")
        return

    section(f"Performance · {custom.start:%d %b %Y} to {sel.end:%d %b %Y} · {REBALANCE_LABELS[rebalance].lower()}")
    k = st.columns(6)
    metrics = [("Total return", "total_return", pct), ("Annualised return", "annualised_return", pct),
               ("Volatility", "annualised_volatility", lambda v: pct(v, signed=False)),
               ("Sharpe ratio", "sharpe_ratio", num), ("Sortino ratio", "sortino_ratio", num),
               ("Max drawdown", "max_drawdown", pct)]
    for col, (label, key, fmt) in zip(k, metrics):
        delta = None
        if equal_perf:
            diff = custom_perf[key] - equal_perf[key]
            delta = (f"{diff:+.2f} vs equal" if fmt is num else f"{diff * 100:+.1f} pp vs equal")
        good_when_lower = key in ("annualised_volatility",)
        kpi(col, label, fmt(custom_perf[key]), delta=delta, delta_color="inverse" if good_when_lower else "normal")
    if custom.carried_holding_days:
        st.caption(f"{custom.carried_holding_days} holding-day(s) were valued at the previous close because that "
                   "stock's exchange was closed while another holding's exchange was open.")

    wealth = {"Custom portfolio": custom.wealth}
    if equal:
        wealth["Equal weight"] = equal.wealth
    bench = panel[sel.benchmark].dropna().loc[custom.start:] if sel.benchmark in panel else pd.Series(dtype=float)
    if len(bench) > 1:
        wealth[sel.benchmark] = bench / bench.iloc[0]
    wealth_frame = pd.DataFrame(wealth)
    palette = SERIES[mode()]
    colours = {"Custom portfolio": palette[0], "Equal weight": palette[1]}
    a, b = st.columns([3, 2])
    with a.container(border=True):
        charts.show(charts.performance_lines(wealth_frame, colours, sel.benchmark, names,
                                             title="Growth of the portfolio vs benchmark (common start)"))
    with b.container(border=True):
        dd = pd.DataFrame({k_: drawdown_series(v.dropna()) for k_, v in wealth.items()})
        charts.show(charts.drawdown_lines(dd, colours, sel.benchmark, "Drawdown", height=380))

    section("Allocation and risk contribution")
    try:
        div = diversification(stocks, weights)
    except InsufficientDataError as exc:
        st.info(str(exc))
        footer()
        return
    c1, c2 = st.columns([3, 2])
    with c1.container(border=True):
        alloc = pd.DataFrame({"Target weight": weights, "Risk contribution": div["risk_contribution"],
                              "Weight at end (drifted)": custom.end_weights})
        charts.show(charts.grouped_bars(alloc, "Weight vs share of portfolio risk", height=340, y_title="Share (%)"))
        st.caption("Risk contribution = w_i·(Σw)_i / σ²_p, which sums to 100%. A holding whose risk share is "
                   "much larger than its weight is concentrating risk.")
    with c2.container(border=True):
        m = st.columns(2)
        kpi(m[0], "Diversification ratio", num(div["diversification_ratio"]),
            help="Weighted average volatility / portfolio volatility. 1.0 = no diversification benefit; higher is better.")
        kpi(m[1], "Effective holdings", num(div["effective_holdings"], 1),
            help="1 / Σw². Equals the number of holdings when equally weighted.")
        kpi(m[0], "Portfolio volatility", pct(div["portfolio_volatility"], signed=False),
            delta=f"vs {div['weighted_avg_volatility']:.1%} weighted avg", delta_color="off")
        kpi(m[1], "Average correlation", num(div["average_correlation"]),
            help="Mean pairwise correlation of the holdings' daily returns.")
    with st.container(border=True):
        charts.show(charts.heatmap(div["correlation"], "Correlation between holdings"))

    section(f"Benchmark comparison · {sel.benchmark_name}")
    try:
        comparison = benchmark_comparison(custom.returns, panel[sel.benchmark])
        rows = {"Custom portfolio": comparison}
        if equal:
            rows["Equal weight"] = benchmark_comparison(equal.returns, panel[sel.benchmark])
        frame = pd.DataFrame(rows).T
        st.dataframe(frame, width="stretch", column_config={
            "benchmark_total_return": st.column_config.NumberColumn("Benchmark return", format="percent"),
            "benchmark_annualised_return": st.column_config.NumberColumn("Benchmark ann. return", format="percent"),
            "excess_annualised_return": st.column_config.NumberColumn("Excess ann. return", format="percent"),
            "beta": st.column_config.NumberColumn("Beta", format="%.2f"),
            "correlation": st.column_config.NumberColumn("Correlation", format="%.2f"),
            "tracking_error": st.column_config.NumberColumn("Tracking error", format="percent"),
            "information_ratio": st.column_config.NumberColumn("Information ratio", format="%.2f"),
            "common_sessions": st.column_config.NumberColumn("Sessions", format="%d"),
        })
        st.caption("Tracking error = std(portfolio − benchmark daily return) × √252. Information ratio = annualised "
                   "mean active return / tracking error. The benchmark is a price index without dividends.")
    except InsufficientDataError as exc:
        st.info(str(exc))
    footer()
