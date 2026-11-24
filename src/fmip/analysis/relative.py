"""Relative strength and benchmark-relative analytics.

Definitions (see docs/METHODOLOGY.md):
- Relative strength line: RS_t = (P_t / P_0) / (B_t / B_0). Above 1 means the stock has
  outperformed the benchmark since the start of the window; a rising line means it is
  currently outperforming.
- Excess return over a period: stock period return - benchmark period return (both simple).
- Rolling beta: Cov(r_s, r_b) / Var(r_b) over a trailing window of daily returns, using only
  dates on which both traded.

Both series are aligned on the dates where both have a price, so different exchange calendars
do not create spurious moves. These are historical descriptions, not forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmip.analysis.returns import TRAILING_PERIODS
from fmip.exceptions import InsufficientDataError


def _aligned(stock: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    frame = pd.concat({"s": stock.astype(float), "b": benchmark.astype(float)}, axis=1).dropna()
    if len(frame) < 2:
        raise InsufficientDataError("Stock and benchmark have fewer than two common sessions")
    return frame


def relative_strength_line(stock: pd.Series, benchmark: pd.Series) -> pd.Series:
    """Ratio of rebased stock to rebased benchmark on common dates (1.0 at the start)."""
    frame = _aligned(stock, benchmark)
    s = frame["s"] / frame["s"].iloc[0]
    b = frame["b"] / frame["b"].iloc[0]
    return (s / b).rename(stock.name)


def excess_returns(prices: pd.DataFrame, benchmark: pd.Series,
                   periods: dict[str, int] = TRAILING_PERIODS) -> pd.DataFrame:
    """Trailing excess return (stock minus benchmark) per ticker and period, on common dates.

    A period needs at least `sessions + 1` common observations; otherwise the cell is NaN.
    """
    rows = {}
    for ticker in prices.columns:
        try:
            frame = _aligned(prices[ticker], benchmark)
        except InsufficientDataError:
            continue
        row = {}
        for label, sessions in periods.items():
            if len(frame) <= sessions:
                row[label] = np.nan
                continue
            window = frame.iloc[-1 - sessions:]
            stock_ret = window["s"].iloc[-1] / window["s"].iloc[0] - 1.0
            bench_ret = window["b"].iloc[-1] / window["b"].iloc[0] - 1.0
            row[label] = float(stock_ret - bench_ret)
        rows[ticker] = row
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=list(periods))


def rolling_relative_return(stock: pd.Series, benchmark: pd.Series, window: int = 63) -> pd.Series:
    """Trailing `window`-session return of the stock minus that of the benchmark."""
    frame = _aligned(stock, benchmark)
    s = frame["s"].pct_change(window)
    b = frame["b"].pct_change(window)
    return (s - b).rename(stock.name)


def beta(stock_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Full-sample beta of daily returns on common dates."""
    frame = pd.concat({"s": stock_returns, "b": benchmark_returns}, axis=1).dropna()
    if len(frame) < 20:
        raise InsufficientDataError("Need at least 20 common returns for beta")
    var_b = frame["b"].var(ddof=1)
    if var_b == 0 or np.isnan(var_b):
        return float("nan")
    return float(frame["s"].cov(frame["b"]) / var_b)


def rolling_beta(stock_returns: pd.Series, benchmark_returns: pd.Series, window: int = 126) -> pd.Series:
    """Trailing-window beta: rolling covariance / rolling benchmark variance."""
    frame = pd.concat({"s": stock_returns, "b": benchmark_returns}, axis=1).dropna()
    cov = frame["s"].rolling(window, min_periods=window).cov(frame["b"])
    var = frame["b"].rolling(window, min_periods=window).var(ddof=1)
    return (cov / var.where(var > 0)).rename(stock_returns.name)


def relative_strength_summary(prices: pd.DataFrame, benchmark: pd.Series,
                              trend_window: int = 63) -> pd.DataFrame:
    """Per-ticker summary: RS ratio at the end of the window, whether the RS line is above its
    `trend_window`-session average (current momentum versus the benchmark), and a status label."""
    rows = {}
    for ticker in prices.columns:
        try:
            line = relative_strength_line(prices[ticker], benchmark)
        except InsufficientDataError:
            continue
        average = line.rolling(trend_window, min_periods=trend_window).mean()
        last = float(line.iloc[-1])
        trend_up = bool(line.iloc[-1] > average.iloc[-1]) if not np.isnan(average.iloc[-1]) else None
        if last >= 1 and trend_up:
            status = "Outperforming, improving"
        elif last >= 1:
            status = "Outperforming, fading" if trend_up is False else "Outperforming"
        elif trend_up:
            status = "Underperforming, improving"
        else:
            status = "Underperforming, weakening" if trend_up is False else "Underperforming"
        # (1 + R_s) / (1 + R_b) - 1: geometric excess return over the whole window.
        rows[ticker] = {"rs_ratio": last, "geometric_excess": last - 1.0,
                        "rs_above_trend": trend_up, "status": status}
    return pd.DataFrame.from_dict(rows, orient="index")
