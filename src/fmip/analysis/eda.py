"""Exploratory data analysis: returns, moving averages, volatility, drawdowns and distributions.

Conventions (see docs/METHODOLOGY.md):
- Prices are adjusted closes; daily returns are simple percentage changes.
- Volatility is the sample standard deviation (ddof=1) annualised by sqrt(252).
- Skewness is the bias-corrected sample skewness; kurtosis is bias-corrected *excess* kurtosis.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from fmip.config import TRADING_DAYS_PER_YEAR
from fmip.exceptions import InsufficientDataError

DEFAULT_MA_WINDOWS: tuple[int, ...] = (20, 50, 100, 200)


def daily_returns(prices: pd.Series) -> pd.Series:
    """Simple daily returns. The first observation is dropped."""
    return prices.astype(float).pct_change().dropna()


def log_returns(prices: pd.Series) -> pd.Series:
    values = prices.astype(float)
    return np.log(values / values.shift(1)).dropna()


def cumulative_returns(prices: pd.Series) -> pd.Series:
    """Growth of the position relative to the first price (0.25 means +25%)."""
    values = prices.astype(float).dropna()
    if values.empty:
        raise InsufficientDataError("No prices available for cumulative returns")
    return values / values.iloc[0] - 1.0


def moving_averages(prices: pd.Series, windows: Sequence[int] = DEFAULT_MA_WINDOWS) -> pd.DataFrame:
    """Simple moving averages of the price, one column per window (e.g. 'MA_50')."""
    values = prices.astype(float)
    return pd.DataFrame({f"MA_{w}": values.rolling(w, min_periods=w).mean() for w in windows})


def annualised_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    clean = returns.dropna()
    if len(clean) < 2:
        raise InsufficientDataError("Need at least two returns to estimate volatility")
    return float(clean.std(ddof=1) * np.sqrt(periods_per_year))


def rolling_volatility(returns: pd.Series, window: int = 20,
                       periods_per_year: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    return returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(periods_per_year)


def rolling_returns(prices: pd.Series, window: int) -> pd.Series:
    """Compounded return over each trailing window of `window` sessions."""
    return prices.astype(float).pct_change(periods=window)


def drawdown_series(prices: pd.Series) -> pd.Series:
    """Percentage drop from the running peak (values are <= 0)."""
    values = prices.astype(float).dropna()
    running_peak = values.cummax()
    return values / running_peak - 1.0


def max_drawdown(prices: pd.Series) -> dict[str, float | pd.Timestamp | None]:
    """Maximum drawdown with the peak and trough dates. `max_drawdown` is negative or zero."""
    dd = drawdown_series(prices)
    if dd.empty:
        raise InsufficientDataError("No prices available for drawdown")
    trough_date = dd.idxmin()
    peak_date = prices.loc[:trough_date].astype(float).idxmax()
    recovery = prices.loc[trough_date:].astype(float)
    peak_value = float(prices.loc[peak_date])
    recovered = recovery[recovery >= peak_value]
    return {
        "max_drawdown": float(dd.min()),
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovered.index[0] if not recovered.empty else None,
    }


def drawdown_episodes(prices: pd.Series, top: int = 5) -> pd.DataFrame:
    """The `top` deepest drawdown episodes: peak, trough, recovery (NaT if not recovered),
    depth, sessions from peak to trough, and sessions from trough to recovery.

    An episode starts at a running peak and ends when the price regains that peak.
    """
    values = prices.astype(float).dropna()
    columns = ["peak_date", "trough_date", "recovery_date", "depth", "decline_sessions", "recovery_sessions"]
    if len(values) < 2:
        return pd.DataFrame(columns=columns)
    peak = values.cummax()
    underwater = values < peak
    episode_id = (~underwater).cumsum()
    rows = []
    positions = {ts: i for i, ts in enumerate(values.index)}
    for _, block in values[underwater].groupby(episode_id[underwater]):
        start_pos = positions[block.index[0]] - 1
        peak_date = values.index[start_pos]
        trough_date = block.idxmin()
        depth = float(block.min() / values.iloc[start_pos] - 1.0)
        end_pos = positions[block.index[-1]] + 1
        recovered = end_pos < len(values)
        rows.append({
            "peak_date": peak_date,
            "trough_date": trough_date,
            "recovery_date": values.index[end_pos] if recovered else pd.NaT,
            "depth": depth,
            "decline_sessions": positions[trough_date] - start_pos,
            "recovery_sessions": (end_pos - positions[trough_date]) if recovered else np.nan,
        })
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values("depth").head(top).reset_index(drop=True)


def distribution_stats(returns: pd.Series) -> dict[str, float]:
    clean = returns.dropna().astype(float)
    if len(clean) < 4:
        raise InsufficientDataError("Need at least four returns for distribution statistics")
    return {
        "observations": float(len(clean)),
        "mean_daily_return": float(clean.mean()),
        "std_daily_return": float(clean.std(ddof=1)),
        "skewness": float(stats.skew(clean, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(clean, fisher=True, bias=False)),
        "min_daily_return": float(clean.min()),
        "max_daily_return": float(clean.max()),
    }


def summarise_ticker(prices: pd.Series) -> dict[str, float | pd.Timestamp | None]:
    """One-row summary used by the Market Overview and Stock Explorer pages."""
    values = prices.astype(float).dropna()
    returns = daily_returns(values)
    dd = max_drawdown(values)
    return {
        "start_date": values.index[0],
        "end_date": values.index[-1],
        "total_return": float(values.iloc[-1] / values.iloc[0] - 1.0),
        "annualised_volatility": annualised_volatility(returns),
        "mean_daily_return": float(returns.mean()),
        "max_drawdown": dd["max_drawdown"],
        "skewness": distribution_stats(returns)["skewness"],
        "excess_kurtosis": distribution_stats(returns)["excess_kurtosis"],
    }


def summary_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker summary for a wide adjusted-close panel."""
    rows = {}
    for ticker in panel.columns:
        series = panel[ticker].dropna()
        if len(series) < 5:
            continue
        rows[ticker] = summarise_ticker(series)
    return pd.DataFrame.from_dict(rows, orient="index")
