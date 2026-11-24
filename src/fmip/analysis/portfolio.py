"""Hypothetical portfolio analytics (long-only).

A portfolio here is a back-calculation on historical prices: what a set of weights would have
returned. It ignores transaction costs, taxes, bid-ask spreads, FX conversion, cash drag and the
fact that the asset list is chosen with hindsight. It is not an executable or investable return.

Conventions:
- Asset returns are simple daily returns on each asset's own calendar. The portfolio starts on
  the first date when every holding has a price. On a day when one holding's exchange is closed,
  that holding is carried at its last close (a 0% return that day), as in normal
  mark-to-market accounting. The number of such holding-days is reported.
- Rebalancing: "daily" keeps weights constant, "monthly" resets them on the first session of
  each month, and "none" is buy-and-hold (weights drift with prices).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fmip.analysis.eda import max_drawdown
from fmip.analysis.relative import beta
from fmip.analysis.returns import asset_returns, common_start
from fmip.analysis.risk import (
    annualised_return,
    conditional_var,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)
from fmip.config import TRADING_DAYS_PER_YEAR
from fmip.exceptions import InsufficientDataError

REBALANCE_OPTIONS = ("daily", "monthly", "none")


def normalise_weights(weights: dict[str, float] | pd.Series) -> pd.Series:
    """Validate long-only weights and scale them to sum to 1."""
    series = pd.Series(weights, dtype=float)
    if series.isna().any():
        raise ValueError("Weights must be numbers")
    if (series < 0).any():
        raise ValueError("Weights must be zero or positive (long-only portfolio)")
    series = series[series > 0]
    total = series.sum()
    if total <= 0:
        raise ValueError("At least one weight must be positive")
    return series / total


def equal_weights(tickers: list[str]) -> pd.Series:
    if not tickers:
        raise ValueError("No tickers supplied")
    return pd.Series(1.0 / len(tickers), index=tickers)


@dataclass(frozen=True)
class PortfolioResult:
    returns: pd.Series            # daily portfolio returns
    wealth: pd.Series             # growth of 1 unit
    end_weights: pd.Series        # weights on the last day (after drift)
    start: pd.Timestamp
    carried_holding_days: int     # holding-days valued at last close because a market was shut


def simulate(prices: pd.DataFrame, weights: pd.Series, rebalance: str = "monthly") -> PortfolioResult:
    """Back-calculate a portfolio's daily returns from asset prices and target weights."""
    if rebalance not in REBALANCE_OPTIONS:
        raise ValueError(f"rebalance must be one of {REBALANCE_OPTIONS}")
    target = normalise_weights(weights)
    missing = [t for t in target.index if t not in prices.columns]
    if missing:
        raise InsufficientDataError(f"No prices for: {', '.join(missing)}")
    panel = prices[target.index].astype(float)
    start = common_start(panel)
    if start is None:
        raise InsufficientDataError("No overlapping history for the holdings")
    rets = asset_returns(panel).loc[start:].iloc[1:]
    if len(rets) < 20:
        raise InsufficientDataError("Fewer than 20 common sessions for the portfolio")
    carried = int(rets.isna().sum().sum())
    rets = rets.fillna(0.0)  # holding carried at its last close while its market is closed

    w = target.values
    holdings = w.copy()            # value of each holding, portfolio starts at 1.0
    months = rets.index.to_period("M")
    port = np.empty(len(rets))
    values = rets.values
    for i in range(len(rets)):
        if i > 0:
            if rebalance == "daily" or (rebalance == "monthly" and months[i] != months[i - 1]):
                holdings = holdings.sum() * w
        before = holdings.sum()
        holdings = holdings * (1.0 + values[i])
        port[i] = holdings.sum() / before - 1.0
    port_returns = pd.Series(port, index=rets.index, name="portfolio")
    wealth = (1.0 + port_returns).cumprod()
    wealth = pd.concat([pd.Series([1.0], index=[start]), wealth])
    end_weights = pd.Series(holdings / holdings.sum(), index=target.index)
    return PortfolioResult(port_returns, wealth, end_weights, start, carried)


def performance(result: PortfolioResult, risk_free_rate: float = 0.0, confidence: float = 0.95) -> dict[str, float]:
    r = result.returns
    dd = max_drawdown(result.wealth)
    return {
        "total_return": float(result.wealth.iloc[-1] - 1.0),
        "annualised_return": annualised_return(r),
        "annualised_volatility": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "sharpe_ratio": sharpe_ratio(r, risk_free_rate),
        "sortino_ratio": sortino_ratio(r, risk_free_rate),
        "max_drawdown": float(dd["max_drawdown"]),
        "var_1d": value_at_risk(r, confidence),
        "cvar_1d": conditional_var(r, confidence),
        "sessions": float(len(r)),
    }


def diversification(prices: pd.DataFrame, weights: pd.Series) -> dict[str, object]:
    """Risk decomposition on common-period daily returns.

    - Diversification ratio = (sum w_i * sigma_i) / sigma_p. 1.0 means no diversification
      benefit; higher is better.
    - Effective number of holdings = 1 / sum(w_i^2) (weight concentration).
    - Risk contribution_i = w_i * (Sigma w)_i / sigma_p^2. The contributions sum to 100%.
    """
    target = normalise_weights(weights)
    panel = prices[target.index].astype(float)
    start = common_start(panel)
    rets = asset_returns(panel).loc[start:].iloc[1:].dropna()
    if len(rets) < 20:
        raise InsufficientDataError("Fewer than 20 common sessions for the risk decomposition")
    cov = rets.cov() * TRADING_DAYS_PER_YEAR
    w = target.values
    port_var = float(w @ cov.values @ w)
    port_vol = float(np.sqrt(port_var))
    asset_vol = np.sqrt(np.diag(cov.values))
    marginal = cov.values @ w
    contributions = pd.Series(w * marginal / port_var if port_var > 0 else np.nan, index=target.index)
    corr = rets.corr()
    n = len(target)
    avg_corr = float(corr.values[np.triu_indices(n, k=1)].mean()) if n > 1 else float("nan")
    return {
        "portfolio_volatility": port_vol,
        "weighted_avg_volatility": float(w @ asset_vol),
        "diversification_ratio": float(w @ asset_vol / port_vol) if port_vol > 0 else float("nan"),
        "effective_holdings": float(1.0 / np.sum(w ** 2)),
        "average_correlation": avg_corr,
        "risk_contribution": contributions,
        "asset_volatility": pd.Series(asset_vol, index=target.index),
        "correlation": corr,
    }


def benchmark_comparison(portfolio_returns: pd.Series, benchmark_prices: pd.Series) -> dict[str, float]:
    """Beta, tracking error, information ratio and correlation versus a benchmark."""
    bench_returns = benchmark_prices.dropna().astype(float).pct_change()
    frame = pd.concat({"p": portfolio_returns, "b": bench_returns}, axis=1).dropna()
    if len(frame) < 20:
        raise InsufficientDataError("Fewer than 20 common sessions with the benchmark")
    active = frame["p"] - frame["b"]
    tracking_error = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    growth_p = float((1 + frame["p"]).prod())
    growth_b = float((1 + frame["b"]).prod())
    years = len(frame) / TRADING_DAYS_PER_YEAR
    ann_p = growth_p ** (1 / years) - 1
    ann_b = growth_b ** (1 / years) - 1
    return {
        "benchmark_total_return": growth_b - 1.0,
        "benchmark_annualised_return": ann_b,
        "excess_annualised_return": ann_p - ann_b,
        "beta": beta(frame["p"], frame["b"]),
        "correlation": float(frame["p"].corr(frame["b"])),
        "tracking_error": tracking_error,
        "information_ratio": float(active.mean() * TRADING_DAYS_PER_YEAR / tracking_error) if tracking_error > 0 else float("nan"),
        "common_sessions": float(len(frame)),
    }
