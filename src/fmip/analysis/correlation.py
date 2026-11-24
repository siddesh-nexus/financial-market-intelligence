"""Correlation and sector analysis on daily returns.

Correlation describes co-movement in the historical sample. It does not imply causation
or predict future co-movement.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from fmip.analysis.eda import annualised_volatility
from fmip.analysis.returns import asset_returns
from fmip.exceptions import InsufficientDataError

CorrelationMethod = Literal["pearson", "spearman"]


def returns_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns for a wide price panel, each asset on its own trading calendar
    (see fmip.analysis.returns.asset_returns). Pairwise correlation handles remaining gaps."""
    return asset_returns(prices).iloc[1:]


def interpret_correlation(value: float) -> str:
    """Plain-language strength label for a correlation coefficient."""
    if pd.isna(value):
        return "n/a"
    magnitude = abs(value)
    if magnitude >= 0.8:
        strength = "very strong"
    elif magnitude >= 0.6:
        strength = "strong"
    elif magnitude >= 0.4:
        strength = "moderate"
    elif magnitude >= 0.2:
        strength = "weak"
    else:
        return "little or no linear relationship"
    return f"{strength} {'positive' if value > 0 else 'negative'}"


def cluster_order(corr: pd.DataFrame) -> list[str]:
    """Order assets so that highly correlated ones sit next to each other (average-linkage
    hierarchical clustering on the distance 1 - rho). Falls back to the input order."""
    if corr.shape[0] < 3 or corr.isna().any().any():
        return list(corr.columns)
    distance = (1.0 - corr.values).clip(min=0.0)
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0
    tree = linkage(squareform(distance, checks=False), method="average")
    return [corr.columns[i] for i in leaves_list(tree)]


def average_pairwise_correlation(corr: pd.DataFrame) -> float:
    """Mean of the off-diagonal correlations (upper triangle)."""
    n = corr.shape[0]
    if n < 2:
        return float("nan")
    upper = corr.values[np.triu_indices(n, k=1)]
    return float(np.nanmean(upper))


def correlation_matrix(returns: pd.DataFrame, method: CorrelationMethod = "pearson",
                       min_periods: int = 60) -> pd.DataFrame:
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")
    if len(returns.dropna(how="all")) < min_periods:
        raise InsufficientDataError(f"Need at least {min_periods} overlapping return rows")
    return returns.corr(method=method, min_periods=min_periods)


def rolling_correlation(returns: pd.DataFrame, first: str, second: str, window: int = 60) -> pd.Series:
    if first not in returns.columns or second not in returns.columns:
        raise KeyError(f"Both {first} and {second} must be in the returns panel")
    return returns[first].rolling(window, min_periods=window).corr(returns[second])


def strongest_pairs(corr: pd.DataFrame, positive_threshold: float = 0.7,
                    negative_threshold: float = -0.3) -> pd.DataFrame:
    """Pairs of assets that are highly positively or negatively correlated (upper triangle only)."""
    columns = corr.columns
    records = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            value = corr.iloc[i, j]
            if pd.isna(value):
                continue
            if value >= positive_threshold:
                relation = "high positive"
            elif value <= negative_threshold:
                relation = "negative"
            else:
                continue
            records.append({"asset_a": columns[i], "asset_b": columns[j],
                            "correlation": float(value), "relationship": relation})
    frame = pd.DataFrame(records, columns=["asset_a", "asset_b", "correlation", "relationship"])
    return frame.sort_values("correlation", ascending=False).reset_index(drop=True)


def sector_returns(returns: pd.DataFrame, sector_map: dict[str, str]) -> pd.DataFrame:
    """Equal-weighted daily sector return: the mean of member-stock returns each day."""
    members: dict[str, list[str]] = {}
    for ticker in returns.columns:
        sector = sector_map.get(ticker)
        if sector:
            members.setdefault(sector, []).append(ticker)
    return pd.DataFrame({s: returns[cols].mean(axis=1) for s, cols in members.items()})


def sector_performance(prices: pd.DataFrame, sector_map: dict[str, str]) -> pd.DataFrame:
    """Per-sector summary over the sample: total return of the equal-weighted sector index,
    annualised volatility, and the number of member stocks."""
    returns = returns_panel(prices)
    sec = sector_returns(returns, sector_map)
    rows = {}
    for sector in sec.columns:
        series = sec[sector].dropna()
        if len(series) < 2:
            continue
        growth = (1.0 + series).prod()
        rows[sector] = {
            "members": sum(1 for t, s in sector_map.items() if s == sector and t in prices.columns),
            "total_return": float(growth - 1.0),
            "annualised_volatility": annualised_volatility(series),
            "mean_daily_return": float(series.mean()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def sector_vs_benchmark(prices: pd.DataFrame, sector_map: dict[str, str], benchmark: pd.Series) -> pd.DataFrame:
    """Sector performance with the benchmark's return over the same sessions and the difference.

    Sector indices are equal-weighted and rebalanced daily (the mean of member returns), so the
    result describes the selected stocks only, not the official sector index.
    """
    perf = sector_performance(prices, sector_map)
    if perf.empty:
        return perf
    bench = benchmark.dropna().astype(float)
    sample = prices.dropna(how="all")
    window = bench.loc[sample.index[0]:sample.index[-1]] if not sample.empty else bench.iloc[0:0]
    perf["benchmark_return"] = float(window.iloc[-1] / window.iloc[0] - 1.0) if len(window) > 1 else np.nan
    perf["relative_return"] = perf["total_return"] - perf["benchmark_return"]
    return perf.sort_values("total_return", ascending=False)
