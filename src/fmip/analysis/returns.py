"""Calendar-safe return calculations for panels that mix exchanges.

A wide price panel built from several exchanges has NaN on days when one market was closed.
Computing `pct_change` on that union calendar loses the first return after every
market-specific holiday. Here each asset's returns are computed on **its own** trading
calendar (from its previous available close), then aligned on the union index. Nothing is
forward-filled or invented.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fmip.exceptions import InsufficientDataError


def asset_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns per column, each on its own trading calendar.

    The value on date t is P_t / P_prev - 1, where P_prev is that asset's previous available
    close. NaN on dates where the asset did not trade. The first row of each asset is NaN.
    """
    frame = prices.astype(float).sort_index()
    out = {}
    for column in frame.columns:
        series = frame[column].dropna()
        out[column] = series.pct_change()
    result = pd.DataFrame(out).reindex(frame.index)
    return result.reindex(columns=frame.columns)


@dataclass(frozen=True)
class Rebased:
    values: pd.DataFrame          # rebased series (1.0 = start) on the union calendar
    common_start: pd.Timestamp | None
    late_starters: dict[str, pd.Timestamp]   # tickers whose data begins after the panel start
    dropped: list[str]            # tickers with no data at all


def common_start(prices: pd.DataFrame) -> pd.Timestamp | None:
    """First date on or after which every column has started trading."""
    firsts = [prices[c].first_valid_index() for c in prices.columns]
    firsts = [f for f in firsts if f is not None]
    return max(firsts) if firsts else None


def rebase(prices: pd.DataFrame, align: str = "common") -> Rebased:
    """Rebase every series to 1.0.

    align="common": all series start at the first date where every series has data, so
    returns are compared over the same period (recommended).
    align="own": each series starts at its own first observation (periods differ).
    Values stay NaN where an asset did not trade, so charts must connect gaps rather than
    treat them as zero.
    """
    frame = prices.astype(float).sort_index()
    dropped = [c for c in frame.columns if frame[c].notna().sum() == 0]
    frame = frame.drop(columns=dropped)
    if frame.empty:
        return Rebased(frame, None, {}, dropped)
    panel_start = frame.index[0]
    firsts = {c: frame[c].first_valid_index() for c in frame.columns}
    late = {c: d for c, d in firsts.items() if d > panel_start}
    start = max(firsts.values()) if align == "common" else None
    if align == "common":
        frame = frame.loc[start:]
    out = {}
    for column in frame.columns:
        series = frame[column]
        base_idx = series.first_valid_index()
        out[column] = series / series.loc[base_idx]
    return Rebased(pd.DataFrame(out, index=frame.index), start, late, dropped)


def total_return(series: pd.Series) -> float:
    values = series.dropna().astype(float)
    if len(values) < 2:
        raise InsufficientDataError("Need at least two prices for a period return")
    return float(values.iloc[-1] / values.iloc[0] - 1.0)


def trailing_return(series: pd.Series, sessions: int) -> float:
    """Return over the last `sessions` trading sessions of the series (NaN if too short)."""
    values = series.dropna().astype(float)
    if len(values) <= sessions:
        return float("nan")
    return float(values.iloc[-1] / values.iloc[-1 - sessions] - 1.0)


TRAILING_PERIODS: dict[str, int] = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}


def trailing_returns_table(prices: pd.DataFrame, periods: dict[str, int] = TRAILING_PERIODS) -> pd.DataFrame:
    """Trailing returns per ticker for each named period (sessions counted per asset)."""
    return pd.DataFrame({label: {c: trailing_return(prices[c], n) for c in prices.columns}
                         for label, n in periods.items()})
