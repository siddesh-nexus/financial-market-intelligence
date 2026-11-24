"""Event-window analysis.

For an event date E, with trading-day offsets:
- Pre-event window:   the `pre_days` sessions ending the day before E.
- Event window:       the `event_days` sessions starting on E.
- Post-event window:  the `post_days` sessions after the event window.
- Recovery:           trading sessions after the event window until the adjusted close
                      regains its pre-event level (last close before E). None if not reached.

Outputs describe what happened around the date. They do not establish that the event caused
the movements; other news and market-wide moves occur in the same windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from fmip.config import TRADING_DAYS_PER_YEAR
from fmip.exceptions import InsufficientDataError


@dataclass(frozen=True)
class EventWindows:
    pre_days: int = 30
    event_days: int = 5
    post_days: int = 30
    recovery_lookahead: int = 250


def _window_return(prices: np.ndarray, start: int, end: int) -> float:
    """Return from the close at position `start` to the close at position `end` (inclusive)."""
    if start < 0 or end >= len(prices):
        return float("nan")
    return float(prices[end] / prices[start] - 1.0)


def _window_volatility(returns: np.ndarray) -> float:
    returns = returns[~np.isnan(returns)]
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def analyse_event(prices: pd.Series, event_date: date | str,
                  windows: EventWindows = EventWindows()) -> dict[str, float | int | None]:
    """Event-window statistics for one adjusted-close series."""
    series = prices.dropna().astype(float).sort_index()
    if len(series) < windows.event_days + 2:
        raise InsufficientDataError("Not enough prices for event-window analysis")

    event_ts = pd.Timestamp(event_date)
    values = series.values
    dates = series.index
    event_pos = int(dates.searchsorted(event_ts))  # first session on/after the event
    if event_pos == 0 or event_pos >= len(values):
        raise InsufficientDataError(f"Event date {event_ts.date()} is outside the price history")

    last_pre = event_pos - 1
    event_end = min(event_pos + windows.event_days - 1, len(values) - 1)
    post_end = min(event_end + windows.post_days, len(values) - 1)
    pre_start = max(last_pre - windows.pre_days, 0)

    returns = np.diff(values) / values[:-1]
    # returns[i] is the return from position i to i+1.
    pre_returns = returns[pre_start:last_pre] if last_pre > pre_start else np.array([])
    event_returns = returns[event_pos - 1:event_end] if event_end >= event_pos else np.array([])
    post_returns = returns[event_end:post_end] if post_end > event_end else np.array([])

    baseline = values[last_pre]
    recovery_days: int | None = None
    search_end = min(event_end + windows.recovery_lookahead, len(values) - 1)
    for pos in range(event_end + 1, search_end + 1):
        if values[pos] >= baseline:
            recovery_days = pos - event_end
            break

    pre_vol = _window_volatility(pre_returns)
    event_vol = _window_volatility(event_returns)
    post_vol = _window_volatility(post_returns)

    return {
        "event_date": event_ts.date(),
        "pre_return": _window_return(values, pre_start, last_pre),
        "event_return": _window_return(values, last_pre, event_end),
        "post_return": _window_return(values, event_end, post_end),
        "pre_volatility": pre_vol,
        "event_volatility": event_vol,
        "post_volatility": post_vol,
        "volatility_ratio_event_to_pre": (event_vol / pre_vol) if pre_vol and not np.isnan(pre_vol) else float("nan"),
        "recovery_sessions": recovery_days,
        "recovered": recovery_days is not None,
    }


def event_study(panel: pd.DataFrame, event_date: date | str,
                windows: EventWindows = EventWindows(),
                benchmark: pd.Series | None = None) -> pd.DataFrame:
    """Event-window statistics for every ticker in a wide adjusted-close panel.

    With a `benchmark` price series, market-adjusted (abnormal) returns are added:
    AR = stock window return - benchmark window return over the same window
    (the market-adjusted model, a simple and transparent event-study baseline).
    """
    rows = {}
    for ticker in panel.columns:
        try:
            rows[ticker] = analyse_event(panel[ticker], event_date, windows)
        except InsufficientDataError:
            continue
    table = pd.DataFrame.from_dict(rows, orient="index")
    if table.empty:
        return table
    table["recovery_sessions"] = table["recovery_sessions"].astype("Int64")
    if benchmark is not None:
        try:
            bench = analyse_event(benchmark, event_date, windows)
        except InsufficientDataError:
            bench = None
        if bench is not None:
            for window in ("pre", "event", "post"):
                table[f"{window}_abnormal_return"] = table[f"{window}_return"] - bench[f"{window}_return"]
    return table


def event_paths(panel: pd.DataFrame, event_date: date | str, pre_days: int = 30,
                post_days: int = 60) -> pd.DataFrame:
    """Each series rebased to 100 at the last close before the event, indexed by session offset.

    Offset 0 is the first session on or after the event date; offset -1 is the base (=100).
    Offsets are counted on each asset's own trading calendar.
    """
    event_ts = pd.Timestamp(event_date)
    paths = {}
    for ticker in panel.columns:
        series = panel[ticker].dropna().astype(float).sort_index()
        pos = int(series.index.searchsorted(event_ts))
        if pos == 0 or pos >= len(series):
            continue
        lo, hi = max(pos - 1 - pre_days, 0), min(pos + post_days, len(series))
        window = series.iloc[lo:hi]
        offsets = np.arange(lo, hi) - pos
        paths[ticker] = pd.Series(window.values / series.iloc[pos - 1] * 100.0, index=offsets)
    frame = pd.DataFrame(paths)
    frame.index.name = "session_offset"
    return frame
