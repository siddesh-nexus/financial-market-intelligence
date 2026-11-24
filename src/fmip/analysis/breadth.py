"""Market breadth computed across a set of stocks.

Breadth measures how broadly a move is shared. Here it is computed **only over the stocks
supplied** (normally the user's selection), not the full exchange, and the dashboard says so.

- Advancers / decliners on day t: stocks whose daily return is > 0 / < 0. Unchanged and
  non-trading stocks are counted separately and not in either group.
- Advance-decline line: cumulative sum of (advancers - decliners).
- % above N-day MA: share of stocks (with at least N prior sessions) whose close is above their
  own N-session simple moving average. Stocks without enough history are excluded from the
  denominator rather than counted as "below".
"""

from __future__ import annotations

import pandas as pd

from fmip.analysis.returns import asset_returns


def advance_decline(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily advancers, decliners, unchanged, participants and the cumulative A-D line."""
    returns = asset_returns(prices).iloc[1:]
    advancers = (returns > 0).sum(axis=1)
    decliners = (returns < 0).sum(axis=1)
    participants = returns.notna().sum(axis=1)
    frame = pd.DataFrame({
        "advancers": advancers,
        "decliners": decliners,
        "unchanged": participants - advancers - decliners,
        "participants": participants,
    })
    frame = frame[frame["participants"] > 0]
    frame["net"] = frame["advancers"] - frame["decliners"]
    frame["ad_line"] = frame["net"].cumsum()
    return frame


def share_above_ma(prices: pd.DataFrame, window: int) -> pd.Series:
    """Fraction (0..1) of eligible stocks trading above their own `window`-day moving average."""
    index = prices.index
    above = pd.DataFrame(index=index)
    eligible = pd.DataFrame(index=index)
    for column in prices.columns:
        series = prices[column].dropna().astype(float)
        ma = series.rolling(window, min_periods=window).mean()
        valid = ma.notna()
        above[column] = ((series > ma) & valid).astype(float).reindex(index, fill_value=0.0)
        eligible[column] = valid.astype(float).reindex(index, fill_value=0.0)
    eligible_n = eligible.sum(axis=1)
    above_n = above.sum(axis=1)
    share = above_n / eligible_n.where(eligible_n > 0)
    return share.rename(f"pct_above_ma{window}")


def breadth_snapshot(prices: pd.DataFrame) -> dict[str, float | int]:
    """Latest-session breadth figures plus period totals, for KPI cards."""
    ad = advance_decline(prices)
    result: dict[str, float | int] = {"stocks": int(prices.shape[1])}
    if ad.empty:
        return result
    last = ad.iloc[-1]
    result.update({
        "as_of": ad.index[-1],
        "advancers": int(last["advancers"]),
        "decliners": int(last["decliners"]),
        "unchanged": int(last["unchanged"]),
        "period_up_days_share": float((ad["net"] > 0).mean()),
    })
    for window in (50, 200):
        share = share_above_ma(prices, window).dropna()
        result[f"pct_above_ma{window}"] = float(share.iloc[-1]) if not share.empty else float("nan")
    return result
