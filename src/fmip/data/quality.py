"""Data-quality checks on stored daily bars.

The checks describe the data; they never modify it. A flagged item is something a user should
know about before trusting a number. It is not necessarily an error: a +30% earnings day is a
real move, but it is worth verifying against a second source.

Severity levels:
- "error":   very likely a data problem (e.g. an unadjusted split, duplicate dates).
- "warning": could distort analysis or needs verification (extreme move, long gap, stale run).
- "info":    worth knowing (short history, zero-volume days).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import pandas as pd

EXTREME_MOVE = 0.25            # |daily return| at or above this is flagged for verification
SPLIT_SIGNATURE_MOVE = 0.40    # unadjusted splits show up as moves of 50%+ in Close
COMMON_SPLIT_RATIOS = (2, 3, 4, 5, 8, 10, 15, 20, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 8, 1 / 10)
ADJ_FACTOR_JUMP = 0.05         # one-day change in adj_close/close larger than a typical dividend
MAX_CALENDAR_GAP_DAYS = 7      # longer gaps than this between sessions are flagged
STALE_RUN = 5                  # this many identical closes in a row is suspicious
SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


@dataclass(frozen=True)
class QualityIssue:
    ticker: str
    check: str
    severity: str
    detail: str
    first_date: date | None = None
    count: int = 1

    def as_dict(self) -> dict:
        return asdict(self)


def _dates(index: pd.Index) -> list[date]:
    return [pd.Timestamp(x).date() for x in index]


def check_bars(ticker: str, bars: pd.DataFrame, *, is_index: bool = False) -> list[QualityIssue]:
    """Run all single-ticker checks. `bars` is indexed by date with columns
    open, high, low, close, adj_close, volume (lower-case, as stored)."""
    issues: list[QualityIssue] = []
    if bars.empty:
        return [QualityIssue(ticker, "no_data", "error", "No stored prices in the selected range.")]

    frame = bars.sort_index()
    duplicated = frame.index.duplicated()
    if duplicated.any():
        issues.append(QualityIssue(ticker, "duplicate_dates", "error",
                                   f"{int(duplicated.sum())} duplicate trading dates.",
                                   _dates(frame.index[duplicated])[0], int(duplicated.sum())))
        frame = frame[~duplicated]

    body_hi = frame[["open", "close"]].max(axis=1)
    body_lo = frame[["open", "close"]].min(axis=1)
    inconsistent = (frame["high"] < body_hi * (1 - 1e-6)) | (frame["low"] > body_lo * (1 + 1e-6))
    if inconsistent.any():
        issues.append(QualityIssue(ticker, "ohlc_inconsistent", "warning",
                                   "High below open/close or low above open/close on "
                                   f"{int(inconsistent.sum())} sessions.",
                                   _dates(frame.index[inconsistent])[0], int(inconsistent.sum())))

    close_ret = frame["close"].pct_change()
    adj_ret = frame["adj_close"].pct_change()

    # Unadjusted split signature: a big jump in Close whose size matches a common split ratio.
    ratio = frame["close"] / frame["close"].shift(1)
    split_like = close_ret.abs() >= SPLIT_SIGNATURE_MOVE
    if split_like.any():
        near_ratio = ratio[split_like].apply(
            lambda r: any(abs(r * k - 1) < 0.03 for k in COMMON_SPLIT_RATIOS))
        suspects = near_ratio[near_ratio].index
        if len(suspects):
            day = suspects[0]
            issues.append(QualityIssue(
                ticker, "possible_unadjusted_split", "error",
                f"Close moved {close_ret.loc[day]:+.1%} on {day.date()}, matching a common split ratio. "
                "Re-download the history (Refresh) before relying on returns.",
                day.date(), len(suspects)))

    extreme = adj_ret.abs() >= EXTREME_MOVE
    if extreme.any():
        days = frame.index[extreme]
        moves = ", ".join(f"{d.date()} {adj_ret.loc[d]:+.1%}" for d in days[:4])
        issues.append(QualityIssue(ticker, "extreme_move", "warning",
                                   f"{int(extreme.sum())} daily move(s) of 25% or more ({moves}). "
                                   "Often genuine (earnings, crises) but worth verifying.",
                                   days[0].date(), int(extreme.sum())))

    factor = frame["adj_close"] / frame["close"]
    factor_jump = factor.pct_change().abs() >= ADJ_FACTOR_JUMP
    if factor_jump.any():
        day = frame.index[factor_jump][0]
        issues.append(QualityIssue(ticker, "adjustment_jump", "error",
                                   f"Adjustment factor (adj close / close) changed by more than 5% on "
                                   f"{day.date()}: mixed adjustment bases in stored history.",
                                   day.date(), int(factor_jump.sum())))

    gaps = pd.Series(frame.index, index=frame.index).diff().dt.days
    long_gaps = gaps > MAX_CALENDAR_GAP_DAYS
    if long_gaps.any():
        day = frame.index[long_gaps][0]
        issues.append(QualityIssue(ticker, "session_gap", "warning",
                                   f"{int(long_gaps.sum())} gap(s) longer than {MAX_CALENDAR_GAP_DAYS} calendar "
                                   f"days between sessions (largest {int(gaps.max())} days). Returns across a gap "
                                   "span several days.", day.date(), int(long_gaps.sum())))

    same = frame["close"].diff().eq(0)
    run_id = (~same).cumsum()
    run_lengths = same.groupby(run_id).sum()
    stale_runs = run_lengths[run_lengths >= STALE_RUN - 1]
    if not stale_runs.empty:
        issues.append(QualityIssue(ticker, "stale_prices", "warning",
                                   f"{len(stale_runs)} run(s) of {STALE_RUN}+ identical closes; the series may be "
                                   "stale or illiquid.", None, len(stale_runs)))

    if not is_index:
        zero_volume = frame["volume"] <= 0
        if zero_volume.any():
            issues.append(QualityIssue(ticker, "zero_volume", "info",
                                       f"{int(zero_volume.sum())} session(s) with zero reported volume.",
                                       _dates(frame.index[zero_volume])[0], int(zero_volume.sum())))
    return issues


def check_coverage(ticker: str, first: pd.Timestamp | None, last: pd.Timestamp | None, sessions: int,
                   start: date, end: date, as_of: date | None = None) -> list[QualityIssue]:
    """Checks on how well stored history covers the requested window and how fresh it is."""
    issues: list[QualityIssue] = []
    if first is None or pd.isna(first):
        return [QualityIssue(ticker, "no_data", "error", "No stored prices for this symbol.")]
    first_d, last_d = pd.Timestamp(first).date(), pd.Timestamp(last).date()
    if first_d > start + pd.Timedelta(days=7):
        issues.append(QualityIssue(ticker, "late_start", "info",
                                   f"History starts {first_d}, after the period start {start}. Comparisons "
                                   "use the common period.", first_d))
    reference = min(end, as_of or date.today())
    lag = int(np.busday_count(last_d, reference))
    if lag > 3:
        issues.append(QualityIssue(ticker, "stale_series", "warning",
                                   f"Last stored session is {last_d} ({lag} business days before {reference}). "
                                   "Use Refresh prices to update.", last_d))
    if sessions < 60:
        issues.append(QualityIssue(ticker, "short_history", "info",
                                   f"Only {sessions} sessions in range; rolling statistics will be unreliable."))
    return issues


def issues_frame(issues: list[QualityIssue]) -> pd.DataFrame:
    columns = ["ticker", "check", "severity", "detail", "first_date", "count"]
    if not issues:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame([i.as_dict() for i in issues])[columns]
    frame["_order"] = frame["severity"].map(SEVERITY_ORDER)
    return frame.sort_values(["_order", "ticker"]).drop(columns="_order").reset_index(drop=True)


def calendar_overlap(panel: pd.DataFrame) -> dict[str, float | int]:
    """How well the tickers' trading calendars line up within a wide price panel."""
    if panel.empty or panel.shape[1] < 2:
        return {"union_sessions": len(panel), "common_sessions": len(panel), "share_common": 1.0}
    present = panel.notna()
    common = int(present.all(axis=1).sum())
    union = int(present.any(axis=1).sum())
    return {"union_sessions": union, "common_sessions": common,
            "share_common": common / union if union else 1.0}
