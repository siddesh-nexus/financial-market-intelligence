"""Cleaning and validation of raw OHLCV price data.

Rules applied (documented in docs/METHODOLOGY.md):
1. Required columns must exist (Open, High, Low, Close, Volume).
2. Index is converted to timezone-naive calendar dates, sorted, and de-duplicated (last wins).
3. Rows with non-positive or non-finite prices, or High < Low, are dropped. Since v0.2 no price
   is forward-filled (MAX_PRICE_FFILL = 0): a missing bar stays missing, and the next return
   spans the gap, so no observation is fabricated. The fill mechanism is kept but disabled.
4. Negative or missing volume is set to 0 and counted in the report (the volume column is NOT NULL);
   zero-volume days are surfaced by the data-quality checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from fmip.exceptions import DataValidationError
from fmip.logging_config import get_logger

log = get_logger(__name__)

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
PRICE_COLUMNS = ("Open", "High", "Low", "Close")
MAX_PRICE_FFILL = 0  # v0.1 used 2; filling invents prices, so it is now off


@dataclass(frozen=True)
class CleaningReport:
    ticker: str
    rows_in: int
    rows_out: int
    duplicates_removed: int
    prices_forward_filled: int
    invalid_rows_dropped: int
    missing_volume_set_to_zero: int

    def as_dict(self) -> dict:
        return asdict(self)


def clean_ohlcv(raw: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, CleaningReport]:
    """Return a cleaned OHLCV frame indexed by date plus a report of what was changed.

    Raises DataValidationError if required columns are missing or the frame is empty.
    """
    if raw is None or raw.empty:
        raise DataValidationError(f"{ticker}: no rows to validate")

    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise DataValidationError(f"{ticker}: missing required columns {missing}")

    rows_in = len(raw)
    df = raw.copy()

    index = pd.to_datetime(df.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    df.index = index.normalize()
    df.index.name = "trade_date"

    df = df.sort_index()
    duplicate_mask = df.index.duplicated(keep="last")
    duplicates_removed = int(duplicate_mask.sum())
    df = df[~duplicate_mask]

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if "Adj Close" in df.columns:
        df["Adj Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")

    invalid_price = ~np.isfinite(df[list(PRICE_COLUMNS)]).all(axis=1) | (df[list(PRICE_COLUMNS)] <= 0).any(axis=1)
    invalid_range = df["High"] < df["Low"]
    invalid_volume = df["Volume"] < 0

    # Only prices are forward-filled. Bad rows are masked first, then short gaps are filled.
    price_block = df[list(PRICE_COLUMNS)].mask(invalid_price | invalid_range)
    masked_cells = price_block.isna()
    if MAX_PRICE_FFILL > 0:
        price_block = price_block.ffill(limit=MAX_PRICE_FFILL)
    forward_filled = int((masked_cells & price_block.notna()).sum().sum())
    df[list(PRICE_COLUMNS)] = price_block

    df.loc[invalid_volume, "Volume"] = np.nan
    missing_volume = int(df["Volume"].isna().sum())
    df["Volume"] = df["Volume"].fillna(0.0)

    still_missing_price = df[list(PRICE_COLUMNS)].isna().any(axis=1)
    invalid_dropped = int((still_missing_price | invalid_range).sum())
    df = df[~(still_missing_price | invalid_range)]

    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    else:
        df["Adj Close"] = df["Adj Close"].fillna(df["Close"])

    cleaned = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].astype(
        {"Open": float, "High": float, "Low": float, "Close": float, "Adj Close": float, "Volume": float}
    )

    report = CleaningReport(
        ticker=ticker,
        rows_in=rows_in,
        rows_out=len(cleaned),
        duplicates_removed=duplicates_removed,
        prices_forward_filled=forward_filled,
        invalid_rows_dropped=invalid_dropped,
        missing_volume_set_to_zero=missing_volume,
    )
    log.debug("Cleaned %s: %s", ticker, report)

    if cleaned.empty:
        raise DataValidationError(f"{ticker}: no valid rows remain after cleaning")
    return cleaned, report
