from datetime import date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine

from fmip.data import service
from fmip.data.repository import MarketRepository, create_schema
from fmip.data.validation import clean_ohlcv
from fmip.exceptions import DataFetchError, DataValidationError, UnknownTickerError
from fmip.data.fetcher import fetch_history
from tests.conftest import make_ohlcv


def _raw_frame(n: int = 10) -> pd.DataFrame:
    frame = make_ohlcv(n=n)
    frame.index = frame.index.tz_localize("America/New_York")
    return frame


def test_clean_rejects_missing_columns():
    with pytest.raises(DataValidationError):
        clean_ohlcv(pd.DataFrame({"Close": [1.0]}, index=pd.bdate_range("2020-01-01", periods=1)), "X")


def test_clean_rejects_empty_frame():
    with pytest.raises(DataValidationError):
        clean_ohlcv(pd.DataFrame(), "X")


def test_clean_removes_duplicates_and_timezones():
    raw = _raw_frame(10)
    duplicated = pd.concat([raw, raw.iloc[[3]]])
    cleaned, report = clean_ohlcv(duplicated, "X")
    assert cleaned.index.is_unique
    assert cleaned.index.tz is None
    assert report.duplicates_removed == 1


def test_clean_drops_invalid_rows_and_never_invents_prices():
    raw = make_ohlcv(n=10)
    raw.iloc[2, raw.columns.get_loc("Close")] = np.nan       # missing price -> row dropped, not filled
    raw.iloc[5, raw.columns.get_loc("High")] = 1.0            # High < Low -> invalid
    raw.iloc[5, raw.columns.get_loc("Low")] = 2.0
    raw.iloc[7, raw.columns.get_loc("Volume")] = -5.0         # negative volume -> zero
    cleaned, report = clean_ohlcv(raw, "X")
    assert report.prices_forward_filled == 0
    assert report.invalid_rows_dropped == 2
    assert raw.index[2] not in cleaned.index and raw.index[5] not in cleaned.index
    assert (cleaned["Volume"] >= 0).all()
    assert cleaned.notna().all().all()


def test_fetch_history_raises_for_unknown_ticker(monkeypatch):
    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            return pd.DataFrame()

    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", FakeTicker)
    with pytest.raises(UnknownTickerError):
        fetch_history("NOPE", "2020-01-01", "2020-02-01", max_attempts=1)


def test_fetch_history_retries_then_fails(monkeypatch):
    class BrokenTicker:
        def __init__(self, symbol):
            pass

        def history(self, **kwargs):
            raise ConnectionError("network down")

    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", BrokenTicker)
    monkeypatch.setattr(service, "fetch_history", service.fetch_history)
    with pytest.raises(DataFetchError):
        fetch_history("AAPL", "2020-01-01", "2020-02-01", max_attempts=2, backoff_seconds=0)


@pytest.fixture
def repo():
    engine = create_engine("sqlite:///:memory:", future=True)
    create_schema(engine)
    return MarketRepository(engine)


def test_upsert_prices_is_idempotent_and_loadable(repo):
    cleaned, _ = clean_ohlcv(make_ohlcv(n=30), "AAPL")
    repo.upsert_prices("AAPL", cleaned)
    repo.upsert_prices("AAPL", cleaned)  # second write must not duplicate rows
    loaded = repo.load_prices(["AAPL"])
    assert len(loaded) == len(cleaned)
    assert repo.latest_price_date("AAPL") == cleaned.index[-1].date()


def test_load_price_panel_pivots_by_ticker(repo):
    for ticker, seed in [("AAA", 1), ("BBB", 2)]:
        cleaned, _ = clean_ohlcv(make_ohlcv(n=40, seed=seed), ticker)
        repo.upsert_prices(ticker, cleaned)
    panel = service.load_price_panel(repo, ["AAA", "BBB"])
    assert list(panel.columns) == ["AAA", "BBB"]
    assert len(panel) == 40


def test_plan_fetch_windows_covers_backfill_and_forward_gaps():
    stored_start, stored_end = date(2020, 6, 1), date(2020, 12, 1)
    windows = service.plan_fetch_windows((stored_start, stored_end), date(2020, 1, 1), date(2021, 6, 1))
    reasons = [reason for _, _, reason in windows]
    assert reasons == ["backfill", "incremental"]
    assert windows[0][:2] == (date(2020, 1, 1), stored_start)
    assert windows[1][1] == date(2021, 6, 1)


def test_plan_fetch_windows_skips_when_up_to_date():
    windows = service.plan_fetch_windows((date(2018, 1, 2), date(2026, 1, 1)), date(2018, 1, 1), date(2026, 1, 1))
    assert windows == []


def test_backfill_fetches_history_older_than_stored_data(repo):
    seen = []

    def recording_fetcher(ticker, start, end):
        seen.append((start, end))
        return make_ohlcv(n=30, seed=2, start=str(start))

    # First run stores a later window only.
    service.ingest_tickers(repo, ["LATE"], "2020-06-01", "2020-07-15", fetcher=recording_fetcher)
    seen.clear()
    # Second run asks for earlier history: it must backfill, not skip.
    results = service.ingest_tickers(repo, ["LATE"], "2020-01-01", "2020-07-15", fetcher=recording_fetcher)
    assert results[0].status == "ok"
    assert seen and seen[0][0] == date(2020, 1, 1)


def test_ingest_is_incremental_and_isolates_failures(repo):
    calls = []

    def fake_fetcher(ticker, start, end):
        calls.append((ticker, start))
        if ticker == "BAD":
            raise UnknownTickerError("no data")
        return make_ohlcv(n=60, seed=4)

    results = service.ingest_tickers(repo, ["GOOD", "BAD"], "2020-01-01", "2020-06-01", fetcher=fake_fetcher)
    statuses = {r.ticker: r.status for r in results}
    assert statuses == {"GOOD": "ok", "BAD": "failed"}

    calls.clear()
    service.ingest_tickers(repo, ["GOOD"], "2020-01-01", "2020-06-01", fetcher=fake_fetcher)
    # Second run should start from the stored latest date minus the overlap, not from scratch.
    assert calls and calls[0][1] > date(2020, 1, 1)
