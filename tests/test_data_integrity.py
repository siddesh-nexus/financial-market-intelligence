"""Data-quality checks, corporate-action refresh, incomplete sessions, metadata and watchlists."""

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect, text

from fmip.data import quality, service
from fmip.data.repository import MarketRepository, create_schema, migrate_schema
from fmip.data.symbols import local_matches, looks_like_ticker
from fmip.exceptions import UnknownTickerError
from fmip.universe import exchange_of, in_market, normalise_sector
from tests.conftest import make_ohlcv


@pytest.fixture
def repo():
    engine = create_engine("sqlite:///:memory:", future=True)
    create_schema(engine)
    return MarketRepository(engine)


def _bars(n: int = 60, seed: int = 0) -> pd.DataFrame:
    frame = make_ohlcv(n=n, seed=seed)
    frame.columns = ["open", "high", "low", "close", "adj_close", "volume"]
    return frame


# ---- quality checks -----------------------------------------------------------------------------
def test_clean_series_has_no_errors():
    issues = quality.check_bars("X", _bars())
    assert not [i for i in issues if i.severity == "error"]


def test_unadjusted_split_is_detected():
    bars = _bars()
    bars.iloc[30:, :5] = bars.iloc[30:, :5] / 4  # a 4-for-1 split that was not back-adjusted
    checks = {i.check for i in quality.check_bars("X", bars)}
    assert "possible_unadjusted_split" in checks


def test_extreme_move_is_flagged_for_verification_not_removed():
    bars = _bars()
    bars.iloc[40:, :5] = bars.iloc[40:, :5] * 1.3
    issues = quality.check_bars("X", bars)
    extreme = [i for i in issues if i.check == "extreme_move"]
    assert extreme and extreme[0].severity == "warning"
    assert len(bars) == 60  # nothing was altered


def test_adjustment_factor_jump_is_an_error():
    bars = _bars()
    bars.loc[bars.index[35]:, "adj_close"] *= 0.9  # mixed adjustment bases
    checks = {i.check: i.severity for i in quality.check_bars("X", bars)}
    assert checks.get("adjustment_jump") == "error"


def test_stale_run_gap_and_duplicates_are_reported():
    bars = _bars()
    bars.iloc[10:16, bars.columns.get_loc("close")] = bars["close"].iloc[10]
    bars = bars.drop(bars.index[40:48])                      # 8 missing sessions -> calendar gap
    bars = pd.concat([bars, bars.iloc[[5]]]).sort_index()    # duplicate date
    checks = {i.check for i in quality.check_bars("X", bars)}
    assert {"stale_prices", "session_gap", "duplicate_dates"} <= checks


def test_coverage_flags_stale_and_late_series():
    issues = quality.check_coverage("X", pd.Timestamp("2024-03-01"), pd.Timestamp("2024-05-01"), 45,
                                    date(2024, 1, 1), date(2024, 6, 28), as_of=date(2024, 6, 28))
    checks = {i.check for i in issues}
    assert {"late_start", "stale_series", "short_history"} <= checks


def test_calendar_overlap():
    index = pd.bdate_range("2024-01-01", periods=10)
    panel = pd.DataFrame({"A": np.ones(10), "B": [1.0] * 8 + [np.nan] * 2}, index=index)
    assert quality.calendar_overlap(panel)["share_common"] == pytest.approx(0.8)


# ---- incomplete sessions --------------------------------------------------------------------------
def _tz_frame(day_count: int, last_day: str, tz: str) -> pd.DataFrame:
    index = pd.bdate_range(end=last_day, periods=day_count).tz_localize(tz)
    frame = make_ohlcv(n=day_count)
    frame.index = index
    return frame


def test_partial_bar_dropped_before_exchange_close():
    raw = _tz_frame(5, "2026-11-23", "America/New_York")
    during = datetime(2026, 11, 23, 18, 0, tzinfo=timezone.utc)     # 13:00 New York
    after = datetime(2026, 11, 23, 22, 0, tzinfo=timezone.utc)      # 17:00 New York
    kept_during, dropped_during = service.drop_incomplete_session(raw, during)
    kept_after, dropped_after = service.drop_incomplete_session(raw, after)
    assert dropped_during == 1 and len(kept_during) == 4
    assert dropped_after == 0 and len(kept_after) == 5


def test_partial_bar_uses_exchange_time_zone():
    raw = _tz_frame(3, "2026-11-23", "Asia/Kolkata")
    # 10:30 UTC = 16:00 IST: NSE closed at 15:30, settled by 16:00.
    _, dropped = service.drop_incomplete_session(raw, datetime(2026, 11, 23, 10, 30, tzinfo=timezone.utc))
    assert dropped == 0
    _, dropped = service.drop_incomplete_session(raw, datetime(2026, 11, 23, 8, 0, tzinfo=timezone.utc))
    assert dropped == 1


# ---- corporate actions ------------------------------------------------------------------------------
def test_adjustment_mismatch_detects_back_adjusted_history():
    cleaned = make_ohlcv(n=10)
    stored = pd.DataFrame({"trade_date": cleaned.index, "close": cleaned["Close"].values,
                           "adj_close": cleaned["Adj Close"].values})
    assert service.adjustment_mismatch(stored, cleaned) is None
    split_adjusted = cleaned.copy()
    split_adjusted[["Close", "Adj Close"]] /= 2
    assert "close changed" in service.adjustment_mismatch(stored, split_adjusted)


def test_corporate_actions_after_date():
    raw = make_ohlcv(n=5)
    raw["Stock Splits"] = [0, 0, 4.0, 0, 0]
    raw["Dividends"] = [0, 0.25, 0, 0, 0]
    notes = service.corporate_actions(raw, after=raw.index[1].date())
    assert notes == [f"split 4 on {raw.index[2].date()}"]


def test_split_after_initial_load_triggers_full_refresh(repo):
    """A 2-for-1 split after the first load: Yahoo halves all earlier prices. The incremental run
    must re-download everything so stored history has one consistent adjustment basis."""
    history = make_ohlcv(n=80, seed=7)
    first_part = history.iloc[:60]
    calls = []

    def first_fetcher(ticker, start, end):
        return first_part

    service.ingest_tickers(repo, ["SPLT"], "2020-01-01", "2020-03-25", fetcher=first_fetcher)

    adjusted = history.copy()
    adjusted[["Open", "High", "Low", "Close", "Adj Close"]] /= 2     # provider's re-adjusted view

    def second_fetcher(ticker, start, end):
        calls.append(start)
        window = adjusted[adjusted.index >= pd.Timestamp(start)]
        return window

    results = service.ingest_tickers(repo, ["SPLT"], "2020-01-01", "2020-06-01", fetcher=second_fetcher)
    assert results[0].full_refresh is True
    assert len(calls) == 2  # incremental window, then the full re-download
    stored = repo.load_prices(["SPLT"]).set_index("trade_date")["close"]
    np.testing.assert_allclose(stored.values, adjusted["Close"].values)
    issues = quality.check_bars("SPLT", repo.load_prices(["SPLT"]).set_index("trade_date"))
    assert not [i for i in issues if i.check in ("possible_unadjusted_split", "adjustment_jump")]


def test_no_refresh_when_overlap_matches(repo):
    data = make_ohlcv(n=60, seed=3)
    service.ingest_tickers(repo, ["SAME"], "2020-01-01", "2020-03-01", fetcher=lambda t, s, e: data.iloc[:40])
    results = service.ingest_tickers(repo, ["SAME"], "2020-01-01", "2020-06-01",
                                     fetcher=lambda t, s, e: data[data.index >= pd.Timestamp(s)])
    assert results[0].full_refresh is False


# ---- metadata, backfill memory, watchlists, migration ----------------------------------------------
def test_failed_ticker_leaves_no_stock_row(repo):
    def failing(ticker, start, end):
        raise UnknownTickerError("nope")

    results = service.ingest_tickers(repo, ["TYPO"], "2020-01-01", "2020-02-01", fetcher=failing)
    assert results[0].status == "failed"
    assert repo.load_stocks(["TYPO"]).empty


def test_metadata_from_search_is_stored_and_not_overwritten(repo):
    # A symbol with no built-in name, so the stored name can only come from search metadata.
    meta = pd.DataFrame([{"ticker": "WIPRO.NS", "company_name": "Wipro Limited", "sector": "Technology",
                          "exchange": "NSE"}])
    service.ingest_tickers(repo, ["WIPRO.NS"], "2020-01-01", "2020-04-01",
                           fetcher=lambda t, s, e: make_ohlcv(n=40), metadata=meta)
    # A later ingest without metadata must not erase the stored name (COALESCE upsert).
    service.ingest_tickers(repo, ["WIPRO.NS"], "2020-01-01", "2020-06-01",
                           fetcher=lambda t, s, e: make_ohlcv(n=40))
    row = repo.load_stocks(["WIPRO.NS"]).iloc[0]
    assert row["company_name"] == "Wipro Limited"
    assert row["sector"] == "Technology"


def test_backfill_not_requested_twice_for_late_listing(repo):
    calls = []

    def late_listing(ticker, start, end):
        calls.append(start)
        return make_ohlcv(n=30, start="2021-06-01")  # listing begins in June 2021 whatever is asked

    service.ingest_tickers(repo, ["IPO"], "2020-01-01", "2021-07-10", fetcher=late_listing, incremental=False)
    calls.clear()
    service.ingest_tickers(repo, ["IPO"], "2020-01-01", "2021-07-10", fetcher=late_listing, incremental=False)
    assert calls == []
    assert repo.history_requested_from("IPO") == date(2020, 1, 1)


def test_watchlist_round_trip(repo):
    repo.save_watchlist("Tech", ["AAPL", "MSFT", "AAPL"], "^NDX")
    repo.save_watchlist("Tech", ["NVDA"], "^GSPC")  # same name updates
    lists = repo.list_watchlists()
    assert lists.loc[0, "tickers"] == ["NVDA"] and lists.loc[0, "benchmark"] == "^GSPC"
    repo.delete_watchlist("Tech")
    assert repo.list_watchlists().empty
    with pytest.raises(ValueError):
        repo.save_watchlist("  ", ["AAPL"])


def test_migrate_schema_adds_new_columns_to_v01_database():
    engine = create_engine("sqlite:///:memory:", future=True)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE stocks (ticker VARCHAR(20) PRIMARY KEY, company_name VARCHAR(200), "
                          "sector_id INTEGER)"))
        conn.execute(text("INSERT INTO stocks (ticker, company_name) VALUES ('AAPL', 'Apple')"))
    added = migrate_schema(engine)
    assert "stocks.exchange" in added and "stocks.history_requested_from" in added
    columns = {c["name"] for c in inspect(engine).get_columns("stocks")}
    assert {"exchange", "currency", "quote_type", "history_requested_from"} <= columns
    with engine.connect() as conn:
        assert conn.execute(text("SELECT company_name FROM stocks")).scalar() == "Apple"


# ---- symbols and universe helpers ------------------------------------------------------------------
@pytest.mark.parametrize("text_, ok", [("AAPL", True), ("tcs.ns", True), ("^NSEI", True), ("BRK-B", True),
                                       ("Apple Inc", False), ("", False), ("$$$", False)])
def test_looks_like_ticker(text_, ok):
    assert looks_like_ticker(text_) is ok


def test_local_matches_search_by_name():
    result = local_matches("tata")
    assert "TCS.NS" in set(result["ticker"])


def test_universe_helpers():
    assert exchange_of("RELIANCE.NS") == "NSE"
    assert exchange_of("AAPL") == "US"
    assert exchange_of("^GSPC") == "Index"
    assert in_market("TCS.NS", "India") and not in_market("AAPL", "India")
    assert in_market("AAPL", "United States") and not in_market("TCS.BO", "United States")
    assert normalise_sector("Financial Services") == "Financials"
    assert normalise_sector("Consumer Cyclical") == "Consumer Discretionary"
