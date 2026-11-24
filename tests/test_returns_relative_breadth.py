"""Calendar-safe returns, rebasing, relative strength, beta and breadth (synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from fmip.analysis import breadth, relative, returns
from fmip.exceptions import InsufficientDataError
from tests.conftest import make_prices


def _two_calendars() -> pd.DataFrame:
    """US-like and India-like series with different holidays on a shared union index."""
    index = pd.bdate_range("2024-01-01", periods=10)
    us = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], index=index, dtype=float)
    india = pd.Series([50, 51, 52, 53, 54, 55, 56, 57, 58, 59], index=index, dtype=float)
    us.iloc[3] = np.nan       # US holiday
    india.iloc[6] = np.nan    # India holiday
    return pd.DataFrame({"US": us, "IN": india})


def test_asset_returns_use_each_assets_own_calendar():
    rets = returns.asset_returns(_two_calendars())
    # The first US session after its holiday measures from the previous available close (102 -> 104).
    assert rets["US"].iloc[4] == pytest.approx(104 / 102 - 1)
    assert np.isnan(rets["US"].iloc[3])
    # A naive union-calendar pct_change would lose this return entirely.
    naive = _two_calendars().pct_change(fill_method=None)
    assert np.isnan(naive["US"].iloc[4])
    assert rets["IN"].iloc[7] == pytest.approx(57 / 55 - 1)


def test_rebase_common_start_and_late_starters():
    index = pd.bdate_range("2024-01-01", periods=6)
    panel = pd.DataFrame({"OLD": [10, 11, 12, 13, 14, 15], "NEW": [np.nan, np.nan, 20, 22, 24, 26]},
                         index=index, dtype=float)
    common = returns.rebase(panel, "common")
    assert common.common_start == index[2]
    assert common.values.iloc[0].tolist() == [1.0, 1.0]
    assert common.values["OLD"].iloc[-1] == pytest.approx(15 / 12)
    assert "NEW" in common.late_starters
    own = returns.rebase(panel, "own")
    assert own.values["OLD"].iloc[-1] == pytest.approx(1.5)
    assert own.values["NEW"].iloc[-1] == pytest.approx(1.3)


def test_rebase_reports_series_without_data():
    index = pd.bdate_range("2024-01-01", periods=3)
    panel = pd.DataFrame({"A": [1.0, 2.0, 3.0], "EMPTY": [np.nan] * 3}, index=index)
    result = returns.rebase(panel)
    assert result.dropped == ["EMPTY"]
    assert list(result.values.columns) == ["A"]


def test_trailing_return_counts_sessions():
    series = pd.Series(np.arange(1, 31, dtype=float), index=pd.bdate_range("2024-01-01", periods=30))
    assert returns.trailing_return(series, 5) == pytest.approx(30 / 25 - 1)
    assert np.isnan(returns.trailing_return(series, 40))


def test_relative_strength_line_is_ratio_of_rebased_series():
    index = pd.bdate_range("2024-01-01", periods=4)
    stock = pd.Series([10, 11, 12, 13], index=index, dtype=float)
    bench = pd.Series([100, 100, 110, 110], index=index, dtype=float)
    line = relative.relative_strength_line(stock, bench)
    assert line.iloc[0] == pytest.approx(1.0)
    assert line.iloc[-1] == pytest.approx(1.3 / 1.1)


def test_excess_returns_match_hand_calculation():
    index = pd.bdate_range("2024-01-01", periods=30)
    stock = pd.Series(np.linspace(100, 130, 30), index=index)
    bench = pd.Series(np.linspace(100, 110, 30), index=index)
    table = relative.excess_returns(stock.to_frame("S"), bench, {"5D": 5, "1Y": 252})
    expected = (stock.iloc[-1] / stock.iloc[-6] - 1) - (bench.iloc[-1] / bench.iloc[-6] - 1)
    assert table.loc["S", "5D"] == pytest.approx(expected)
    assert np.isnan(table.loc["S", "1Y"])  # not enough history: reported as missing, not invented


def test_beta_of_leveraged_series_is_two():
    bench = make_prices(seed=1).pct_change().dropna()
    stock = 2 * bench
    assert relative.beta(stock, bench) == pytest.approx(2.0)
    rolling = relative.rolling_beta(stock, bench, window=60).dropna()
    assert np.allclose(rolling.values, 2.0)


def test_beta_requires_overlap():
    bench = make_prices(n=10).pct_change().dropna()
    with pytest.raises(InsufficientDataError):
        relative.beta(bench, bench)


def test_relative_strength_summary_status():
    index = pd.bdate_range("2024-01-01", periods=120)
    bench = pd.Series(100.0, index=index)
    winner = pd.Series(np.linspace(100, 150, 120), index=index)
    loser = pd.Series(np.linspace(100, 70, 120), index=index)
    summary = relative.relative_strength_summary(pd.DataFrame({"W": winner, "L": loser}), bench)
    assert summary.loc["W", "status"] == "Outperforming, improving"
    assert summary.loc["L", "status"] == "Underperforming, weakening"
    assert summary.loc["W", "geometric_excess"] == pytest.approx(0.5)


def test_advance_decline_counts():
    index = pd.bdate_range("2024-01-01", periods=3)
    panel = pd.DataFrame({"A": [10, 11, 12], "B": [10, 9, 9], "C": [10, 10.5, 10]}, index=index, dtype=float)
    ad = breadth.advance_decline(panel)
    assert ad.loc[index[1], ["advancers", "decliners", "unchanged"]].tolist() == [2, 1, 0]
    assert ad.loc[index[2], ["advancers", "decliners", "unchanged"]].tolist() == [1, 1, 1]
    assert ad["ad_line"].iloc[-1] == 1


def test_share_above_ma_excludes_stocks_without_enough_history():
    index = pd.bdate_range("2024-01-01", periods=30)
    rising = pd.Series(np.arange(30, dtype=float) + 1, index=index)
    young = pd.Series([np.nan] * 25 + [5.0, 6.0, 7.0, 8.0, 9.0], index=index)
    share = breadth.share_above_ma(pd.DataFrame({"R": rising, "Y": young}), window=10)
    # Only R is eligible; it is above its rising MA, so the share is 100%, not 50%.
    assert share.iloc[-1] == pytest.approx(1.0)
    assert np.isnan(share.iloc[5])  # nobody eligible yet
