import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fmip.analysis import eda
from fmip.exceptions import InsufficientDataError


def test_daily_returns_match_manual_calculation():
    prices = pd.Series([100.0, 110.0, 99.0])
    returns = eda.daily_returns(prices)
    assert returns.tolist() == pytest.approx([0.10, -0.10])


def test_cumulative_returns_start_at_zero_and_compound():
    prices = pd.Series([100.0, 120.0, 90.0])
    cum = eda.cumulative_returns(prices)
    assert cum.iloc[0] == 0
    assert cum.iloc[-1] == pytest.approx(-0.10)


def test_moving_average_uses_only_past_values():
    prices = pd.Series(np.arange(1.0, 31.0))
    ma = eda.moving_averages(prices, windows=(5,))
    assert ma["MA_5"].iloc[4] == pytest.approx(3.0)  # mean of 1..5
    assert np.isnan(ma["MA_5"].iloc[3])


def test_annualised_volatility_scales_with_sqrt_252():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.0])
    expected = returns.std(ddof=1) * np.sqrt(252)
    assert eda.annualised_volatility(returns) == pytest.approx(expected)


def test_max_drawdown_known_example():
    index = pd.bdate_range("2021-01-01", periods=5)
    prices = pd.Series([100.0, 120.0, 90.0, 95.0, 130.0], index=index)
    result = eda.max_drawdown(prices)
    assert result["max_drawdown"] == pytest.approx(90.0 / 120.0 - 1.0)
    assert result["peak_date"] == index[1]
    assert result["trough_date"] == index[2]
    assert result["recovery_date"] == index[4]


def test_max_drawdown_is_never_positive(prices):
    assert eda.max_drawdown(prices)["max_drawdown"] <= 0


def test_rolling_returns_window():
    prices = pd.Series([100.0, 101.0, 102.0, 103.0])
    rr = eda.rolling_returns(prices, 2)
    assert rr.iloc[2] == pytest.approx(102.0 / 100.0 - 1)


def test_distribution_stats_match_scipy(prices):
    returns = eda.daily_returns(prices)
    stats_out = eda.distribution_stats(returns)
    assert stats_out["skewness"] == pytest.approx(stats.skew(returns, bias=False))
    assert stats_out["excess_kurtosis"] == pytest.approx(stats.kurtosis(returns, bias=False))


def test_insufficient_data_raises():
    with pytest.raises(InsufficientDataError):
        eda.distribution_stats(pd.Series([0.01, 0.02]))
