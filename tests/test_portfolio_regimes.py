"""Portfolio back-calculation, diversification and regime classification (synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from fmip.analysis import portfolio, regimes
from fmip.exceptions import InsufficientDataError
from tests.conftest import make_prices


@pytest.fixture
def two_assets() -> pd.DataFrame:
    return pd.DataFrame({"A": make_prices(n=300, seed=1), "B": make_prices(n=300, seed=2, vol=0.02)})


def test_normalise_weights_scales_and_rejects_shorts():
    weights = portfolio.normalise_weights({"A": 30, "B": 10})
    assert weights.to_dict() == pytest.approx({"A": 0.75, "B": 0.25})
    with pytest.raises(ValueError):
        portfolio.normalise_weights({"A": 1, "B": -1})
    with pytest.raises(ValueError):
        portfolio.normalise_weights({"A": 0, "B": 0})


def test_daily_rebalanced_return_is_weighted_sum(two_assets):
    w = pd.Series({"A": 0.6, "B": 0.4})
    result = portfolio.simulate(two_assets, w, rebalance="daily")
    asset = two_assets.pct_change().iloc[1:]
    expected = asset["A"] * 0.6 + asset["B"] * 0.4
    assert np.allclose(result.returns.values, expected.values)
    assert result.wealth.iloc[0] == 1.0


def test_buy_and_hold_matches_value_weighted_growth(two_assets):
    w = pd.Series({"A": 0.5, "B": 0.5})
    result = portfolio.simulate(two_assets, w, rebalance="none")
    growth = 0.5 * two_assets["A"] / two_assets["A"].iloc[0] + 0.5 * two_assets["B"] / two_assets["B"].iloc[0]
    assert result.wealth.iloc[-1] == pytest.approx(growth.iloc[-1])
    # Weights drift with prices when nothing is rebalanced.
    assert result.end_weights["A"] == pytest.approx(0.5 * two_assets["A"].iloc[-1] / two_assets["A"].iloc[0]
                                                    / growth.iloc[-1])


def test_monthly_rebalance_resets_weights(two_assets):
    w = pd.Series({"A": 0.5, "B": 0.5})
    monthly = portfolio.simulate(two_assets, w, rebalance="monthly")
    hold = portfolio.simulate(two_assets, w, rebalance="none")
    assert not np.isclose(monthly.wealth.iloc[-1], hold.wealth.iloc[-1])


def test_portfolio_uses_common_period_and_carries_closed_markets():
    index = pd.bdate_range("2024-01-01", periods=60)
    a = pd.Series(np.linspace(100, 120, 60), index=index)
    b = pd.Series(np.linspace(50, 55, 60), index=index)
    b.iloc[:10] = np.nan    # listed later
    b.iloc[30] = np.nan     # market holiday
    result = portfolio.simulate(pd.DataFrame({"A": a, "B": b}), pd.Series({"A": 0.5, "B": 0.5}), "daily")
    assert result.start == index[10]
    assert result.carried_holding_days == 1
    assert result.returns.notna().all()


def test_diversification_metrics(two_assets):
    w = pd.Series({"A": 0.5, "B": 0.5})
    div = portfolio.diversification(two_assets, w)
    assert div["risk_contribution"].sum() == pytest.approx(1.0)
    assert div["diversification_ratio"] >= 1.0 - 1e-9
    assert div["effective_holdings"] == pytest.approx(2.0)
    # The more volatile asset contributes more risk at equal weight.
    assert div["risk_contribution"]["B"] > div["risk_contribution"]["A"]


def test_benchmark_comparison_against_itself():
    prices = make_prices(n=200, seed=5)
    rets = prices.pct_change().dropna()
    comparison = portfolio.benchmark_comparison(rets, prices)
    assert comparison["beta"] == pytest.approx(1.0)
    assert comparison["tracking_error"] == pytest.approx(0.0, abs=1e-12)
    assert comparison["excess_annualised_return"] == pytest.approx(0.0, abs=1e-12)


def test_simulate_needs_enough_common_history():
    short = pd.DataFrame({"A": make_prices(n=10), "B": make_prices(n=10, seed=3)})
    with pytest.raises(InsufficientDataError):
        portfolio.simulate(short, pd.Series({"A": 0.5, "B": 0.5}))


def _trend(n: int, start: float, end: float, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    base = np.linspace(start, end, n) * (1 + rng.normal(0, 0.002, n))
    return pd.Series(base, index=pd.bdate_range("2018-01-01", periods=n))


def test_rule_based_trend_regimes():
    up = regimes.rule_based_regimes(_trend(400, 100, 200))
    down = regimes.rule_based_regimes(_trend(400, 200, 100))
    assert up["trend_regime"].dropna().iloc[-1] == "Bull"
    assert down["trend_regime"].dropna().iloc[-1] == "Bear"
    assert up["trend_regime"].iloc[:150].isna().all()  # no label before the 200-day MA exists


def test_volatility_regime_has_no_lookahead():
    calm = make_prices(n=600, vol=0.005, seed=1)
    stressed = calm.copy()
    stressed.iloc[500:] = stressed.iloc[500:] * (1 + np.random.default_rng(9).normal(0, 0.05, 100)).cumprod()
    a = regimes.rule_based_regimes(calm)
    b = regimes.rule_based_regimes(stressed)
    # Changing the future must not change labels of earlier sessions.
    pd.testing.assert_series_equal(a["vol_regime"].iloc[:490], b["vol_regime"].iloc[:490])
    assert (b["vol_regime"].iloc[520:] == "High").mean() > 0.5


def test_gmm_states_are_ordered_by_volatility():
    rng = np.random.default_rng(0)
    calm = rng.normal(0.0005, 0.005, 400)
    wild = rng.normal(-0.001, 0.03, 200)
    prices = pd.Series(100 * np.cumprod(1 + np.concatenate([calm, wild, calm])),
                       index=pd.bdate_range("2015-01-01", periods=1000))
    frame, params = regimes.gmm_regimes(prices, n_states=2)
    assert list(params.index) == ["Calm", "Stressed"]
    assert params.loc["Stressed", "mean_volatility"] > params.loc["Calm", "mean_volatility"]
    assert frame["probability"].between(0, 1).all()
    assert (frame["state"].iloc[450:550] == "Stressed").mean() > 0.7


def test_regime_summary_shares_sum_to_one():
    labels = pd.Series(["Bull"] * 5 + ["Bear"] * 3 + ["Bull"] * 2)
    rets = pd.Series(np.linspace(-0.01, 0.01, 10))
    summary = regimes.regime_summary(labels, rets)
    assert summary["share_of_time"].sum() == pytest.approx(1.0)
    assert summary.loc["Bull", "episodes"] == 2
    assert summary.loc["Bull", "avg_episode_sessions"] == pytest.approx(3.5)
