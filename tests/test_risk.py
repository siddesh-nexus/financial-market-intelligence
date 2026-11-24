import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fmip.analysis import risk
from fmip.exceptions import InsufficientDataError


@pytest.fixture
def returns():
    rng = np.random.default_rng(7)
    return pd.Series(rng.normal(0.0004, 0.012, 1000))


def test_sharpe_ratio_matches_definition(returns):
    rf = 0.04
    excess = returns - rf / 252
    expected = excess.mean() / excess.std(ddof=1) * np.sqrt(252)
    assert risk.sharpe_ratio(returns, rf) == pytest.approx(expected)


def test_sortino_uses_downside_deviation(returns):
    rf = 0.0
    downside = np.minimum(returns.values, 0.0)
    expected = returns.mean() / np.sqrt(np.mean(downside ** 2)) * np.sqrt(252)
    assert risk.sortino_ratio(returns, rf) == pytest.approx(expected)


def test_historical_var_is_empirical_quantile(returns):
    expected = -np.quantile(returns.values, 0.05)
    assert risk.value_at_risk(returns, 0.95, "historical") == pytest.approx(expected)


def test_parametric_var_matches_normal_formula(returns):
    z = stats.norm.ppf(0.05)
    expected = -(returns.mean() + z * returns.std(ddof=1))
    assert risk.value_at_risk(returns, 0.95, "parametric") == pytest.approx(expected)


def test_cvar_is_at_least_var(returns):
    var = risk.value_at_risk(returns, 0.95)
    cvar = risk.conditional_var(returns, 0.95)
    assert cvar >= var


def test_cvar_equals_mean_of_tail(returns):
    cutoff = np.quantile(returns.values, 0.05)
    expected = -returns[returns <= cutoff].mean()
    assert risk.conditional_var(returns, 0.95) == pytest.approx(expected)


def test_invalid_confidence_rejected(returns):
    with pytest.raises(ValueError):
        risk.value_at_risk(returns, confidence=1.2)


def test_too_few_observations_rejected():
    with pytest.raises(InsufficientDataError):
        risk.sharpe_ratio(pd.Series([0.01] * 5))


def test_risk_profile_reports_assumptions(prices):
    profile = risk.risk_profile(prices, risk_free_rate=0.03, confidence=0.99)
    assert profile["risk_free_rate_annual"] == 0.03
    assert profile["confidence_level"] == 0.99
    assert profile["max_drawdown"] <= 0
    assert profile["cvar_1d"] >= profile["var_historical_1d"]
