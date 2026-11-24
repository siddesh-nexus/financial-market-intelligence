import numpy as np
import pandas as pd
import pytest

from fmip.analysis import clustering, correlation, events
from fmip.exceptions import InsufficientDataError
from tests.conftest import make_prices


def test_perfectly_correlated_pair_detected():
    base = make_prices(seed=3)
    panel = pd.DataFrame({"A": base, "B": base * 2.0, "C": make_prices(seed=99)})
    returns = correlation.returns_panel(panel)
    corr = correlation.correlation_matrix(returns)
    pairs = correlation.strongest_pairs(corr, positive_threshold=0.99)
    assert ("A", "B") in set(zip(pairs["asset_a"], pairs["asset_b"]))
    assert pairs.iloc[0]["correlation"] == pytest.approx(1.0)


def test_negative_correlation_detected():
    base = make_prices(seed=5)
    panel = pd.DataFrame({"A": base, "B": 2 * base.iloc[0] - (base - base.iloc[0]) + base.iloc[0]})
    returns = correlation.returns_panel(panel)
    corr = correlation.correlation_matrix(returns)
    assert corr.loc["A", "B"] < -0.9


def test_spearman_and_pearson_both_run(panel):
    returns = correlation.returns_panel(panel)
    assert correlation.correlation_matrix(returns, "spearman").shape == (6, 6)
    assert correlation.correlation_matrix(returns, "pearson").shape == (6, 6)


def test_rolling_correlation_window_length(panel):
    returns = correlation.returns_panel(panel)
    rc = correlation.rolling_correlation(returns, "T0", "T1", window=60)
    assert rc.dropna().between(-1, 1).all()


def test_sector_performance_uses_mapping(panel):
    sector_map = {"T0": "Tech", "T1": "Tech", "T2": "Energy", "T3": "Energy", "T4": "Health", "T5": "Health"}
    perf = correlation.sector_performance(panel, sector_map)
    assert set(perf.index) == {"Tech", "Energy", "Health"}
    assert (perf["members"] == 2).all()


def test_clustering_produces_requested_number_of_groups(panel):
    volumes = pd.DataFrame({c: np.linspace(1e6, 2e6, len(panel)) for c in panel.columns}, index=panel.index)
    features = clustering.stock_features(panel, volumes)
    assert len(features) == 6
    labels = clustering.cluster_stocks(features, k=2)
    assert labels.nunique() == 2
    evaluation = clustering.evaluate_k(features, [2, 3])
    assert set(evaluation["k"]) == {2, 3}
    assert evaluation["silhouette"].between(-1, 1).all()


def test_pca_explained_variance_sums_below_one(panel):
    volumes = pd.DataFrame(1e6, index=panel.index, columns=panel.columns)
    features = clustering.stock_features(panel, volumes)
    coords, ratio = clustering.pca_projection(features)
    assert coords.shape == (6, 2)
    assert 0 < ratio.sum() <= 1.0 + 1e-9


def test_clustering_requires_minimum_stocks(panel):
    volumes = pd.DataFrame(1e6, index=panel.index, columns=panel.columns)
    features = clustering.stock_features(panel[["T0", "T1"]], volumes[["T0", "T1"]])
    with pytest.raises(InsufficientDataError):
        clustering.cluster_stocks(features, k=2)


def test_event_window_recovery_detected():
    index = pd.bdate_range("2020-01-01", periods=120)
    values = np.full(120, 100.0)
    values[60:65] = 80.0          # crash during the event window
    values[65:] = 100.0           # full recovery afterwards
    series = pd.Series(values, index=index)
    result = events.analyse_event(series, index[60], events.EventWindows(pre_days=10, event_days=5, post_days=10))
    assert result["recovered"] is True
    assert result["recovery_sessions"] == 1
    assert result["event_return"] < 0


def test_event_window_not_recovered_returns_none():
    index = pd.bdate_range("2020-01-01", periods=100)
    values = np.concatenate([np.full(50, 100.0), np.full(50, 70.0)])
    series = pd.Series(values, index=index)
    result = events.analyse_event(series, index[50], events.EventWindows(recovery_lookahead=20))
    assert result["recovered"] is False
    assert result["recovery_sessions"] is None


def test_event_outside_history_raises():
    series = make_prices(n=100)
    with pytest.raises(InsufficientDataError):
        events.analyse_event(series, "2030-01-01")
