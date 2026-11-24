"""Extensions to existing modules: drawdown episodes, anomaly explanations, event paths,
abnormal returns, correlation ordering, clustering peers and rolling risk."""

import numpy as np
import pandas as pd
import pytest

from fmip.analysis import anomalies, clustering, correlation, eda, events, risk
from tests.conftest import make_ohlcv, make_prices


def test_drawdown_episodes_hand_built():
    values = [100, 90, 80, 100, 110, 99, 110, 120, 60]
    series = pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)
    episodes = eda.drawdown_episodes(series, top=5)
    worst = episodes.iloc[0]
    assert worst["depth"] == pytest.approx(-0.5)            # 120 -> 60, not recovered
    assert pd.isna(worst["recovery_date"])
    second = episodes.iloc[1]
    assert second["depth"] == pytest.approx(-0.2)           # 100 -> 80, recovered at 100
    assert second["decline_sessions"] == 2 and second["recovery_sessions"] == 1
    assert len(episodes) == 3


def test_anomaly_annotation_has_severity_and_reason():
    ohlcv = make_ohlcv(n=300, seed=4)
    ohlcv.iloc[250, ohlcv.columns.get_loc("Adj Close")] *= 1.25
    ohlcv.iloc[250, ohlcv.columns.get_loc("Close")] *= 1.25
    ohlcv.iloc[250, ohlcv.columns.get_loc("High")] *= 1.25
    found = anomalies.annotate(anomalies.detect_all("T", ohlcv))
    jump = found[(found["category"] == "return_zscore")
                 & (pd.to_datetime(found["detection_date"]) == ohlcv.index[250])]
    assert not jump.empty
    row = jump.iloc[0]
    assert row["severity"] == "Extreme"
    assert "Daily return" in row["explanation"] and "z =" in row["explanation"]
    assert set(found["severity"]) <= {"Moderate", "High", "Extreme"}


@pytest.mark.parametrize("method, score, label", [
    ("zscore_rolling", 3.2, "Moderate"), ("zscore_rolling", -4.5, "High"), ("zscore_rolling", 7, "Extreme"),
    ("isolation_forest", 0.02, "Moderate"), ("isolation_forest", 0.07, "High"), ("isolation_forest", 0.2, "Extreme"),
])
def test_severity_bands(method, score, label):
    assert anomalies.severity_label(method, score) == label


def test_event_abnormal_returns_and_paths():
    index = pd.bdate_range("2024-01-01", periods=100)
    bench = pd.Series(100.0, index=index)
    bench.iloc[50:] = 95.0                      # market falls 5% on the event
    stock = pd.Series(100.0, index=index)
    stock.iloc[50:] = 80.0                      # stock falls 20%
    windows = events.EventWindows(pre_days=10, event_days=5, post_days=10)
    table = events.event_study(stock.to_frame("S"), index[50], windows, benchmark=bench)
    assert table.loc["S", "event_return"] == pytest.approx(-0.20)
    assert table.loc["S", "event_abnormal_return"] == pytest.approx(-0.15)
    paths = events.event_paths(pd.DataFrame({"S": stock, "B": bench}), index[50], pre_days=5, post_days=5)
    assert paths.loc[-1, "S"] == pytest.approx(100.0)
    assert paths.loc[0, "S"] == pytest.approx(80.0)
    assert paths.index.min() == -6 and paths.index.max() == 4


def test_cluster_order_groups_correlated_assets():
    base_a, base_b = make_prices(seed=1), make_prices(seed=2)
    panel = pd.DataFrame({"A1": base_a, "B1": base_b, "A2": base_a * 1.01, "B2": base_b * 0.99})
    corr = correlation.correlation_matrix(correlation.returns_panel(panel))
    order = correlation.cluster_order(corr)
    assert abs(order.index("A1") - order.index("A2")) == 1
    assert abs(order.index("B1") - order.index("B2")) == 1


@pytest.mark.parametrize("value, text", [(0.85, "very strong positive"), (-0.65, "strong negative"),
                                         (0.45, "moderate positive"), (0.1, "little or no linear relationship")])
def test_interpret_correlation(value, text):
    assert correlation.interpret_correlation(value) == text


def test_sector_vs_benchmark_relative_return():
    index = pd.bdate_range("2024-01-01", periods=10)
    prices = pd.DataFrame({"A": np.linspace(100, 110, 10), "B": np.linspace(100, 110, 10)}, index=index)
    bench = pd.Series(np.linspace(100, 105, 10), index=index)
    perf = correlation.sector_vs_benchmark(prices, {"A": "Tech", "B": "Tech"}, bench)
    assert perf.loc["Tech", "benchmark_return"] == pytest.approx(0.05)
    assert perf.loc["Tech", "relative_return"] == pytest.approx(perf.loc["Tech", "total_return"] - 0.05)


def test_nearest_peers_and_profiles(panel):
    volumes = pd.DataFrame(1e6, index=panel.index, columns=panel.columns)
    features = clustering.stock_features(panel, volumes)
    peers = clustering.nearest_peers(features, "T0", n=3)
    assert "T0" not in peers.index and len(peers) == 3 and peers.is_monotonic_increasing
    labels = clustering.cluster_stocks(features, 2)
    profiles = clustering.standardised_profiles(features, labels)
    assert profiles.shape == (2, len(clustering.FEATURE_COLUMNS))


def test_rolling_sharpe_matches_full_sample_on_last_window():
    rets = make_prices(n=300, seed=8).pct_change().dropna()
    rolling = risk.rolling_sharpe(rets, window=126, risk_free_rate=0.02)
    assert rolling.iloc[-1] == pytest.approx(risk.sharpe_ratio(rets.iloc[-126:], 0.02))
    var = risk.rolling_var(rets, window=100, confidence=0.95)
    assert var.iloc[-1] == pytest.approx(risk.value_at_risk(rets.iloc[-100:], 0.95))
