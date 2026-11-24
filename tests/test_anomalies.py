import json

import numpy as np
import pandas as pd
import pytest

from fmip.analysis import anomalies
from fmip.exceptions import InsufficientDataError
from tests.conftest import make_ohlcv


def test_rolling_zscore_excludes_current_value_from_baseline():
    series = pd.Series(np.zeros(30))
    series.iloc[25] = 1000.0
    scored = anomalies.rolling_zscore(series, window=20)
    # Baseline at position 25 is built from earlier zeros only, so spread is zero and z is masked.
    assert np.isnan(scored["zscore"].iloc[25])
    # Next value sees the spike in its baseline window, so no false claim about the future.
    assert scored["baseline"].iloc[26] > 0


def test_rolling_zscore_flags_planted_spike():
    rng = np.random.default_rng(1)
    series = pd.Series(rng.normal(0, 1, 200))
    series.iloc[150] = 25.0
    scored = anomalies.rolling_zscore(series, window=60)
    assert scored["zscore"].iloc[150] > 10


def test_statistical_detection_finds_planted_return_shock(ohlcv):
    shocked = ohlcv.copy()
    position = 300
    shocked.iloc[position, shocked.columns.get_loc("Adj Close")] *= 1.25
    shocked.iloc[position, shocked.columns.get_loc("Close")] *= 1.25
    shocked.iloc[position, shocked.columns.get_loc("High")] *= 1.25
    shocked.iloc[position, shocked.columns.get_loc("Low")] *= 1.25
    result = anomalies.detect_statistical_anomalies("TEST", shocked, window=60, threshold=3.0)
    shock_date = shocked.index[position].date()
    flagged = result[(result["detection_date"] == shock_date) & (result["category"] == "return_zscore")]
    assert len(flagged) == 1
    assert flagged.iloc[0]["observed_value"] == pytest.approx(0.25, abs=0.05)
    assert flagged.iloc[0]["method"] == "zscore_rolling"


def test_statistical_output_has_required_columns(ohlcv):
    result = anomalies.detect_statistical_anomalies("TEST", ohlcv, window=60, threshold=2.0)
    assert list(result.columns) == anomalies.ANOMALY_COLUMNS
    if not result.empty:
        details = json.loads(result.iloc[0]["details"])
        assert details["window"] == 60
        assert details["threshold"] == 2.0


def test_isolation_forest_flags_roughly_contamination_share(ohlcv):
    result = anomalies.detect_isolation_forest_anomalies("TEST", ohlcv, contamination=0.05)
    features = anomalies.isolation_forest_features(ohlcv)
    share = len(result) / len(features)
    assert 0.02 <= share <= 0.08
    assert set(result["method"]) == {"isolation_forest"}


def test_isolation_forest_is_reproducible(ohlcv):
    first = anomalies.detect_isolation_forest_anomalies("TEST", ohlcv, random_state=1)
    second = anomalies.detect_isolation_forest_anomalies("TEST", ohlcv, random_state=1)
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))


def test_invalid_parameters_rejected(ohlcv):
    with pytest.raises(ValueError):
        anomalies.detect_isolation_forest_anomalies("TEST", ohlcv, contamination=0.9)
    with pytest.raises(ValueError):
        anomalies.rolling_zscore(ohlcv["Close"], window=5)


def test_too_short_series_raises():
    short = make_ohlcv(n=30)
    with pytest.raises(InsufficientDataError):
        anomalies.detect_statistical_anomalies("TEST", short, window=60)
