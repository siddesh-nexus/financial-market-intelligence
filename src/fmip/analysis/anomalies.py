"""Anomaly detection engine.

Two approaches are implemented:

1. Statistical (rolling z-score). For each signal, the baseline mean and standard deviation
   are computed from the PREVIOUS `window` sessions only (shifted by one day) so the current
   observation is never part of its own baseline. A day is flagged when |z| >= threshold.
   Signals: daily return, volume (on log scale), intraday range relative to close, and a
   volatility spike (rolling 20-day volatility compared with its own history).

2. Machine learning (Isolation Forest). Multivariate features per day are scaled and
   scored; the `contamination` parameter sets the expected share of flagged days.

A flagged day is a statistical outlier relative to recent history. It is NOT evidence of
market manipulation, insider activity, or a future price move. Results describe the data only.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from fmip.config import TRADING_DAYS_PER_YEAR
from fmip.exceptions import InsufficientDataError

ANOMALY_COLUMNS = [
    "ticker", "detection_date", "category", "method", "observed_value",
    "baseline_value", "anomaly_score", "details",
]
ZSCORE_METHOD = "zscore_rolling"
IFOREST_METHOD = "isolation_forest"
MIN_WINDOW_OBSERVATIONS = 20


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=ANOMALY_COLUMNS)


def rolling_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.DataFrame:
    """Compare each value with the mean and std of the previous `window` values.

    Returns columns: observed, baseline (mean), spread (std), zscore.
    """
    if window < MIN_WINDOW_OBSERVATIONS:
        raise ValueError(f"window must be at least {MIN_WINDOW_OBSERVATIONS}")
    min_obs = min_periods or max(window // 2, MIN_WINDOW_OBSERVATIONS)
    values = series.astype(float)
    baseline = values.shift(1).rolling(window, min_periods=min_obs).mean()
    spread = values.shift(1).rolling(window, min_periods=min_obs).std(ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = (values - baseline) / spread
    zscore = zscore.where(spread > 0)
    return pd.DataFrame({"observed": values, "baseline": baseline, "spread": spread, "zscore": zscore})


def _signals(ohlcv: pd.DataFrame, window: int) -> dict[str, tuple[str, pd.Series]]:
    """Build the four univariate signals for the statistical engine."""
    adj = ohlcv["Adj Close"].astype(float)
    returns = adj.pct_change()
    volatility_20 = returns.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return {
        "return_zscore": ("daily_return", returns),
        "volume_zscore": ("log_volume", np.log1p(ohlcv["Volume"].astype(float))),
        "price_range_zscore": ("intraday_range_pct", (ohlcv["High"] - ohlcv["Low"]) / ohlcv["Close"]),
        "volatility_spike": ("rolling_volatility_20d", volatility_20),
    }


def detect_statistical_anomalies(
    ticker: str,
    ohlcv: pd.DataFrame,
    window: int = 60,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Rolling z-score anomalies for one ticker. `ohlcv` must be indexed by date and contain
    High, Low, Close, Adj Close and Volume."""
    required = {"High", "Low", "Close", "Adj Close", "Volume"}
    if not required.issubset(ohlcv.columns):
        raise ValueError(f"ohlcv must contain columns {sorted(required)}")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if len(ohlcv) < window + 2:
        raise InsufficientDataError(f"{ticker}: need more than {window + 2} rows for window {window}")

    frames = []
    for category, (label, signal) in _signals(ohlcv, window).items():
        scored = rolling_zscore(signal, window)
        flagged = scored[scored["zscore"].abs() >= threshold].dropna(subset=["zscore"])
        if flagged.empty:
            continue
        baseline = flagged["baseline"]
        observed = flagged["observed"]
        if category == "volume_zscore":
            observed = np.expm1(observed)
            baseline = np.expm1(baseline)
        frames.append(pd.DataFrame({
            "ticker": ticker,
            "detection_date": flagged.index,
            "category": category,
            "method": ZSCORE_METHOD,
            "observed_value": observed.values,
            "baseline_value": baseline.values,
            "anomaly_score": flagged["zscore"].abs().values,
            "details": [
                json.dumps({
                    "signal": label,
                    "zscore": round(float(z), 4),
                    "window": window,
                    "threshold": threshold,
                    "baseline_std": round(float(s), 6),
                })
                for z, s in zip(flagged["zscore"], flagged["spread"])
            ],
        }))
    if not frames:
        return _empty_frame()
    result = pd.concat(frames, ignore_index=True)
    result["detection_date"] = pd.to_datetime(result["detection_date"]).dt.date
    return result[ANOMALY_COLUMNS]


def isolation_forest_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Multivariate daily features used by the Isolation Forest."""
    adj = ohlcv["Adj Close"].astype(float)
    returns = adj.pct_change()
    log_volume = np.log1p(ohlcv["Volume"].astype(float))
    return pd.DataFrame({
        "daily_return": returns,
        "abs_return": returns.abs(),
        "volume_change": log_volume.diff(),
        "intraday_range_pct": (ohlcv["High"] - ohlcv["Low"]) / ohlcv["Close"],
        "volatility_20d": returns.rolling(20, min_periods=20).std(ddof=1),
    }, index=ohlcv.index).dropna()


def detect_isolation_forest_anomalies(
    ticker: str,
    ohlcv: pd.DataFrame,
    contamination: float = 0.01,
    random_state: int = 42,
) -> pd.DataFrame:
    """Flag days whose combined feature profile is unusual, using Isolation Forest."""
    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5")
    features = isolation_forest_features(ohlcv)
    if len(features) < 50:
        raise InsufficientDataError(f"{ticker}: need at least 50 feature rows, got {len(features)}")

    scaled = StandardScaler().fit_transform(features.values)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state)
    labels = model.fit_predict(scaled)
    # decision_function is higher for normal points; negate so higher = more anomalous.
    scores = -model.decision_function(scaled)

    mask = labels == -1
    if not mask.any():
        return _empty_frame()
    flagged = features[mask]
    return pd.DataFrame({
        "ticker": ticker,
        "detection_date": flagged.index.date,
        "category": "multivariate",
        "method": IFOREST_METHOD,
        "observed_value": flagged["daily_return"].values,
        "baseline_value": float(features["daily_return"].mean()),
        "anomaly_score": scores[mask],
        "details": [
            json.dumps({
                "features": {k: round(float(v), 6) for k, v in row.items()},
                "contamination": contamination,
            })
            for _, row in flagged.iterrows()
        ],
    })[ANOMALY_COLUMNS]


CATEGORY_LABELS = {
    "return_zscore": "Price move",
    "volume_zscore": "Volume surge",
    "price_range_zscore": "Intraday range",
    "volatility_spike": "Volatility spike",
    "multivariate": "Multivariate (Isolation Forest)",
}


def severity_label(method: str, score: float) -> str:
    """Heuristic severity bands.

    z-score: |z| 3-4 Moderate, 4-6 High, 6+ Extreme. Isolation Forest (negated decision
    function, > 0 for flagged days): < 0.05 Moderate, 0.05-0.10 High, > 0.10 Extreme.
    """
    if pd.isna(score):
        return "n/a"
    if method == IFOREST_METHOD:
        return "Extreme" if score > 0.10 else "High" if score > 0.05 else "Moderate"
    magnitude = abs(score)
    return "Extreme" if magnitude >= 6 else "High" if magnitude >= 4 else "Moderate"


def _fmt_volume(value: float) -> str:
    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= size:
            return f"{value / size:.1f}{suffix}"
    return f"{value:.0f}"


def explain_anomaly(category: str, observed: float, baseline: float, details: str | None) -> str:
    """One-sentence, plain-language reason why an observation was flagged."""
    try:
        info = json.loads(details) if details else {}
    except (TypeError, ValueError):
        info = {}
    z = info.get("zscore")
    window = info.get("window", "")
    spread = info.get("baseline_std")
    z_text = f"; z = {z:+.1f}" if z is not None else ""
    if category == "return_zscore":
        sd = f", σ {spread:.2%}" if spread is not None else ""
        return f"Daily return {observed:+.2%} against a {window}-day average of {baseline:+.2%}{sd}{z_text}."
    if category == "volume_zscore":
        ratio = observed / baseline if baseline else float("nan")
        return (f"Volume {_fmt_volume(observed)} vs typical {_fmt_volume(baseline)} "
                f"({ratio:.1f}× the {window}-day log-scale average{z_text}).")
    if category == "price_range_zscore":
        return f"High-low range {observed:.2%} of close vs {window}-day average {baseline:.2%}{z_text}."
    if category == "volatility_spike":
        return f"20-day volatility {observed:.0%} (annualised) vs {window}-day average {baseline:.0%}{z_text}."
    if category == "multivariate":
        features = info.get("features", {})
        parts = []
        if "daily_return" in features:
            parts.append(f"return {features['daily_return']:+.2%}")
        if "volume_change" in features:
            parts.append(f"log-volume change {features['volume_change']:+.2f}")
        if "intraday_range_pct" in features:
            parts.append(f"range {features['intraday_range_pct']:.2%}")
        if "volatility_20d" in features:
            parts.append(f"20d daily σ {features['volatility_20d']:.2%}")
        return "Unusual combination of " + ", ".join(parts) + " relative to this stock's history."
    return "Flagged as unusual relative to recent history."


def annotate(anomalies: pd.DataFrame) -> pd.DataFrame:
    """Add `severity`, `signal` (readable category) and `explanation` columns."""
    if anomalies.empty:
        return anomalies.assign(severity=pd.Series(dtype=object), signal=pd.Series(dtype=object),
                                explanation=pd.Series(dtype=object))
    frame = anomalies.copy()
    frame["severity"] = [severity_label(m, s) for m, s in zip(frame["method"], frame["anomaly_score"])]
    frame["signal"] = frame["category"].map(CATEGORY_LABELS).fillna(frame["category"])
    frame["explanation"] = [
        explain_anomaly(c, o, b, d) for c, o, b, d in
        zip(frame["category"], frame["observed_value"], frame["baseline_value"], frame["details"])
    ]
    return frame


def detect_all(
    ticker: str,
    ohlcv: pd.DataFrame,
    window: int = 60,
    threshold: float = 3.0,
    contamination: float = 0.01,
) -> pd.DataFrame:
    """Run both engines for one ticker and return the combined, date-sorted results."""
    statistical = detect_statistical_anomalies(ticker, ohlcv, window, threshold)
    multivariate = detect_isolation_forest_anomalies(ticker, ohlcv, contamination)
    combined = pd.concat([statistical, multivariate], ignore_index=True)
    return combined.sort_values(["detection_date", "method", "category"]).reset_index(drop=True)
