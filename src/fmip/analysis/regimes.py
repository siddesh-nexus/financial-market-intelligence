"""Historical market-regime classification.

Two complementary, *descriptive* approaches:

1. Rule-based (transparent, no look-ahead). Every input at date t uses data up to t only.
   - Trend: Bull if close > MA200 and MA50 > MA200; Bear if close < MA200 and MA50 < MA200;
     otherwise Sideways / transition.
   - Volatility: 21-session annualised volatility compared with its own *expanding* history
     (only past values): High if above the 80th percentile, Low if below the 20th, else Normal.

2. Gaussian Mixture Model on [21-session return, 21-session volatility], standardised.
   The model is fitted on the whole sample, so states are an in-sample description of the
   history, not a real-time signal. States are ordered by average volatility.

Regimes label the past. They are not a prediction of future market behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from fmip.config import TRADING_DAYS_PER_YEAR
from fmip.exceptions import InsufficientDataError

VOL_WINDOW = 21
FAST_MA, SLOW_MA = 50, 200
HIGH_VOL_PCT, LOW_VOL_PCT = 0.80, 0.20
MIN_VOL_HISTORY = 252
GMM_STATE_NAMES = {
    2: ["Calm", "Stressed"],
    3: ["Calm", "Normal", "Stressed"],
    4: ["Calm", "Normal", "Elevated", "Stressed"],
}


def rule_based_regimes(prices: pd.Series) -> pd.DataFrame:
    """Trend and volatility regime per session (NaN-labelled until enough history exists)."""
    close = prices.dropna().astype(float)
    if len(close) < SLOW_MA + 5:
        raise InsufficientDataError(f"Need more than {SLOW_MA + 5} sessions for trend regimes")
    returns = close.pct_change()
    ma_fast = close.rolling(FAST_MA, min_periods=FAST_MA).mean()
    ma_slow = close.rolling(SLOW_MA, min_periods=SLOW_MA).mean()
    vol = returns.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    hist = vol.shift(1)  # thresholds from strictly earlier observations
    upper = hist.expanding(min_periods=MIN_VOL_HISTORY).quantile(HIGH_VOL_PCT)
    lower = hist.expanding(min_periods=MIN_VOL_HISTORY).quantile(LOW_VOL_PCT)

    trend = pd.Series("Sideways", index=close.index, dtype=object)
    trend[(close > ma_slow) & (ma_fast > ma_slow)] = "Bull"
    trend[(close < ma_slow) & (ma_fast < ma_slow)] = "Bear"
    trend[ma_slow.isna()] = None

    vol_regime = pd.Series("Normal", index=close.index, dtype=object)
    vol_regime[vol > upper] = "High"
    vol_regime[vol < lower] = "Low"
    vol_regime[upper.isna() | vol.isna()] = None

    return pd.DataFrame({
        "close": close, "return": returns, "ma_fast": ma_fast, "ma_slow": ma_slow,
        "volatility": vol, "vol_upper": upper, "vol_lower": lower,
        "trend_regime": trend, "vol_regime": vol_regime,
    })


def gmm_regimes(prices: pd.Series, n_states: int = 3, random_state: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a Gaussian Mixture on rolling return/volatility features.

    Returns (per-session frame with state label and probability, per-state parameters).
    """
    if n_states not in GMM_STATE_NAMES:
        raise ValueError(f"n_states must be one of {sorted(GMM_STATE_NAMES)}")
    close = prices.dropna().astype(float)
    returns = close.pct_change()
    features = pd.DataFrame({
        "ret_21": close.pct_change(VOL_WINDOW),
        "vol_21": returns.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR),
    }).dropna()
    if len(features) < 100:
        raise InsufficientDataError("Need at least 100 sessions of rolling features for the mixture model")
    scaled = StandardScaler().fit_transform(features.values)
    model = GaussianMixture(n_components=n_states, covariance_type="full", n_init=3, random_state=random_state)
    raw_labels = model.fit_predict(scaled)
    probabilities = model.predict_proba(scaled)

    # Order states by average volatility so names are stable across runs.
    mean_vol = pd.Series(features["vol_21"].values).groupby(raw_labels).mean()
    order = list(mean_vol.sort_values().index)
    rank = {state: position for position, state in enumerate(order)}
    names = GMM_STATE_NAMES[n_states]
    labels = np.array([names[rank[s]] for s in raw_labels])
    confidence = probabilities.max(axis=1)

    frame = pd.DataFrame({"close": close.reindex(features.index), "return": returns.reindex(features.index),
                          "ret_21": features["ret_21"], "vol_21": features["vol_21"],
                          "state": labels, "probability": confidence}, index=features.index)
    params = frame.groupby("state").agg(
        sessions=("state", "size"),
        mean_21d_return=("ret_21", "mean"),
        mean_volatility=("vol_21", "mean"),
    )
    params = params.reindex([n for n in names if n in params.index])
    return frame, params


def regime_summary(labels: pd.Series, returns: pd.Series) -> pd.DataFrame:
    """Share of time, annualised mean return and volatility, and episode statistics per regime.

    Returns are those of the same sessions (descriptive), not forward returns.
    """
    frame = pd.DataFrame({"label": labels, "ret": returns}).dropna(subset=["label"])
    if frame.empty:
        return pd.DataFrame()
    episodes = (frame["label"] != frame["label"].shift()).cumsum()
    lengths = frame.groupby(episodes)["label"].agg(["first", "size"])
    rows = {}
    for label, group in frame.groupby("label"):
        ret = group["ret"].dropna()
        label_episodes = lengths[lengths["first"] == label]["size"]
        rows[label] = {
            "share_of_time": len(group) / len(frame),
            "sessions": len(group),
            "annualised_mean_return": float(ret.mean() * TRADING_DAYS_PER_YEAR) if len(ret) else np.nan,
            "annualised_volatility": float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(ret) > 1 else np.nan,
            "episodes": int(len(label_episodes)),
            "avg_episode_sessions": float(label_episodes.mean()) if len(label_episodes) else np.nan,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def current_regime(rules: pd.DataFrame) -> dict[str, str | None]:
    """Latest available trend and volatility labels."""
    trend = rules["trend_regime"].dropna()
    vol = rules["vol_regime"].dropna()
    return {"trend": trend.iloc[-1] if not trend.empty else None,
            "volatility": vol.iloc[-1] if not vol.empty else None,
            "as_of": rules.index[-1]}
