"""Financial risk metrics.

Assumptions (also documented in docs/METHODOLOGY.md):
- Returns are simple daily returns; annualisation uses 252 trading days.
- Risk-free rate is an annual rate, converted to a daily rate as rf / 252 (simple convention).
- Sharpe ratio = mean(excess daily return) / std(daily return) * sqrt(252).
- Sortino ratio uses downside deviation over ALL observations: sqrt(mean(min(excess, 0)^2)).
- VaR and CVaR are reported as POSITIVE loss numbers for a one-day horizon.
  Historical VaR is the empirical (1 - confidence) quantile of returns.
  Parametric VaR assumes normal returns: -(mean + z * std), where z = Phi^-1(1 - confidence).
  CVaR (expected shortfall) = -mean of returns at or below the historical VaR cut-off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from fmip.analysis.eda import annualised_volatility, max_drawdown
from fmip.config import TRADING_DAYS_PER_YEAR
from fmip.exceptions import InsufficientDataError

MIN_OBSERVATIONS = 20


def _clean_returns(returns: pd.Series) -> pd.Series:
    clean = returns.dropna().astype(float)
    if len(clean) < MIN_OBSERVATIONS:
        raise InsufficientDataError(f"Need at least {MIN_OBSERVATIONS} returns, got {len(clean)}")
    return clean


def _validate_confidence(confidence: float) -> None:
    if not 0.5 <= confidence < 1.0:
        raise ValueError("confidence must be in [0.5, 1.0)")


def annualised_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Geometric (compound) annual growth rate implied by the return series."""
    clean = _clean_returns(returns)
    growth = float(np.prod(1.0 + clean.values))
    return growth ** (periods_per_year / len(clean)) - 1.0


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                 periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    clean = _clean_returns(returns)
    excess = clean - risk_free_rate / periods_per_year
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                  periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    clean = _clean_returns(returns)
    excess = clean - risk_free_rate / periods_per_year
    downside = np.minimum(excess.values, 0.0)
    downside_dev = np.sqrt(np.mean(downside ** 2))
    if downside_dev == 0:
        return float("nan")
    return float(excess.mean() / downside_dev * np.sqrt(periods_per_year))


def value_at_risk(returns: pd.Series, confidence: float = 0.95, method: str = "historical") -> float:
    """One-day VaR as a positive loss fraction (0.02 = 2% loss)."""
    _validate_confidence(confidence)
    clean = _clean_returns(returns)
    if method == "historical":
        return float(-np.quantile(clean.values, 1.0 - confidence))
    if method == "parametric":
        z = stats.norm.ppf(1.0 - confidence)
        return float(-(clean.mean() + z * clean.std(ddof=1)))
    raise ValueError("method must be 'historical' or 'parametric'")


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected shortfall: average loss on the days at or beyond the historical VaR cut-off."""
    _validate_confidence(confidence)
    clean = _clean_returns(returns)
    cutoff = np.quantile(clean.values, 1.0 - confidence)
    tail = clean[clean <= cutoff]
    return float(-tail.mean())


def risk_profile(
    prices: pd.Series,
    risk_free_rate: float = 0.0,
    confidence: float = 0.95,
) -> dict[str, float]:
    """All risk metrics for one price series, plus the parameters used."""
    values = prices.astype(float).dropna()
    returns = values.pct_change().dropna()
    dd = max_drawdown(values)
    return {
        "annualised_volatility": annualised_volatility(returns),
        "annualised_return": annualised_return(returns),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate),
        "max_drawdown": float(dd["max_drawdown"]),
        "var_historical_1d": value_at_risk(returns, confidence, "historical"),
        "var_parametric_1d": value_at_risk(returns, confidence, "parametric"),
        "cvar_1d": conditional_var(returns, confidence),
        "observations": float(len(returns)),
        "risk_free_rate_annual": float(risk_free_rate),
        "confidence_level": float(confidence),
    }


def rolling_sharpe(returns: pd.Series, window: int = 126, risk_free_rate: float = 0.0,
                   periods_per_year: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    """Trailing-window Sharpe ratio (same definition as sharpe_ratio, per window)."""
    excess = returns.astype(float) - risk_free_rate / periods_per_year
    mean = excess.rolling(window, min_periods=window).mean()
    std = excess.rolling(window, min_periods=window).std(ddof=1)
    return (mean / std.where(std > 0) * np.sqrt(periods_per_year)).rename(returns.name)


def rolling_var(returns: pd.Series, window: int = 252, confidence: float = 0.95) -> pd.Series:
    """Trailing-window one-day historical VaR as a positive loss fraction."""
    _validate_confidence(confidence)
    return (-returns.astype(float).rolling(window, min_periods=window).quantile(1.0 - confidence)).rename(returns.name)


def risk_comparison(panel: pd.DataFrame, risk_free_rate: float = 0.0,
                    confidence: float = 0.95) -> pd.DataFrame:
    """Risk profile for each column of a wide price panel. Tickers with too little data are skipped."""
    rows = {}
    for ticker in panel.columns:
        try:
            rows[ticker] = risk_profile(panel[ticker], risk_free_rate, confidence)
        except InsufficientDataError:
            continue
    return pd.DataFrame.from_dict(rows, orient="index")
