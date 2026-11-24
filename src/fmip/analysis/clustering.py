"""Behavioural segmentation of stocks: K-means on standardised risk/return features,
PCA for visualisation, and silhouette scores for choosing k.

Clusters group stocks with similar historical profiles in this sample. They are descriptive,
not a forecast or a trading signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from fmip.analysis.eda import annualised_volatility, daily_returns, max_drawdown
from fmip.config import TRADING_DAYS_PER_YEAR
from fmip.exceptions import InsufficientDataError

FEATURE_COLUMNS = [
    "avg_return_annual",
    "volatility_annual",
    "max_drawdown",
    "log_avg_volume",
    "skewness",
    "excess_kurtosis",
]
MIN_STOCKS = 4


def stock_features(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker with the features in FEATURE_COLUMNS.

    `prices` is a wide adjusted-close panel and `volumes` a wide volume panel with the same tickers.
    """
    rows = {}
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        returns = daily_returns(series)
        if len(returns) < 60 or ticker not in volumes.columns:
            continue
        vol = volumes[ticker].dropna()
        if vol.empty:
            continue
        mean_daily = float(returns.mean())
        rows[ticker] = {
            "avg_return_annual": mean_daily * TRADING_DAYS_PER_YEAR,
            "volatility_annual": annualised_volatility(returns),
            "max_drawdown": max_drawdown(series)["max_drawdown"],
            "log_avg_volume": float(np.log1p(vol.mean())),
            "skewness": float(returns.skew()),
            "excess_kurtosis": float(returns.kurt()),
        }
    features = pd.DataFrame.from_dict(rows, orient="index")
    return features[FEATURE_COLUMNS].dropna() if not features.empty else features


def _scaled(features: pd.DataFrame) -> np.ndarray:
    if len(features) < MIN_STOCKS:
        raise InsufficientDataError(f"Clustering needs at least {MIN_STOCKS} stocks, got {len(features)}")
    return StandardScaler().fit_transform(features[FEATURE_COLUMNS].values)


def evaluate_k(features: pd.DataFrame, k_values: range | list[int] = range(2, 6),
               random_state: int = 42) -> pd.DataFrame:
    """Silhouette score and inertia for each candidate k."""
    scaled = _scaled(features)
    rows = []
    for k in k_values:
        if k < 2 or k >= len(features):
            continue
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(scaled)
        rows.append({"k": k, "silhouette": float(silhouette_score(scaled, labels)),
                     "inertia": float(model.inertia_)})
    return pd.DataFrame(rows)


def cluster_stocks(features: pd.DataFrame, k: int, random_state: int = 42) -> pd.Series:
    """Cluster label per ticker (0..k-1)."""
    scaled = _scaled(features)
    labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(scaled)
    return pd.Series(labels, index=features.index, name="cluster")


def pca_projection(features: pd.DataFrame, n_components: int = 2) -> tuple[pd.DataFrame, np.ndarray]:
    """Project standardised features onto principal components. Returns (coordinates, explained variance ratio)."""
    scaled = _scaled(features)
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(scaled)
    columns = [f"PC{i + 1}" for i in range(n_components)]
    return pd.DataFrame(coords, index=features.index, columns=columns), pca.explained_variance_ratio_


def standardised_profiles(features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Mean z-score of each feature per cluster (0 = average stock in the sample)."""
    scaled = pd.DataFrame(_scaled(features), index=features.index, columns=FEATURE_COLUMNS)
    return scaled.assign(cluster=labels.values).groupby("cluster")[FEATURE_COLUMNS].mean()


def nearest_peers(features: pd.DataFrame, ticker: str, n: int = 5) -> pd.Series:
    """The `n` stocks closest to `ticker` in standardised feature space (Euclidean distance)."""
    if ticker not in features.index:
        raise KeyError(ticker)
    scaled = pd.DataFrame(_scaled(features), index=features.index)
    distance = np.sqrt(((scaled - scaled.loc[ticker]) ** 2).sum(axis=1)).drop(ticker)
    return distance.sort_values().head(n).rename("distance")


def describe_clusters(features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Mean of each feature per cluster, plus cluster size."""
    joined = features.assign(cluster=labels.values)
    summary = joined.groupby("cluster")[FEATURE_COLUMNS].mean()
    summary.insert(0, "members", joined.groupby("cluster").size())
    return summary
