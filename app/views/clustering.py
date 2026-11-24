"""Stock Clustering: K-means on standardised risk/return features, PCA view, profiles and peers."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import charts
from common import cached_panel, display_names, require_database, sector_map
from components import footer, kpi, require_stocks, section
from fmip.analysis.clustering import (
    FEATURE_COLUMNS,
    MIN_STOCKS,
    cluster_stocks,
    describe_clusters,
    evaluate_k,
    nearest_peers,
    pca_projection,
    standardised_profiles,
    stock_features,
)
from fmip.exceptions import InsufficientDataError
from state import current

FEATURE_LABELS = {
    "avg_return_annual": "Mean return (ann.)",
    "volatility_annual": "Volatility (ann.)",
    "max_drawdown": "Max drawdown",
    "log_avg_volume": "log avg volume",
    "skewness": "Skewness",
    "excess_kurtosis": "Excess kurtosis",
}


def render() -> None:
    require_database()
    sel = current()
    if not require_stocks(sel, MIN_STOCKS, " for clustering"):
        return
    names = display_names(sel.tickers)
    prices = cached_panel(tuple(sel.tickers), sel.start, sel.end, sel.price_column)
    volumes = cached_panel(tuple(sel.tickers), sel.start, sel.end, "volume")
    features = stock_features(prices, volumes) if not prices.empty else pd.DataFrame()
    if len(features) < MIN_STOCKS:
        st.info(f"Need at least {MIN_STOCKS} stocks with 60+ sessions in the period.")
        return

    max_k = min(6, len(features) - 1)
    evaluation = evaluate_k(features, range(2, max_k + 1))
    best_k = int(evaluation.loc[evaluation["silhouette"].idxmax(), "k"]) if not evaluation.empty else 2
    c = st.columns([2, 3])
    k = c[0].slider("Number of clusters (k)", 2, max_k, best_k, key="cl_k",
                    help=f"The silhouette score is highest at k = {best_k}. It ranges from −1 to 1; higher means "
                         "better-separated clusters.")
    c[1].caption("Features per stock (standardised before clustering): " + ", ".join(FEATURE_LABELS.values())
                 + ". Volume enters on a log scale so very liquid names do not dominate.")
    try:
        labels = cluster_stocks(features, k)
        coords, variance = pca_projection(features)
    except InsufficientDataError as exc:
        st.info(str(exc))
        return
    score = float(evaluation.set_index("k").loc[k, "silhouette"]) if k in set(evaluation["k"]) else float("nan")
    m = st.columns(4)
    kpi(m[0], "Stocks clustered", str(len(features)))
    kpi(m[1], "Silhouette at chosen k", f"{score:.2f}",
        help="Above 0.5 is strong structure; 0.25–0.5 is weak or moderate; below 0.25 means little structure.")
    kpi(m[2], "Variance explained by 2 PCs", f"{variance.sum():.0%}")
    kpi(m[3], "Largest cluster", f"{labels.value_counts().max()} stocks")

    a, b = st.columns([3, 2])
    with a.container(border=True):
        charts.show(charts.scatter_labeled(coords, "PC1", "PC2", "PCA projection of standardised features",
                                           f"PC1 ({variance[0]:.0%})", f"PC2 ({variance[1]:.0%})",
                                           groups=labels, height=420))
        st.caption("Distance on this map approximates similarity in the six features. The axes are combinations of "
                   "features, not single metrics.")
    with b.container(border=True):
        profiles = standardised_profiles(features, labels).rename(columns=FEATURE_LABELS)
        profiles.index = [f"Cluster {i}" for i in profiles.index]
        charts.show(charts.heatmap(profiles, "Cluster profiles (mean z-score)", zmin=-2, zmax=2, height=300))
        st.caption("Blue = above the sample average, red = below. For drawdown, a higher value means shallower.")
        st.dataframe(evaluation, hide_index=True, width="stretch",
                     column_config={"silhouette": st.column_config.NumberColumn("Silhouette", format="%.3f"),
                                    "inertia": st.column_config.NumberColumn("Inertia", format="%.1f")})

    section("Cluster members and characteristics")
    summary = describe_clusters(features, labels)
    sectors = sector_map(list(features.index))
    members = pd.DataFrame({"cluster": labels, "name": [names.get(t, t) for t in labels.index],
                            "sector": [sectors.get(t, "n/a") for t in labels.index]}).join(features)
    left, right = st.columns([3, 2])
    pct_cols = ["avg_return_annual", "volatility_annual", "max_drawdown"]
    config = {c_: st.column_config.NumberColumn(FEATURE_LABELS[c_], format="percent") for c_ in pct_cols}
    config.update({c_: st.column_config.NumberColumn(FEATURE_LABELS[c_], format="%.2f")
                   for c_ in ("log_avg_volume", "skewness", "excess_kurtosis")})
    with left:
        st.dataframe(members.sort_values("cluster"), width="stretch", column_config=config)
    with right:
        st.dataframe(summary, width="stretch", column_config=config)
        focus = st.selectbox("Find peers for", list(features.index), key="cl_focus",
                             format_func=lambda t: f"{t} · {names.get(t, t)}")
        peers = nearest_peers(features, focus, n=min(5, len(features) - 1))
        st.dataframe(peers.to_frame().assign(cluster=labels.reindex(peers.index)), width="stretch",
                     column_config={"distance": st.column_config.NumberColumn("Distance (std units)", format="%.2f")})
    st.caption("Clusters group stocks with similar historical profiles in this sample. They are descriptive, not "
               "predictive, and can change with the period or selection.")
    footer()
