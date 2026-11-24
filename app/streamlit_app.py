"""Streamlit entry point.

Run from the project root (venv active), so `.streamlit/config.toml` is picked up:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from fmip.config import load_settings
from fmip.logging_config import setup_logging

setup_logging(load_settings().log_level)
st.set_page_config(page_title="FMIP · Market Intelligence", page_icon=":material/monitoring:", layout="wide",
                   initial_sidebar_state="auto")  # open on desktop, collapsed on phones

from state import render_header, render_sidebar  # noqa: E402  (after set_page_config)
from theme import CSS  # noqa: E402
from views import (  # noqa: E402
    anomalies,
    clustering,
    correlation,
    events,
    explorer,
    overview,
    portfolio,
    regimes,
    relative,
    risk,
)

# Every view exposes a function named `render`, so each page needs an explicit, unique url_path.
pages = {
    "Markets": [
        st.Page(overview.render, title="Market Overview", icon=":material/dashboard:", url_path="overview", default=True),
        st.Page(explorer.render, title="Stock Explorer", icon=":material/candlestick_chart:", url_path="explorer"),
        st.Page(relative.render, title="Relative Strength", icon=":material/compare_arrows:", url_path="relative-strength"),
    ],
    "Analytics": [
        st.Page(correlation.render, title="Correlation Analysis", icon=":material/hub:", url_path="correlation"),
        st.Page(anomalies.render, title="Anomaly Detection", icon=":material/crisis_alert:", url_path="anomalies"),
        st.Page(risk.render, title="Risk Analytics", icon=":material/shield:", url_path="risk"),
        st.Page(regimes.render, title="Market Regimes", icon=":material/waterfall_chart:", url_path="regimes"),
        st.Page(clustering.render, title="Stock Clustering", icon=":material/bubble_chart:", url_path="clustering"),
        st.Page(events.render, title="Event Analysis", icon=":material/event:", url_path="events"),
    ],
    "Portfolio": [
        st.Page(portfolio.render, title="Portfolio Analytics", icon=":material/pie_chart:", url_path="portfolio"),
    ],
}
page = st.navigation(pages, position="sidebar")  # before any other element, so deep links resolve
st.html(CSS)
selection = render_sidebar()
st.session_state["_selection"] = selection
render_header(page.title, selection)
page.run()
