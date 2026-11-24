"""Event Analysis: returns, abnormal returns, volatility and recovery around historical events."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

import charts
from common import cached_panel, display_names, get_repository, ingest, missing_history, require_database
from components import footer, kpi, pct, require_stocks, section
from fmip.analysis.events import EventWindows, event_paths, event_study
from fmip.universe import DEFAULT_EVENTS
from state import current
from theme import entity_colours

CUSTOM_LABEL = "Custom event…"


def _event_catalog() -> list[dict]:
    events = {e.name: {"name": e.name, "date": e.event_date, "category": e.category, "description": e.description}
              for e in DEFAULT_EVENTS}
    repository = get_repository()
    if repository is not None:
        stored = repository.load_events()
        for row in stored.itertuples(index=False):
            events.setdefault(row.name, {"name": row.name, "date": pd.Timestamp(row.event_date).date(),
                                         "category": row.category, "description": row.description or ""})
    return sorted(events.values(), key=lambda e: e["date"], reverse=True)


def render() -> None:
    require_database()
    sel = current()
    if not require_stocks(sel):
        return
    names = display_names([*sel.tickers, sel.benchmark])
    catalog = _event_catalog()
    labels = [f"{e['date']:%d %b %Y} · {e['name']}" for e in catalog] + [CUSTOM_LABEL]

    with st.container(border=True):
        c = st.columns([3, 1, 1, 1])
        choice = c[0].selectbox("Event", labels, key="ev_choice")
        pre_days = c[1].number_input("Pre-event sessions", 5, 120, 30, key="ev_pre")
        event_days = c[2].number_input("Event sessions", 1, 20, 5, key="ev_len")
        post_days = c[3].number_input("Post-event sessions", 5, 250, 60, key="ev_post")
        if choice == CUSTOM_LABEL:
            d = st.columns([2, 1, 1])
            name = d[0].text_input("Event name", "Custom event", key="ev_name")
            event_day = d[1].date_input("First session to analyse", value=date(2023, 3, 10), max_value=date.today(),
                                        key="ev_date")
            if d[2].button("Save event", width="stretch", icon=":material/bookmark_add:"):
                get_repository().upsert_events(pd.DataFrame([{"name": name.strip()[:200], "event_date": event_day,
                                                              "category": "Custom", "description": ""}]))
                st.toast(f"Saved '{name}'.")
            description = ""
        else:
            selected = catalog[labels.index(choice)]
            name, event_day, description = selected["name"], selected["date"], selected["description"]
            st.caption(f"{selected['category']} · {description}")
    windows = EventWindows(pre_days=int(pre_days), event_days=int(event_days), post_days=int(post_days))

    symbols = [*sel.tickers, sel.benchmark]
    load_start = event_day - timedelta(days=int(pre_days * 1.6) + 30)
    load_end = min(date.today(), event_day + timedelta(days=int((post_days + windows.recovery_lookahead) * 1.5) + 30))
    uncovered = missing_history(symbols, load_start)
    if uncovered:
        st.info(f"Stored history does not reach back to {load_start:%d %b %Y} for {', '.join(uncovered)}.")
        if st.button(f"Download history from {load_start:%d %b %Y}", icon=":material/download:"):
            with st.spinner("Downloading earlier history from Yahoo Finance…"):
                results = ingest(uncovered, load_start, incremental=False)
            failed = [r.ticker for r in results if r.status == "failed"]
            if failed:
                st.warning(f"Could not load earlier history for {', '.join(failed)} (it may not exist that far back).")
            st.rerun()

    panel = cached_panel(tuple(symbols), load_start, load_end, sel.price_column)
    if panel.empty:
        st.warning("No stored data around this event.")
        return
    stocks = panel[[t for t in sel.tickers if t in panel.columns]]
    bench = panel[sel.benchmark] if sel.benchmark in panel.columns else None
    table = event_study(stocks, event_day, windows, benchmark=bench)
    if table.empty:
        st.info("The event falls outside the stored history for the selected stocks.")
        footer()
        return
    bench_row = event_study(panel[[sel.benchmark]], event_day, windows) if bench is not None else pd.DataFrame()

    section(f"{name} · first session on or after {event_day:%d %b %Y}")
    k = st.columns(4)
    if not bench_row.empty:
        b = bench_row.iloc[0]
        kpi(k[0], f"{sel.benchmark} event-window return", pct(b["event_return"]),
            help=f"From the last close before the event to the end of the {event_days}-session event window.")
        rec = b["recovery_sessions"]
        kpi(k[1], f"{sel.benchmark} recovery", f"{int(rec)} sessions" if pd.notna(rec) else "Not recovered",
            help="Sessions after the event window until the pre-event close was regained "
                 f"(searched up to {windows.recovery_lookahead} sessions).")
    kpi(k[2], "Median stock event return", pct(table["event_return"].median()))
    if "event_abnormal_return" in table:
        kpi(k[3], "Median abnormal return", pct(table["event_abnormal_return"].median()),
            help="Stock event-window return minus the benchmark's over the same window (market-adjusted).")

    with st.container(border=True):
        paths = event_paths(panel, event_day, int(pre_days), int(post_days))
        charts.show(charts.event_paths(paths, entity_colours(sel.tickers), sel.benchmark, int(event_days)))
        st.caption("The shaded band is the event window. Each line is rebased to 100 at the last close before the event.")

    a, b_ = st.columns([1, 1])
    with a.container(border=True):
        charts.show(charts.grouped_bars(table[["pre_return", "event_return", "post_return"]]
                                        .rename(columns={"pre_return": "Before", "event_return": "During",
                                                         "post_return": "After"}),
                                        "Return before, during and after", height=330, y_title="Return (%)"))
    with b_.container(border=True):
        if "event_abnormal_return" in table:
            charts.show(charts.ranked_bars(table["event_abnormal_return"],
                                           f"Event-window return vs {sel.benchmark}", names=names))

    display = table.copy()
    display.insert(0, "name", [names.get(t, t) for t in display.index])
    cols = ["name", "pre_return", "event_return", "post_return", "event_abnormal_return", "post_abnormal_return",
            "pre_volatility", "event_volatility", "post_volatility", "volatility_ratio_event_to_pre",
            "recovered", "recovery_sessions"]
    pct_fmt = {c_: st.column_config.NumberColumn(c_.replace("_", " ").capitalize(), format="percent")
               for c_ in cols if c_.endswith(("_return", "_volatility"))}
    pct_fmt["volatility_ratio_event_to_pre"] = st.column_config.NumberColumn("Vol ratio (event / pre)", format="%.2f")
    pct_fmt["recovery_sessions"] = st.column_config.NumberColumn("Recovery (sessions)")
    pct_fmt["name"] = "Company"
    pct_fmt["recovered"] = "Recovered"
    st.dataframe(display[[c_ for c_ in cols if c_ in display.columns]], width="stretch", column_config=pct_fmt)
    st.caption("Volatility is annualised from daily returns inside each window. Recovery counts sessions after the "
               "event window until the close regains its pre-event level; blank means not within the search horizon. "
               "These are observed differences around the date, not causal effects of the event.")
    footer()
