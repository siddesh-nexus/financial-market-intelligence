"""Visual tokens shared by charts and custom HTML.

Colours come from a validated categorical palette (colour-vision-deficiency safe on adjacent
pairs; see docs/RESEARCH.md). Each mode has its own selected steps. They are not
automatically inverted.

Rules:
- Colour follows the entity, never its rank. A ticker keeps its slot while it stays selected.
- The benchmark is always drawn in neutral ink, dashed.
- Up/down (polarity) uses the reserved status green/red and always comes with a +/- sign.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"],
}
MAX_SERIES = len(SERIES["light"])


@dataclass(frozen=True)
class Tokens:
    mode: str
    surface: str
    page: str
    ink: str
    ink_2: str
    muted: str
    grid: str
    axis: str
    border: str
    up: str
    down: str
    warning: str
    serious: str
    critical: str
    overflow: str        # colour for series beyond the eight palette slots
    benchmark: str
    diverging_neg: str
    diverging_mid: str
    diverging_pos: str
    band_bull: str
    band_bear: str
    band_side: str


LIGHT = Tokens(
    mode="light", surface="#fcfcfb", page="#f6f6f4", ink="#141413", ink_2="#52514e", muted="#898781",
    grid="#e7e6e0", axis="#c3c2b7", border="rgba(11,11,11,0.10)", up="#0a7d0a", down="#c43333",
    warning="#fab219", serious="#ec835a", critical="#d03b3b", overflow="#b9b8b0", benchmark="#3a3936",
    diverging_neg="#d03b3b", diverging_mid="#f0efec", diverging_pos="#256abf",
    band_bull="rgba(12,163,12,0.10)", band_bear="rgba(208,59,59,0.12)", band_side="rgba(137,135,129,0.08)",
)
DARK = Tokens(
    mode="dark", surface="#1a1a19", page="#111110", ink="#f0efec", ink_2="#c3c2b7", muted="#898781",
    grid="#2a2a28", axis="#45453f", border="rgba(255,255,255,0.10)", up="#0ca30c", down="#e66767",
    warning="#fab219", serious="#ec835a", critical="#e05555", overflow="#5c5b56", benchmark="#d8d6cf",
    diverging_neg="#e05555", diverging_mid="#383835", diverging_pos="#3987e5",
    band_bull="rgba(12,163,12,0.14)", band_bear="rgba(230,103,103,0.16)", band_side="rgba(195,194,183,0.07)",
)

SEVERITY_COLOURS = {"Moderate": "warning", "High": "serious", "Extreme": "critical"}


def mode() -> str:
    """'light' or 'dark', following the active Streamlit theme (which follows the OS)."""
    try:
        theme_type = st.context.theme.type
    except Exception:  # outside a script run (tests, bare mode)
        theme_type = None
    return "dark" if theme_type == "dark" else "light"


def tokens() -> Tokens:
    return DARK if mode() == "dark" else LIGHT


def entity_colours(tickers: list[str]) -> dict[str, str]:
    """Stable colour per ticker, based on its position in the global selection order."""
    palette = SERIES[mode()]
    t = tokens()
    order = list(st.session_state.get("tickers", [])) if _has_session() else []
    for ticker in tickers:  # tickers outside the global selection go after it
        if ticker not in order:
            order.append(ticker)
    return {ticker: (palette[i] if i < MAX_SERIES else t.overflow) for i, ticker in enumerate(order)}


def _has_session() -> bool:
    try:
        return st.runtime.exists()
    except Exception:
        return False


# CSS for the few custom HTML elements (header bar, chips, section labels). It uses
# prefers-color-scheme because, with both theme palettes configured and no settings menu,
# Streamlit's own theme follows the operating-system setting too.
CSS = """
<style>
:root {
  --fm-surface: #fcfcfb; --fm-ink: #141413; --fm-ink-2: #52514e; --fm-muted: #7a7973;
  --fm-border: rgba(11,11,11,0.10); --fm-chip: #f0efec; --fm-accent: #2a78d6;
  --fm-good: #0a7d0a; --fm-warn: #a86b00; --fm-bad: #c43333;
}
@media (prefers-color-scheme: dark) {
  :root {
    --fm-surface: #1a1a19; --fm-ink: #f0efec; --fm-ink-2: #c3c2b7; --fm-muted: #9a9890;
    --fm-border: rgba(255,255,255,0.10); --fm-chip: #242422; --fm-accent: #3987e5;
    --fm-good: #2fbf2f; --fm-warn: #fab219; --fm-bad: #f07a7a;
  }
}
.fm-header { display:flex; flex-wrap:wrap; align-items:center; gap:10px 16px;
  padding:8px 12px; margin:0 0 2px 0; background:var(--fm-surface);
  border:1px solid var(--fm-border); border-radius:8px; }
.fm-brand { display:flex; align-items:baseline; gap:10px; min-width:0; }
.fm-mark { font-weight:700; letter-spacing:.04em; color:var(--fm-accent); font-size:.8rem; }
.fm-title { font-weight:600; color:var(--fm-ink); font-size:1.05rem; white-space:nowrap; }
.fm-chips { display:flex; flex-wrap:wrap; gap:6px; margin-left:auto; }
.fm-chip { display:inline-flex; align-items:center; gap:6px; padding:3px 9px; border-radius:999px;
  background:var(--fm-chip); color:var(--fm-ink-2); font-size:.76rem; line-height:1.3;
  border:1px solid var(--fm-border); white-space:nowrap; }
.fm-chip b { color:var(--fm-ink); font-weight:600; }
.fm-dot { width:7px; height:7px; border-radius:50%; display:inline-block; }
.fm-good { background:var(--fm-good); } .fm-warn { background:var(--fm-warn); } .fm-bad { background:var(--fm-bad); }
.fm-section { font-size:.72rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em;
  color:var(--fm-muted); margin:8px 0 2px 0; line-height:1.4; }
.fm-note { font-size:.78rem; color:var(--fm-muted); }
.fm-list { margin:0; padding:0; list-style:none; font-size:.84rem; }
.fm-list li { display:flex; justify-content:space-between; align-items:baseline; gap:8px; padding:4px 0;
  border-bottom:1px solid var(--fm-border); color:var(--fm-ink-2); }
.fm-list li:last-child { border-bottom:none; }
.fm-list li > span:first-child { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.fm-list li > span:last-child { flex:none; }
.fm-list .t { color:var(--fm-ink); font-weight:600; }
.fm-up { color:var(--fm-good); font-variant-numeric: tabular-nums; }
.fm-down { color:var(--fm-bad); font-variant-numeric: tabular-nums; }
.fm-num { font-variant-numeric: tabular-nums; color:var(--fm-ink); }
</style>
"""
