"""Plotly figure builders. Pages call these; they hold no analytics logic themselves.

Conventions (see app/theme.py): one y-axis per chart, thin 2px lines, entity-stable colours,
benchmark dashed in neutral ink, recessive grid, hover enabled on every chart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from theme import SERIES, SEVERITY_COLOURS, mode, tokens

FONT = "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
PLOT_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def show(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, theme=None, width="stretch", config=PLOT_CONFIG, key=key)


def _style(fig: go.Figure, title: str | None = None, height: int = 340, y_title: str | None = None,
           y_format: str | None = None, hover: str = "x unified", legend: bool = True) -> go.Figure:
    t = tokens()
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=16, t=48 if title else 28, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=t.ink_2),
        title=dict(text=title, x=0, xanchor="left", y=0.98, font=dict(size=13, color=t.ink)) if title else None,
        hovermode=hover,
        hoverlabel=dict(bgcolor=t.surface, bordercolor=t.axis, font=dict(color=t.ink, family=FONT, size=12)),
        legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", font=dict(size=11, color=t.ink_2),
                    bgcolor="rgba(0,0,0,0)", itemclick="toggle", itemdoubleclick="toggleothers"),
        showlegend=legend,
    )
    fig.update_xaxes(showgrid=False, linecolor=t.axis, tickcolor=t.axis, ticks="outside", ticklen=4,
                     zeroline=False, tickfont=dict(color=t.muted), automargin=True)
    fig.update_yaxes(gridcolor=t.grid, zeroline=False, linecolor=t.axis, tickfont=dict(color=t.muted),
                     automargin=True,
                     title=dict(text=y_title, font=dict(size=11, color=t.muted)) if y_title else None)
    if y_format:
        fig.update_yaxes(tickformat=y_format)
    return fig


def empty(message: str, height: int = 200) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(color=tokens().muted, size=13))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, height=height, legend=False)


# ---- performance -----------------------------------------------------------------------------
def performance_lines(rebased: pd.DataFrame, colours: dict[str, str], benchmark: str | None,
                      names: dict[str, str], as_index: bool = False, title: str | None = None,
                      height: int = 380) -> go.Figure:
    """Rebased lines. `rebased` is 1.0 at the start. Shows % change or an index (=100)."""
    t = tokens()
    values = rebased * 100 if as_index else (rebased - 1.0) * 100
    fig = go.Figure()
    for column in values.columns:
        is_bench = column == benchmark
        series = values[column]
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, name=f"{column} (benchmark)" if is_bench else column,
            mode="lines", connectgaps=True,
            line=dict(color=t.benchmark if is_bench else colours.get(column, t.overflow),
                      width=2.2 if is_bench else 1.8, dash="dash" if is_bench else "solid"),
            customdata=np.full(len(series), names.get(column, column)),
            hovertemplate=("%{y:.1f}" if as_index else "%{y:+.1f}%") + " · %{customdata}<extra>" + column + "</extra>",
        ))
    if as_index:
        fig.add_hline(y=100, line=dict(color=t.axis, width=1, dash="dot"))
    else:
        fig.add_hline(y=0, line=dict(color=t.axis, width=1, dash="dot"))
    return _style(fig, title, height, "Index (start = 100)" if as_index else "Change since start (%)",
                  y_format=None if as_index else "+.0f")


def lines(frame: pd.DataFrame, title: str | None, y_title: str | None, colours: dict[str, str] | None = None,
          percent: bool = False, zero_line: bool = False, ref_line: float | None = None, height: int = 300,
          benchmark: str | None = None, value_format: str | None = None) -> go.Figure:
    t = tokens()
    palette = SERIES[mode()]
    fig = go.Figure()
    scale = 100 if percent else 1
    fmt = value_format or ("%{y:+.1f}%" if percent else "%{y:.2f}")
    for i, column in enumerate(frame.columns):
        is_bench = column == benchmark
        colour = t.benchmark if is_bench else (colours or {}).get(column, palette[i % len(palette)] if i < len(palette) else t.overflow)
        series = frame[column].dropna()
        fig.add_trace(go.Scatter(x=series.index, y=series.values * scale, name=str(column), mode="lines",
                                 line=dict(color=colour, width=1.8, dash="dash" if is_bench else "solid"),
                                 hovertemplate=fmt + f"<extra>{column}</extra>"))
    if zero_line:
        fig.add_hline(y=0, line=dict(color=t.axis, width=1, dash="dot"))
    if ref_line is not None:
        fig.add_hline(y=ref_line, line=dict(color=t.axis, width=1, dash="dot"))
    return _style(fig, title, height, y_title, legend=frame.shape[1] > 1)


def drawdown_lines(drawdowns: pd.DataFrame, colours: dict[str, str], benchmark: str | None = None,
                   title: str | None = "Drawdown from running peak", height: int = 300) -> go.Figure:
    t = tokens()
    fig = go.Figure()
    single = drawdowns.shape[1] == 1
    for column in drawdowns.columns:
        series = drawdowns[column].dropna() * 100
        is_bench = column == benchmark
        colour = t.benchmark if is_bench else colours.get(column, t.overflow)
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, name=column, mode="lines",
            line=dict(color=colour, width=1.6, dash="dash" if is_bench else "solid"),
            fill="tozeroy" if single else None,
            fillcolor=t.band_bear if single else None,
            hovertemplate="%{y:.1f}%<extra>" + column + "</extra>"))
    return _style(fig, title, height, "Drawdown (%)", legend=not single)


# ---- price -----------------------------------------------------------------------------------
def candlestick(ohlcv: pd.DataFrame, ticker: str, mas: pd.DataFrame | None = None,
                anomalies: pd.DataFrame | None = None, volume_ma: int = 20, height: int = 520) -> go.Figure:
    """Candles (split-adjusted OHLC) with optional moving averages and anomaly markers, plus a
    volume panel with its moving average. The two panels share the x-axis but each has its own
    y-axis because they measure different things (not a dual-axis chart)."""
    t = tokens()
    palette = SERIES[mode()]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=ohlcv.index, open=ohlcv["Open"], high=ohlcv["High"], low=ohlcv["Low"], close=ohlcv["Close"],
        name=ticker, increasing=dict(line=dict(color=t.up, width=1), fillcolor=t.up),
        decreasing=dict(line=dict(color=t.down, width=1), fillcolor=t.down), showlegend=False,
    ), row=1, col=1)
    ma_colours = [palette[0], palette[1], palette[6], palette[3]]
    for i, column in enumerate((mas if mas is not None else pd.DataFrame()).columns):
        fig.add_trace(go.Scatter(x=mas.index, y=mas[column], name=column.replace("_", " "), mode="lines",
                                 line=dict(width=1.4, color=ma_colours[i % len(ma_colours)]),
                                 hovertemplate="%{y:.2f}<extra>" + column.replace("_", " ") + "</extra>"),
                      row=1, col=1)
    if anomalies is not None and not anomalies.empty:
        marks = anomalies.drop_duplicates("detection_date").set_index("detection_date")
        marks = marks[marks.index.isin(ohlcv.index)]
        if not marks.empty:
            fig.add_trace(go.Scatter(
                x=marks.index, y=ohlcv.loc[marks.index, "High"] * 1.025, mode="markers", name="Anomaly",
                marker=dict(symbol="triangle-down", size=9, color=t.serious, line=dict(color=t.surface, width=1)),
                customdata=marks[["signal", "severity"]].values,
                hovertemplate="%{customdata[0]} · %{customdata[1]}<extra>Anomaly</extra>"), row=1, col=1)
    volume_colours = np.where(ohlcv["Close"] >= ohlcv["Open"], t.up, t.down)
    fig.add_trace(go.Bar(x=ohlcv.index, y=ohlcv["Volume"], marker_color=volume_colours, marker_line_width=0,
                         opacity=0.55, name="Volume", showlegend=False,
                         hovertemplate="%{y:.3s}<extra>Volume</extra>"), row=2, col=1)
    vol_ma = ohlcv["Volume"].rolling(volume_ma, min_periods=volume_ma).mean()
    fig.add_trace(go.Scatter(x=ohlcv.index, y=vol_ma, name=f"Volume {volume_ma}d avg", mode="lines",
                             line=dict(color=t.ink_2, width=1.2),
                             hovertemplate="%{y:.3s}<extra>" + f"{volume_ma}d avg</extra>"), row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, bargap=0.1)
    _style(fig, None, height, None, hover="x unified")
    fig.update_yaxes(title=dict(text="Price", font=dict(size=11, color=t.muted)), row=1, col=1)
    fig.update_yaxes(title=dict(text="Volume", font=dict(size=11, color=t.muted)), tickformat=".2s", row=2, col=1)
    return fig


def histogram(returns: pd.Series, title: str, var: float | None = None, cvar: float | None = None,
              height: int = 300) -> go.Figure:
    t = tokens()
    fig = go.Figure(go.Histogram(x=returns * 100, nbinsx=70, marker=dict(color=SERIES[mode()][0],
                                 line=dict(color=t.surface, width=1)),
                                 hovertemplate="%{x}%: %{y} days<extra></extra>", name="Daily returns"))
    if var is not None:
        fig.add_vline(x=-var * 100, line=dict(color=t.serious, width=1.5, dash="dash"),
                      annotation_text=f"VaR {var:.1%}", annotation_font_color=t.ink_2)
    if cvar is not None:
        fig.add_vline(x=-cvar * 100, line=dict(color=t.critical, width=1.5, dash="dot"),
                      annotation_text=f"CVaR {cvar:.1%}", annotation_position="bottom left",
                      annotation_font_color=t.ink_2)
    fig.update_xaxes(title=dict(text="Daily return (%)", font=dict(size=11, color=t.muted)))
    return _style(fig, title, height, "Sessions", hover="closest", legend=False)


# ---- comparisons -----------------------------------------------------------------------------
def ranked_bars(values: pd.Series, title: str | None, percent: bool = True, signed: bool = True,
                height: int | None = None, colour: str | None = None, value_format: str | None = None,
                names: dict[str, str] | None = None) -> go.Figure:
    """Horizontal bars sorted by value; signed values use up/down colours plus a sign."""
    t = tokens()
    data = values.dropna().sort_values()
    scaled = data * 100 if percent else data
    if signed:
        colours = [t.up if v >= 0 else t.down for v in scaled]
    else:
        colours = colour or SERIES[mode()][0]
    fmt = value_format or ("{:+.1f}%" if (percent and signed) else "{:.1f}%" if percent else "{:.2f}")
    text = [fmt.format(v) for v in scaled]
    custom = [names.get(i, i) for i in data.index] if names else list(data.index)
    fig = go.Figure(go.Bar(x=scaled.values, y=list(data.index), orientation="h", marker_color=colours,
                           marker_line_width=0, text=text, textposition="outside", cliponaxis=False,
                           textfont=dict(color=t.ink_2, size=11), customdata=custom,
                           hovertemplate="%{customdata}: %{text}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=t.axis, width=1))
    span = float(scaled.abs().max()) if len(scaled) else 1.0
    lo = min(0.0, float(scaled.min())) if len(scaled) else 0.0
    hi = max(0.0, float(scaled.max())) if len(scaled) else 1.0
    pad = span * 0.22 or 1.0   # room for the outside value labels
    _style(fig, title, height or max(160, 28 * len(data) + 60), None, hover="closest", legend=False)
    fig.update_xaxes(showticklabels=False, showgrid=False, range=[lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0)])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=t.ink_2), ticksuffix="  ")
    return fig


def grouped_bars(frame: pd.DataFrame, title: str | None, percent: bool = True, height: int = 320,
                 y_title: str | None = None) -> go.Figure:
    """One bar group per row, one bar per column (columns take categorical slots in order)."""
    palette = SERIES[mode()]
    fig = go.Figure()
    scale = 100 if percent else 1
    for i, column in enumerate(frame.columns):
        fig.add_trace(go.Bar(x=list(frame.index), y=frame[column] * scale, name=str(column).replace("_", " "),
                             marker_color=palette[i % len(palette)], marker_line_width=0,
                             hovertemplate=("%{y:+.1f}%" if percent else "%{y:.2f}") + f"<extra>{column}</extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    return _style(fig, title, height, y_title, hover="x unified")


def heatmap(matrix: pd.DataFrame, title: str | None, height: int | None = None, zmin: float = -1, zmax: float = 1,
            percent: bool = False) -> go.Figure:
    t = tokens()
    n = matrix.shape[0]
    scale = [[0.0, t.diverging_neg], [0.5, t.diverging_mid], [1.0, t.diverging_pos]]
    show_text = n <= 16
    text = np.where(np.isnan(matrix.values), "", np.vectorize(lambda v: f"{v:.0%}" if percent else f"{v:.2f}")(
        np.nan_to_num(matrix.values)))
    fig = go.Figure(go.Heatmap(
        z=matrix.values, x=list(matrix.columns), y=list(matrix.index), zmin=zmin, zmax=zmax, colorscale=scale,
        text=text if show_text else None, texttemplate="%{text}" if show_text else None,
        textfont=dict(size=10, color=t.ink), xgap=2, ygap=2,
        colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(color=t.muted, size=10)),
        hovertemplate="%{y} · %{x}: %{z:.2f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-45 if n > 8 else 0)
    return _style(fig, title, height or max(320, 34 * n + 90), None, hover="closest", legend=False)


def scatter_labeled(frame: pd.DataFrame, x: str, y: str, title: str | None, x_title: str, y_title: str,
                    colours: dict[str, str] | None = None, groups: pd.Series | None = None,
                    x_percent: bool = False, y_percent: bool = False, height: int = 380,
                    highlight: str | None = None) -> go.Figure:
    """Labelled scatter. Colour by `groups` (categorical slots) or by entity `colours`."""
    t = tokens()
    palette = SERIES[mode()]
    fig = go.Figure()
    xs = frame[x] * (100 if x_percent else 1)
    ys = frame[y] * (100 if y_percent else 1)
    fmt_x = "%{x:.1f}%" if x_percent else "%{x:.2f}"
    fmt_y = "%{y:.1f}%" if y_percent else "%{y:.2f}"
    if groups is not None:
        for i, group in enumerate(sorted(groups.unique())):
            members = groups[groups == group].index
            fig.add_trace(go.Scatter(
                x=xs.loc[members], y=ys.loc[members], mode="markers+text", text=list(members),
                textposition="top center", textfont=dict(size=10, color=t.ink_2), name=f"Cluster {group}",
                marker=dict(size=11, color=palette[i % len(palette)], line=dict(color=t.surface, width=2)),
                hovertemplate=f"%{{text}}<br>{x_title}: {fmt_x}<br>{y_title}: {fmt_y}<extra>Cluster {group}</extra>"))
    else:
        for ticker in frame.index:
            colour = (colours or {}).get(ticker, palette[0])
            size = 14 if ticker == highlight else 10
            fig.add_trace(go.Scatter(
                x=[xs.loc[ticker]], y=[ys.loc[ticker]], mode="markers+text", text=[ticker], name=ticker,
                textposition="top center", textfont=dict(size=10, color=t.ink_2), showlegend=False,
                marker=dict(size=size, color=colour, line=dict(color=t.surface, width=2)),
                hovertemplate=f"%{{text}}<br>{x_title}: {fmt_x}<br>{y_title}: {fmt_y}<extra></extra>"))
    fig.update_xaxes(title=dict(text=x_title, font=dict(size=11, color=t.muted)), showgrid=True, gridcolor=t.grid)
    return _style(fig, title, height, y_title, hover="closest", legend=groups is not None)


# ---- anomalies -------------------------------------------------------------------------------
def anomaly_timeline(anomalies: pd.DataFrame, height: int = 360) -> go.Figure:
    """Each flagged day as a point: x = date, y = ticker, colour = severity (status palette),
    marker shape = signal type. Severity is also in the hover text, never colour alone."""
    t = tokens()
    if anomalies.empty:
        return empty("No anomalies in this period with the current settings.")
    symbols = {"Price move": "circle", "Volume surge": "square", "Intraday range": "diamond",
               "Volatility spike": "triangle-up", "Multivariate (Isolation Forest)": "x"}
    fig = go.Figure()
    for severity in ("Moderate", "High", "Extreme"):
        group = anomalies[anomalies["severity"] == severity]
        if group.empty:
            continue
        fig.add_trace(go.Scatter(
            x=group["detection_date"], y=group["ticker"], mode="markers", name=severity,
            marker=dict(size=9 if severity == "Moderate" else 11 if severity == "High" else 13,
                        color=getattr(t, SEVERITY_COLOURS[severity]),
                        symbol=[symbols.get(s, "circle") for s in group["signal"]],
                        line=dict(color=t.surface, width=1), opacity=0.9),
            customdata=group[["signal", "explanation"]].values,
            hovertemplate="<b>%{y}</b> · %{x|%d %b %Y}<br>%{customdata[0]} (" + severity
                          + ")<br>%{customdata[1]}<extra></extra>"))
    n = anomalies["ticker"].nunique()
    fig.update_yaxes(showgrid=True, gridcolor=t.grid, categoryorder="category descending")
    return _style(fig, None, max(height, 30 * n + 80), None, hover="closest")


# ---- regimes ---------------------------------------------------------------------------------
def _episodes(labels: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    clean = labels.dropna()
    if clean.empty:
        return []
    groups = (clean != clean.shift()).cumsum()
    out = []
    for _, block in clean.groupby(groups):
        out.append((block.index[0], block.index[-1], block.iloc[0]))
    return out


def regime_bands(rules: pd.DataFrame, title: str | None, height: int = 420) -> go.Figure:
    """Price with MA50/MA200 and shaded trend regimes (Bull / Bear / Sideways)."""
    t = tokens()
    palette = SERIES[mode()]
    fills = {"Bull": t.band_bull, "Bear": t.band_bear, "Sideways": t.band_side}
    fig = go.Figure()
    for start, end, label in _episodes(rules["trend_regime"]):
        fig.add_vrect(x0=start, x1=end, fillcolor=fills.get(label, t.band_side), line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=rules.index, y=rules["close"], name="Close", mode="lines",
                             line=dict(color=t.ink, width=1.6), hovertemplate="%{y:,.2f}<extra>Close</extra>"))
    fig.add_trace(go.Scatter(x=rules.index, y=rules["ma_fast"], name="MA 50", mode="lines",
                             line=dict(color=palette[0], width=1.3), hovertemplate="%{y:,.2f}<extra>MA 50</extra>"))
    fig.add_trace(go.Scatter(x=rules.index, y=rules["ma_slow"], name="MA 200", mode="lines",
                             line=dict(color=palette[1], width=1.3), hovertemplate="%{y:,.2f}<extra>MA 200</extra>"))
    # Legend keys for the shaded bands (added after the date traces so the axis stays a date axis).
    first = rules.index[0]
    for label, colour in (("Bull", t.up), ("Bear", t.down), ("Sideways", t.muted)):
        fig.add_trace(go.Scatter(x=[first], y=[None], mode="markers", name=f"{label} regime", hoverinfo="skip",
                                 marker=dict(size=10, symbol="square", color=colour, opacity=0.5)))
    fig.update_xaxes(type="date")
    return _style(fig, title, height, "Price")


def volatility_regime(rules: pd.DataFrame, title: str | None, height: int = 260) -> go.Figure:
    t = tokens()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rules.index, y=rules["vol_upper"] * 100, name="80th pct (past only)", mode="lines",
                             line=dict(color=t.serious, width=1, dash="dot"),
                             hovertemplate="%{y:.0f}%<extra>80th pct</extra>"))
    fig.add_trace(go.Scatter(x=rules.index, y=rules["vol_lower"] * 100, name="20th pct (past only)", mode="lines",
                             line=dict(color=t.muted, width=1, dash="dot"),
                             hovertemplate="%{y:.0f}%<extra>20th pct</extra>"))
    fig.add_trace(go.Scatter(x=rules.index, y=rules["volatility"] * 100, name="21-day volatility", mode="lines",
                             line=dict(color=SERIES[mode()][0], width=1.6),
                             hovertemplate="%{y:.1f}%<extra>21d vol</extra>"))
    return _style(fig, title, height, "Annualised volatility (%)")


def gmm_states(frame: pd.DataFrame, order: list[str], title: str | None, height: int = 380) -> go.Figure:
    t = tokens()
    palette = SERIES[mode()]
    state_colours = {"Calm": palette[2], "Normal": palette[0], "Elevated": palette[3], "Stressed": palette[7]}
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.index, y=frame["close"], mode="lines", name="Close", showlegend=False,
                             line=dict(color=t.axis, width=1), hoverinfo="skip"))
    for state in order:
        group = frame[frame["state"] == state]
        fig.add_trace(go.Scatter(
            x=group.index, y=group["close"], mode="markers", name=state,
            marker=dict(size=4, color=state_colours.get(state, palette[4])),
            customdata=np.c_[group["vol_21"] * 100, group["probability"] * 100],
            hovertemplate="%{y:,.2f} · " + state + "<br>21d vol %{customdata[0]:.0f}% · p=%{customdata[1]:.0f}%<extra></extra>"))
    return _style(fig, title, height, "Price", hover="closest")


# ---- events ----------------------------------------------------------------------------------
def event_paths(paths: pd.DataFrame, colours: dict[str, str], benchmark: str | None,
                event_days: int, height: int = 380) -> go.Figure:
    t = tokens()
    fig = go.Figure()
    fig.add_vrect(x0=-0.5, x1=event_days - 0.5, fillcolor=t.band_side, line_width=0, layer="below")
    for column in paths.columns:
        is_bench = column == benchmark
        series = paths[column].dropna()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines",
                                 name=f"{column} (benchmark)" if is_bench else column,
                                 line=dict(color=t.benchmark if is_bench else colours.get(column, t.overflow),
                                           width=2.2 if is_bench else 1.6, dash="dash" if is_bench else "solid"),
                                 hovertemplate="%{y:.1f}<extra>" + column + "</extra>"))
    fig.add_hline(y=100, line=dict(color=t.axis, width=1, dash="dot"))
    fig.add_vline(x=0, line=dict(color=t.ink_2, width=1))
    fig.update_xaxes(title=dict(text="Sessions relative to event (0 = first session on/after the date)",
                                font=dict(size=11, color=t.muted)))
    return _style(fig, None, height, "Index (last pre-event close = 100)")
