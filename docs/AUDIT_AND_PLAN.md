# Audit and Implementation Plan (v0.2 redesign)

Date of audit: 24 Nov 2026. Baseline: v0.1.0, 49 passing tests, SQLite dev database with
30 tickers × 2,236 sessions (2018-01-02 to 2026-11-23) and 6,576 stored anomaly rows.
A copy of the v0.1 tree (without `.venv`) was backed up before editing.

## 1. What already works (kept)

- **Layering.** `fmip/analysis/*` holds pure pandas functions with no database or UI code, and
  `app/` only renders. This separation is good and is kept.
- **Financial formulas.** Sharpe, Sortino (full-sample downside deviation), historical and
  parametric VaR, CVaR, max drawdown with peak, trough and recovery dates, skew and kurtosis are
  correct and unit-tested.
- **Anomaly engine.** The rolling z-score uses a *shifted* baseline, so a day is never part of
  its own baseline (no look-ahead). Isolation Forest runs on standardised features.
- **Ingestion.** Per-ticker failure isolation, retries, a backfill and incremental planner, and
  idempotent upserts that work on both PostgreSQL and SQLite.
- **Cleaning.** Duplicates are removed, invalid rows dropped, and forward-fill is capped at 2
  sessions with a cleaning report.

## 2. Problems found

### Data and correctness

| # | Problem | Effect | Fix |
|---|---|---|---|
| D1 | **Corporate actions after ingestion are never re-applied.** Yahoo back-adjusts *all* history when a split or dividend occurs, but incremental updates only rewrite the last 5 days. | A split after the first load would leave an artificial −50% to −95% one-day "return" in stored data. Each new dividend leaves a small step in `adj_close`. | On every incremental fetch, compare the overlap with stored values and check the split/dividend columns. On mismatch, re-download the full history for that ticker. |
| D2 | **Incomplete sessions can be stored.** Ingesting while an exchange is open stores a partial bar for today. | That day shows a wrong close and volume until the next run. | Drop bars for the current session unless the exchange (by its own time zone) has closed. |
| D3 | **Mixed exchange calendars.** `returns_panel` computes `pct_change` on the *union* calendar, so the first return after a market-specific holiday becomes NaN and is lost. Cumulative charts divide by `panel.iloc[0]`, which is NaN for any ticker without data on the first date. | NSE and NYSE stocks together give missing returns or a blank line. | Compute returns per asset on its own trading calendar, then align. Rebase charts at the first *common* date and report late starters. |
| D4 | **Price-return index vs total-return stocks.** `^GSPC` is a price index (`adj_close == close`), while stocks use dividend-adjusted closes. | Stocks look better than the benchmark by roughly the dividend yield (about 1–2% a year). | Add a global *return basis* switch (total return vs price return), defaulting to price return when a benchmark is compared, with clear labelling. |
| D5 | Upserting a stock without metadata overwrote a known company name with `NULL`. | Names found by search would be lost. | `COALESCE` in the upsert. |
| D6 | No visible data-quality checks in the UI, and no staleness indicator. | Users cannot tell whether a big number comes from the market or from the data. | New `fmip/data/quality.py` with checks for duplicates, gaps, stale runs, extreme moves, unadjusted-split signatures, adjustment-factor jumps and staleness, plus a freshness badge in the header. |

**Large cumulative returns, investigated.** The largest stored daily moves are ORCL +36% on 10 Sep
2025, NVDA +24% on 25 May 2023, UNH −22% on 17 Apr 2025, NKE −20% on 28 Jun 2024, and BA ±24% in
March 2020. All of these are documented earnings or COVID moves. No adjustment-factor jumps were
found, and the GE spin-offs are applied by Yahoo as split factors, so GE has no artificial drop.
NVDA +31.5% on 23 Nov 2026 comes after the −14% on 20 Nov, and cannot be checked against an
independent source here (Stooq's CSV endpoint now requires a JavaScript challenge). The quality
panel flags it as "extreme move – verify". The large five-year returns (GE +417%, NVDA +407%) are
real price histories.

### UI and UX

- Large `st.title` with no context. Nothing shows the period, benchmark, data source or freshness.
- The sidebar holds a 29-item multiselect showing 8 pre-selected chips, which wastes space. There
  is no search, no free ticker entry, no watchlists, a fixed benchmark and only a raw date picker.
- Every page calls `sidebar_controls()` itself. It works because widget keys are implicit, but
  state handling is fragile.
- Deploy, Rerun, Print and Record controls are visible (no `.streamlit/config.toml`).
- `plotly_white` is hard-coded, so charts stay white in dark mode. The palette is cycled, not fixed
  per entity.
- Metrics often have no context (for example "Anomalies in last 30 days" ignores the chosen
  period).

### Engineering

- Stored anomalies exist only for tickers that `run_analysis.py` processed. Any new ticker would
  have none, and parameters cannot be tuned in the UI.
- The per-ticker freshness lookup costs one query per ticker, with no aggregated query.
- `analytical_metrics` is unused (kept for future scheduled metrics).

## 3. Research summary (details in `docs/RESEARCH.md`)

- **Data source.** Keep Yahoo Finance via `yfinance` as the primary source. It is the only free
  option here with multi-exchange daily OHLCV, split and dividend adjustment, and symbol search.
  Alpha Vantage's free tier is limited to 25 requests a day and puts the adjusted daily series
  behind premium. Nasdaq Data Link's free WIKI prices have been frozen since 2018, and current EOD
  tables are premium. Stooq's CSV download is behind a browser challenge. Yahoo data is
  unofficial and meant for personal or research use, which is documented in the README.
- **Interaction patterns kept from professional tools** (TradingView, Koyfin, Yahoo Finance): a
  compact context bar (symbol set, benchmark, period, as-of), preset period chips (1M to 5Y and
  Max), percentage-rebased comparison charts, leader and laggard tables, breadth (percentage above
  the 50/200-day MA, advance/decline), relative-strength lines, and watchlists.

## 4. Plan

1. **UI shell.** Add `.streamlit/config.toml` with `toolbarMode = "minimal"`, light and dark
   themes and compact padding. Add a compact context header, sidebar navigation in sections, and
   global controls (market, search, selection, benchmark, period, return basis, watchlists) held
   in `st.session_state` so they persist across pages. Use one fixed palette that follows each
   entity, selected separately for light and dark.
2. **Dynamic selection.** Add `fmip/data/symbols.py` (Yahoo search, sector normalisation,
   exchange inference), on-demand ingestion of missing tickers and backfill ranges, and a
   `watchlists` table.
3. **Analytics.** Add `returns.py` (calendar-safe returns and rebasing), `relative.py` (relative
   strength, excess returns, rolling beta), `breadth.py`, `regimes.py` (rule-based and Gaussian
   Mixture), and `portfolio.py` (weights, rebalancing, diversification, risk contribution,
   tracking error). Extend events (market-adjusted returns and paths), risk (beta, rolling
   metrics), clustering (nearest peers) and anomalies (live, tunable, severity and explanations).
4. **Data quality.** Apply fixes D1–D6 and add tests for each.
5. **Validation.** Render every page headlessly with different selections, benchmarks and periods,
   including missing tickers. Take screenshots, then update the README, METHODOLOGY and RESEARCH
   docs.
