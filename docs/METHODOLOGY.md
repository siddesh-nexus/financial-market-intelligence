# Methodology and Assumptions

This document records how every metric in the platform is calculated, so results can be checked and
reproduced. All statistics describe historical data. None of them is a forecast, a trading signal,
or evidence of intent.

## 1. Data

| Item | Choice | Why |
|---|---|---|
| Source | Yahoo Finance via `yfinance` (unofficial API) | Free daily OHLCV for many exchanges, with split and dividend adjustment and symbol search |
| Frequency | Daily sessions | Matches the depth of history needed for risk and event work |
| `Close` | Split-adjusted close | Used for candles, moving averages and **price-return** mode |
| `Adj Close` | Split- and dividend-adjusted close | Used in **total-return** mode and by the anomaly engine |
| Return basis | User choice: price return (default) or total return | Benchmark indices such as `^GSPC` are price indices (`Adj Close == Close`). Comparing total-return stocks with a price index flatters the stocks by roughly the dividend yield. |
| Volume | Raw `Volume` | Adjusted volume is not provided consistently |
| Time zone | Converted to the exchange's calendar dates (naive) | Each bar keeps its own exchange date |

**Cleaning rules** (`fmip/data/validation.py`):

1. Required columns `Open, High, Low, Close, Volume` must exist; otherwise the dataset is rejected.
2. Duplicate dates are removed; the last occurrence is kept.
3. Rows with non-positive or non-finite prices, or with `High < Low`, are **dropped**. Since v0.2 no
   price is forward-filled (v0.1 filled up to 2 sessions). A missing bar stays missing, and the next
   return spans the gap. No observation is fabricated.
4. Negative or missing volume is set to 0 and counted in the cleaning report. Zero-volume days are
   surfaced by the data-quality checks.

**Caching and incremental updates** (`fmip/data/service.py`):

- The database is the cache. For each ticker the loader reads the stored earliest and latest dates.
- **Backfill:** if history starts more than 7 days after the requested start, the earlier range is
  downloaded once. The request is recorded in `stocks.history_requested_from`, so a listing that
  starts later than requested is not requested again.
- **Incremental:** if the latest stored date is behind the requested end, the loader fetches forward
  from `latest - 5 days`, so recently revised bars are refreshed.
- **Corporate actions:** Yahoo back-adjusts the *whole* history when a split or dividend occurs. Each
  incremental download is compared with the stored overlap rows (`close` and `adj_close`, tolerance
  0.01%). On a mismatch, or on a split or dividend with no overlap to compare, the ticker's full history
  is re-downloaded. Every stored bar then uses one adjustment basis. Without this step, a split after
  the first load would appear as an artificial one-day crash.
- **Unfinished sessions:** a bar dated today (in the exchange's own time zone) is dropped until 30
  minutes after that exchange's regular close (e.g. 16:00 New York, 15:30 India), so intraday partial
  bars are never stored as daily closes.
- **Dashboard:** symbols the user adds are downloaded on demand, with at least five years of history so
  rolling windows have warm-up data. Stored symbols are updated only when the user presses *Refresh
  prices*, which avoids provider calls on every interaction.
- Each ticker is handled separately, so one failing symbol does not stop a batch. Metadata (name,
  sector, exchange) is written only after prices were stored, so mistyped symbols leave no rows, and
  empty values never overwrite stored metadata (`COALESCE` upsert).

**Mixed exchange calendars** (`fmip/analysis/returns.py`): price panels use the union of all
symbols' trading dates, with NaN where a market was closed. Returns are computed per asset on **its
own** calendar (from its previous available close), then aligned. Comparisons are rebased at the
first **common** date (or each asset's own start, if the user chooses). Nothing is forward-filled for
analysis. Charts connect gaps visually. Returns are in local currency, with no FX conversion; the
dashboard warns when a selection spans several currencies.

**Known data limitations**

- The default universe is a fixed list of current large-cap names. A study of long history on this
  list has **survivorship bias**: companies that fell out of the index or failed are not included.
- Yahoo Finance data is unofficial and occasionally revised or incomplete. Compare key figures
  against a primary source before citing them.

## 2. Return and volatility conventions

| Metric | Formula |
|---|---|
| Daily return | `r_t = P_t / P_{t-1} - 1` (simple) |
| Log return | `ln(P_t / P_{t-1})` (available, used for reference) |
| Cumulative return | `P_t / P_0 - 1` |
| Rolling N-session return | `P_t / P_{t-N} - 1` |
| Annualised volatility | `std(r, ddof=1) * sqrt(252)` |
| Moving average (N) | Simple mean of the last N closes |

- **252** trading days per year is used throughout.
- Moving averages use only past data at each date.

## 3. Drawdown

`DD_t = P_t / max(P_0..P_t) - 1`. Maximum drawdown is `min(DD_t)`, always zero or negative.
The peak date is the last high before the trough. The recovery date is the first later date where
the price is at or above that peak. The recovery date is blank if it has not happened.

## 4. Distribution statistics

- **Skewness**: sample skewness with bias correction (`scipy.stats.skew(bias=False)`).
- **Excess kurtosis**: Fisher definition, so a normal distribution scores 0
  (`scipy.stats.kurtosis(fisher=True, bias=False)`).

## 5. Risk metrics (`fmip/analysis/risk.py`)

**Assumptions**

- Simple daily returns, annualised with 252 sessions.
- Risk-free rate `rf` is an **annual** rate, set in `.env` (`RISK_FREE_RATE`, default 4%) or in
  the dashboard. The daily rate is `rf / 252` (simple convention, not compounded).
- Confidence level for VaR and CVaR is user-chosen (90%, 95% or 99%; default 95%).
- One-day horizon. Losses are reported as **positive** numbers (a 0.016 VaR means a 1.6% loss).
- Minimum of 20 returns per series.

| Metric | Definition |
|---|---|
| Annualised return | Geometric: `(prod(1 + r))^(252/n) - 1` |
| Sharpe ratio | `mean(r - rf/252) / std(r - rf/252) * sqrt(252)` |
| Sortino ratio | `mean(r - rf/252) / sqrt(mean(min(r - rf/252, 0)^2)) * sqrt(252)` |
| Historical VaR | `-quantile(r, 1 - c)` |
| Parametric VaR | `-(mean(r) + z * std(r))`, `z = Phi^-1(1 - c)`; a normal-distribution comparison |
| CVaR (expected shortfall) | `-mean(r | r <= quantile(r, 1 - c))` |

Historical VaR does not assume a distribution, but it depends on the sample. Fat tails
(large positive excess kurtosis) mean the parametric VaR can understate tail losses. The
platform shows both so the difference is visible.

## 6. Correlation and sectors (`fmip/analysis/correlation.py`)

- Correlations use **daily simple returns**. Pearson measures linear co-movement; Spearman uses
  ranks and is less sensitive to outliers.
- Matrices use pairwise-complete observations, requiring at least 60 overlapping sessions.
- Rolling correlation: Pearson over a trailing window (default 60 sessions).
- "Strongest pairs" lists pairs at or above the positive threshold (default 0.7) or at or below the
  negative threshold (default -0.3). **Correlation in a sample does not imply a causal link,
  shared exposure, or that the relationship will continue.**
- Sector returns are **equal-weighted** averages of member-stock daily returns. Sector labels are
  approximate, GICS-style groupings defined in `fmip/universe.py`. For searched symbols, they come from
  Yahoo's sector field, mapped to the same labels (e.g. "Financial Services" → "Financials").
- **Sector vs benchmark:** sector total return minus the benchmark's return over the same sessions.
  This describes the *selected* stocks only, not the official sector index.
- **Ordering:** the heatmap can be ordered by average-linkage hierarchical clustering on the distance
  `1 - rho`, which places co-moving assets next to each other.
- **Interpretation bands** (shown in the UI): |rho| ≥ 0.8 very strong, ≥ 0.6 strong, ≥ 0.4 moderate,
  ≥ 0.2 weak, otherwise little linear relationship. For periods with fewer than 60 overlapping
  returns, the matrix uses a minimum of 15 and shows a small-sample warning.
- Daily correlations between markets in different time zones (e.g. NSE and NYSE) are understated
  because the markets close at different times (asynchronous trading).

## 7. Anomaly detection (`fmip/analysis/anomalies.py`)

### 7a. Statistical: rolling z-score

For each signal `x_t`:

- Baseline mean and standard deviation use the previous `W` observations only
  (`x_{t-W}..x_{t-1}`; `W` defaults to 60, minimum 20). The current observation is never part of its
  own baseline, so the test does not look ahead.
- `z_t = (x_t - mean_baseline) / std_baseline`. A day is flagged when `|z_t| >= threshold`
  (default 3.0).

Four signals are tested:

| Category | Signal `x_t` | What it captures |
|---|---|---|
| `return_zscore` | Daily simple return | Unusual price moves |
| `price_range_zscore` | `(High - Low) / Close` | Unusual intraday movement |
| `volume_zscore` | `ln(1 + Volume)` | Unusually high or low trading volume |
| `volatility_spike` | Annualised 20-day rolling volatility | Sudden jumps in volatility |

Observed and baseline values are reported on the original scale (volume uses `expm1` to convert back).

The rolling z-score is also the "rolling-window threshold": the flag limits are
`baseline ± threshold * spread`.

Statistical anomalies are **rare-event flags relative to recent history**. With `|z| >= 3`,
about 0.3% of observations would be flagged if returns were normal, and more when returns are
fat-tailed. Flags are not confirmed events.

### 7b. Machine learning: Isolation Forest

- Features per day: daily return, absolute return, change in log volume, intraday range,
  20-day volatility. Features are standardised (zero mean, unit variance) on the full sample.
- Model: `IsolationForest(n_estimators=200, contamination=c, random_state=42)`.
  `c` is the expected share of flagged days (default 1%).
- Each flagged row stores a score where higher means more anomalous.

Isolation Forest scores the full sample at once, so the model can use future rows to set the scale.
It is useful for multivariate outliers but is not a real-time, look-ahead-free detector.

### 7c. Output schema

Each record: ticker, detection date, category, method (`zscore_rolling` or `isolation_forest`),
observed value, baseline value, anomaly score, and a JSON `details` field with the parameters used.

### 7d. Severity and explanations (dashboard)

- **Severity** (heuristic bands): z-score |z| 3–4 *Moderate*, 4–6 *High*, ≥ 6 *Extreme*. Isolation
  Forest score < 0.05 *Moderate*, 0.05–0.10 *High*, > 0.10 *Extreme*.
- **Explanation:** a sentence built from the stored values, e.g. "Daily return +31.48% against a
  60-day average of −0.94%, σ 3.03%; z = +10.7".
- The dashboard runs detection **live** on each stock's full stored history, so baselines are warm at
  the period start. The user can tune the threshold, window and contamination. Results are cached
  per ticker and invalidated when a new session is stored. `scripts/run_analysis.py` still writes the
  default-parameter results to the `anomalies` table for other consumers.

## 8. Behavioural clustering (`fmip/analysis/clustering.py`)

Features per stock (requires at least 60 sessions):

- Average annual return (mean daily return × 252)
- Annualised volatility
- Maximum drawdown
- Log of average daily volume
- Skewness and excess kurtosis of daily returns

Method:

1. Standardise features (z-scores across stocks).
2. K-means (`n_init=10`, `random_state=42`).
3. Choose k by highest **silhouette score** over k = 2..6 (shown in the dashboard; the user can override).
4. PCA to two components for display. Explained variance is shown.

Silhouette ranges from -1 to 1. Values near 0 mean clusters overlap, so the groups are weak.
With 29 stocks, groups are sensitive to the feature choice and the sample period.

## 9. Event-window analysis (`fmip/analysis/events.py`)

For event date `E` (first trading session on or after `E` is the event day):

| Window | Definition |
|---|---|
| Pre | `pre_days` sessions ending the day before the event (default 30) |
| Event | `event_days` sessions starting on the event day (default 5) |
| Post | `post_days` sessions after the event window (default 30) |
| Recovery | Sessions after the event window until the adjusted close is at or above the last pre-event close (search limit 250 sessions) |

Reported: window returns, annualised volatility in each window, the event-to-pre volatility ratio,
and recovery sessions. Missing windows (insufficient history) are shown as `n/a`, not estimated.

- **Abnormal (market-adjusted) return** = stock window return − benchmark window return over the same
  window. This is the simplest event-study baseline. It does not adjust for beta. A market-model
  version (estimating alpha and beta in a clean estimation window) is a possible extension.
- **Event paths:** each series is rebased to 100 at the last close before the event and plotted
  against session offset (0 = first session on or after the date), counted on its own calendar.
- If stored history does not reach the event, the page offers a one-click download of the earlier
  range for the selected symbols.

**Causation caveat.** These are observed differences in the same period. Other news, earnings,
rates and market-wide moves happen in the same windows. Event-window results do not show that the
named event caused any movement. The only causal claim the platform supports is a description of
what was observed.

Default event dates are anchors for analysis and should be checked against primary sources before
publication:

| Event | Anchor date |
|---|---|
| Lehman Brothers bankruptcy | 2008-09-15 |
| Brexit referendum result | 2016-06-24 |
| COVID-19 market crash start | 2020-02-24 |
| Russia invades Ukraine | 2022-02-24 |
| First Fed hike of the 2022 cycle | 2022-03-16 |
| Silicon Valley Bank failure | 2023-03-10 |
| US "reciprocal" tariff announcement | 2025-04-03 (first session after the 2 Apr announcement) |

Users can add and save custom events (stored in `market_events`).

## 10. Relative strength (`fmip/analysis/relative.py`)

All calculations use dates on which both the stock and the benchmark traded.

| Metric | Definition |
|---|---|
| RS line | `RS_t = (P_t / P_0) / (B_t / B_0)`. Above 1 means outperformance since the period start; a rising line means current outperformance. |
| Geometric excess | `RS_T - 1 = (1 + R_s) / (1 + R_b) - 1` over the whole period |
| Trailing excess return | `R_s(n) - R_b(n)` over the last `n` sessions: 1W = 5, 1M = 21, 3M = 63, 6M = 126, 1Y = 252. NaN if fewer than `n + 1` common sessions (never extrapolated). |
| Rolling relative return | `R_s(w) - R_b(w)` over a trailing window `w` |
| Beta | `Cov(r_s, r_b) / Var(r_b)` of daily returns; rolling beta uses a trailing window (≥ 63 sessions) |
| Trend status | Outperforming / underperforming (RS ≥ 1 or < 1) × improving / weakening (RS above or below its 63-session average) |

## 11. Market breadth (`fmip/analysis/breadth.py`)

Breadth is computed **only over the selected stocks**, and the dashboard labels it that way.

- Advancers / decliners: stocks with a daily return > 0 / < 0 that session. Unchanged and non-trading
  stocks are counted separately. The A-D line is the cumulative sum of (advancers − decliners).
- % above N-day MA: share of *eligible* stocks (those with at least N prior sessions) closing above
  their own N-session simple moving average. Stocks without enough history are excluded from the
  denominator, not counted as "below". The dashboard loads about 320 extra calendar days so MA200
  exists at the period start.

## 12. Market regimes (`fmip/analysis/regimes.py`)

**Rule-based (no look-ahead).** Every input at date t uses data up to t only.

- Trend: *Bull* if close > MA200 and MA50 > MA200; *Bear* if close < MA200 and MA50 < MA200;
  otherwise *Sideways / transition*. No label until MA200 exists.
- Volatility: 21-session annualised volatility versus the 20th and 80th percentiles of its own
  **past** values (expanding window, shifted by one day, at least 252 observations). *High*, *Low*
  or *Normal*.

**Gaussian Mixture.** Features: 21-session return and 21-session annualised volatility,
standardised. A GMM with 2–4 components (full covariance, 3 initialisations, fixed seed) is fitted on
the window shown. States are named by average volatility (Calm → Stressed), and each session gets its
posterior probability. The fit is **in-sample**, so labels can change when the window changes. GMM was
preferred to K-means because it allows elliptical clusters and gives probabilities.

Per-regime statistics (share of time, annualised mean return and volatility, episode count and
average length) use the **same** sessions as the label, so they describe the regime and are not
forward returns. Regimes are descriptions of the past, not forecasts.

## 13. Portfolio analytics (`fmip/analysis/portfolio.py`)

- Long-only weights, scaled to sum to 1. Negative weights are rejected.
- The portfolio starts on the first date when every holding has a price. Asset returns are computed
  on each asset's own calendar. On a day when a holding's exchange is closed, that holding is carried
  at its last close (a 0% return), as in mark-to-market accounting, and the count of such holding-days
  is shown.
- Rebalancing: *daily* (constant weights), *monthly* (reset on the first session of each month) or
  *none* (buy-and-hold; weights drift).
- Metrics reuse the risk definitions in section 5, applied to portfolio daily returns. Max drawdown
  uses the wealth curve.
- **Diversification:** portfolio volatility `sqrt(w' Σ w)` (annualised covariance); diversification
  ratio `Σ w_i σ_i / σ_p` (1 = no benefit); effective holdings `1 / Σ w_i²`; risk contribution
  `w_i (Σ w)_i / σ_p²`, which sums to 100%; and the average pairwise correlation.
- **Versus benchmark:** beta, correlation, tracking error `std(r_p − r_b) × √252`, information ratio
  `mean(r_p − r_b) × 252 / TE`, and annualised excess return.
- **Limits:** these are hypothetical back-calculations. There are no transaction costs, taxes,
  spreads, cash drag or FX conversion, and the holdings were chosen with hindsight. The results are
  not investable or executable returns.

## 14. Data-quality checks (`fmip/data/quality.py`)

Checks describe the data; they never modify it.

| Check | Severity | Rule |
|---|---|---|
| Duplicate dates | error | More than one bar per date |
| Possible unadjusted split | error | Close-to-close move ≥ 40% whose ratio is within 3% of a common split ratio (2, 3, 4, 5, 8, 10, 15, 20 or the reciprocals) |
| Adjustment-factor jump | error | `adj_close / close` changes by ≥ 5% in one day (mixed adjustment bases) |
| Extreme move | warning | \|daily return\| ≥ 25%. Often genuine (earnings, crises), but worth verifying |
| OHLC inconsistency | warning | High below open/close, or low above open/close |
| Session gap | warning | More than 7 calendar days between consecutive sessions |
| Stale prices | warning | 5 or more identical closes in a row |
| Stale series | warning | Last session more than 3 business days before the period end or today |
| Late start / short history | info | History starts after the period start; fewer than 60 sessions |
| Zero volume | info | Sessions with zero reported volume (not checked for indices) |

The header shows a summary (errors, warnings), and Market Overview lists every note with its first
date.

## 15. Verification

- Unit tests (`tests/`) check the formulas against independent calculations (for example, the Sharpe
  ratio, historical and parametric VaR, CVaR, drawdown examples, and event recovery on a constructed series).
- Cleaning, incremental ingestion and backfill are tested with fake data sources and an
  in-memory database.
- v0.2 adds tests for: own-calendar returns across mismatched holidays; common-start rebasing;
  relative strength, excess returns and beta (a 2× leveraged series has beta 2); breadth eligibility;
  portfolio maths (daily rebalancing = weighted sum, buy-and-hold = value-weighted growth, risk
  contributions sum to 1, self-benchmark gives beta 1 and tracking error 0); no look-ahead in
  volatility regimes (changing the future leaves past labels unchanged); GMM state ordering; every
  data-quality rule; partial-bar removal by exchange time zone; and a simulated split after the first
  load, which must trigger a full re-download that leaves no split signature.
- 116 tests in total, all deterministic and offline.
- A spot check on stored real data reproduced the known S&P 500 2020 drawdown: peak 19 Feb 2020,
  trough 23 Mar 2020, about -33.9%, regained on 18 Aug 2020.
