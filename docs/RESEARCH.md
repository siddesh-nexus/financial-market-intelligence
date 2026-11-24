# Research Notes

This document explains why the platform measures what it measures, what the methods assume, and
where they break down. Formulas are in [METHODOLOGY.md](METHODOLOGY.md). The audit that led to v0.2 is
in [AUDIT_AND_PLAN.md](AUDIT_AND_PLAN.md).

---

## 1. Data sources

### 1.1 Comparison

| Source | History and coverage | Access limits (checked Nov 2026) | Adjustments | Licensing | Verdict |
|---|---|---|---|---|---|
| **Yahoo Finance via `yfinance`** | Decades of daily OHLCV for US, India (NSE `.NS`, BSE `.BO`), Europe and Asia; indices; symbol search | No published quota. Unofficial endpoints that throttle heavy use and have changed without notice | `Close` is split-adjusted; `Adj Close` is split- and dividend-adjusted; dividend and split events are available | Unofficial; the project README says the data is for personal or research use, and Yahoo's terms apply | **Chosen** for this project |
| Alpha Vantage | Long US history; some global symbols | Free key: 25 requests a day; the daily *adjusted* series is premium-only | Adjusted series on premium | Commercial terms | Too restrictive for multi-stock dashboards on the free tier |
| Nasdaq Data Link (ex-Quandl) | The free `WIKI` US prices dataset stopped updating in 2018; current end-of-day tables (e.g. QuoteMedia, Sharadar) are premium | Free API key with rate limits | Depends on dataset | Per-dataset licences | Good for paid, audited data; not free for current prices |
| Stooq | Broad daily history | The CSV download endpoint now serves a JavaScript proof-of-work challenge, so scripted downloads fail. The project does not try to get around it. | Adjusted closes | Personal use | Not usable programmatically here |
| Exchange data (NSE, NYSE, Nasdaq) | Authoritative | Bhavcopy and end-of-day files exist but formats, access and terms differ by exchange; real-time feeds are licensed | Unadjusted; adjustments must be applied by the user | Exchange licences | Best for production; out of scope |

### 1.2 Decision and limits

Yahoo Finance is the only free source that covers several exchanges, provides both split-adjusted
and total-return closes, and offers name search. The platform treats it as **unofficial,
revisable and not licensed for redistribution**. Consequences built into the design:

- **Adjusted vs unadjusted.** Both are stored. The user chooses price return or total return.
  Indices are price indices, so price return is the default when comparing with a benchmark.
- **Corporate actions.** Yahoo back-adjusts history after splits and dividends, so the platform
  re-downloads a ticker's full history when stored and fresh values disagree (METHODOLOGY §1).
- **Spin-offs.** Yahoo applies some spin-offs (e.g. GE's 2023 and 2024 separations) as split
  factors. The stored GE series has no artificial drop, but the spun-off shares are not included in
  the return.
- **Unfinished sessions.** Bars for sessions that have not closed are not stored.
- **Verification.** Large moves are flagged for checking against a second source. During this
  audit, NVDA's +31.5% on 23 Nov 2026 (after −14.0% on 20 Nov) could not be confirmed independently,
  because Stooq was unreachable and the session is after the knowledge cutoff of the tools used. It is
  kept and flagged, not removed.
- **Commercial use** would need a licensed feed. The fetcher is isolated in `fmip/data/fetcher.py`,
  so a provider can be swapped without touching analytics.

---

## 2. Design references and patterns adopted

Professional tools were studied for interaction patterns, not copied:

| Pattern | Seen in | Adapted here as |
|---|---|---|
| A compact context bar showing symbol set, benchmark, period and as-of time | Koyfin dashboards, Bloomberg launchpads | A one-line header with benchmark, period, stock count, return basis, data freshness, data-check status and source chips |
| Preset period chips (1M … 5Y, Max) plus a custom range | TradingView, Yahoo Finance charts | A segmented period control in the sidebar, applied to every page |
| Percent-rebased comparison charts against an index | TradingView "compare", Koyfin graphs | A Performance chart with *Change %* and *Index = 100* views and common-start alignment |
| Leaders / laggards lists | Yahoo Finance market movers | Five compact ranked lists (return, volatility, drawdown, volume change) |
| Breadth gauges (percentage above MA) | StockCharts, TradingView breadth indicators | Breadth metrics clearly scoped to *your* selection |
| Watchlists | All of the above | Watchlists saved in the database, with load, save and delete |
| Relative-strength lines | StockCharts "price relative" | An RS line per stock and an excess-return table across horizons |

Visual design: a restrained neutral surface, one accent colour, and a **validated categorical
palette** (eight hues, ordered so adjacent series stay distinguishable under common colour-vision
deficiencies), with separately tuned light and dark steps. Up and down are always shown with a sign
as well as colour. Each chart has one y-axis, entity colours stay stable while a ticker is selected,
and the benchmark is always neutral and dashed.

Framework chrome is handled through configuration, not CSS hacks. `client.toolbarMode = "minimal"`
removes Deploy, Rerun, Settings, Print and Record. `theme.paddingTop` tightens the top space. The
`[theme.light]` and `[theme.dark]` sections follow the operating-system setting.

---

## 3. Why these metrics

| Metric | Question it answers | Why it is useful | Main caveat |
|---|---|---|---|
| Cumulative return (common start) | What did each stock do over the same dates? | The basic comparison; a common start removes bias from different listing dates | Depends heavily on start and end dates |
| Annualised volatility | How much does it typically move? | Scale-free risk measure used everywhere | Assumes a stable volatility; returns are heavy-tailed (Cont, 2001) |
| Sharpe ratio | Return per unit of total risk | Standard and comparable (Sharpe, 1966, 1994) | Penalises upside volatility; unstable on short samples |
| Sortino ratio | Return per unit of *downside* risk | Investors dislike losses more than gains (Sortino & Price, 1994) | Needs enough negative days to be stable |
| Max drawdown and recovery | Worst peak-to-trough loss and how long it took to recover | Closest metric to the experience of holding the stock | A single path statistic; depends on the sample |
| Historical VaR / CVaR | How bad is a bad day? | CVaR is a coherent risk measure, VaR is not (Artzner et al., 1999) | Historical VaR assumes the past tail represents the future; the normal VaR understates fat tails |
| Beta | Sensitivity to the benchmark | Separates market exposure from stock-specific moves | Changes over time, so a rolling view is shown |
| Correlation | Do two stocks move together? | Drives diversification | Rises in sell-offs (Longin & Solnik, 2001); understated across time zones (Scholes & Williams, 1977) |
| Relative strength | Is the stock beating its benchmark, and is that improving? | Basis of momentum research (Jegadeesh & Titman, 1993) | Momentum also reverses; this is descriptive |
| Breadth | Is a move broad or narrow? | Narrow rallies led by a few names are a known fragility signal | Here it covers only the selected stocks |
| Regimes | Which kind of market was this period? | Risk and correlations differ by regime (Ang & Timmermann, 2012) | In-sample labels; not a forecast |
| Diversification ratio and risk contribution | Where does a portfolio's risk come from? | Shows concentration that equal *weights* can hide (Choueifaty & Coignard, 2008; Maillard et al., 2010) | Based on historical covariance |
| Tracking error and information ratio | How differently and how efficiently does the portfolio behave versus the benchmark? | Standard active-management measures | Unstable on short samples |

---

## 4. Statistical methods

**Rolling z-score anomalies.** Each day is compared with the mean and standard deviation of the
previous *W* days only, so there is no look-ahead. It is simple, explainable and parameter-light.
*Strengths:* each flag can be explained in one sentence; it adapts to changing volatility.
*Weaknesses:* returns are fat-tailed, so |z| ≥ 3 happens far more often than the 0.3% a normal
distribution implies. After a shock the baseline inflates, so later shocks look less unusual
(masking).

**Isolation Forest** (Liu, Ting & Zhou, 2008) isolates points with random splits. Unusual
*combinations* (e.g. a modest return on extreme volume and range) need fewer splits. *Strengths:*
multivariate, no distribution assumption. *Weaknesses:* the contamination parameter fixes the share
flagged; scores are relative within a stock's own history; the model is fitted on the full sample,
so it is not a real-time detector.

**Both methods flag statistical outliers.** A flag is not evidence of manipulation, insider
trading or a coming move. Earnings releases, index rebalances and macro news explain most flags.
Flags confirmed by both methods are shown separately because they are more robust.

**K-means and PCA.** Stocks are clustered on standardised risk and return features. *k* is suggested
by the silhouette score (Rousseeuw, 1987). PCA is used only to draw the result in two dimensions.
With a few dozen stocks, clusters are sensitive to the feature set and period, so silhouette values
are shown so users can judge whether any structure exists.

**Regimes.** The rule-based classifier (moving-average trend and expanding-percentile volatility)
is fully causal and transparent; the 200-day rule is widely used in practice (e.g. Faber, 2007). The
Gaussian Mixture alternative is a static, in-sample cousin of Hamilton's (1989) regime-switching
models. It gives probabilities but no transition dynamics. A Markov-switching model
(`statsmodels.tsa.MarkovRegression`, already a dependency) is a natural next step.

**Event studies.** Windows and recovery follow standard practice (Brown & Warner, 1985; MacKinlay,
1997). Abnormal returns use the market-adjusted model: stock return minus benchmark return. This
removes market-wide moves but not beta differences. The output describes what happened, not what
caused it.

---

## 5. Assumptions behind the financial calculations

- 252 trading days a year; simple daily returns; arithmetic risk-free conversion rf/252.
- Returns are in each stock's local currency; there is no FX conversion.
- Portfolios: long-only, frictionless, mark-to-market at the last close while a market is shut.
- Benchmarks are price indices unless a total-return index or ETF is chosen.
- Statistics are reported only when there is enough data (e.g. at least 20 returns for risk, 60
  sessions for clustering features, 205 sessions for trend regimes). Otherwise the UI says so, rather
  than estimating.

---

## 6. Limitations of historical market data

- **Survivorship bias:** today's large caps are survivors. A universe chosen today makes the past
  look better than an investable universe would have (Elton, Gruber & Blake, 1996).
- **Look-ahead in selection:** choosing stocks you know did well, then measuring how they did.
- **Non-stationarity:** volatility, correlation and return distributions change over time. A sample
  estimate describes that sample.
- **Data revisions and errors:** free data is revised and occasionally wrong. Quality checks surface
  suspicious points but cannot prove correctness.
- **Asynchronous markets:** daily returns from exchanges that close at different times are not
  simultaneous. Cross-market correlations are biased toward zero.

---

## 7. Why the additional features were added

Each new feature fills a gap that the original seven pages could not answer:

1. **User-driven selection, search and watchlists.** A fixed list of 29 US large caps limited
   analysis. Search and free entry make the platform usable for any market Yahoo covers, including
   India.
2. **Benchmark choice and return basis.** Relative statements need the right yardstick. Price and
   total return must match the benchmark's basis.
3. **Relative strength.** Absolute returns alone cannot say whether a stock merely rode the market.
4. **Breadth.** It distinguishes broad moves from narrow ones, using data already loaded.
5. **Regimes.** Risk metrics averaged over calm and stressed periods hide both. Labelling regimes
   makes that visible.
6. **Portfolio analytics.** It turns single-stock statistics into a combined view, with
   diversification and risk contribution, and shows explicitly what the numbers do not include.
7. **Data-quality layer.** Large numbers must be trustworthy before they are interesting.

Features considered and **not** added: price forecasts and buy/sell signals (outside a descriptive
analytics platform, and easy to misread as advice); intraday data (free sources are unreliable and
licensing is restrictive); sentiment from news (adds a dependency and an unverified source).

---

## 8. References

- Ang, A. & Timmermann, A. (2012). Regime changes and financial markets. *Annual Review of Financial Economics*, 4, 313–337.
- Artzner, P., Delbaen, F., Eber, J.-M. & Heath, D. (1999). Coherent measures of risk. *Mathematical Finance*, 9(3), 203–228.
- Brown, S. & Warner, J. (1985). Using daily stock returns: the case of event studies. *Journal of Financial Economics*, 14(1), 3–31.
- Choueifaty, Y. & Coignard, Y. (2008). Toward maximum diversification. *Journal of Portfolio Management*, 35(1), 40–51.
- Cont, R. (2001). Empirical properties of asset returns: stylized facts and statistical issues. *Quantitative Finance*, 1(2), 223–236.
- Elton, E., Gruber, M. & Blake, C. (1996). Survivorship bias and mutual fund performance. *Review of Financial Studies*, 9(4), 1097–1120.
- Faber, M. (2007). A quantitative approach to tactical asset allocation. *Journal of Wealth Management*, 9(4), 69–79.
- Hamilton, J. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357–384.
- Jegadeesh, N. & Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance*, 48(1), 65–91.
- Jorion, P. (2006). *Value at Risk: The New Benchmark for Managing Financial Risk* (3rd ed.). McGraw-Hill.
- Liu, F. T., Ting, K. M. & Zhou, Z.-H. (2008). Isolation forest. *Proc. IEEE ICDM*, 413–422.
- Longin, F. & Solnik, B. (2001). Extreme correlation of international equity markets. *Journal of Finance*, 56(2), 649–676.
- MacKinlay, A. C. (1997). Event studies in economics and finance. *Journal of Economic Literature*, 35(1), 13–39.
- Maillard, S., Roncalli, T. & Teïletche, J. (2010). The properties of equally weighted risk contribution portfolios. *Journal of Portfolio Management*, 36(4), 60–70.
- Rousseeuw, P. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.
- Scholes, M. & Williams, J. (1977). Estimating betas from nonsynchronous data. *Journal of Financial Economics*, 5(3), 309–327.
- Sharpe, W. (1966). Mutual fund performance. *Journal of Business*, 39(1), 119–138; and (1994) The Sharpe ratio. *Journal of Portfolio Management*, 21(1), 49–58.
- Sortino, F. & Price, L. (1994). Performance measurement in a downside risk framework. *Journal of Investing*, 3(3), 59–64.

Documentation consulted:

- Streamlit configuration reference (`toolbarMode`, theming): <https://docs.streamlit.io/develop/api-reference/configuration/config.toml>
- Streamlit app chrome: <https://docs.streamlit.io/develop/concepts/architecture/app-chrome>
- yfinance project README and disclaimer: <https://github.com/ranaroussi/yfinance>
- Alpha Vantage premium and free-tier limits: <https://www.alphavantage.co/premium/>
- Nasdaq Data Link documentation: <https://docs.data.nasdaq.com/>
- scikit-learn `IsolationForest`, `KMeans`, `GaussianMixture` user guides: <https://scikit-learn.org/stable/>
