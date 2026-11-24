-- PostgreSQL schema for the Financial Market Intelligence platform.
-- Apply with:  psql "$DATABASE_URL" -f sql/schema.sql
-- (scripts/init_db.py applies the equivalent SQLAlchemy definitions in code.)

CREATE TABLE IF NOT EXISTS sectors (
    id    SERIAL PRIMARY KEY,
    name  VARCHAR(100) NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS stocks (
    ticker        VARCHAR(20) PRIMARY KEY,
    company_name  VARCHAR(200),
    sector_id     INTEGER REFERENCES sectors(id) ON DELETE SET NULL,
    -- v0.2 columns (scripts/init_db.py adds them to older databases automatically):
    exchange               VARCHAR(40),
    currency               VARCHAR(10),
    quote_type             VARCHAR(20),
    history_requested_from DATE      -- earliest start date ever requested; prevents repeat backfills
);
CREATE INDEX IF NOT EXISTS ix_stocks_sector_id ON stocks(sector_id);
-- Upgrading a v0.1 database by hand:
-- ALTER TABLE stocks ADD COLUMN IF NOT EXISTS exchange VARCHAR(40);
-- ALTER TABLE stocks ADD COLUMN IF NOT EXISTS currency VARCHAR(10);
-- ALTER TABLE stocks ADD COLUMN IF NOT EXISTS quote_type VARCHAR(20);
-- ALTER TABLE stocks ADD COLUMN IF NOT EXISTS history_requested_from DATE;

-- Saved watchlists (v0.2). tickers is a JSON array of symbols.
CREATE TABLE IF NOT EXISTS watchlists (
    name        VARCHAR(80) PRIMARY KEY,
    tickers     TEXT        NOT NULL,
    benchmark   VARCHAR(20),
    updated_at  TIMESTAMP   NOT NULL DEFAULT NOW()
);

-- Daily OHLCV bars. adj_close is dividend- and split-adjusted and is used for return calculations.
CREATE TABLE IF NOT EXISTS ohlcv_daily (
    ticker       VARCHAR(20) NOT NULL REFERENCES stocks(ticker) ON DELETE CASCADE,
    trade_date   DATE        NOT NULL,
    open         DOUBLE PRECISION NOT NULL CHECK (open > 0),
    high         DOUBLE PRECISION NOT NULL CHECK (high > 0),
    low          DOUBLE PRECISION NOT NULL CHECK (low > 0),
    close        DOUBLE PRECISION NOT NULL CHECK (close > 0),
    adj_close    DOUBLE PRECISION NOT NULL CHECK (adj_close > 0),
    volume       DOUBLE PRECISION NOT NULL CHECK (volume >= 0),
    ingested_at  TIMESTAMP   NOT NULL DEFAULT NOW(),
    PRIMARY KEY (ticker, trade_date),
    CHECK (high >= low)
);
CREATE INDEX IF NOT EXISTS ix_ohlcv_trade_date ON ohlcv_daily(trade_date);

-- Output of the anomaly engine. One row per flagged event per method.
CREATE TABLE IF NOT EXISTS anomalies (
    id              SERIAL PRIMARY KEY,
    ticker          VARCHAR(20) NOT NULL REFERENCES stocks(ticker) ON DELETE CASCADE,
    detection_date  DATE        NOT NULL,
    category        VARCHAR(50) NOT NULL,   -- return_zscore, volume_zscore, volatility_spike, price_range_zscore, multivariate
    method          VARCHAR(50) NOT NULL,   -- zscore_rolling or isolation_forest
    observed_value  DOUBLE PRECISION,
    baseline_value  DOUBLE PRECISION,
    anomaly_score   DOUBLE PRECISION,
    details         TEXT,
    CONSTRAINT uq_anomaly_event UNIQUE (ticker, detection_date, category, method)
);
CREATE INDEX IF NOT EXISTS ix_anomalies_ticker_date ON anomalies(ticker, detection_date);

-- Computed metrics (risk, volatility, correlation summaries) keyed by ticker, date and name.
CREATE TABLE IF NOT EXISTS analytical_metrics (
    ticker       VARCHAR(20) NOT NULL REFERENCES stocks(ticker) ON DELETE CASCADE,
    as_of_date   DATE        NOT NULL,
    metric_name  VARCHAR(80) NOT NULL,
    value        DOUBLE PRECISION,
    PRIMARY KEY (ticker, as_of_date, metric_name)
);

CREATE TABLE IF NOT EXISTS market_events (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(200) NOT NULL UNIQUE,
    event_date  DATE         NOT NULL,
    category    VARCHAR(100),
    description TEXT
);
