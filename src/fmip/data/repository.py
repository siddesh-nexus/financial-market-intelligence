"""PostgreSQL persistence layer (SQLAlchemy Core).

The same code runs against PostgreSQL in production and SQLite in development and unit tests,
because the upsert helpers pick the matching dialect-specific INSERT ... ON CONFLICT implementation.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import date, datetime

import pandas as pd
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    func,
    inspect,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine

from fmip.logging_config import get_logger

log = get_logger(__name__)

metadata = MetaData()

sectors_table = Table(
    "sectors", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(100), nullable=False, unique=True),
)

stocks_table = Table(
    "stocks", metadata,
    Column("ticker", String(20), primary_key=True),
    Column("company_name", String(200)),
    Column("sector_id", Integer, ForeignKey("sectors.id"), nullable=True, index=True),
    # Added in v0.2 (applied to older databases by migrate_schema):
    Column("exchange", String(40)),
    Column("currency", String(10)),
    Column("quote_type", String(20)),
    # Earliest start date ever requested from the provider. Prevents repeated backfill
    # requests for listings that simply start later than the requested date.
    Column("history_requested_from", Date),
)

prices_table = Table(
    "ohlcv_daily", metadata,
    Column("ticker", String(20), ForeignKey("stocks.ticker"), primary_key=True),
    Column("trade_date", Date, primary_key=True),
    Column("open", Float, nullable=False),
    Column("high", Float, nullable=False),
    Column("low", Float, nullable=False),
    Column("close", Float, nullable=False),
    Column("adj_close", Float, nullable=False),
    Column("volume", Float, nullable=False),
    Column("ingested_at", DateTime, server_default=func.now()),
)

anomalies_table = Table(
    "anomalies", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ticker", String(20), ForeignKey("stocks.ticker"), nullable=False, index=True),
    Column("detection_date", Date, nullable=False, index=True),
    Column("category", String(50), nullable=False),
    Column("method", String(50), nullable=False),
    Column("observed_value", Float),
    Column("baseline_value", Float),
    Column("anomaly_score", Float),
    Column("details", Text),
    UniqueConstraint("ticker", "detection_date", "category", "method", name="uq_anomaly_event"),
)

metrics_table = Table(
    "analytical_metrics", metadata,
    Column("ticker", String(20), ForeignKey("stocks.ticker"), primary_key=True),
    Column("as_of_date", Date, primary_key=True),
    Column("metric_name", String(80), primary_key=True),
    Column("value", Float),
)

events_table = Table(
    "market_events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(200), nullable=False, unique=True),
    Column("event_date", Date, nullable=False),
    Column("category", String(100)),
    Column("description", Text),
)

watchlists_table = Table(
    "watchlists", metadata,
    Column("name", String(80), primary_key=True),
    Column("tickers", Text, nullable=False),        # JSON array of symbols
    Column("benchmark", String(20)),
    Column("updated_at", DateTime, server_default=func.now()),
)

PRICE_VALUE_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")
STOCK_META_COLUMNS = ("company_name", "exchange", "currency", "quote_type")
CHUNK_SIZE = 500  # keeps multi-row INSERTs under SQLite's bound-parameter limit


def _chunks(rows: list[dict], size: int = CHUNK_SIZE):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def create_db_engine(database_url: str) -> Engine:
    if not database_url:
        raise ValueError("DATABASE_URL is not set. See .env.example.")
    return create_engine(database_url, pool_pre_ping=True, future=True)


def create_schema(engine: Engine) -> None:
    """Create missing tables and add columns introduced after v0.1. Safe to run repeatedly."""
    metadata.create_all(engine)
    migrate_schema(engine)
    log.info("Database schema ensured")


def migrate_schema(engine: Engine) -> list[str]:
    """Additive migration: add any column defined above that an existing table lacks.

    Only ADD COLUMN is performed (never drops or type changes), so it is safe on live data.
    Returns the list of 'table.column' names that were added.
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    added: list[str] = []
    with engine.begin() as conn:
        for table in metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            present = {col["name"] for col in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in present:
                    continue
                ddl_type = column.type.compile(dialect=engine.dialect)
                conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {column.name} {ddl_type}'))
                added.append(f"{table.name}.{column.name}")
    if added:
        log.info("Schema migration added columns: %s", ", ".join(added))
    return added


class MarketRepository:
    """Read and write market data and analytical results."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    # ---- sectors / stocks -------------------------------------------------
    def upsert_sectors(self, sector_names: Iterable[str]) -> None:
        names = sorted({name for name in sector_names if name})
        if not names:
            return
        rows = [{"name": name} for name in names]
        with self.engine.begin() as conn:
            stmt = self._insert(conn, sectors_table).values(rows)
            conn.execute(stmt.on_conflict_do_nothing(index_elements=["name"]))

    def upsert_stocks(self, stocks: pd.DataFrame) -> None:
        """`stocks` columns: ticker, plus any of company_name, sector (name), exchange, currency,
        quote_type. Missing or empty values never overwrite stored metadata."""
        if stocks.empty:
            return
        frame = stocks.copy()
        for column in ("company_name", "sector", *STOCK_META_COLUMNS):
            if column not in frame.columns:
                frame[column] = None
        frame = frame.astype(object).where(frame.notna(), None)
        self.upsert_sectors(frame["sector"].dropna())
        with self.engine.begin() as conn:
            sector_ids = dict(conn.execute(select(sectors_table.c.name, sectors_table.c.id)).all())
            rows = [
                {
                    "ticker": row["ticker"],
                    "sector_id": sector_ids.get(row["sector"]),
                    **{c: (row[c] or None) for c in STOCK_META_COLUMNS},
                }
                for row in frame.to_dict(orient="records")
            ]
            stmt = self._insert(conn, stocks_table).values(rows)
            set_ = {c: func.coalesce(getattr(stmt.excluded, c), stocks_table.c[c])
                    for c in ("sector_id", *STOCK_META_COLUMNS)}
            conn.execute(stmt.on_conflict_do_update(index_elements=["ticker"], set_=set_))

    def load_stocks(self, tickers: list[str] | None = None) -> pd.DataFrame:
        """Stock metadata joined with sector names."""
        stmt = (
            select(stocks_table.c.ticker, stocks_table.c.company_name,
                   sectors_table.c.name.label("sector"), stocks_table.c.exchange,
                   stocks_table.c.currency, stocks_table.c.quote_type,
                   stocks_table.c.history_requested_from)
            .select_from(stocks_table.outerjoin(sectors_table, stocks_table.c.sector_id == sectors_table.c.id))
        )
        if tickers:
            stmt = stmt.where(stocks_table.c.ticker.in_(tickers))
        return pd.read_sql(stmt.order_by(stocks_table.c.ticker), self.engine)

    def history_requested_from(self, ticker: str) -> date | None:
        with self.engine.connect() as conn:
            return conn.execute(select(stocks_table.c.history_requested_from)
                                .where(stocks_table.c.ticker == ticker)).scalar()

    def mark_history_requested(self, ticker: str, start: date) -> None:
        """Record that history from `start` has been requested (keeps the earliest date)."""
        current = self.history_requested_from(ticker)
        if current is not None and current <= start:
            return
        with self.engine.begin() as conn:
            conn.execute(update(stocks_table).where(stocks_table.c.ticker == ticker)
                         .values(history_requested_from=start))

    # ---- prices -------------------------------------------------------------
    def upsert_prices(self, ticker: str, cleaned: pd.DataFrame) -> int:
        """Insert or update daily bars for one ticker. `cleaned` is indexed by date."""
        if cleaned.empty:
            return 0
        rows = [
            {
                "ticker": ticker,
                "trade_date": idx.date() if hasattr(idx, "date") else idx,
                "open": float(r["Open"]),
                "high": float(r["High"]),
                "low": float(r["Low"]),
                "close": float(r["Close"]),
                "adj_close": float(r["Adj Close"]),
                "volume": float(r["Volume"]),
            }
            for idx, r in cleaned.iterrows()
        ]
        with self.engine.begin() as conn:
            # Ensure the parent stock row exists so the foreign key is satisfied.
            ensure_stock = self._insert(conn, stocks_table).values(ticker=ticker)
            conn.execute(ensure_stock.on_conflict_do_nothing(index_elements=["ticker"]))
            for chunk in _chunks(rows):
                stmt = self._insert(conn, prices_table).values(chunk)
                conn.execute(
                    stmt.on_conflict_do_update(
                        index_elements=["ticker", "trade_date"],
                        set_={c: getattr(stmt.excluded, c) for c in PRICE_VALUE_COLUMNS},
                    )
                )
        return len(rows)

    def latest_price_date(self, ticker: str) -> date | None:
        with self.engine.connect() as conn:
            value = conn.execute(
                select(func.max(prices_table.c.trade_date)).where(prices_table.c.ticker == ticker)
            ).scalar()
        return value

    def price_date_bounds(self, ticker: str) -> tuple[date | None, date | None]:
        """(earliest, latest) stored trade date for a ticker, or (None, None) if nothing is stored."""
        with self.engine.connect() as conn:
            earliest, latest = conn.execute(
                select(func.min(prices_table.c.trade_date), func.max(prices_table.c.trade_date))
                .where(prices_table.c.ticker == ticker)
            ).one()
        return earliest, latest

    def coverage(self, tickers: list[str] | None = None) -> pd.DataFrame:
        """One aggregated query: first/last stored date, row count and last ingestion per ticker."""
        stmt = select(
            prices_table.c.ticker,
            func.min(prices_table.c.trade_date).label("first_date"),
            func.max(prices_table.c.trade_date).label("last_date"),
            func.count().label("sessions"),
            func.max(prices_table.c.ingested_at).label("last_ingested"),
        ).group_by(prices_table.c.ticker)
        if tickers:
            stmt = stmt.where(prices_table.c.ticker.in_(tickers))
        frame = pd.read_sql(stmt, self.engine)
        for column in ("first_date", "last_date", "last_ingested"):
            frame[column] = pd.to_datetime(frame[column])
        return frame.set_index("ticker")

    def load_prices(self, tickers: list[str], start: date | str | None = None,
                    end: date | str | None = None) -> pd.DataFrame:
        """Return long-format prices: columns ticker, trade_date, open, high, low, close, adj_close, volume."""
        columns = [prices_table.c.ticker, prices_table.c.trade_date,
                   *(prices_table.c[c] for c in PRICE_VALUE_COLUMNS)]
        stmt = select(*columns).where(prices_table.c.ticker.in_(tickers))
        if start is not None:
            stmt = stmt.where(prices_table.c.trade_date >= pd.Timestamp(start).date())
        if end is not None:
            stmt = stmt.where(prices_table.c.trade_date <= pd.Timestamp(end).date())
        stmt = stmt.order_by(prices_table.c.ticker, prices_table.c.trade_date)
        return pd.read_sql(stmt, self.engine, parse_dates=["trade_date"])

    # ---- analytical results --------------------------------------------------
    def replace_anomalies(self, anomalies: pd.DataFrame) -> int:
        """Replace stored anomalies for the tickers present in `anomalies`."""
        if anomalies.empty:
            return 0
        tickers = sorted(anomalies["ticker"].unique())
        rows = anomalies.to_dict(orient="records")
        with self.engine.begin() as conn:
            conn.execute(delete(anomalies_table).where(anomalies_table.c.ticker.in_(tickers)))
            for chunk in _chunks(rows):
                conn.execute(self._insert(conn, anomalies_table).values(chunk))
        return len(rows)

    def load_anomalies(self, tickers: list[str] | None = None) -> pd.DataFrame:
        stmt = select(anomalies_table)
        if tickers:
            stmt = stmt.where(anomalies_table.c.ticker.in_(tickers))
        return pd.read_sql(stmt.order_by(anomalies_table.c.detection_date), self.engine,
                           parse_dates=["detection_date"])

    def upsert_events(self, events: pd.DataFrame) -> int:
        if events.empty:
            return 0
        rows = events.to_dict(orient="records")
        with self.engine.begin() as conn:
            stmt = self._insert(conn, events_table).values(rows)
            conn.execute(
                stmt.on_conflict_do_update(
                    index_elements=["name"],
                    set_={"event_date": stmt.excluded.event_date,
                          "category": stmt.excluded.category,
                          "description": stmt.excluded.description},
                )
            )
        return len(rows)

    def load_events(self) -> pd.DataFrame:
        return pd.read_sql(select(events_table).order_by(events_table.c.event_date), self.engine,
                           parse_dates=["event_date"])

    # ---- watchlists -------------------------------------------------------------
    def save_watchlist(self, name: str, tickers: list[str], benchmark: str | None = None) -> None:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Watchlist name must not be empty")
        if not tickers:
            raise ValueError("A watchlist needs at least one ticker")
        row = {"name": clean_name[:80], "tickers": json.dumps(list(dict.fromkeys(tickers))),
               "benchmark": benchmark, "updated_at": datetime.now()}
        with self.engine.begin() as conn:
            stmt = self._insert(conn, watchlists_table).values(row)
            conn.execute(stmt.on_conflict_do_update(
                index_elements=["name"],
                set_={"tickers": stmt.excluded.tickers, "benchmark": stmt.excluded.benchmark,
                      "updated_at": stmt.excluded.updated_at},
            ))

    def list_watchlists(self) -> pd.DataFrame:
        frame = pd.read_sql(select(watchlists_table).order_by(watchlists_table.c.name), self.engine)
        if not frame.empty:
            frame["tickers"] = frame["tickers"].map(json.loads)
        return frame

    def delete_watchlist(self, name: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(delete(watchlists_table).where(watchlists_table.c.name == name))

    def ping(self) -> bool:
        with self.engine.connect() as conn:
            return conn.execute(text("SELECT 1")).scalar() == 1

    # ---- internals ------------------------------------------------------------
    @staticmethod
    def _insert(conn, table: Table):
        dialect = conn.dialect.name
        if dialect == "postgresql":
            return pg_insert(table)
        if dialect == "sqlite":
            return sqlite_insert(table)
        raise NotImplementedError(f"Unsupported database dialect: {dialect}")
