"""Create the PostgreSQL schema and load the default market events.

Usage (from the project root, with the venv active):
    python scripts/init_db.py
"""

from __future__ import annotations

import sys

import pandas as pd

from fmip.config import load_settings
from fmip.data.repository import MarketRepository, create_db_engine, create_schema
from fmip.logging_config import get_logger, setup_logging
from fmip.universe import DEFAULT_EVENTS

log = get_logger("scripts.init_db")


def main() -> int:
    settings = load_settings()
    setup_logging(settings.log_level)
    if not settings.database_configured:
        log.error("DATABASE_URL is not set. Copy .env.example to .env and fill it in.")
        return 1

    engine = create_db_engine(settings.database_url)
    create_schema(engine)
    repository = MarketRepository(engine)
    events = pd.DataFrame([
        {"name": e.name, "event_date": e.event_date, "category": e.category, "description": e.description}
        for e in DEFAULT_EVENTS
    ])
    repository.upsert_events(events)
    log.info("Schema ready and %d default events loaded", len(events))
    return 0


if __name__ == "__main__":
    sys.exit(main())
