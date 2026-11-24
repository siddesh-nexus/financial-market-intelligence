"""Reference lists: default stock universe, example symbols, benchmarks, markets and events.

Sector labels are approximate, GICS-style groupings chosen for analytical illustration.
These lists only *seed* the dashboard. Users can search for and add any symbol that the data
provider supports. Nothing here is analytical output.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

BENCHMARK_INDEX = "^GSPC"  # S&P 500, the default benchmark

# Default universe loaded by `scripts/ingest.py`. ticker -> (company name, sector)
STOCK_UNIVERSE: dict[str, tuple[str, str]] = {
    "AAPL": ("Apple Inc.", "Technology"),
    "MSFT": ("Microsoft Corp.", "Technology"),
    "NVDA": ("NVIDIA Corp.", "Technology"),
    "ORCL": ("Oracle Corp.", "Technology"),
    "AMZN": ("Amazon.com Inc.", "Consumer Discretionary"),
    "HD": ("Home Depot Inc.", "Consumer Discretionary"),
    "NKE": ("Nike Inc.", "Consumer Discretionary"),
    "PG": ("Procter & Gamble Co.", "Consumer Staples"),
    "KO": ("Coca-Cola Co.", "Consumer Staples"),
    "WMT": ("Walmart Inc.", "Consumer Staples"),
    "JPM": ("JPMorgan Chase & Co.", "Financials"),
    "BAC": ("Bank of America Corp.", "Financials"),
    "GS": ("Goldman Sachs Group Inc.", "Financials"),
    "XOM": ("Exxon Mobil Corp.", "Energy"),
    "CVX": ("Chevron Corp.", "Energy"),
    "JNJ": ("Johnson & Johnson", "Health Care"),
    "PFE": ("Pfizer Inc.", "Health Care"),
    "UNH": ("UnitedHealth Group Inc.", "Health Care"),
    "CAT": ("Caterpillar Inc.", "Industrials"),
    "BA": ("Boeing Co.", "Industrials"),
    "GE": ("GE Aerospace", "Industrials"),
    "NEE": ("NextEra Energy Inc.", "Utilities"),
    "DUK": ("Duke Energy Corp.", "Utilities"),
    "LIN": ("Linde plc", "Materials"),
    "FCX": ("Freeport-McMoRan Inc.", "Materials"),
    "PLD": ("Prologis Inc.", "Real Estate"),
    "AMT": ("American Tower Corp.", "Real Estate"),
    "T": ("AT&T Inc.", "Communication Services"),
    "DIS": ("Walt Disney Co.", "Communication Services"),
}

# Well-known symbols offered as suggestions in the stock picker. They are downloaded on demand
# the first time a user selects them; they are not part of the default ingestion batch.
EXAMPLE_SYMBOLS: dict[str, tuple[str, str]] = {
    "GOOGL": ("Alphabet Inc.", "Communication Services"),
    "META": ("Meta Platforms Inc.", "Communication Services"),
    "TSLA": ("Tesla Inc.", "Consumer Discretionary"),
    "RELIANCE.NS": ("Reliance Industries Ltd.", "Energy"),
    "TCS.NS": ("Tata Consultancy Services Ltd.", "Technology"),
    "HDFCBANK.NS": ("HDFC Bank Ltd.", "Financials"),
    "INFY.NS": ("Infosys Ltd.", "Technology"),
    "ICICIBANK.NS": ("ICICI Bank Ltd.", "Financials"),
    "HINDUNILVR.NS": ("Hindustan Unilever Ltd.", "Consumer Staples"),
    "ITC.NS": ("ITC Ltd.", "Consumer Staples"),
    "LT.NS": ("Larsen & Toubro Ltd.", "Industrials"),
    "SBIN.NS": ("State Bank of India", "Financials"),
    "BHARTIARTL.NS": ("Bharti Airtel Ltd.", "Communication Services"),
}

# Benchmarks offered in the dashboard. symbol -> display name
BENCHMARKS: dict[str, str] = {
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ Composite",
    "^NDX": "NASDAQ-100",
    "^DJI": "Dow Jones Industrial Average",
    "^RUT": "Russell 2000",
    "^NSEI": "NIFTY 50",
    "^BSESN": "BSE SENSEX",
    "^FTSE": "FTSE 100",
    "^GDAXI": "DAX",
    "^N225": "Nikkei 225",
}


@dataclass(frozen=True)
class Market:
    name: str
    suffixes: tuple[str, ...]   # ticker suffixes that identify the market ("" = no suffix, US)
    default_benchmark: str
    defaults: tuple[str, ...]   # suggested starting selection


MARKETS: dict[str, Market] = {
    "United States": Market("United States", ("",), "^GSPC",
                            ("AAPL", "MSFT", "NVDA", "AMZN", "JPM", "XOM", "JNJ", "KO")),
    "India": Market("India", (".NS", ".BO"), "^NSEI",
                    ("RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "ITC.NS")),
    "All markets": Market("All markets", (), "^GSPC",
                          ("AAPL", "MSFT", "NVDA", "AMZN", "JPM", "XOM", "JNJ", "KO")),
}
DEFAULT_MARKET = "United States"

# Yahoo Finance sector names -> the GICS-style labels used in this project.
_SECTOR_ALIASES = {
    "technology": "Technology",
    "information technology": "Technology",
    "financial services": "Financials",
    "financial": "Financials",
    "financials": "Financials",
    "consumer cyclical": "Consumer Discretionary",
    "consumer discretionary": "Consumer Discretionary",
    "consumer defensive": "Consumer Staples",
    "consumer staples": "Consumer Staples",
    "healthcare": "Health Care",
    "health care": "Health Care",
    "basic materials": "Materials",
    "materials": "Materials",
    "industrials": "Industrials",
    "energy": "Energy",
    "utilities": "Utilities",
    "real estate": "Real Estate",
    "communication services": "Communication Services",
}

# Exchange suffixes recognised for display. Unknown suffixes are shown as-is.
_SUFFIX_EXCHANGE = {
    ".NS": "NSE", ".BO": "BSE", ".L": "LSE", ".DE": "XETRA", ".PA": "Euronext Paris",
    ".AS": "Euronext Amsterdam", ".T": "Tokyo", ".HK": "Hong Kong", ".TO": "Toronto",
    ".AX": "ASX", ".SW": "SIX", ".SS": "Shanghai", ".SZ": "Shenzhen", ".KS": "Korea",
}


def normalise_sector(raw: str | None) -> str | None:
    """Map a provider sector name onto this project's labels. Unknown names pass through."""
    if not raw:
        return None
    return _SECTOR_ALIASES.get(raw.strip().lower(), raw.strip())


def ticker_suffix(ticker: str) -> str:
    """'.NS' for 'TCS.NS', '' for 'AAPL' or for an index such as '^GSPC'."""
    if ticker.startswith("^") or "." not in ticker:
        return ""
    return ticker[ticker.rindex("."):].upper()


def exchange_of(ticker: str) -> str:
    """Best-effort exchange label from the ticker format (Yahoo conventions)."""
    if ticker.startswith("^"):
        return "Index"
    suffix = ticker_suffix(ticker)
    if not suffix:
        return "US"
    return _SUFFIX_EXCHANGE.get(suffix, suffix.lstrip("."))


def in_market(ticker: str, market: str) -> bool:
    """True if the ticker belongs to the named market (always True for 'All markets')."""
    spec = MARKETS.get(market)
    if spec is None or not spec.suffixes:
        return True
    if ticker.startswith("^"):
        return False
    return ticker_suffix(ticker) in spec.suffixes


def known_names() -> dict[str, tuple[str, str]]:
    """All built-in ticker -> (name, sector) pairs, including the benchmark indices."""
    names = {**STOCK_UNIVERSE, **EXAMPLE_SYMBOLS}
    for symbol, label in BENCHMARKS.items():
        names[symbol] = (label, "Index")
    return names


@dataclass(frozen=True)
class MarketEvent:
    name: str
    event_date: date
    category: str
    description: str


# Anchor dates for event-window analysis: the first session in which the news could be priced.
# Users can add custom events in the dashboard. Verify dates against primary sources before
# citing them in published work.
DEFAULT_EVENTS: tuple[MarketEvent, ...] = (
    MarketEvent("Lehman Brothers bankruptcy", date(2008, 9, 15), "Financial crisis",
                "Lehman Brothers files for Chapter 11 bankruptcy protection."),
    MarketEvent("Brexit referendum result", date(2016, 6, 24), "Geopolitical",
                "UK votes to leave the European Union."),
    MarketEvent("COVID-19 market crash start", date(2020, 2, 24), "Pandemic",
                "Start of the sharp global equity sell-off linked to COVID-19."),
    MarketEvent("Russia invades Ukraine", date(2022, 2, 24), "Geopolitical",
                "Russian full-scale invasion of Ukraine begins."),
    MarketEvent("First Fed hike of the 2022 cycle", date(2022, 3, 16), "Interest rates",
                "FOMC raises the federal funds target range by 25 basis points."),
    MarketEvent("Silicon Valley Bank failure", date(2023, 3, 10), "Financial crisis",
                "Silicon Valley Bank is closed by regulators; bank stocks sell off."),
    MarketEvent("US 'reciprocal' tariff announcement", date(2025, 4, 3), "Trade policy",
                "Broad US import tariffs announced after the close on 2 Apr 2025; "
                "the anchor is the first session after the announcement."),
)
