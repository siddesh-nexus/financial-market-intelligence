"""Domain-specific exceptions raised by the platform."""


class FmipError(Exception):
    """Base class for all platform errors."""


class DataFetchError(FmipError):
    """Raised when a market data provider fails or returns nothing usable."""


class UnknownTickerError(DataFetchError):
    """Raised when a ticker returns no data (delisted, mistyped, or unavailable)."""


class DataValidationError(FmipError):
    """Raised when a dataset is structurally invalid (e.g. missing OHLCV columns)."""


class InsufficientDataError(FmipError):
    """Raised when there are too few observations to run an analysis."""
