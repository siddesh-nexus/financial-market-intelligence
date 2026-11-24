"""Shared fixtures. All data here is SYNTHETIC and generated with a fixed seed, so tests are
deterministic. No test depends on live market data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_prices(n: int = 400, drift: float = 0.0005, vol: float = 0.01, start_price: float = 100.0,
                seed: int = 0, start: str = "2020-01-01") -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, vol, n)
    prices = start_price * np.cumprod(1.0 + returns)
    index = pd.bdate_range(start=start, periods=n)
    return pd.Series(prices, index=index, name="adj_close")


def make_ohlcv(n: int = 400, seed: int = 0, start: str = "2020-01-01") -> pd.DataFrame:
    close = make_prices(n=n, seed=seed, start=start)
    rng = np.random.default_rng(seed + 1)
    spread = np.abs(rng.normal(0.01, 0.003, n)) * close.values
    high = close.values + spread
    low = close.values - spread
    open_ = close.values * (1 + rng.normal(0, 0.002, n))
    volume = rng.lognormal(mean=14, sigma=0.2, size=n)
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close.values,
        "Adj Close": close.values, "Volume": volume,
    }, index=close.index)


@pytest.fixture
def prices() -> pd.Series:
    return make_prices()


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture
def panel() -> pd.DataFrame:
    frames = {f"T{i}": make_prices(seed=i, vol=0.008 + 0.002 * i) for i in range(6)}
    return pd.DataFrame(frames)
