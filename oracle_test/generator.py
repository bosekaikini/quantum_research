"""Synthetic price generation helpers for oracle_test."""

from __future__ import annotations

from collections.abc import Sequence
import random


def generate_price_series(
    length: int,
    start_price: float = 100.0,
    drift: float = 0.0015,
    volatility: float = 0.02,
    seed: int | None = None,
) -> list[float]:
    """Generate a full synthetic price series."""

    if length <= 0:
        return []

    rng = random.Random(seed)
    prices = [float(start_price)]

    for _ in range(1, length):
        shock = rng.gauss(drift, volatility)
        next_price = max(0.01, prices[-1] * (1.0 + shock))
        prices.append(float(next_price))

    return prices


def generate_price_stream(price_series_or_length: int | Sequence[float]) -> float:
    """Return only the latest price value from a series."""

    if isinstance(price_series_or_length, int):
        price_series = generate_price_series(price_series_or_length)
    else:
        price_series = list(price_series_or_length)

    if not price_series:
        return 0.0

    return float(price_series[-1])
