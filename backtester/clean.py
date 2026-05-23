"""
Normalize market data from external sources (S&P list CSV, yfinance).

Fetch and clean once here; strategies and backtests consume the returned structures.
"""

from __future__ import annotations

import concurrent.futures
from functools import lru_cache

import pandas as pd
import yfinance as yf

SP500_TICKERS_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"

Fundamentals = dict[str, float]
StockData = dict[str, Fundamentals]


def safe_float(value, default: float = 0.0) -> float:
    """Coerce a yfinance field to float; map None/NaN/bad values to ``default``."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fetch_fundamentals(stock: str) -> tuple[str, Fundamentals]:
    """Pull and clean EPS, P/E, dividend yield, and price for one ticker."""
    ticker = yf.Ticker(stock)
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    return stock, {
        "eps": safe_float(info.get("trailingEps") or info.get("forwardEps")),
        "pe_ratio": safe_float(info.get("trailingPE") or info.get("forwardPE")),
        "dividend_yield": safe_float(info.get("dividendYield")),
        "price": safe_float(info.get("currentPrice") or info.get("regularMarketPrice")),
    }


def build_stock_data(stocks: list[str], max_workers: int = 20) -> StockData:
    """Fetch cleaned fundamentals for many tickers in parallel."""
    stock_data: StockData = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for stock, data in executor.map(fetch_fundamentals, stocks):
            stock_data[stock] = data
    return stock_data


@lru_cache(maxsize=1)
def _load_sp500_symbols() -> list[str]:
    table = pd.read_csv(SP500_TICKERS_URL)
    return [symbol.replace(".", "-") for symbol in table["Symbol"].tolist()]


def get_sp500_tickers(max_tickers: int | None = 50) -> list[str]:
    """S&P 500 symbols from the constituents CSV (Yahoo-style tickers)."""
    tickers = _load_sp500_symbols()
    if max_tickers is None:
        return tickers
    return tickers[:max_tickers]


def clean_close_prices(
    price_data: pd.DataFrame,
    *,
    single_ticker_fallback: str | None = None,
) -> pd.DataFrame:
    """
    Extract the Close panel from ``yf.download`` output and drop bad columns.

    Forward-fills gaps, removes all-NaN columns, and keeps symbols with at least
    one valid close.
    """
    close_prices = price_data["Close"] if "Close" in price_data else pd.DataFrame()
    if isinstance(close_prices, pd.Series):
        name = single_ticker_fallback or "UNKNOWN"
        close_prices = close_prices.to_frame(name=name)
    close_prices = close_prices.dropna(how="all").ffill().dropna(how="all")
    if close_prices.empty:
        return close_prices

    valid_symbols = [
        symbol for symbol in close_prices.columns if close_prices[symbol].notna().any()
    ]
    return close_prices[valid_symbols]
