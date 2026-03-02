"""
TODO:

1. Import financial data
2a. Call random implementation random_stock_selector(stocks, stock_data, num, previous_selection)
2b. Call combination implementation combination_stock_selector(stocks, stock_data, num)
3. backtest over last year and produce results for both strategies + index results
4. output summary results (win number, total return, return %, per trade)
5. plot each of the baselines as well as the index on a graph
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from classsical_baseline.implement_classical import implement_classical

start_date = "2023-01-01"
end_date = "2025-01-01"
num = 5
budget = 10000
MAX_TICKERS = 500
REBALANCE_EVERY_N_DAYS = 21


def get_sp500_tickers(max_tickers: int = MAX_TICKERS) -> list[str]:
    fallback_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    table = pd.read_csv(fallback_url)
    symbols = table["Symbol"].tolist()
    tickers = [symbol.replace(".", "-") for symbol in symbols]
    return tickers[:max_tickers]


def _safe_float(value, default: float = 0.0) -> float:
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


def build_stock_data(stocks: list[str]) -> dict[str, dict[str, float]]:
    stock_data: dict[str, dict[str, float]] = {}
    for stock in stocks:
        ticker = yf.Ticker(stock)
        info: dict[str, object] = {}
        try:
            fast_info = getattr(ticker, "fast_info", None)
            if fast_info:
                info = {
                    "currentPrice": _safe_float(fast_info.get("last_price")),
                }
        except Exception:
            info = {}

        if not info:
            try:
                info = ticker.info or {}
            except Exception:
                info = {}

        stock_data[stock] = {
            "eps": _safe_float(info.get("trailingEps") or info.get("forwardEps")),
            "pe_ratio": _safe_float(info.get("trailingPE") or info.get("forwardPE")),
            "dividend_yield": _safe_float(info.get("dividendYield")),
            "price": _safe_float(info.get("currentPrice") or info.get("regularMarketPrice")),
        }
    return stock_data


def _portfolio_value(cash: float, portfolio: dict[str, int], prices: pd.Series) -> float:
    holdings_value = sum(
        int(shares) * float(prices.get(symbol, 0.0))
        for symbol, shares in portfolio.items()
        if float(prices.get(symbol, 0.0)) > 0
    )
    return float(cash + holdings_value)


def _summary_from_curve(name: str, curve: pd.Series, initial_budget: float) -> dict[str, float | int | str]:
    if curve.empty or initial_budget <= 0:
        return {"name": name, "win_number": 0, "total_return": 0.0, "return_pct": 0.0, "per_day": 0.0}

    daily_returns = curve.pct_change().dropna()
    total_return = (float(curve.iloc[-1]) / initial_budget) - 1.0
    return {
        "name": name,
        "win_number": int((daily_returns > 0).sum().item() if hasattr((daily_returns > 0).sum(), "item") else (daily_returns > 0).sum()),
        "total_return": total_return,
        "return_pct": total_return * 100,
        "per_day": total_return / max(len(daily_returns), 1),
    }


def get_results(start_date: str, end_date: str, num_stocks: int = num, budget: float = budget) -> dict[str, object]:
    stocks = get_sp500_tickers()

    price_data = yf.download(stocks, start=start_date, end=end_date, auto_adjust=True, progress=False)
    close_prices = price_data["Close"] if "Close" in price_data else pd.DataFrame()
    if isinstance(close_prices, pd.Series):
        close_prices = close_prices.to_frame(name=stocks[0])
    close_prices = close_prices.dropna(how="all").ffill().dropna(how="all")
    if not close_prices.empty:
        valid_symbols = [symbol for symbol in close_prices.columns if close_prices[symbol].notna().any()]
        close_prices = close_prices[valid_symbols]

    if close_prices.empty:
        raise ValueError("No price data available for backtest window")

    stock_data = build_stock_data(list(close_prices.columns))

    states = {
        "random": {"cash": float(budget), "portfolio": {}, "previous_selection": tuple(), "last_changes": tuple()},
        "combination": {"cash": float(budget), "portfolio": {}, "previous_selection": tuple(), "last_changes": tuple()},
    }
    history = {"random": [], "combination": []}

    for index, date in enumerate(close_prices.index):
        prices_today = close_prices.loc[date].dropna()
        if prices_today.empty:
            continue

        should_rebalance = (index % REBALANCE_EVERY_N_DAYS == 0)
        if should_rebalance:
            for strategy in ("random", "combination"):
                state = states[strategy]
                result = implement_classical(
                    stocks=list(prices_today.index),
                    stock_data=stock_data,
                    current_portfolio=state["portfolio"],
                    cash=state["cash"],
                    current_prices=prices_today,
                    strategy=strategy,
                    num_stocks=num_stocks,
                    previous_selection=state["previous_selection"],
                )
                state["portfolio"] = result["portfolio"]
                state["cash"] = result["cash"]
                state["previous_selection"] = result["selection"]
                state["last_changes"] = result["changes"]

        history["random"].append(_portfolio_value(states["random"]["cash"], states["random"]["portfolio"], prices_today))
        history["combination"].append(_portfolio_value(states["combination"]["cash"], states["combination"]["portfolio"], prices_today))

    timeline = close_prices.index[: len(history["random"])]
    random_curve = pd.Series(history["random"], index=timeline, name="Random Portfolio Value")
    combination_curve = pd.Series(history["combination"], index=timeline, name="Combination Portfolio Value")

    index_data = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=True, progress=False)
    index_curve = pd.Series(dtype=float)
    if not index_data.empty and "Close" in index_data:
        index_close = index_data["Close"]
        if isinstance(index_close, pd.DataFrame):
            index_close = index_close.iloc[:, 0]
        index_curve = (index_close / float(index_close.iloc[0])) * budget
        index_curve = index_curve.reindex(timeline).ffill().dropna()

    budget_line = pd.Series(float(budget), index=timeline, name="Budget")

    summary = pd.DataFrame(
        [
            _summary_from_curve("Random Baseline", random_curve, budget),
            _summary_from_curve("Combination Baseline", combination_curve, budget),
            _summary_from_curve("S&P 500 Index", index_curve, budget),
        ]
    )

    print(summary[["name", "win_number", "total_return", "return_pct", "per_day"]].to_string(index=False))
    plot_df = pd.DataFrame(
        {
            "budget": budget_line,
            "random val": random_curve,
            "comb val": combination_curve,
            "s&p val": index_curve,
        }
    ).ffill().dropna(how="all")
    ax = plot_df.plot(title="Budget + Portfolio Value")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD")
    plt.tight_layout()
    plt.show()

    return {
        "random_selection": states["random"]["previous_selection"],
        "combination_selection": states["combination"]["previous_selection"],
        "random_changes": states["random"]["last_changes"],
        "combination_changes": states["combination"]["last_changes"],
        "summary": summary,
    }


if __name__ == "__main__":
    get_results(start_date, end_date, num)
