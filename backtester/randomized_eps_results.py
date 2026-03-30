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


import concurrent.futures

def _fetch_info(stock: str) -> tuple[str, dict[str, float]]:
    ticker = yf.Ticker(stock)
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    return stock, {
        "eps": _safe_float(info.get("trailingEps") or info.get("forwardEps")),
        "pe_ratio": _safe_float(info.get("trailingPE") or info.get("forwardPE")),
        "dividend_yield": _safe_float(info.get("dividendYield")),
        "price": _safe_float(info.get("currentPrice") or info.get("regularMarketPrice")),
    }

def build_stock_data(stocks: list[str]) -> dict[str, dict[str, float]]:
    stock_data: dict[str, dict[str, float]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = executor.map(_fetch_info, stocks)
        for stock, data in results:
            stock_data[stock] = data
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


def get_results(start_date: str, end_date: str, num_stocks: int = num, budget: float = budget, run_idx: int = 1) -> dict[str, object]:
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

    strategies = (
        "randomized_eps",
    )
    states = {
        s: {"cash": float(budget), "portfolio": {}, "previous_selection": tuple(), "last_changes": tuple()}
        for s in strategies
    }
    history = {s: [] for s in strategies}

    for index, date in enumerate(close_prices.index):
        prices_today = close_prices.loc[date].dropna()
        if prices_today.empty:
            continue

        should_rebalance = (index % REBALANCE_EVERY_N_DAYS == 0)
        if should_rebalance:
            for strategy in strategies:
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

        for strategy in strategies:
            history[strategy].append(_portfolio_value(states[strategy]["cash"], states[strategy]["portfolio"], prices_today))

    timeline = close_prices.index[: len(history[strategies[0]])]
    curves = {
        s: pd.Series(history[s], index=timeline, name=f"{s.replace('_', ' ').title()} Value")
        for s in strategies
    }

    index_data = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=True, progress=False)
    index_curve = pd.Series(dtype=float)
    if not index_data.empty and "Close" in index_data:
        index_close = index_data["Close"]
        if isinstance(index_close, pd.DataFrame):
            index_close = index_close.iloc[:, 0]
        index_curve = (index_close / float(index_close.iloc[0])) * budget
        index_curve = index_curve.reindex(timeline).ffill().dropna()

    budget_line = pd.Series(float(budget), index=timeline, name="Budget")

    summary_dfs = [_summary_from_curve(f"{s.replace('_', ' ').title()}", curves[s], budget) for s in strategies]
    summary_dfs.append(_summary_from_curve("S&P 500 Index", index_curve, budget))
    summary = pd.DataFrame(summary_dfs)

    print(summary[["name", "win_number", "total_return", "return_pct", "per_day"]].to_string(index=False))
    
    plot_dict = {"budget": budget_line, "s&p val": index_curve}
    for s in strategies:
        plot_dict[f"{s} val"] = curves[s]
    
    plot_df = pd.DataFrame(plot_dict).ffill().dropna(how="all")

    return {
        **{f"{s}_selection": states[s]["previous_selection"] for s in strategies},
        **{f"{s}_changes": states[s]["last_changes"] for s in strategies},
        "summary": summary,
        "plot_df": plot_df,
    }


if __name__ == "__main__":
    runs_data = []
    for i in range(1, 4):
        print(f"--- Running Backtest Pass {i} ---")
        res = get_results(start_date, end_date, num, budget, run_idx=i)
        runs_data.append(res["plot_df"])
        
    global_min = min(df.min().min() for df in runs_data)
    global_max = max(df.max().max() for df in runs_data)
    
    for i, plot_df in enumerate(runs_data, 1):
        styles = {col: '-' for col in plot_df.columns}
        styles['budget'] = 'k--'
        styles['s&p val'] = 'k--'
        ax = plot_df.plot(title=f"Budget + Portfolio Value (Randomized EPS)", style=styles)
        ax.set_xlabel("Date")
        ax.set_ylabel("USD")
        ax.set_ylim(global_min, global_max)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(f"randomized_eps_results_{i}.png", bbox_inches="tight")
        print(f"Saved plot to randomized_eps_results_{i}.png")
        plt.close(ax.figure)
