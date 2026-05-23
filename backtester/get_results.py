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

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backtester.clean import build_stock_data, clean_close_prices, get_sp500_tickers
from classical_baseline.implement_classical import implement_classical

start_date = "2023-01-01"
end_date = "2025-01-01"
num = 5
budget = 10000
MAX_TICKERS = 50
REBALANCE_EVERY_N_DAYS = 21


def _portfolio_value(
    cash: float, portfolio: dict[str, int], prices: pd.Series
) -> float:
    holdings_value = sum(
        int(shares) * float(prices.get(symbol, 0.0))
        for symbol, shares in portfolio.items()
        if float(prices.get(symbol, 0.0)) > 0
    )
    return float(cash + holdings_value)


def _summary_from_curve(
    name: str, curve: pd.Series, initial_budget: float
) -> dict[str, float | int | str]:
    if curve.empty or initial_budget <= 0:
        return {
            "name": name,
            "win_number": 0,
            "total_return": 0.0,
            "return_pct": 0.0,
            "per_day": 0.0,
        }

    daily_returns = curve.pct_change().dropna()
    total_return = (float(curve.iloc[-1]) / initial_budget) - 1.0
    return {
        "name": name,
        "win_number": int(
            (daily_returns > 0).sum().item()
            if hasattr((daily_returns > 0).sum(), "item")
            else (daily_returns > 0).sum()
        ),
        "total_return": total_return,
        "return_pct": total_return * 100,
        "per_day": total_return / max(len(daily_returns), 1),
    }


def get_results(
    start_date: str,
    end_date: str,
    num_stocks: int = num,
    budget: float = budget,
    run_idx: int = 1,
) -> dict[str, object]:
    stocks = get_sp500_tickers(MAX_TICKERS)

    price_data = yf.download(
        stocks, start=start_date, end=end_date, auto_adjust=True, progress=False
    )
    close_prices = clean_close_prices(
        price_data, single_ticker_fallback=stocks[0] if stocks else None
    )

    if close_prices.empty:
        raise ValueError("No price data available for backtest window")

    stock_data = build_stock_data(list(close_prices.columns))

    strategies = (
        "random",
        "combination",
        "metric_eps",
        "metric_pe",
        "metric_div",
        "stock_number",
        "composition",
        "selection",
        "selection_and_composition",
        "fully_random",
    )
    states = {
        s: {
            "cash": float(budget),
            "portfolio": {},
            "previous_selection": tuple(),
            "last_changes": tuple(),
        }
        for s in strategies
    }
    history = {s: [] for s in strategies}

    for index, date in enumerate(close_prices.index):
        prices_today = close_prices.loc[date].dropna()
        if prices_today.empty:
            continue

        should_rebalance = index % REBALANCE_EVERY_N_DAYS == 0
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
            history[strategy].append(
                _portfolio_value(
                    states[strategy]["cash"],
                    states[strategy]["portfolio"],
                    prices_today,
                )
            )

    timeline = close_prices.index[: len(history["random"])]
    curves = {
        s: pd.Series(
            history[s], index=timeline, name=f"{s.replace('_', ' ').title()} Value"
        )
        for s in strategies
    }

    index_data = yf.download(
        "^GSPC", start=start_date, end=end_date, auto_adjust=True, progress=False
    )
    index_curve = pd.Series(dtype=float)
    if not index_data.empty and "Close" in index_data:
        index_close = index_data["Close"]
        if isinstance(index_close, pd.DataFrame):
            index_close = index_close.iloc[:, 0]
        index_curve = (index_close / float(index_close.iloc[0])) * budget
        index_curve = index_curve.reindex(timeline).ffill().dropna()

    budget_line = pd.Series(float(budget), index=timeline, name="Budget")

    summary_dfs = [
        _summary_from_curve(f"{s.replace('_', ' ').title()}", curves[s], budget)
        for s in strategies
    ]
    summary_dfs.append(_summary_from_curve("S&P 500 Index", index_curve, budget))
    summary = pd.DataFrame(summary_dfs)

    print(
        summary[
            ["name", "win_number", "total_return", "return_pct", "per_day"]
        ].to_string(index=False)
    )

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
        styles = {col: "-" for col in plot_df.columns}
        styles["budget"] = "k--"
        styles["s&p val"] = "k--"
        ax = plot_df.plot(title=f"Budget + Portfolio Value (Pass {i})", style=styles)
        ax.set_xlabel("Date")
        ax.set_ylabel("USD")
        ax.set_ylim(global_min, global_max)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.savefig(
            ROOT_DIR / "plots" / "general_backtest" / f"backtest_results_{i}.png",
            bbox_inches="tight",
        )
        print(f"Saved plot to backtest_results_{i}.png")
        plt.close(ax.figure)

    import seaborn as sns

    avg_corr = sum(df.corr() for df in runs_data) / len(runs_data)

    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Average Correlation between Strategies")
    plt.savefig(
        ROOT_DIR / "plots" / "general_backtest" / "correlation_heatmap.png",
        bbox_inches="tight",
    )
    print("Saved correlation heatmap to correlation_heatmap.png")
    plt.close()
