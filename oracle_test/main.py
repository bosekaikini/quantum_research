"""Entry point for the oracle test simulation."""

from __future__ import annotations

import random
from statistics import fmean

from backtest_call import run_backtest
from firefly_call import run_firefly_strategy
from generator import generate_price_series, generate_price_stream
from oracle_strat import run_oracle_strategy


def _strategy_return_pct(final_value: float, budget: float) -> float:
    if budget <= 0:
        return 0.0
    return ((final_value / float(budget)) - 1.0) * 100.0


def run_buy_and_hold_strategy(price_series: list[float], budget: float = 10_000.0) -> dict[str, float | str]:
    """Buy the first available price and hold until the end."""

    if len(price_series) < 2:
        return {
            "final_value": float(budget),
            "return_pct": 0.0,
            "methodology": "Buy once at the first price and hold until the last price.",
        }

    entry_price = float(price_series[0])
    exit_price = float(price_series[-1])
    final_value = float(budget) * (exit_price / entry_price) if entry_price else float(budget)

    return {
        "final_value": final_value,
        "return_pct": _strategy_return_pct(final_value, budget),
        "methodology": "Buy once at the first price and hold until the last price.",
    }


def run_random_strategy(
    price_series: list[float],
    budget: float = 10_000.0,
    seed: int | None = None,
) -> dict[str, float | int | str]:
    """Pick a random entry and a later random exit, then fully invest between them."""

    if len(price_series) < 2:
        return {
            "final_value": float(budget),
            "return_pct": 0.0,
            "buy_index": 0,
            "sell_index": 0,
            "methodology": "Choose a random buy day and a later random sell day, then invest fully between them.",
        }

    rng = random.Random(seed)
    buy_index = rng.randint(0, len(price_series) - 2)
    sell_index = rng.randint(buy_index + 1, len(price_series) - 1)

    buy_price = float(price_series[buy_index])
    sell_price = float(price_series[sell_index])
    final_value = float(budget) * (sell_price / buy_price) if buy_price else float(budget)

    return {
        "final_value": final_value,
        "return_pct": _strategy_return_pct(final_value, budget),
        "buy_index": buy_index,
        "sell_index": sell_index,
        "methodology": "Choose a random buy day and a later random sell day, then invest fully between them.",
    }


def _capture_pct(strategy_return_pct: float, oracle_return_pct: float) -> float:
    if oracle_return_pct <= 0:
        return 0.0
    return (strategy_return_pct / oracle_return_pct) * 100.0


def _oracle_dominates(trial_result: dict[str, object], tolerance: float = 1e-9) -> bool:
    oracle_final_value = float(trial_result["oracle_final_value"])
    competitor_final_values = [
        float(trial_result["firefly_final_value"]),
        float(trial_result["random_final_value"]),
        float(trial_result["buy_hold_final_value"]),
        float(trial_result["backtest_final_value"]),
    ]
    return oracle_final_value + tolerance >= max(competitor_final_values)


def run_single_trial(series_length: int, budget: float, seed: int) -> dict[str, object]:
    price_series = generate_price_series(series_length, seed=seed)
    price_stream = generate_price_stream(price_series)

    oracle_result = run_oracle_strategy(price_series, budget=budget)
    firefly_result = run_firefly_strategy(price_series, price_stream, budget=budget)
    backtest_result = run_backtest(price_series, budget=budget)
    random_result = run_random_strategy(price_series, budget=budget, seed=seed)
    buy_and_hold_result = run_buy_and_hold_strategy(price_series, budget=budget)

    oracle_return_pct = float(oracle_result["return_pct"])
    firefly_return_pct = float(firefly_result["return_pct"])

    return {
        "oracle_final_value": float(oracle_result["final_value"]),
        "oracle_return_pct": oracle_return_pct,
        "oracle_methodology": "Use full foresight to buy every up-swing between local valleys and peaks.",
        "firefly_final_value": float(firefly_result["final_value"]),
        "firefly_return_pct": firefly_return_pct,
        "firefly_methodology": "Estimate exposure from momentum, trend, and recent price context, then scale into the series.",
        "firefly_capture_pct": _capture_pct(firefly_return_pct, oracle_return_pct),
        "random_final_value": float(random_result["final_value"]),
        "random_return_pct": float(random_result["return_pct"]),
        "random_methodology": random_result["methodology"],
        "buy_hold_final_value": float(buy_and_hold_result["final_value"]),
        "buy_hold_return_pct": float(buy_and_hold_result["return_pct"]),
        "buy_hold_methodology": buy_and_hold_result["methodology"],
        "backtest_final_value": float(backtest_result["final_value"]),
        "backtest_return_pct": float(backtest_result["return_pct"]),
        "backtest_methodology": "Use a moving-average signal: buy above the lookback average and sell below it.",
    }


def main(runs: int = 30, series_length: int = 60, budget: float = 10_000.0) -> dict[str, object]:
    trial_results = [run_single_trial(series_length, budget, seed=1_000 + index) for index in range(runs)]

    if not all(_oracle_dominates(result) for result in trial_results):
        failing_index = next(
            index for index, result in enumerate(trial_results) if not _oracle_dominates(result)
        )
        failing_result = trial_results[failing_index]
        raise AssertionError(
            "Oracle did not dominate every strategy on trial "
            f"{failing_index}: oracle={failing_result['oracle_final_value']}, "
            f"firefly={failing_result['firefly_final_value']}, "
            f"random={failing_result['random_final_value']}, "
            f"buy_hold={failing_result['buy_hold_final_value']}, "
            f"backtest={failing_result['backtest_final_value']}"
        )

    average_oracle = fmean(result["oracle_final_value"] for result in trial_results)
    average_oracle_return = fmean(result["oracle_return_pct"] for result in trial_results)
    average_firefly = fmean(result["firefly_final_value"] for result in trial_results)
    average_firefly_return = fmean(result["firefly_return_pct"] for result in trial_results)
    average_firefly_capture = fmean(result["firefly_capture_pct"] for result in trial_results)
    average_random = fmean(result["random_final_value"] for result in trial_results)
    average_random_return = fmean(result["random_return_pct"] for result in trial_results)
    average_buy_hold = fmean(result["buy_hold_final_value"] for result in trial_results)
    average_buy_hold_return = fmean(result["buy_hold_return_pct"] for result in trial_results)
    average_backtest = fmean(result["backtest_final_value"] for result in trial_results)
    average_backtest_return = fmean(result["backtest_return_pct"] for result in trial_results)

    summary = {
        "oracle_average_final_value": average_oracle,
        "oracle_average_return_pct": average_oracle_return,
        "firefly_average_final_value": average_firefly,
        "firefly_average_return_pct": average_firefly_return,
        "firefly_average_capture_pct": average_firefly_capture,
        "random_average_final_value": average_random,
        "random_average_return_pct": average_random_return,
        "buy_hold_average_final_value": average_buy_hold,
        "buy_hold_average_return_pct": average_buy_hold_return,
        "backtest_average_final_value": average_backtest,
        "backtest_average_return_pct": average_backtest_return,
    }

    print("Oracle methodology: Use full foresight to buy every up-swing between local valleys and peaks.")
    print("Oracle average return %:", round(summary["oracle_average_return_pct"], 2))
    print("Firefly methodology: Estimate exposure from momentum, trend, and recent price context, then scale into the series.")
    print("Firefly average return %:", round(summary["firefly_average_return_pct"], 2))
    print("Firefly return captured vs oracle %:", round(summary["firefly_average_capture_pct"], 2))
    print("Random methodology: Choose a random buy day and a later random sell day, then invest fully between them.")
    print("Random average return %:", round(summary["random_average_return_pct"], 2))
    print("Buy-and-hold methodology: Buy once at the first price and hold until the last price.")
    print("Buy-and-hold average return %:", round(summary["buy_hold_average_return_pct"], 2))
    print("Moving-average backtest methodology: Use a moving-average signal: buy above the lookback average and sell below it.")
    print("Moving-average backtest average return %:", round(summary["backtest_average_return_pct"], 2))
    print("Oracle dominance check: passed for all trials.")

    return summary


if __name__ == "__main__":
    main()