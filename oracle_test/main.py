"""Entry point for the oracle test simulation."""

from __future__ import annotations

from statistics import fmean

from backtest_call import run_backtest
from firefly_call import run_firefly_strategy
from generator import generate_price_series, generate_price_stream
from oracle_strat import run_oracle_strategy


def run_single_trial(series_length: int, budget: float, seed: int) -> dict[str, float]:
    price_series = generate_price_series(series_length, seed=seed)
    price_stream = generate_price_stream(price_series)

    oracle_result = run_oracle_strategy(price_series, budget=budget)
    firefly_result = run_firefly_strategy(price_series, price_stream, budget=budget)
    backtest_result = run_backtest(price_series, budget=budget)

    combined_final_value = fmean(
        [
            float(oracle_result["final_value"]),
            float(firefly_result["final_value"]),
            float(backtest_result["final_value"]),
        ]
    )

    return {
        "oracle_final_value": float(oracle_result["final_value"]),
        "firefly_final_value": float(firefly_result["final_value"]),
        "backtest_final_value": float(backtest_result["final_value"]),
        "combined_final_value": combined_final_value,
    }


def main(runs: int = 30, series_length: int = 60, budget: float = 10_000.0) -> dict[str, float]:
    trial_results = [run_single_trial(series_length, budget, seed=1_000 + index) for index in range(runs)]

    average_oracle = fmean(result["oracle_final_value"] for result in trial_results)
    average_firefly = fmean(result["firefly_final_value"] for result in trial_results)
    average_backtest = fmean(result["backtest_final_value"] for result in trial_results)
    average_combined = fmean(result["combined_final_value"] for result in trial_results)

    summary = {
        "oracle_average_final_value": average_oracle,
        "firefly_average_final_value": average_firefly,
        "backtest_average_final_value": average_backtest,
        "final_value": average_combined,
    }

    print("Oracle average final value:", round(summary["oracle_average_final_value"], 2))
    print("Firefly average final value:", round(summary["firefly_average_final_value"], 2))
    print("Backtest average final value:", round(summary["backtest_average_final_value"], 2))
    print("Final averaged value:", round(summary["final_value"], 2))

    return summary


if __name__ == "__main__":
    main()