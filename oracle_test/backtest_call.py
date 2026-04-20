"""Simple backtest wrapper used by oracle_test."""

from __future__ import annotations


def run_backtest(price_series: list[float], budget: float = 10_000.0, lookback: int = 5) -> dict[str, float | int]:
	"""Run a small moving-average style backtest over the generated series."""

	if len(price_series) < 2:
		return {
			"final_value": float(budget),
			"total_return": 0.0,
			"return_pct": 0.0,
			"trades": 0,
		}

	series = [float(price) for price in price_series]
	cash = float(budget)
	shares = 0.0
	trades = 0

	for index in range(1, len(series)):
		current_price = series[index]
		window_start = max(0, index - lookback)
		window = series[window_start:index]
		average_price = sum(window) / len(window) if window else current_price

		if current_price > average_price and cash > 0.0:
			shares = cash / current_price
			cash = 0.0
			trades += 1
		elif current_price < average_price and shares > 0.0:
			cash = shares * current_price
			shares = 0.0
			trades += 1

	final_value = cash + (shares * series[-1])
	total_return = (final_value / float(budget)) - 1.0 if budget else 0.0

	return {
		"final_value": final_value,
		"total_return": total_return,
		"return_pct": total_return * 100.0,
		"trades": trades,
	}