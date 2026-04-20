"""Oracle-style strategy that can see the entire generated price series."""

from __future__ import annotations


def run_oracle_strategy(price_series: list[float], budget: float = 10_000.0) -> dict[str, float | int]:
	"""Trade every known up-swing in the series with full foresight."""

	if len(price_series) < 2:
		return {
			"final_value": float(budget),
			"total_return": 0.0,
			"return_pct": 0.0,
			"trades": 0,
		}

	capital = float(budget)
	trades = 0
	index = 0
	series = [float(price) for price in price_series]

	while index < len(series) - 1:
		while index < len(series) - 1 and series[index + 1] <= series[index]:
			index += 1

		valley = series[index]
		while index < len(series) - 1 and series[index + 1] >= series[index]:
			index += 1

		peak = series[index]
		if peak > valley:
			capital *= peak / valley
			trades += 1

	total_return = (capital / float(budget)) - 1.0 if budget else 0.0
	return {
		"final_value": capital,
		"total_return": total_return,
		"return_pct": total_return * 100.0,
		"trades": trades,
	}
