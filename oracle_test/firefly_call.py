"""Lightweight firefly wrapper used by oracle_test."""

from __future__ import annotations

from statistics import fmean


def run_firefly_strategy(
	price_series: list[float],
	price_stream: float,
	budget: float = 10_000.0,
) -> dict[str, float]:
	"""Estimate performance using the latest stream value plus historical context."""

	if len(price_series) < 2:
		return {
			"final_value": float(budget),
			"total_return": 0.0,
			"return_pct": 0.0,
			"exposure": 0.0,
		}

	series = [float(price) for price in price_series]
	latest_price = float(price_stream)
	recent_window = series[-min(5, len(series)) :]
	recent_average = fmean(recent_window)
	baseline_average = fmean(series)

	momentum = (latest_price - recent_average) / recent_average if recent_average else 0.0
	trend = (latest_price - series[0]) / series[0] if series[0] else 0.0

	exposure = 0.5 + (2.25 * momentum) + (0.35 * trend) + (0.15 * ((latest_price - baseline_average) / baseline_average if baseline_average else 0.0))
	exposure = max(0.0, min(1.0, exposure))

	final_value = float(budget) * (1.0 + exposure * trend)
	total_return = (final_value / float(budget)) - 1.0 if budget else 0.0

	return {
		"final_value": final_value,
		"total_return": total_return,
		"return_pct": total_return * 100.0,
		"exposure": exposure,
	}