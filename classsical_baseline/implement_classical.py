"""
TODO:
1. Get trading data
2. Loop through per 1 minute ticker
3. Call stock selector
4. Call stock composition
5. return results

"""
from __future__ import annotations
from typing import Iterable
import pandas as pd
from classsical_baseline.combination_stock_selector import combination_stock_selector
from classsical_baseline.random_stock_selector import random_stock_selector
from classsical_baseline import stock_composition as stock_composition_module


def _normalize_selection(selection: Iterable[str] | None, current_prices: pd.Series) -> tuple[str, ...]:
	if selection is None:
		return tuple()
	cleaned: list[str] = []
	for symbol in selection:
		price = float(current_prices.get(symbol, 0.0))
		if price > 0 and symbol not in cleaned:
			cleaned.append(symbol)
	return tuple(cleaned)


def _default_stock_composition(
	selection: tuple[str, ...],
	current_portfolio: dict[str, int],
	cash: float,
	current_prices: pd.Series,
) -> tuple[tuple[str, str, int], ...]:
	if not selection:
		return tuple(("SELL", symbol, shares) for symbol, shares in current_portfolio.items() if shares > 0)

	portfolio_value = cash + sum(
		shares * float(current_prices.get(symbol, 0.0))
		for symbol, shares in current_portfolio.items()
		if float(current_prices.get(symbol, 0.0)) > 0
	)
	target_per_stock = portfolio_value / len(selection)
	changes: list[tuple[str, str, int]] = []

	for symbol, shares in current_portfolio.items():
		if shares > 0 and symbol not in selection:
			changes.append(("SELL", symbol, int(shares)))

	for symbol in selection:
		price = float(current_prices.get(symbol, 0.0))
		if price <= 0:
			continue
		target_shares = int(target_per_stock // price)
		current_shares = int(current_portfolio.get(symbol, 0))
		delta = target_shares - current_shares
		if delta > 0:
			changes.append(("BUY", symbol, delta))
		elif delta < 0:
			changes.append(("SELL", symbol, abs(delta)))

	sell_orders = [change for change in changes if change[0] == "SELL"]
	buy_orders = [change for change in changes if change[0] == "BUY"]
	return tuple(sell_orders + buy_orders)


def _compose_changes(
	selection: tuple[str, ...],
	current_portfolio: dict[str, int],
	cash: float,
	current_prices: pd.Series,
) -> tuple[tuple[str, str, int], ...]:
	composition_fn = getattr(stock_composition_module, "stock_composition", None)
	if callable(composition_fn):
		call_options = [
			{
				"selection": selection,
				"current_portfolio": current_portfolio,
				"cash": cash,
				"current_prices": current_prices,
			},
			{
				"selected_stocks": selection,
				"current_portfolio": current_portfolio,
				"cash": cash,
				"current_prices": current_prices,
			},
		]
		for kwargs in call_options:
			try:
				result = composition_fn(**kwargs)
				if result is not None:
					return tuple(result)
			except TypeError:
				continue

	return _default_stock_composition(selection, current_portfolio, cash, current_prices)


def _execute_changes(
	changes: tuple[tuple[str, str, int], ...],
	current_portfolio: dict[str, int],
	cash: float,
	current_prices: pd.Series,
) -> tuple[dict[str, int], float]:
	updated_portfolio = dict(current_portfolio)
	updated_cash = float(cash)

	for action, symbol, quantity in changes:
		qty = int(max(quantity, 0))
		if qty == 0:
			continue
		price = float(current_prices.get(symbol, 0.0))
		if price <= 0:
			continue

		if action == "SELL":
			held = int(updated_portfolio.get(symbol, 0))
			executed = min(held, qty)
			if executed > 0:
				updated_portfolio[symbol] = held - executed
				if updated_portfolio[symbol] == 0:
					updated_portfolio.pop(symbol, None)
				updated_cash += executed * price
		elif action == "BUY":
			affordable = int(updated_cash // price)
			executed = min(affordable, qty)
			if executed > 0:
				updated_portfolio[symbol] = int(updated_portfolio.get(symbol, 0)) + executed
				updated_cash -= executed * price

	return updated_portfolio, updated_cash


def implement_classical(
	stocks: list[str],
	stock_data: dict[str, dict[str, float]],
	current_portfolio: dict[str, int],
	cash: float,
	current_prices: pd.Series,
	strategy: str,
	num_stocks: int = 5,
	previous_selection: tuple[str, ...] = tuple(),
) -> dict[str, object]:
	if strategy == "random":
		selection = random_stock_selector(stocks, stock_data, list(previous_selection))
	elif strategy == "combination":
		selection = combination_stock_selector(stocks, stock_data, num_stocks)
	else:
		raise ValueError("strategy must be 'random' or 'combination'")

	normalized_selection = _normalize_selection(selection, current_prices)
	changes = _compose_changes(normalized_selection, current_portfolio, cash, current_prices)
	updated_portfolio, updated_cash = _execute_changes(changes, current_portfolio, cash, current_prices)

	return {
		"selection": normalized_selection,
		"changes": changes,
		"portfolio": updated_portfolio,
		"cash": updated_cash,
	}


