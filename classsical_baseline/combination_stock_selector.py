"""
TODO

1. from stock list
2. A) Use either a math.comb for best performing mixes by brute forcing (defined by a rudimentary filterof eps, pe ratio, dividend yield))
3. Return the selected stocks in a usable format (tuple)
"""
def _safe_value(value: float | None) -> float:
    if value is None:
        return 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if value == value else 0.0


def _stock_score(stock: str, stocks_data: dict[str, dict[str, float]]) -> float:
    fundamentals = stocks_data.get(stock, {})
    eps = _safe_value(fundamentals.get("eps"))
    pe_ratio = _safe_value(fundamentals.get("pe_ratio"))
    dividend_yield = _safe_value(fundamentals.get("dividend_yield"))
    return eps * pe_ratio * dividend_yield


def combination_stock_selector(stocks, stocks_data, num):
    candidates: list[tuple[str, float]] = []
    for stock in stocks:
        candidates.append((stock, _stock_score(stock, stocks_data)))

    if not candidates:
        return tuple()

    candidates.sort(key=lambda item: item[1], reverse=True)
    num_stocks = max(1, min(int(num), len(candidates)))
    selection = [stock for stock, _ in candidates[:num_stocks]]
    return tuple(selection)