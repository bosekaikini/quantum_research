import random

def get_eps_div_score(stock: str, stocks_data: dict[str, dict[str, float]]) -> float:
    data = stocks_data.get(stock, {})
    eps = float(data.get("eps", 0.0) or 0.0)
    div = float(data.get("dividend_yield", 0.0) or 0.0)
    return eps * div

def select_stocks(stocks: list[str], stocks_data: dict[str, dict[str, float]], n: int = 1) -> tuple[str, ...]:
    """
    Method 2: Fixed n stocks chosen by score.
    Selection is deterministic, composition will be independent (randomized).
    """
    candidates = []
    for stock in stocks:
        candidates.append((stock, get_eps_div_score(stock, stocks_data)))
        
    candidates.sort(key=lambda x: x[1], reverse=True)
    num_to_select = min(max(1, n), len(candidates))
    selection = [stock for stock, score in candidates[:num_to_select]]
    return tuple(selection)

def get_composition_weights(selection: tuple[str, ...]) -> dict[str, float]:
    """
    Method 2: Composition is the independent variable.
    Returns randomized weights for the selected stocks.
    """
    if not selection:
        return {}
    
    weights = [random.random() for _ in selection]
    total = sum(weights)
    
    if total <= 0:
        return {s: 1.0 / len(selection) for s in selection}
    
    return {s: w / total for s, w in zip(selection, weights)}
