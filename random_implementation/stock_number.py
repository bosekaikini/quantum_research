import random

def get_eps_div_score(stock: str, stocks_data: dict[str, dict[str, float]]) -> float:
    data = stocks_data.get(stock, {})
    eps = float(data.get("eps", 0.0) or 0.0)
    div = float(data.get("dividend_yield", 0.0) or 0.0)
    return eps * div

def select_stocks(stocks: list[str], stocks_data: dict[str, dict[str, float]], n: int = 1) -> tuple[str, ...]:
    """
    Method 1: Stock Number is random, Selection by EPS * Div Yield.
    Number of stocks is the independent variable.
    """
    candidates = []
    for stock in stocks:
        candidates.append((stock, get_eps_div_score(stock, stocks_data)))
        
    # Sort candidates by the metric descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # The number of stocks is randomized between 1 and n. 
    # If run with n=1, it will cleanly return 1 stock.
    num_to_select = random.randint(1, max(1, n))
    num_to_select = min(num_to_select, len(candidates))
    
    selection = [stock for stock, score in candidates[:num_to_select]]
    return tuple(selection)

def get_composition_weights(selection: tuple[str, ...]) -> dict[str, float]:
    """Proportional (equal) composition."""
    if not selection:
        return {}
    weight = 1.0 / len(selection)
    return {s: weight for s in selection}
