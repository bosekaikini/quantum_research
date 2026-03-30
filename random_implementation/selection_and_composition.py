import random

def select_stocks(stocks: list[str], stocks_data: dict[str, dict[str, float]], n: int = 1) -> tuple[str, ...]:
    """
    Method 4: Selection AND Composition are independent variables.
    Randomly select n stocks from the available pool.
    """
    available = [s for s in stocks if s in stocks_data]
    if not available:
        return tuple()
        
    num_to_select = min(max(1, n), len(available))
    selection = random.sample(available, num_to_select)
    return tuple(selection)

def get_composition_weights(selection: tuple[str, ...]) -> dict[str, float]:
    """
    Method 4: Composition is an independent variable.
    Returns randomized weights for the randomly selected stocks.
    """
    if not selection:
        return {}
    
    weights = [random.random() for _ in selection]
    total = sum(weights)
    
    if total <= 0:
        return {s: 1.0 / len(selection) for s in selection}
    
    return {s: w / total for s, w in zip(selection, weights)}
