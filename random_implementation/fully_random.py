import random

def select_stocks(stocks: list[str], stocks_data: dict[str, dict[str, float]], n: int = 1) -> tuple[str, ...]:
    """
    Method 5: Fully random. Stock Number AND Selection are independent variables.
    Randomly pick the number of stocks to choose (between 1 and n),
    and randomly select them from the available pool.
    """
    available = [s for s in stocks if s in stocks_data]
    if not available:
        return tuple()
        
    num_limit = min(max(1, n), len(available))
    num_to_select = random.randint(1, num_limit)
    
    selection = random.sample(available, num_to_select)
    return tuple(selection)

def get_composition_weights(selection: tuple[str, ...]) -> dict[str, float]:
    """
    Method 5: Composition is an independent variable.
    Returns randomized weights for the selected stocks.
    """
    if not selection:
        return {}
    
    weights = [random.random() for _ in selection]
    total = sum(weights)
    
    if total <= 0:
        return {s: 1.0 / len(selection) for s in selection}
    
    return {s: w / total for s, w in zip(selection, weights)}
