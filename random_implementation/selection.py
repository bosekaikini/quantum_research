import random

def select_stocks(stocks: list[str], stocks_data: dict[str, dict[str, float]], n: int = 1) -> tuple[str, ...]:
    """
    Method 3: Selection is the independent variable.
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
    Method 3: Composition is proportional (equal weighting).
    """
    if not selection:
        return {}
    weight = 1.0 / len(selection)
    return {s: weight for s in selection}
