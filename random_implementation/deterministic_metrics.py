def select_by_eps(stocks: list[str], stocks_data: dict[str, dict[str, float]], n: int = 5) -> tuple[str, ...]:
    """Select the top n stocks with the highest Earnings Per Share (EPS)."""
    candidates = []
    for stock in stocks:
        data = stocks_data.get(stock, {})
        eps = float(data.get("eps", 0.0) or 0.0)
        candidates.append((stock, eps))
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    selection = [s for s, _ in candidates[:max(1, n)]]
    return tuple(selection)


def select_by_pe(stocks: list[str], stocks_data: dict[str, dict[str, float]], n: int = 5) -> tuple[str, ...]:
    """Select the top n stocks with the lowest positive PE Ratio."""
    candidates = []
    for stock in stocks:
        data = stocks_data.get(stock, {})
        pe = float(data.get("pe_ratio", 0.0) or 0.0)
        if pe > 0:
            candidates.append((stock, pe))
            
    candidates.sort(key=lambda x: x[1])  # Lowest PE first
    selection = [s for s, _ in candidates]
    
    if len(selection) < n:
        remaining = [s for s in stocks if s not in selection]
        selection.extend(remaining[:n - len(selection)])
        
    return tuple(selection[:max(1, n)])


def select_by_div_yield(stocks: list[str], stocks_data: dict[str, dict[str, float]], n: int = 5) -> tuple[str, ...]:
    """Select the top n stocks with the highest Dividend Yield."""
    candidates = []
    for stock in stocks:
        data = stocks_data.get(stock, {})
        dy = float(data.get("dividend_yield", 0.0) or 0.0)
        candidates.append((stock, dy))
        
    candidates.sort(key=lambda x: x[1], reverse=True)
    selection = [s for s, _ in candidates[:max(1, n)]]
    return tuple(selection)
