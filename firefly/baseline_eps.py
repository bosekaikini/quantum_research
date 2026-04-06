def baseline_eps(stock_tickers, eps_values, top_n=10, budget=10000.0):
    """
    Creates a deterministic baseline portfolio strategy by selecting the top `n` stocks 
    ranked purely by their Earnings Per Share (EPS).
    
    Args:
        stock_tickers (list): List of stock tickers in the universe.
        eps_values (list): Corresponding EPS values for each stock.
        top_n (int): Number of top stocks to include in the baseline (default is 10).
        budget (float): The total budget allocated for this portfolio (default is 10,000).
        
    Returns:
        dict: A dictionary containing the established budget and the portfolio strategy.
              The portfolio is a list of tuples: (stock_ticker, allocation_weight).
    """
    
    # 1. Combine tickers with their respective EPS values
    eps_scores = list(zip(stock_tickers, eps_values))
    
    # 2. Sort the stocks descending strictly based on EPS
    eps_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 3. Isolate the top N highest EPS stocks
    top_stocks = eps_scores[:top_n]
    
    # 4. Normalize their scores to determine portfolio allocation weights
    positive_eps = [max(eps, 0) for ticker, eps in top_stocks]
    total_eps = sum(positive_eps)
    
    portfolio = []
    
    # Allocate proportional to the EPS value, fallback to equal weighting if necessary
    if total_eps > 0:
        for ticker, eps in top_stocks:
            weight = max(eps, 0) / total_eps
            portfolio.append((ticker, weight))
    else:
        num_stocks = len(top_stocks)
        fallback_weight = 1.0 / num_stocks if num_stocks > 0 else 0
        for ticker, _ in top_stocks:
            portfolio.append((ticker, fallback_weight))
            
    # Include the budget so it mathematically represents the full backtesting conditions
    return {
        "budget": budget,
        "portfolio": portfolio
    }
