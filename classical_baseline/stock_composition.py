"""
TODO:

1. Import stock selector
2. If changes from before -- look at nets of each trade option
3. Submit sell/buy orders accordingly
"""

from typing import Dict, Tuple, Optional
import math
import pulp

def composition(
    portfolio: Tuple[str, ...],
    stock_data: Dict[str, Dict[str, float]],
    suggested: Tuple[str, ...],
    budget: float,
    max_assets: Optional[int] = None,
    max_shares_per_asset: int = 10_000,
    min_dollar_per_asset: Optional[float] = None,
    time_limit_sec: Optional[float] = None,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    MILP composition function.

    Args:
        portfolio: tuple of tickers (kept for API compatibility; not used in this simple model).
        stock_data: dict[ticker] -> {"eps", "pe_ratio", "dividend_yield", "price"} (price required)
        suggested: tuple of candidate tickers (from your selector)
        budget: available cash to spend (dollars)
        max_assets: optional cap on number of distinct tickers to purchase
        max_shares_per_asset: general upper bound for integer shares (falls back to per-ticker M)
        min_dollar_per_asset: optional minimum dollar invested per selected asset
        time_limit_sec: optional solver time limit (seconds)

    Returns:
        shares: dict[ticker] -> integer number of shares to buy (0 if none)
        weights: dict[ticker] -> fraction of invested cash (sums to 1 over invested assets)
    """

    # Filter suggested tickers to those present in stock_data with a valid positive price
    tickers = [t for t in suggested if (t in stock_data and stock_data[t].get("price") is not None and stock_data[t]["price"] > 0)]
    if not tickers or budget <= 0:
        return {}, {}

    # Build model
    model = pulp.LpProblem("composition", pulp.LpMaximize)

    # Decision variables
    s = {t: pulp.LpVariable(f"s_{t}", lowBound=0, upBound=max_shares_per_asset, cat="Integer") for t in tickers}
    y = {t: pulp.LpVariable(f"y_{t}", cat="Binary") for t in tickers}

    # Objective: maximize invested dollars (neutral when no expected-return provided)
    model += pulp.lpSum(stock_data[t]["price"] * s[t] for t in tickers), "max_invested"

    # Budget constraint
    model += pulp.lpSum(stock_data[t]["price"] * s[t] for t in tickers) <= float(budget), "budget"

    # Big-M per ticker: choose M_t = min(max_shares_per_asset, floor(budget / price))
    for t in tickers:
        price = stock_data[t]["price"]
        # safe per-ticker upper bound (how many shares could we possibly buy of this ticker)
        per_ticker_M = min(max_shares_per_asset, int(math.floor(budget / price)) if price > 0 else 0)
        # ensure at least 1 if price <= budget
        per_ticker_M = max(per_ticker_M, 0)
        model += s[t] <= per_ticker_M * y[t]

    # Cardinality constraint (optional)
    if max_assets is not None:
        model += pulp.lpSum(y[t] for t in tickers) <= int(max_assets)

    # Minimum dollar per selected asset (optional)
    if min_dollar_per_asset is not None:
        for t in tickers:
            model += stock_data[t]["price"] * s[t] >= float(min_dollar_per_asset) * y[t]

    # Solve with optional time limit
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_sec) if time_limit_sec is not None else pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    # Extract solution
    shares: Dict[str, int] = {}
    invested: Dict[str, float] = {}
    total_invested = 0.0

    for t in tickers:
        val = int(pulp.value(s[t]) or 0)
        shares[t] = val
        invested_amount = stock_data[t]["price"] * val
        invested[t] = invested_amount
        total_invested += invested_amount

    # Compute weights (fractions of invested cash)
    if total_invested > 0:
        weights = {t: invested[t] / total_invested for t in tickers if invested[t] > 0}
    else:
        weights = {}

    return shares, weights