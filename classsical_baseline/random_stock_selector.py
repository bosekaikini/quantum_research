from typing import Dict, Tuple, Optional, Iterable
import math
import pulp

def composition(
    portfolio: Tuple[str, ...],
    stock_data: Dict[str, Dict[str, float]],
    suggested: Tuple[str, ...],
    budget: float,
    max_assets: Optional[int] = None,
    alpha: float = 1.0,     # weight for EPS
    beta: float = 5.0,      # weight for dividend yield
    gamma: float = 0.1,     # penalty for PE ratio
    max_shares_per_asset: int = 10_000,
    time_limit_sec: Optional[float] = None,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Choose integer shares to buy from `suggested` to best use `budget` according to a
    linear fundamental score built from eps, pe_ratio, dividend_yield, and price.

    Args:
        portfolio: tuple of tickers (kept for API compatibility; not used here).
        stock_data: dict[ticker] -> {"eps","pe_ratio","dividend_yield","price"}
        suggested: tuple of candidate tickers
        budget: available cash to spend (dollars)
        max_assets: optional cap on number of distinct tickers to purchase
        alpha/beta/gamma: weights for eps, dividend_yield, and pe_ratio respectively
        max_shares_per_asset: global big-M fallback
        time_limit_sec: optional solver time limit (seconds)

    Returns:
        (shares, weights)
          shares: dict[ticker] -> integer shares to buy (0 if none)
          weights: dict[ticker] -> fraction of invested capital (sums to 1 over invested assets)
    """

    # --- prepare and validate tickers ---
    tickers = [
        t for t in suggested
        if t in stock_data and stock_data[t].get("price") is not None and stock_data[t]["price"] > 0
    ]
    if not tickers or budget <= 0:
        return {}, {}

    # --- extract raw feature lists (handle missing values using median fallback) ---
    def _get_all(field: str):
        vals = [stock_data[t].get(field) for t in tickers]
        # treat None as missing; compute median of present values
        present = [v for v in vals if v is not None]
        if not present:
            # no info at all -> all zeros
            return {t: 0.0 for t in tickers}
        sorted_p = sorted(present)
        mid = sorted_p[len(sorted_p)//2]
        out = {}
        for t, v in zip(tickers, vals):
            out[t] = v if v is not None else mid
        return out

    eps_map = _get_all("eps")
    pe_map = _get_all("pe_ratio")
    div_map = _get_all("dividend_yield")
    price_map = {t: float(stock_data[t]["price"]) for t in tickers}

    # --- normalize features (min-max) to [0,1] to make weights meaningful ---
    def _minmax_map(m: Dict[str, float]):
        vals = [m[t] for t in tickers]
        vmin, vmax = min(vals), max(vals)
        if vmax == vmin:
            return {t: 0.0 for t in tickers}
        return {t: (m[t] - vmin) / (vmax - vmin) for t in tickers}

    norm_eps = _minmax_map(eps_map)
    norm_pe = _minmax_map(pe_map)
    norm_div = _minmax_map(div_map)

    # --- build linear score per ticker ---
    # note: we subtract gamma * norm_pe because lower PE is preferable
    score = {t: alpha * norm_eps[t] + beta * norm_div[t] - gamma * norm_pe[t] for t in tickers}

    # Optional: if all scores are zero (e.g., no useful info), fall back to neutral score = 1.0
    if all(abs(s) < 1e-12 for s in score.values()):
        score = {t: 1.0 for t in tickers}

    # --- MILP model (integer shares) ---
    model = pulp.LpProblem("composition", pulp.LpMaximize)

    s = {t: pulp.LpVariable(f"s_{t}", lowBound=0, upBound=max_shares_per_asset, cat="Integer") for t in tickers}
    y = {t: pulp.LpVariable(f"y_{t}", cat="Binary") for t in tickers}

    # Objective: maximize sum(score_t * price_t * s_t)
    model += pulp.lpSum(score[t] * price_map[t] * s[t] for t in tickers)

    # Budget constraint: invested dollars <= budget
    model += pulp.lpSum(price_map[t] * s[t] for t in tickers) <= float(budget)

    # Big-M linking: per-ticker M_t = min(global, floor(budget/price))
    for t in tickers:
        per_t_M = min(max_shares_per_asset, int(math.floor(budget / price_map[t])))
        # ensure non-negative M
        per_t_M = max(per_t_M, 0)
        model += s[t] <= per_t_M * y[t]

    # Cardinality constraint
    if max_assets is not None:
        model += pulp.lpSum(y[t] for t in tickers) <= int(max_assets)

    # Solve with optional time limit
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_sec) if time_limit_sec is not None else pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    # --- collect solution ---
    shares: Dict[str, int] = {}
    invested: Dict[str, float] = {}
    total_invested = 0.0
    for t in tickers:
        val = int(pulp.value(s[t]) or 0)
        shares[t] = val
        invested_amount = price_map[t] * val
        invested[t] = invested_amount
        total_invested += invested_amount

    # weights: share of invested capital
    weights: Dict[str, float]
    if total_invested > 0:
        weights = {t: invested[t] / total_invested for t in tickers if invested[t] > 0}
    else:
        weights = {}

    return shares, weights