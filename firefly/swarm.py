from firefly import Firefly
from movement import calculate_movement
from brightness import calculate_brightness, calculate_portfolio_performance_from_prices
from cluster_brightness import cluster_brightness_values, cluster_cumulative_values, select_cluster_indexes
import pandas as pd
from functools import lru_cache
from copy import deepcopy
import yfinance as yf
import concurrent.futures
import random



@lru_cache(maxsize=1)
def get_sp500_tickers(max_tickers: int | None = None) -> list[str]:
    fallback_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    table = pd.read_csv(fallback_url)
    symbols = table["Symbol"].tolist()
    tickers = [symbol.replace(".", "-") for symbol in symbols]
    return tickers if max_tickers is None else tickers[:max_tickers]


def _safe_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fetch_fundamentals(stock: str) -> tuple[str, dict[str, float]]:
    ticker = yf.Ticker(stock)
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    return stock, {
        "eps": _safe_float(info.get("trailingEps") or info.get("forwardEps")),
        "pe_ratio": _safe_float(info.get("trailingPE") or info.get("forwardPE")),
        "dividend_yield": _safe_float(info.get("dividendYield")),
    }


def _normalize_series(values: pd.Series, invert: bool = False) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=float)

    clean = values.replace([pd.NA, pd.NaT], 0.0).fillna(0.0).astype(float)
    if invert:
        clean = 1.0 / (clean + 1e-9)
    if float(clean.max()) > float(clean.min()):
        normalized = (clean - clean.min()) / (clean.max() - clean.min())
    else:
        normalized = clean
    return normalized.astype(float).fillna(0.0)


def _scale_positive_series(values: pd.Series, cap: float) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=float)
    clipped = values.clip(lower=0.0, upper=cap)
    return (clipped / cap).astype(float).fillna(0.0)


def _build_market_signals(price_frame: pd.DataFrame, fundamentals: dict[str, dict[str, float]], tickers: list[str]) -> tuple[list[float], list[float]]:
    if price_frame.empty:
        return [1.0] * len(tickers), [1.0] * len(tickers)

    if hasattr(price_frame.columns, "nlevels") and price_frame.columns.nlevels > 1:
        price_frame = price_frame.droplevel(0, axis=1)

    price_frame = price_frame.ffill().bfill().dropna(axis=1, how="all")
    if price_frame.empty:
        return [1.0] * len(tickers), [1.0] * len(tickers)

    returns = price_frame.pct_change().dropna()
    if returns.empty:
        return [1.0] * len(tickers), [1.0] * len(tickers)

    momentum = (price_frame.iloc[-1] / price_frame.iloc[0] - 1.0).clip(lower=-1.0)
    volatility = returns.std().replace(0.0, pd.NA).fillna(0.0)
    market_breadth = float((momentum > 0).mean()) if len(momentum) else 0.0
    market_trend = float(momentum.mean()) if len(momentum) else 0.0
    regime_factor = max(0.05, min(1.0, 0.65 * market_breadth + 0.35 * max(0.0, market_trend / 0.05)))

    eps = pd.Series({ticker: fundamentals.get(ticker, {}).get("eps", 0.0) for ticker in tickers})
    pe_ratio = pd.Series({ticker: fundamentals.get(ticker, {}).get("pe_ratio", 0.0) for ticker in tickers})
    dividend_yield = pd.Series({ticker: fundamentals.get(ticker, {}).get("dividend_yield", 0.0) for ticker in tickers})

    momentum = momentum.reindex(tickers).fillna(0.0)
    volatility = volatility.reindex(tickers).fillna(0.0)
    eps = eps.reindex(tickers).fillna(0.0)
    pe_ratio = pe_ratio.reindex(tickers).fillna(0.0)
    dividend_yield = dividend_yield.reindex(tickers).fillna(0.0)

    momentum_score = _scale_positive_series(momentum, cap=0.2)
    volatility_score = _normalize_series(volatility, invert=True)
    eps_score = _normalize_series(eps.clip(lower=0.0))
    dividend_score = _normalize_series(dividend_yield.clip(lower=0.0))
    pe_score = _normalize_series(pe_ratio.clip(lower=0.0), invert=True)

    combined = (0.35 * momentum_score + 0.2 * volatility_score + 0.2 * eps_score + 0.15 * dividend_score + 0.1 * pe_score)
    confidence = (0.6 * volatility_score + 0.4 * (momentum_score + eps_score) / 2.0)

    metrics = [float(regime_factor * value) for value in combined.fillna(0.0).tolist()]
    weights = [float(regime_factor * (0.5 + 0.5 * value)) for value in confidence.fillna(0.0).tolist()]
    return metrics, weights


def _extract_price_frame(downloaded: pd.DataFrame) -> pd.DataFrame:
    if downloaded.empty:
        return pd.DataFrame()

    price_key = "Adj Close" if "Adj Close" in downloaded else "Close" if "Close" in downloaded else None
    if price_key is None:
        return pd.DataFrame()

    price_frame = downloaded[price_key]
    if isinstance(price_frame, pd.DataFrame) and hasattr(price_frame.columns, "nlevels") and price_frame.columns.nlevels > 1:
        price_frame = price_frame.droplevel(0, axis=1)
    elif isinstance(price_frame, pd.Series):
        price_frame = price_frame.to_frame()

    if isinstance(price_frame.index, pd.DatetimeIndex) and price_frame.index.tz is not None:
        price_frame.index = price_frame.index.tz_localize(None)
    return price_frame


def _build_rolling_windows(end_date: pd.Timestamp, iterations: int, lookback_days: int, step_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start_anchor = end_date - pd.Timedelta(days=lookback_days + step_days * max(iterations - 1, 0))
    for iteration in range(iterations):
        window_start = start_anchor + pd.Timedelta(days=step_days * iteration)
        window_end = window_start + pd.Timedelta(days=lookback_days)
        windows.append((window_start, window_end))
    return windows


def run_swarm(
    num_fireflies, 
    iterations, 
    metrics, 
    weights, 
    stock_tickers=None, 
    budget=0.0, 
    bounds=(0.0, 1.0),
    cluster_radius=0.25,
    cluster_chosen=3,
    noise_floor=None,
    lookback_days=63,
    step_days=21,
    top_n=None,
    mutation_prob=0.2,
    immigrant_fraction=0.25,
):
    """
    Orchestrates the Firefly algorithm over a given number of iterations.
    
    Args:
        num_fireflies (int): The number of fireflies in the swarm.
        iterations (int): The number of iterations to run.
        metrics (list): Metric data for the stock universe.
        weights (list): Weightings for those metrics.
        stock_tickers (list): The list of stock tickers.
        budget (float): The budget for the portfolios.
        bounds (tuple): Coordinate boundaries for the firefly [x, y] spawn mapping.
        
    Returns:
        dict: iteration history, cumulative return, and final best portfolio information
    """
    if stock_tickers is None:
        stock_tickers = get_sp500_tickers()

    fundamentals: dict[str, dict[str, float]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        for stock, data in executor.map(_fetch_fundamentals, stock_tickers):
            fundamentals[stock] = data

    end_date = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    windows = _build_rolling_windows(end_date, iterations, lookback_days, step_days)

    global_start = windows[0][0] - pd.Timedelta(days=5)
    price_history = yf.download(list(stock_tickers), start=global_start, end=end_date + pd.Timedelta(days=1), auto_adjust=True, progress=False)
    benchmark_history = yf.download("^GSPC", start=global_start, end=end_date + pd.Timedelta(days=1), auto_adjust=True, progress=False)
    full_price_frame = _extract_price_frame(price_history)
    benchmark_price_frame = _extract_price_frame(benchmark_history)

    first_window_price_frame = full_price_frame.loc[windows[0][0]:windows[0][1]].dropna(how="all")
    derived_metrics, derived_weights = _build_market_signals(first_window_price_frame, fundamentals, list(stock_tickers))

    if metrics is None or not any(metrics) or len(set(metrics)) <= 1:
        metrics = derived_metrics
    if weights is None or not any(weights) or len(set(weights)) <= 1:
        weights = derived_weights
    
    # 1. Initialize the entire swarm of fireflies
    swarm = [Firefly(metrics, weights, stock_tickers, budget, bounds, top_n=top_n) for _ in range(num_fireflies)]
    history: list[dict[str, object]] = []
    cumulative_value = float(budget)
    best_firefly = None
    best_performance = {
        "fitness": float("-inf"),
        "cumulative_return": 0.0,
        "portfolio_value": float(budget),
        "avg_daily_return": 0.0,
        "volatility": 0.0,
    }

    # 2. Iteration Loop
    for iteration, (window_start, window_end) in enumerate(windows):
        window_price_frame = full_price_frame.loc[window_start:window_end].dropna(how="all")
        window_benchmark_frame = benchmark_price_frame.loc[window_start:window_end].dropna(how="all")

        iteration_metrics, iteration_weights = _build_market_signals(window_price_frame, fundamentals, list(stock_tickers))
        if iteration_metrics:
            for firefly in swarm:
                firefly.metrics = iteration_metrics
                firefly.weights = iteration_weights
        for firefly in swarm:
            firefly.rebuild_portfolio()

        raw_brightness = [
            calculate_portfolio_performance_from_prices(firefly, window_price_frame, window_benchmark_frame)["fitness"]
            for firefly in swarm
        ]
        for firefly, brightness in zip(swarm, raw_brightness):
            firefly.brightness = brightness

        cluster_scores = cluster_brightness_values(swarm, raw_brightness, cluster_radius)
        cluster_totals = cluster_cumulative_values(swarm, raw_brightness, cluster_radius)
        selected_indexes = select_cluster_indexes(cluster_scores, noise_floor, cluster_chosen)
        selected_fireflies = [swarm[index] for index in selected_indexes]

        iteration_records = []
        for firefly in selected_fireflies:
            performance = calculate_portfolio_performance_from_prices(firefly, window_price_frame, window_benchmark_frame)
            iteration_records.append((firefly, performance))

        if iteration_records:
            iteration_best_firefly, iteration_best_performance = max(
                iteration_records,
                key=lambda item: item[1]["fitness"],
            )
        else:
            iteration_best_firefly = swarm[0]
            iteration_best_performance = calculate_portfolio_performance_from_prices(iteration_best_firefly, window_price_frame, window_benchmark_frame)

        if iteration_best_performance["fitness"] > best_performance["fitness"]:
            best_firefly = deepcopy(iteration_best_firefly)
            best_performance = dict(iteration_best_performance)

        cumulative_value *= (1.0 + iteration_best_performance["cumulative_return"])
        cumulative_return = (cumulative_value / float(budget)) - 1.0 if budget else 0.0

        history.append(
            {
                "iteration": iteration + 1,
                "window_start": window_start.date().isoformat(),
                "window_end": window_end.date().isoformat(),
                "cluster_indexes": selected_indexes,
                "cluster_scores": [cluster_scores[index] for index in selected_indexes],
                "cluster_totals": [cluster_totals[index] for index in selected_indexes],
                "best_portfolio": iteration_best_firefly.portfolio,
                "iteration_return": iteration_best_performance["cumulative_return"],
                "iteration_portfolio_value": iteration_best_performance["portfolio_value"],
                "cumulative_return": cumulative_return,
                "cumulative_value": cumulative_value,
                "fitness": iteration_best_performance["fitness"],
            }
        )

        next_swarm: list[Firefly] = []
        if selected_fireflies:
            elite_count = max(1, min(len(selected_fireflies), int(num_fireflies * 0.25)))
            for elite in selected_fireflies[:elite_count]:
                elite_copy = Firefly(
                    elite.metrics,
                    elite.weights,
                    stock_tickers,
                    budget,
                    bounds,
                    position=tuple(elite.position),
                    top_n=top_n,
                )
                elite_copy.rebuild_portfolio()
                next_swarm.append(elite_copy)

            while len(next_swarm) < num_fireflies:
                if random.random() < immigrant_fraction:
                    immigrant = Firefly(metrics, weights, stock_tickers, budget, bounds, top_n=top_n)
                    immigrant.rebuild_portfolio()
                    next_swarm.append(immigrant)
                    continue

                parent = selected_fireflies[len(next_swarm) % len(selected_fireflies)]
                if len(selected_fireflies) > 1:
                    partner = selected_fireflies[(len(next_swarm) + 1) % len(selected_fireflies)]
                    new_position = calculate_movement(parent, partner)
                else:
                    new_position = tuple(parent.position)

                if random.random() < mutation_prob:
                    new_position = (
                        min(1.0, max(0.0, new_position[0] + random.uniform(-0.15, 0.15))),
                        min(1.0, max(0.0, new_position[1] + random.uniform(-0.15, 0.15))),
                    )

                child = Firefly(
                    metrics,
                    weights,
                    stock_tickers,
                    budget,
                    bounds,
                    position=new_position,
                    top_n=top_n,
                )
                child.rebuild_portfolio()
                next_swarm.append(child)
        else:
            next_swarm = [Firefly(metrics, weights, stock_tickers, budget, bounds, top_n=top_n) for _ in range(num_fireflies)]

        swarm = next_swarm

    return {
        "history": history,
        "best_firefly": best_firefly,
        "best_portfolio": best_firefly.portfolio if best_firefly is not None else [],
        "best_performance": best_performance,
        "cumulative_return": (cumulative_value / float(budget)) - 1.0 if budget else 0.0,
        "cumulative_value": cumulative_value,
    }


if __name__ == "__main__":
    demo_tickers = get_sp500_tickers(30)
    result = run_swarm(
        num_fireflies=16,
        iterations=6,
        metrics=None,
        weights=None,
        stock_tickers=demo_tickers,
        budget=10_000,
        top_n=8,
    )
    for entry in result["history"]:
        print(
            f"Iteration {entry['iteration']} ({entry['window_start']} -> {entry['window_end']}): return={entry['iteration_return']:.4f}, "
            f"cumulative_return={entry['cumulative_return']:.4f}, portfolio_value={entry['iteration_portfolio_value']:.2f}"
        )
        print("Best portfolio:", entry["best_portfolio"])

    print("Final best fitness:", result["best_performance"]["fitness"])
    print("Final cumulative return:", result["cumulative_return"])
    print("Final cumulative value:", result["cumulative_value"])
