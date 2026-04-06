import numpy as np
import yfinance as yf
from functools import lru_cache


@lru_cache(maxsize=8)
def _load_price_data(
    tickers: tuple[str, ...],
    start_date: str | None,
    end_date: str | None,
    period: str | None,
) -> tuple[np.ndarray, list[str]]:
    download_kwargs: dict[str, object] = {"interval": "1d", "progress": False}
    if start_date is not None and end_date is not None:
        download_kwargs["start"] = start_date
        download_kwargs["end"] = end_date
    elif period is not None:
        download_kwargs["period"] = period

    data = yf.download(list(tickers), **download_kwargs)
    price_key = "Adj Close" if "Adj Close" in data else "Close" if "Close" in data else None
    if price_key is None:
        return np.empty((0, 0)), []

    price_frame = data[price_key]
    price_frame = price_frame.dropna(axis=1, thresh=int(len(price_frame) * 0.8))
    price_frame = price_frame.ffill().bfill()

    returns_df = price_frame.pct_change().dropna()
    matrix = returns_df.values.T
    final_tickers = returns_df.columns.tolist()
    return matrix, final_tickers


@lru_cache(maxsize=8)
def _benchmark_average_daily_return(
    start_date: str | None,
    end_date: str | None,
    period: str | None,
) -> float:
    download_kwargs: dict[str, object] = {"interval": "1d", "progress": False}
    if start_date is not None and end_date is not None:
        download_kwargs["start"] = start_date
        download_kwargs["end"] = end_date
    elif period is not None:
        download_kwargs["period"] = period

    data = yf.download("^GSPC", **download_kwargs)
    price_key = "Adj Close" if "Adj Close" in data else "Close" if "Close" in data else None
    if price_key is None or data.empty:
        return 0.0

    prices = data[price_key]
    if hasattr(prices, "columns"):
        prices = prices.iloc[:, 0]

    prices = prices.dropna()
    returns = prices.pct_change().dropna()
    if returns.empty:
        return 0.0

    mean_value = returns.mean()
    if hasattr(mean_value, "iloc"):
        mean_value = mean_value.iloc[0]
    return float(mean_value)


def calculate_brightness(firefly, tickers, period="1mo", start_date: str | None = None, end_date: str | None = None):
    performance = calculate_portfolio_performance(
        firefly,
        tickers,
        period=period,
        start_date=start_date,
        end_date=end_date,
    )
    return performance["fitness"]


def _portfolio_performance_from_returns(
    firefly,
    matrix: np.ndarray,
    final_tickers: list[str],
    benchmark_avg_daily_return: float,
) -> dict[str, float]:
    if matrix.size == 0 or not final_tickers:
        return {
            "fitness": 0.0,
            "cumulative_return": 0.0,
            "portfolio_value": float(getattr(firefly, "budget", 0.0)),
            "avg_daily_return": 0.0,
            "volatility": 0.0,
        }

    ticker_map = {ticker: i for i, ticker in enumerate(final_tickers)}
    portfolio = firefly.get_list() if hasattr(firefly, "get_list") else getattr(firefly, "portfolio", [])

    aligned = [item for item in portfolio if item[0] in ticker_map and item[1] >= 0]
    if not aligned:
        return {
            "fitness": 0.0,
            "cumulative_return": 0.0,
            "portfolio_value": float(getattr(firefly, "budget", 0.0)),
            "avg_daily_return": 0.0,
            "volatility": 0.0,
        }

    weights = np.array([item[1] for item in aligned], dtype=float)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        weights = np.full(len(aligned), 1.0 / len(aligned), dtype=float)
    elif weight_sum > 1.0:
        weights = weights / weight_sum

    indices = [ticker_map[item[0]] for item in aligned]
    sub_returns = matrix[indices]
    port_rets = np.dot(weights, sub_returns)

    cumulative_return = float(np.prod(1.0 + port_rets) - 1.0)
    avg_daily_return = float(np.mean(port_rets))
    volatility = float(np.std(port_rets))
    excess_avg_daily_return = avg_daily_return - benchmark_avg_daily_return
    portfolio_value = float(getattr(firefly, "budget", 0.0)) * (1.0 + cumulative_return)

    fitness = cumulative_return * 100.0
    if volatility > 0:
        fitness += (avg_daily_return / volatility) * 2.0

    return {
        "fitness": fitness,
        "cumulative_return": cumulative_return,
        "portfolio_value": portfolio_value,
        "avg_daily_return": avg_daily_return,
        "benchmark_avg_daily_return": benchmark_avg_daily_return,
        "excess_avg_daily_return": excess_avg_daily_return,
        "volatility": volatility,
    }


def calculate_portfolio_performance(
    firefly,
    tickers,
    period="1mo",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, float]:
    matrix, final_tickers = _load_price_data(tuple(tickers), start_date, end_date, period)
    benchmark_avg_daily_return = _benchmark_average_daily_return(start_date, end_date, period)
    return _portfolio_performance_from_returns(firefly, matrix, final_tickers, benchmark_avg_daily_return)


def calculate_portfolio_performance_from_prices(
    firefly,
    price_frame,
    benchmark_prices,
) -> dict[str, float]:
    if price_frame is None or getattr(price_frame, "empty", True):
        return {
            "fitness": 0.0,
            "cumulative_return": 0.0,
            "portfolio_value": float(getattr(firefly, "budget", 0.0)),
            "avg_daily_return": 0.0,
            "benchmark_avg_daily_return": 0.0,
            "excess_avg_daily_return": 0.0,
            "volatility": 0.0,
        }

    if hasattr(price_frame, "columns") and getattr(price_frame.columns, "nlevels", 1) > 1:
        price_frame = price_frame.droplevel(0, axis=1)

    price_frame = price_frame.ffill().bfill().dropna(axis=1, how="all")
    returns_df = price_frame.pct_change().dropna()
    matrix = returns_df.values.T
    final_tickers = returns_df.columns.tolist()

    benchmark_avg_daily_return = 0.0
    if benchmark_prices is not None and not getattr(benchmark_prices, "empty", True):
        if hasattr(benchmark_prices, "columns"):
            benchmark_prices = benchmark_prices.iloc[:, 0]
        benchmark_prices = benchmark_prices.dropna()
        benchmark_returns = benchmark_prices.pct_change().dropna()
        if not benchmark_returns.empty:
            benchmark_avg_daily_return = float(benchmark_returns.mean())

    return _portfolio_performance_from_returns(firefly, matrix, final_tickers, benchmark_avg_daily_return)