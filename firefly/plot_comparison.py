import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import concurrent.futures
import sys

from swarm import get_sp500_tickers, run_swarm, _extract_price_frame, _build_rolling_windows, _fetch_fundamentals
from baseline_eps import baseline_eps
from brightness import calculate_portfolio_performance_from_prices

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from random_implementation.selection_and_composition import select_stocks as sc_select, get_composition_weights as sc_weights
from random_implementation.fully_random import select_stocks as fr_select, get_composition_weights as fr_weights

def main(run_idx=1):
    print(f"--- Running Backtest Pass {run_idx} ---")
    tickers = get_sp500_tickers(30)
    budget = 10000.0
    
    # Match the backtest period from get_results.py
    iterations = 32
    end_date = pd.Timestamp("2025-01-01")
    
    swarm_result = run_swarm(
        num_fireflies=16,
        iterations=iterations,
        metrics=None,
        weights=None,
        stock_tickers=tickers,
        budget=budget,
        top_n=8,
        end_date=end_date,
    )
    
    # 2. Re-fetch identical market info over identical dimensions for a 1:1 identical backtest comparison
    windows = _build_rolling_windows(end_date, iterations, 63, 21)
    global_start = windows[0][0] - pd.Timedelta(days=5)
    
    price_history = yf.download(list(tickers), start=global_start, end=end_date + pd.Timedelta(days=1), auto_adjust=True, progress=False)
    benchmark_history = yf.download("^GSPC", start=global_start, end=end_date + pd.Timedelta(days=1), auto_adjust=True, progress=False)
    
    full_price_frame = _extract_price_frame(price_history)
    benchmark_price_frame = _extract_price_frame(benchmark_history)
    
    fundamentals = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        for stock, data in executor.map(_fetch_fundamentals, tickers):
            fundamentals[stock] = data
            
    eps_values = [fundamentals.get(t, {}).get("eps", 0.0) for t in tickers]
    
    # Generate EPS baseline portfolio manually using the user's isolated function
    eps_portfolio_result = baseline_eps(tickers, eps_values, top_n=8, budget=budget)
    eps_portfolio = eps_portfolio_result["portfolio"]
    
    class MockFirefly:
        def __init__(self, port, b):
            self.portfolio = port
            self.budget = b
            
    eps_mock = MockFirefly(eps_portfolio, budget)
    
    # Generate mock random portfolios initialized once dynamically identically
    sc_selection = sc_select(tickers, fundamentals, n=8)
    sc_portfolio_dict = sc_weights(sc_selection)
    sc_portfolio = [(t, w) for t, w in sc_portfolio_dict.items()]
    sc_mock = MockFirefly(sc_portfolio, budget)
    
    fr_selection = fr_select(tickers, fundamentals, n=8)
    fr_portfolio_dict = fr_weights(fr_selection)
    fr_portfolio = [(t, w) for t, w in fr_portfolio_dict.items()]
    fr_mock = MockFirefly(fr_portfolio, budget)
    
    eps_cumulative_value = budget
    eps_history_values = [eps_cumulative_value]
    
    sc_cumulative = budget
    sc_history = [sc_cumulative]
    
    fr_cumulative = budget
    fr_history = [fr_cumulative]
    
    sp500_cumulative_value = budget
    sp500_history_values = [sp500_cumulative_value]
    
    swarm_history_values = [budget]
    window_labels = ["Start"]
    
    # Run the identical iterating evaluation timeframe loop logic
    for idx, (window_start, window_end) in enumerate(windows):
        window_price_frame = full_price_frame.loc[window_start:window_end].dropna(how="all")
        window_benchmark_frame = benchmark_price_frame.loc[window_start:window_end].dropna(how="all")
        
        if idx < len(windows) - 1:
            segment_end = windows[idx + 1][0]
        else:
            segment_end = window_start + pd.Timedelta(days=21)
            
        real_pf = full_price_frame.loc[window_start:segment_end].dropna(how="all")
        real_bf = benchmark_price_frame.loc[window_start:segment_end].dropna(how="all")
        
        perf = calculate_portfolio_performance_from_prices(eps_mock, real_pf, real_bf)
        eps_cumulative_value *= (1.0 + perf["cumulative_return"])
        eps_history_values.append(eps_cumulative_value)
        
        # Calculate native S&P 500 Buy & Hold return for identical rolling multiplication
        if not real_bf.empty:
            start_val = float(real_bf.iloc[0].squeeze())
            end_val = float(real_bf.iloc[-1].squeeze())
            sp500_return = (end_val / start_val) - 1.0 if start_val > 0 else 0.0
        else:
            sp500_return = 0.0
        sp500_cumulative_value *= (1.0 + sp500_return)
        sp500_history_values.append(sp500_cumulative_value)
        
        perf_sc = calculate_portfolio_performance_from_prices(sc_mock, real_pf, real_bf)
        sc_cumulative *= (1.0 + perf_sc["cumulative_return"])
        sc_history.append(sc_cumulative)
        
        perf_fr = calculate_portfolio_performance_from_prices(fr_mock, real_pf, real_bf)
        fr_cumulative *= (1.0 + perf_fr["cumulative_return"])
        fr_history.append(fr_cumulative)
        
        swarm_history_values.append(swarm_result["history"][idx]["cumulative_value"])
        window_labels.append(window_end.strftime("%Y-%m-%d"))

    # Produce integer baseline outputs required
    eps_total_return = ((eps_cumulative_value / budget) - 1.0) * 100.0
    print(f"EPS_RETURN:{eps_total_return:.2f}")
    
    # 3. Create cleanly formatted graph mapping both arrays natively
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(window_labels, swarm_history_values, label="Firefly Method", color="blue", linewidth=2)
    ax.plot(window_labels, eps_history_values, label="Baseline EPS Strategy (Top 10)", color="black", linestyle="dotted", linewidth=2)
    ax.plot(window_labels, sp500_history_values, label="S&P 500 (Buy & Hold)", color="red", linestyle="--", linewidth=2)
    ax.plot(window_labels, sc_history, label="Random: Select + Compose", color="green", linestyle="dashdot", linewidth=1.5)
    ax.plot(window_labels, fr_history, label="Random: Fully Random", color="purple", linestyle="dashed", linewidth=1.5)
    
    ax.set_title("Quantum Research: Firefly Backtest vs EPS Baseline")
    ax.set_ylabel("Total Portfolio Equity ($10k Genesis)")
    ax.set_xlabel("Rolling Investment Window Timeframe")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Directly save plot into the current directory
    out_file = ROOT_DIR / "plots" / "firefly_comparison" / ("backtest_plot.png" if run_idx == 1 else f"backtest_plot_{run_idx}.png")
    plt.savefig(out_file, dpi=300)
    print(f"Saved plot to {out_file}")
    
    # Calculate detailed performance metrics across all arrays
    import numpy as np
    def get_metrics(history_arr, name):
        arr = np.array(history_arr)
        pct_returns = (arr[1:] / arr[:-1]) - 1.0
        tot_return = (arr[-1] / arr[0]) - 1.0
        # rough daily return est where each window represents ~21 days of delta
        expected_daily = np.mean(pct_returns) / 21
        vol = np.std(pct_returns) * np.sqrt(252/21)
        ann_return = (1 + tot_return) ** (252 / 731) - 1.0
        sharpe = (ann_return - 0.04) / vol if vol > 0 else 0
        run_max = np.maximum.accumulate(arr)
        drawdowns = (arr - run_max) / run_max
        max_dd = np.min(drawdowns)
        fake_fitness = (expected_daily * 252) / (vol + 1e-6) - abs(max_dd) * 0.5 # proxy matching internal params
        return f"| {name} | {tot_return*100:.2f}% | {ann_return*100:.2f}% | {sharpe:.2f} | {max_dd*100:.2f}% | {vol*100:.2f}% | {fake_fitness:.3f} |"

    print("\n\n### Strategy Performance Metrics Matrix (Pass {})".format(run_idx))
    print("| Strategy | Total Return | Ann. Return | Sharpe Ratio | Max Drawdown | Ann. Volatility | Est. Fitness |")
    print("|----------|--------------|-------------|--------------|--------------|-----------------|--------------|")
    print(get_metrics(swarm_history_values, "Firefly Optimization Swarm"))
    print(get_metrics(eps_history_values, "Baseline: Top 8 EPS"))
    print(get_metrics(sp500_history_values, "Baseline: S&P 500 Index"))
    print(get_metrics(sc_history, "Random: Select + Compose"))
    print(get_metrics(fr_history, "Random: Fully Random"))
    print("\n\n")

if __name__ == "__main__":
    for i in [1, 2, 3]:
        main(i)
