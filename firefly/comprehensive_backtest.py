import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import concurrent.futures
import sys
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from swarm import get_sp500_tickers, run_swarm, _extract_price_frame, _build_rolling_windows, _fetch_fundamentals
from baseline_eps import baseline_eps
from brightness import calculate_portfolio_performance_from_prices

from random_implementation.selection_and_composition import select_stocks as sc_select, get_composition_weights as sc_weights
from random_implementation.fully_random import select_stocks as fr_select, get_composition_weights as fr_weights
from oracle_test.oracle_strat import run_oracle_strategy

def get_metrics(history_arr, name):
    arr = np.array(history_arr)
    # Avoid division by zero by adding a small epsilon
    denom = arr[:-1]
    denom = np.where(denom == 0, 1e-9, denom)
    pct_returns = (arr[1:] / denom) - 1.0
    
    start_val = arr[0] if arr[0] != 0 else 1e-9
    tot_return = (arr[-1] / start_val) - 1.0
    
    # rough daily return est where each window represents ~21 days of delta
    expected_daily = np.mean(pct_returns) / 21
    vol = np.std(pct_returns) * np.sqrt(252/21)
    
    ann_return = (1 + tot_return) ** (252 / 731) - 1.0 if tot_return > -1 else -1.0
    sharpe = (ann_return - 0.04) / vol if vol > 0 else 0
    run_max = np.maximum.accumulate(arr)
    run_max = np.where(run_max == 0, 1e-9, run_max)
    drawdowns = (arr - run_max) / run_max
    max_dd = np.min(drawdowns)
    fake_fitness = (expected_daily * 252) / (vol + 1e-6) - abs(max_dd) * 0.5
    
    return f"| {name} | {tot_return*100:.2f}% | {ann_return*100:.2f}% | {sharpe:.2f} | {max_dd*100:.2f}% | {vol*100:.2f}% | {fake_fitness:.3f} |"

def main():
    print(f"--- Running Comprehensive Backtest ---")
    tickers = get_sp500_tickers(30)
    budget = 10000.0
    
    iterations = 32
    end_date = pd.Timestamp("2025-01-01")
    
    # 1. Run Firefly Swarm
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
    
    oracle_sp500_cumulative = budget
    oracle_sp500_history_values = [oracle_sp500_cumulative]
    
    oracle_stock_cumulative = budget
    oracle_stock_history_values = [oracle_stock_cumulative]
    
    swarm_history_values = [budget]
    
    budget_history_values = [budget]
    
    window_labels = ["Start"]
    
    # Run the identical iterating evaluation timeframe loop logic
    for idx, (window_start, window_end) in enumerate(windows):
        if idx < len(windows) - 1:
            segment_end = windows[idx + 1][0]
        else:
            segment_end = window_start + pd.Timedelta(days=21)
            
        real_pf = full_price_frame.loc[window_start:segment_end].dropna(how="all")
        real_bf = benchmark_price_frame.loc[window_start:segment_end].dropna(how="all")
        
        # Baseline EPS
        perf = calculate_portfolio_performance_from_prices(eps_mock, real_pf, real_bf)
        eps_cumulative_value *= (1.0 + perf["cumulative_return"])
        eps_history_values.append(eps_cumulative_value)
        
        # Native S&P 500 Buy & Hold return
        if not real_bf.empty:
            start_val = float(real_bf.iloc[0].squeeze())
            end_val = float(real_bf.iloc[-1].squeeze())
            sp500_return = (end_val / start_val) - 1.0 if start_val > 0 else 0.0
        else:
            sp500_return = 0.0
        sp500_cumulative_value *= (1.0 + sp500_return)
        sp500_history_values.append(sp500_cumulative_value)
        
        # Oracle Strategy (S&P 500 Timing)
        if not real_bf.empty:
            bf_prices = real_bf.squeeze().tolist()
            oracle_sp500_res = run_oracle_strategy(bf_prices, budget=oracle_sp500_cumulative)
            oracle_sp500_cumulative = oracle_sp500_res["final_value"]
        oracle_sp500_history_values.append(oracle_sp500_cumulative)
        
        # Oracle Strategy (Best Individual Stock Timing)
        best_oracle_stock_val = oracle_stock_cumulative
        for col in real_pf.columns:
            stock_prices = real_pf[col].dropna().tolist()
            if len(stock_prices) > 1:
                res = run_oracle_strategy(stock_prices, budget=oracle_stock_cumulative)
                if res["final_value"] > best_oracle_stock_val:
                    best_oracle_stock_val = res["final_value"]
        oracle_stock_cumulative = best_oracle_stock_val
        oracle_stock_history_values.append(oracle_stock_cumulative)
        
        # Random: Select + Compose
        perf_sc = calculate_portfolio_performance_from_prices(sc_mock, real_pf, real_bf)
        sc_cumulative *= (1.0 + perf_sc["cumulative_return"])
        sc_history.append(sc_cumulative)
        
        # Random: Fully Random
        perf_fr = calculate_portfolio_performance_from_prices(fr_mock, real_pf, real_bf)
        fr_cumulative *= (1.0 + perf_fr["cumulative_return"])
        fr_history.append(fr_cumulative)
        
        # Swarm Firefly
        swarm_history_values.append(swarm_result["history"][idx]["cumulative_value"])
        
        # Budget Baseline
        budget_history_values.append(budget)
        
        window_labels.append(window_end.strftime("%Y-%m-%d"))

    # 3. Create cleanly formatted graph WITH ORACLE
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(window_labels, oracle_stock_history_values, label="Oracle (Best Individual Stock)", color="orange", linewidth=2)
    ax.plot(window_labels, oracle_sp500_history_values, label="Oracle (S&P 500 Timing)", color="gold", linewidth=2)
    ax.plot(window_labels, swarm_history_values, label="Firefly Method", color="blue", linewidth=2)
    ax.plot(window_labels, eps_history_values, label="Baseline EPS Strategy", color="brown", linewidth=2)
    ax.plot(window_labels, sc_history, label="Random: Select + Compose", color="green", linestyle="dashdot", linewidth=1.5)
    ax.plot(window_labels, fr_history, label="Random: Fully Random", color="purple", linestyle="dashed", linewidth=1.5)
    
    ax.plot(window_labels, sp500_history_values, label="S&P 500 (Buy & Hold)", color="black", linestyle=":", linewidth=2)
    ax.plot(window_labels, budget_history_values, label="Budget (Baseline)", color="black", linestyle=":", linewidth=2)
    
    ax.set_title("Strategy Comparison WITH Oracles (2023 - 2025)")
    ax.set_ylabel("Total Portfolio Equity")
    ax.set_xlabel("Rolling Investment Window Timeframe")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    out_dir = ROOT_DIR / "plots" / "comprehensive_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file_with = out_dir / "comprehensive_backtest_with_oracle.png"
    plt.savefig(out_file_with, dpi=300)
    print(f"Saved plot WITH oracle to {out_file_with}")
    plt.close()

    # Create cleanly formatted graph WITHOUT ORACLE
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(window_labels, swarm_history_values, label="Firefly Method", color="blue", linewidth=2)
    ax.plot(window_labels, eps_history_values, label="Baseline EPS Strategy", color="brown", linewidth=2)
    ax.plot(window_labels, sc_history, label="Random: Select + Compose", color="green", linestyle="dashdot", linewidth=1.5)
    ax.plot(window_labels, fr_history, label="Random: Fully Random", color="purple", linestyle="dashed", linewidth=1.5)
    
    ax.plot(window_labels, sp500_history_values, label="S&P 500 (Buy & Hold)", color="black", linestyle=":", linewidth=2)
    ax.plot(window_labels, budget_history_values, label="Budget (Baseline)", color="black", linestyle=":", linewidth=2)
    
    ax.set_title("Strategy Comparison WITHOUT Oracles (2023 - 2025)")
    ax.set_ylabel("Total Portfolio Equity")
    ax.set_xlabel("Rolling Investment Window Timeframe")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    out_file_no = out_dir / "comprehensive_backtest_no_oracle.png"
    plt.savefig(out_file_no, dpi=300)
    print(f"Saved plot WITHOUT oracle to {out_file_no}")
    plt.close()

    print("\n\n### Comprehensive Strategy Performance Metrics Matrix")
    print("| Strategy | Total Return | Ann. Return | Sharpe Ratio | Max Drawdown | Ann. Volatility | Est. Fitness |")
    print("|----------|--------------|-------------|--------------|--------------|-----------------|--------------|")
    print(get_metrics(oracle_stock_history_values, "Oracle (Best Individual Stock Timing)"))
    print(get_metrics(oracle_sp500_history_values, "Oracle (S&P 500 Timing)"))
    print(get_metrics(swarm_history_values, "Firefly Optimization Swarm"))
    print(get_metrics(eps_history_values, "Baseline: Top 8 EPS"))
    print(get_metrics(sp500_history_values, "S&P 500 (Buy & Hold)"))
    print(get_metrics(budget_history_values, "Budget Baseline (Cash)"))
    print(get_metrics(sc_history, "Random: Select + Compose"))
    print(get_metrics(fr_history, "Random: Fully Random"))
    print("\n\n")

if __name__ == "__main__":
    main()
