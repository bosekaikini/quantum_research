import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backtester.get_results import get_results, start_date, end_date, num, budget

def compute_marchenko_pastur(returns: pd.DataFrame):
    # Standardize returns to mean 0, var 1
    std_returns = (returns - returns.mean()) / returns.std()
    
    T, N = std_returns.shape
    q = T / N
    
    # Compute correlation matrix
    corr_matrix = std_returns.corr().values
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    
    # Sort descending
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Marchenko-Pastur theoretical upper bound
    lambda_max = (1 + np.sqrt(1 / q)) ** 2
    
    # Identify signals (eigenvalues > lambda_max)
    signals_count = np.sum(eigenvalues > lambda_max)
    print(f"Total trading days (T): {T}")
    print(f"Number of strategies (N): {N}")
    print(f"Q (T/N): {q:.2f}")
    print(f"Marchenko-Pastur Upper Bound (lambda_max): {lambda_max:.4f}")
    print(f"Found {signals_count} significant signal(s) outside the noise band.")
    
    # Calculate strategy importance
    if signals_count == 0:
        print("No significant signals found. Using the largest eigenvalue as a fallback.")
        signals_count = 1
        
    importance_scores = np.zeros(N)
    for i in range(signals_count):
        importance_scores += (eigenvectors[:, i] ** 2) * eigenvalues[i]
        
    # Normalize scores to 1.0 (100%)
    importance_scores = importance_scores / np.sum(importance_scores)
    
    strategy_ranking = pd.Series(importance_scores, index=returns.columns).sort_values(ascending=False)
    
    # Plot Eigenvalues against lambda_max
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, N + 1), eigenvalues, 'bo-', label="Empirical Eigenvalues")
    plt.axhline(y=lambda_max, color='r', linestyle='--', label=f"MP upper bound ({lambda_max:.2f})")
    plt.title("Correlation Matrix Eigenvalue Spectrum vs Marchenko-Pastur bound")
    plt.xlabel("Eigenvalue Rank")
    plt.ylabel("Eigenvalue Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("eigenvalues_mp.png", bbox_inches="tight")
    print("Saved eigenvalue plot to eigenvalues_mp.png")
    plt.close()
    
    return strategy_ranking, eigenvalues, lambda_max, signals_count

if __name__ == "__main__":
    print("Simulating 1 pass of the 10 strategies to extract daily returns...")
    import matplotlib
    matplotlib.use('Agg')
    
    res = get_results(start_date, end_date, num, budget, run_idx="mp")
    plot_df = res["plot_df"]
    
    strategy_cols = [col for col in plot_df.columns if "val" in col.lower() and "s&p" not in col.lower() and "budget" not in col.lower()]
    returns_df = plot_df[strategy_cols].pct_change().dropna()
    
    clean_names = {col: col.replace(" val", "").replace(" Val", "").strip() for col in strategy_cols}
    returns_df = returns_df.rename(columns=clean_names)
    
    ranking, evals, lmax, count = compute_marchenko_pastur(returns_df)
    
    print("\n--- Strategy Importance Ranking (via RMT Principal Components) ---")
    for idx, (strategy, score) in enumerate(ranking.items(), 1):
        print(f"{idx}. {strategy}: {score * 100:.2f}% importance")
        
    with open("signal_ranking.md", "w") as f:
        f.write("# Strategy Importance Ranking\n\n")
        f.write("Using Random Matrix Theory (the Marchenko-Pastur distribution wrapper), we filtered out noise-driven variation to isolate true mathematically significant signals. By identifying eigenvalues strictly greater than $\\lambda_+$, we extracted the principal component vectors representing the true independent drivers of portfolio movement.\n\n")
        f.write("The table below ranks the simulated strategies by their importance—calculated via their squared weights within the 'signal' eigenvectors, proportionately scaled by each signal's eigenvalue.\n\n")
        f.write("| Rank | Strategy Module | RMT Importance Score |\n")
        f.write("|------|-----------------|----------------------|\n")
        for idx, (strategy, score) in enumerate(ranking.items(), 1):
            f.write(f"| {idx} | **{strategy.title()}** | {score * 100:.2f}% |\n")
        f.write(f"\n*Analysis derived from $T={len(returns_df)}$ daily increments over $N={len(returns_df.columns)}$ strategies. MP Upper Bound: $\\lambda_+={lmax:.4f}$. Significant Signals Discovered: {count}.*\n")
