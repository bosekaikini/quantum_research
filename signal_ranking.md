# Strategy Importance Ranking

Using Random Matrix Theory (the Marchenko-Pastur distribution wrapper), we filtered out noise-driven variation to isolate true mathematically significant signals. By identifying eigenvalues strictly greater than $\lambda_+$, we extracted the principal component vectors representing the true independent drivers of portfolio movement.

The table below ranks the simulated strategies by their importance—calculated via their squared weights within the 'signal' eigenvectors, proportionately scaled by each signal's eigenvalue.

| Rank | Strategy Module | RMT Importance Score |
|------|-----------------|----------------------|
| 1 | **Metric_Eps** | 14.20% |
| 2 | **Metric_Pe** | 14.20% |
| 3 | **Metric_Div** | 14.20% |
| 4 | **Combination** | 14.20% |
| 5 | **Composition** | 12.58% |
| 6 | **Stock_Number** | 10.67% |
| 7 | **Selection** | 6.37% |
| 8 | **Selection_And_Composition** | 4.85% |
| 9 | **Random** | 4.83% |
| 10 | **Fully_Random** | 3.91% |

*Analysis derived from $T=501$ daily increments over $N=10$ strategies. MP Upper Bound: $\lambda_+=1.3025$. Significant Signals Discovered: 1.*
