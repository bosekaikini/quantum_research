# Independent Parameter SHAP Valuation

By dissecting the 10 strategy implementations down to their sub-level boolean logic gates (e.g. 'Is Selection Random?' vs 'Does it use EPS?'), we successfully mapped out **5,010 specific observations** of 'Parameter Logic' → 'Daily P&L Return'.

We injected the S&P 500 Daily Return as the fundamental control variable (`Market_Return`) and fed the matrix into a Random Forest Regressor algorithm. Below are the **absolute global SHAP values** (average magnitude feature impact per day) confirming exactly which independent parameter is the prime driver of portfolio swings.

| Rank | Structural Parameter Limit | Global SHAP Importance (Mean Absolute $|\phi|$) |
|------|----------------------------|----------------------------------------------------|
| 1 | `Market_Return` | **0.005480** |
| 2 | `Random_Number` | **0.000116** |
| 3 | `Metric_MIXED` | **0.000079** |
| 4 | `Random_Selection` | **0.000058** |
| 5 | `Random_Composition` | **0.000038** |
| 6 | `Metric_EPS` | **0.000000** |
| 7 | `Metric_PE` | **0.000000** |
| 8 | `Metric_DIV` | **0.000000** |
