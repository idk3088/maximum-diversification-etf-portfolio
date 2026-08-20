# Stage 4 — Portfolio evaluation

This stage evaluates the fixed Stage 3 MDP weights against two benchmarks:

- an equal-weight portfolio of the same sector ETFs;
- SPY using Tiingo adjusted end-of-day prices.

It reports cumulative and annualized return, annualized volatility, zero-rate
Sharpe ratio, SPY correlation, maximum drawdown, allocation statistics, and
the achieved diversification ratio. A second script calculates the 252-day
rolling MDP/SPY correlation.

## Run

```powershell
$env:TIINGO_API_TOKEN = "your_token"
python ".\Stage 4 - Portfolio Evaluation\scripts\stage4_portfolio_evaluation.py"
python ".\Stage 4 - Portfolio Evaluation\scripts\stage4_rolling_correlation_analysis.py"
```

The provider cache and all generated results are ignored by Git.
