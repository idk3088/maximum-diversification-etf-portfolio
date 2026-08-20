# Appendix A — Covariance-window robustness

This optional analysis tests how sensitive the MDP is to the covariance
estimation window. It compares 63, 126, 252, 504, and 756 trading days with the
full available history while keeping the Stage 3 objective, constraints, and
optimizer settings unchanged.

The analysis compares weights, diversification ratio, annualized return,
annualized volatility, Sharpe ratio, SPY correlation, maximum drawdown, and
pairwise L1 distances between weight vectors.

## Run

Run Stages 1–4 first, then:

```powershell
python ".\Appendix A - Covariance Window Robustness Test\scripts\appendix_window_robustness_test.py"
```

All appendix outputs are ignored by Git.
