# Stage 3 — Maximum Diversification Portfolio

This stage constructs the long-only Maximum Diversification Portfolio from the
Stage 2 volatility vector and covariance matrix.

The script calculates an unconstrained analytical benchmark and then uses
SciPy SLSQP to maximize the diversification ratio subject to nonnegative
weights that sum to one. It uses equal weights and 25 reproducible random
feasible initializations, then selects the best successful feasible solution.

## Run

```powershell
python ".\Stage 3 - Maximum Diversification Portfolio Construction\scripts\stage3_mdp_optimization.py"
```

Generated weights, optimizer diagnostics, and summary statistics are written
to `outputs/`, which is ignored by Git.
