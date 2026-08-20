# Iteration 1 — Equity portfolio prototype

## Research hypothesis

A hand-selected group of eight equities could be combined into a portfolio
with attractive return, drawdown, volatility, and SPY-correlation properties
by optimizing a weighted score under simple position limits.

## What this iteration introduced

- Pairwise company comparisons within several industries.
- An eight-asset portfolio rather than isolated stock selection.
- Walk-forward validation instead of relying only on a full-sample optimum.
- Explicit long-only weight bounds and multiple portfolio diagnostics.

## What was not rigorous enough

The optimizer encoded several research judgments as fixed numbers without a
theoretical or empirical justification. In the preserved notebook, each asset
was constrained to **5%–25%** and the objective used fixed penalties of 0.20
for volatility, 0.35 for maximum drawdown, and 0.05 for positive SPY
correlation.

Those parameters make the result look disciplined, but they do not prove that
the chosen bounds or trade-offs are economically correct. Changing them can
change the recommended portfolio.

The asset universe was also selected manually before optimization. Individual
stock comparisons included benchmark correlations, but the research design did
not systematically search or justify the full cross-asset correlation
structure of the portfolio universe.

> Earlier notes recalled a 2%–20% bound. The surviving walk-forward notebook
> actually records 5%–25%; the methodological lesson is the same: the bounds
> were discretionary.

## Evidence

See [`evidence/design_parameters.csv`](evidence/design_parameters.csv) and the
configuration cell in
[`eight_stock_walk_forward_optimizer.ipynb`](notebooks/eight_stock_walk_forward_optimizer.ipynb).

## Falsified claim

The experiment did not establish that a multi-metric score with hand-tuned
coefficients represents a robust portfolio objective. The optimizer could only
solve the preferences supplied to it; it could not validate those preferences.

## Lesson carried forward

Make security selection more systematic and investable before refining the
portfolio optimizer. This motivated the move from hand-selected stocks to a
sector-ETF universe and liquidity-based selection.

## Curated files

- `notebooks/eight_stock_portfolio_weighting.ipynb`
- `notebooks/eight_stock_walk_forward_optimizer.ipynb`
- `evidence/design_parameters.csv`

Notebook outputs and raw price CSVs are intentionally excluded.
