# Iteration 4 — Minimum-correlation objective

## Research hypothesis

A sector-ETF portfolio designed to minimize correlation with SPY would produce
a meaningfully diversified allocation.

## What this iteration improved

- The universe covered one selected ETF for each of the 11 GICS sectors.
- Alternative correlation-estimation windows were compared.
- Regularization strength was tested through a walk-forward framework.
- Concentration, HHI, active holdings, and constraint effects were measured
  explicitly rather than left implicit.

## Evidence that exposed objective misalignment

The selected regularized solution used only two ETFs:

| ETF | Sector | Weight |
|---|---|---:|
| XLE | Energy | **81.01%** |
| XLP | Consumer Staples | 18.99% |
| Remaining nine ETFs | — | 0.00% |

The pre-constraint solution in a related diagnostic had a maximum individual
weight of 84.66%, a top-three weight sum of 100%, and an effective number of
holdings of only 1.35.

These results are preserved in the small aggregate files under `evidence/`.
No raw return or price series is included.

## What actually failed

This was not primarily an optimizer bug. The optimizer found an efficient way
to satisfy the objective it was given. The problem was that **minimum SPY
correlation is not the same thing as a well-diversified portfolio**.

The objective did not adequately reward breadth or balanced risk contribution.
A tiny set of assets with unusual recent correlation properties could dominate
the solution even when the resulting portfolio was economically undesirable.

Adding an arbitrary maximum-weight constraint can hide the symptom, but it does
not repair the underlying objective.

## Falsified claim

Very low benchmark correlation, optimized in isolation, is not a sufficient
definition of diversification.

## Lesson carried forward

Choose an objective that measures the desired portfolio property directly and
has a defensible theoretical interpretation. This led to the literature-based
Maximum Diversification Portfolio objective used in ETF Portfolio 4.0, where
diversification is evaluated through the relationship between weighted asset
volatility and total portfolio volatility.

## Curated files

- `scripts/` — liquidity selection, correlation-window selection, and
  minimum-correlation optimization.
- `evidence/optimal_regularized_portfolio_weights.csv`.
- `evidence/portfolio_constraint_comparison.csv`.
- `evidence/lambda_summary_results.csv`.

Raw prices, daily return matrices, and caches are excluded.
