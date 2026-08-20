# Iteration 3 — Daily machine-learning forecasting

## Research hypothesis

Machine-learning models trained on engineered daily ETF features could predict
future daily returns well enough to support portfolio construction.

## What this iteration improved

- The notebook workflow was refactored into modular Python source files.
- OLS, Ridge, Lasso, Random Forest, and XGBoost were evaluated under a common
  model-selection framework.
- Purged time-series splits and explicit target-alignment checks were added.
- The one-standard-error rule was used to prefer a simpler model when its
  validation error was statistically competitive.
- Dedicated tests were written for look-ahead, target alignment, fold
  construction, and model selection.

This was a substantial improvement in software and validation discipline.

## Evidence that rejected the hypothesis

The selected-model evidence covers five ETFs and six monthly selection dates,
for 30 selected models in total. Their overall mean out-of-sample R² was:

**-0.5240**

A negative out-of-sample R² means the forecasts were worse than the relevant
fold-specific training-mean benchmark. The result was not a small shortfall
from a strong signal; it was evidence that the daily prediction setup did not
provide reliable incremental forecasting power.

| ETF | Selected-model observations | Mean OOS R² |
|---|---:|---:|
| XLE | 6 | -0.3200 |
| XLF | 6 | -0.6800 |
| XLK | 6 | -0.3800 |
| XLP | 6 | -0.6600 |
| XLV | 6 | -0.5800 |
| **Overall** | **30** | **-0.5240** |

The compact evidence is preserved in
[`evidence/selected_model_oos_r2_summary.csv`](evidence/selected_model_oos_r2_summary.csv).

## Falsified claim

High-frequency machine-learning ideas cannot be transferred mechanically to a
small daily ETF dataset. With the chosen features, horizon, and sample, market
noise dominated the learnable signal.

## Lesson carried forward

Do not treat model complexity as evidence of economic signal. The next version
abandoned prediction-first portfolio construction and returned to a transparent
portfolio-level objective that could be inspected directly.

## Curated files

- `code/src/` — feature, validation, model-selection, and leakage-control modules.
- `code/tests/` — tests for temporal leakage and model-selection behavior.
- `code/scripts/` — historical orchestration scripts.
- `code/config.yaml` and `code/requirements.txt`.
- `evidence/selected_model_oos_r2_summary.csv`.

The large price-data cache and detailed prediction files are excluded.
