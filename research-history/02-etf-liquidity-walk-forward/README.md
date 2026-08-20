# Iteration 2 — ETF liquidity and walk-forward selection

## Research hypothesis

Replacing hand-selected stocks with liquid sector ETFs would create a more
systematic, investable, and historically testable portfolio universe.

## What this iteration improved

- ETF candidates were screened using average dollar volume rather than an
  informal preference for familiar securities.
- Five-year eligibility and data-quality rules were made explicit.
- Holdings were selected and evaluated through a monthly walk-forward process.
- Later observations were excluded from each holding month's training window.
- Audit files and immutable timestamped result folders were introduced.

These changes materially improved the implementation and the time ordering of
the calculations.

## The remaining leakage problem

Point-in-time return calculations are not enough if the research universe is
defined with today's knowledge. Sector sleeves, ETF classifications, and the
candidate pool were still partly based on a present-day perspective and an
incomplete historical ETF database.

That creates residual **survivorship and classification leakage**: a backtest
can evaluate past dates using a set of funds or sector labels that was easier to
identify because they are known today. A liquid fund selected from a biased
candidate set is still selected from a biased candidate set.

The notebooks themselves began to recognize this limitation. Historical notes
state that a manually declared research pool can retain survivorship or
classification bias even when the monthly calculations are point-in-time.

## Falsified claim

Using average dollar volume made the selection rule more objective, but it did
not by itself make the entire investment universe point-in-time or
survivor-bias-free.

## Lesson carried forward

Separate three questions that had previously been mixed together:

1. Was the asset genuinely available at the historical decision date?
2. Was its sector classification known and valid at that date?
3. Was it sufficiently liquid using only information available then?

This motivated more explicit leakage tests, historical metadata work, and
purged time-series validation in the next iteration.

## Curated files

- `notebooks/01_point_in_time_universe.ipynb`
- `notebooks/02_rolling_etf_selection.ipynb`
- `notebooks/03_monthly_walk_forward_optimizer.ipynb`

Notebook outputs, provider caches, generated workbooks, and raw market data are
excluded.
