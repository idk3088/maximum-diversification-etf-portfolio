# Data and publication notes

This repository contains code and project structure, not bundled market data.

The workflow downloads data from Twelve Data and Tiingo using credentials
provided through environment variables. Provider caches, aligned price and
return series, covariance estimates, portfolio results, and generated charts
are ignored by Git so that a public clone does not redistribute licensed data.

Before publishing any generated dataset, result table, chart, report, or web
page, review the current terms of the relevant provider and obtain any required
redistribution permission. Keep attribution with any publication when the
provider requires it.

Credentials must never be placed in scripts, notebooks, CSV files, screenshots,
README examples, commit messages, or GitHub Actions secrets unless the intended
workflow and repository visibility have been reviewed carefully.
