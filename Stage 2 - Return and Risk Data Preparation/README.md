# Stage 2 — Return and risk preparation

This stage reads the Stage 1 selection, downloads adjusted end-of-day prices
from Tiingo, aligns the common trading history, and calculates:

- simple daily returns;
- annualized volatility using `sqrt(252)`;
- the annualized sample covariance matrix;
- a validated coverage summary.

It does not select ETFs or optimize portfolio weights.

## Run

```powershell
$env:TIINGO_API_TOKEN = "your_token"
python ".\Stage 2 - Return and Risk Data Preparation\scripts\stage2_return_risk_preparation.py"
```

Provider caches are written under `data/raw_tiingo/`, while analytical files
are written under `outputs/`. Both locations are ignored by Git.
