# Stage 1 — ETF selection

This stage screens a candidate universe organized across the 11 GICS sectors
and selects the most liquid ETF in each sector.

For every candidate, the script downloads recent raw close and volume data from
Twelve Data and computes average daily dollar volume:

`DollarVolume = Close × Volume`

`ADV = mean(DollarVolume)`

## Files

- `data/sector_etf_universe.csv` — candidate ETF metadata and source links.
- `scripts/stage1_liquidity_selection.py` — download, validation, ranking, and selection workflow.
- `data/liquidity_ranking.csv` — generated full ranking; ignored by Git.
- `data/selected_sector_etfs.csv` — generated one-ETF-per-sector selection; ignored by Git.
- `data/data_cache/` — generated provider cache; ignored by Git.

## Run

```powershell
$env:TWELVE_DATA_API_KEY = "your_key"
python ".\Stage 1 - ETF Selection\scripts\stage1_liquidity_selection.py"
```
