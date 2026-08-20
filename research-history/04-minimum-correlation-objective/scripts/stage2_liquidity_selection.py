"""Stage 2: select the most liquid ETF in each GICS sector.

Market data is downloaded from Twelve Data's documented ``/time_series`` API.
The API key is read from the ``TWELVE_DATA_API_KEY`` environment variable and
is never stored in this script. For each ETF, the request explicitly sets
``adjust=none`` so the script uses raw daily Close and Volume fields:

    DollarVolume = Close * Volume
    ADV = mean(DollarVolume)

Downloaded observations are cached as one CSV file per ticker in ``data_cache``.
On later runs, a ticker is requested only when its cache is absent or does not
adequately cover the configured observation window. Each missing ticker is
requested once per run; there is no automatic retry loop.

This script intentionally does not calculate returns, correlations, Sharpe
ratios, or portfolio weights.

Local usage:
    python -m pip install pandas
    # PowerShell:
    $env:TWELVE_DATA_API_KEY="paste_your_key_here"
    python stage2_liquidity_selection.py
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install it with: python -m pip install pandas"
    ) from exc


# ---------------------------------------------------------------------------
# User-configurable settings
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "sector_etf_universe.csv"
LIQUIDITY_RANKING_FILE = BASE_DIR / "liquidity_ranking.csv"
SELECTED_ETFS_FILE = BASE_DIR / "selected_sector_etfs.csv"
CACHE_DIR = BASE_DIR / "data_cache"

# Three recent calendar months normally contain about 60-66 US trading days.
# Set END_DATE to "YYYY-MM-DD" to reproduce a historical snapshot, or leave it
# as None to use the computer's local run date.
LOOKBACK_MONTHS = 3
END_DATE: str | None = None
MIN_VALID_TRADING_DAYS = 40

# A cache whose latest observation is at least this recent is accepted. A
# one-business-day allowance handles weekends, holidays, and vendor publication
# lag without downloading the same ticker repeatedly.
CACHE_ALLOWED_BUSINESS_DAY_LAG = 1

# Twelve Data's Basic plan currently permits 8 API credits per minute, with one
# /time_series symbol costing one credit. The default batches therefore contain
# at most 8 symbols and wait for the next quota window. Paid-plan users may
# increase SYMBOLS_PER_REQUEST and reduce BATCH_PAUSE_SECONDS if appropriate.
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
TWELVE_DATA_API_URL = "https://api.twelvedata.com/time_series"
SYMBOLS_PER_REQUEST = 8
BATCH_PAUSE_SECONDS = 65
HTTP_TIMEOUT_SECONDS = 30
HTTP_USER_AGENT = "sector-etf-liquidity-research/1.0"
DATA_SOURCE = "Twelve Data time_series API (adjust=none)"

EXPECTED_GICS_SECTORS = (
    "Information Technology",
    "Health Care",
    "Financials",
    "Consumer Discretionary",
    "Communication Services",
    "Industrials",
    "Consumer Staples",
    "Energy",
    "Utilities",
    "Real Estate",
    "Materials",
)

INPUT_COLUMNS = (
    "ticker",
    "etf_name",
    "issuer",
    "underlying_index",
    "gics_sector",
    "source",
)

RANKING_COLUMNS = (
    "ticker",
    "etf_name",
    "issuer",
    "gics_sector",
    "ADV",
    "observation_period",
    "data_source",
)

SELECTION_COLUMNS = (
    "gics_sector",
    "ticker",
    "etf_name",
    "issuer",
    "ADV",
    "observation_period",
)

CACHE_COLUMNS = ("Date", "Close", "Volume")


class Stage2Error(RuntimeError):
    """Raised when an input, download, cache, or output validation fails."""


def load_etf_universe(input_file: Path = INPUT_FILE) -> pd.DataFrame:
    """Load and validate the Stage 1 ETF universe."""
    if not input_file.exists():
        raise Stage2Error(f"Input file not found: {input_file}")

    try:
        universe = pd.read_csv(input_file, dtype=str)
    except Exception as exc:
        raise Stage2Error(f"Could not read {input_file}: {exc}") from exc

    missing_columns = sorted(set(INPUT_COLUMNS) - set(universe.columns))
    if missing_columns:
        raise Stage2Error(
            "Stage 1 input is missing required columns: "
            + ", ".join(missing_columns)
        )

    universe = universe.loc[:, INPUT_COLUMNS].copy()
    for column in INPUT_COLUMNS:
        universe[column] = universe[column].fillna("").str.strip()

    blank_columns = [column for column in INPUT_COLUMNS if universe[column].eq("").any()]
    if blank_columns:
        raise Stage2Error(
            "Stage 1 input contains blank values in required columns: "
            + ", ".join(blank_columns)
        )

    universe["ticker"] = universe["ticker"].str.upper()
    duplicates = sorted(
        universe.loc[universe["ticker"].duplicated(keep=False), "ticker"].unique()
    )
    if duplicates:
        raise Stage2Error(
            "Duplicate tickers found in Stage 1 input: " + ", ".join(duplicates)
        )

    actual_sectors = set(universe["gics_sector"])
    expected_sectors = set(EXPECTED_GICS_SECTORS)
    missing_sectors = sorted(expected_sectors - actual_sectors)
    unexpected_sectors = sorted(actual_sectors - expected_sectors)
    if missing_sectors or unexpected_sectors:
        details: list[str] = []
        if missing_sectors:
            details.append("missing sectors: " + ", ".join(missing_sectors))
        if unexpected_sectors:
            details.append("unexpected sectors: " + ", ".join(unexpected_sectors))
        raise Stage2Error("Invalid GICS sector coverage (" + "; ".join(details) + ")")

    return universe


def resolve_observation_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the requested start date and inclusive end date."""
    if LOOKBACK_MONTHS <= 0:
        raise Stage2Error("LOOKBACK_MONTHS must be a positive integer.")
    if MIN_VALID_TRADING_DAYS <= 0:
        raise Stage2Error("MIN_VALID_TRADING_DAYS must be a positive integer.")
    if CACHE_ALLOWED_BUSINESS_DAY_LAG < 0:
        raise Stage2Error("CACHE_ALLOWED_BUSINESS_DAY_LAG cannot be negative.")
    if SYMBOLS_PER_REQUEST <= 0:
        raise Stage2Error("SYMBOLS_PER_REQUEST must be positive.")
    if BATCH_PAUSE_SECONDS < 0:
        raise Stage2Error("BATCH_PAUSE_SECONDS cannot be negative.")

    if END_DATE is None:
        end_inclusive = pd.Timestamp(date.today())
    else:
        try:
            end_inclusive = pd.Timestamp(END_DATE)
        except Exception as exc:
            raise Stage2Error("END_DATE must use YYYY-MM-DD format.") from exc
        if pd.isna(end_inclusive):
            raise Stage2Error("END_DATE must use YYYY-MM-DD format.")

    if end_inclusive.tzinfo is not None:
        end_inclusive = end_inclusive.tz_localize(None)
    end_inclusive = end_inclusive.normalize()
    start = (end_inclusive - pd.DateOffset(months=LOOKBACK_MONTHS)).normalize()

    if start >= end_inclusive:
        raise Stage2Error("The configured observation window is empty.")
    return start, end_inclusive


def _cache_path(ticker: str) -> Path:
    """Return the deterministic cache path for a ticker."""
    return CACHE_DIR / f"{ticker.upper()}.csv"


def _normalize_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a provider/cache frame to a dated Close/Volume DataFrame."""
    empty = pd.DataFrame(columns=["Close", "Volume"])
    if frame is None or frame.empty:
        return empty

    normalized = frame.copy()
    normalized.columns = [str(column).strip().title() for column in normalized.columns]
    if not set(CACHE_COLUMNS).issubset(normalized.columns):
        return empty

    normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
    normalized["Close"] = pd.to_numeric(normalized["Close"], errors="coerce")
    normalized["Volume"] = pd.to_numeric(normalized["Volume"], errors="coerce")
    normalized = normalized.dropna(subset=["Date"])
    normalized = normalized.set_index("Date").loc[:, ["Close", "Volume"]]
    normalized.index = pd.DatetimeIndex(normalized.index).tz_localize(None).normalize()
    normalized = normalized.loc[~normalized.index.duplicated(keep="last")]
    return normalized.sort_index()


def load_cached_market_data(ticker: str) -> pd.DataFrame:
    """Load one ticker's local cache; invalid cache files are ignored."""
    path = _cache_path(ticker)
    if not path.exists():
        return pd.DataFrame(columns=["Close", "Volume"])

    try:
        cached = _normalize_history(pd.read_csv(path))
    except Exception as exc:
        logging.warning("Ignoring unreadable cache for %s: %s", ticker, exc)
        return pd.DataFrame(columns=["Close", "Volume"])

    if cached.empty:
        logging.warning("Ignoring empty or invalid cache for %s.", ticker)
    return cached


def _valid_window_rows(
    history: pd.DataFrame, start: pd.Timestamp, end_inclusive: pd.Timestamp
) -> pd.DataFrame:
    """Return valid positive-price, nonnegative-volume rows in the window."""
    if history.empty or not {"Close", "Volume"}.issubset(history.columns):
        return pd.DataFrame(columns=["Close", "Volume"])

    window = history.loc[
        (history.index >= start) & (history.index <= end_inclusive),
        ["Close", "Volume"],
    ].copy()
    return window.loc[
        window["Close"].notna()
        & window["Volume"].notna()
        & (window["Close"] > 0)
        & (window["Volume"] >= 0)
    ]


def cache_covers_window(
    history: pd.DataFrame, start: pd.Timestamp, end_inclusive: pd.Timestamp
) -> bool:
    """Return True when cached data is sufficiently complete and recent."""
    valid = _valid_window_rows(history, start, end_inclusive)
    if len(valid) < MIN_VALID_TRADING_DAYS:
        return False

    required_latest = (
        end_inclusive - pd.offsets.BDay(CACHE_ALLOWED_BUSINESS_DAY_LAG)
    ).normalize()
    return valid.index.min() <= start + pd.offsets.BDay(3) and valid.index.max() >= required_latest


def _ticker_batches(tickers: list[str], batch_size: int):
    """Yield deterministic ticker batches."""
    for index in range(0, len(tickers), batch_size):
        yield tickers[index : index + batch_size]


def _twelve_data_url(
    tickers: list[str],
    start: pd.Timestamp,
    end_inclusive: pd.Timestamp,
    api_key: str,
) -> str:
    """Build one Twelve Data batch time-series request."""
    query = urlencode(
        {
            "symbol": ",".join(tickers),
            "interval": "1day",
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end_inclusive.strftime("%Y-%m-%d"),
            "order": "asc",
            # The provider defaults to split adjustment. Liquidity uses the
            # raw as-traded close, so adjustment is explicitly disabled.
            "adjust": "none",
            "apikey": api_key,
        }
    )
    return f"{TWELVE_DATA_API_URL}?{query}"


def _parse_twelve_data_series(ticker: str, result: object) -> pd.DataFrame:
    """Parse one symbol result from a Twelve Data response."""
    if not isinstance(result, dict):
        raise Stage2Error(f"{ticker}: malformed Twelve Data response")

    if result.get("status") == "error":
        message = str(result.get("message", "unknown API error"))
        raise Stage2Error(f"{ticker}: Twelve Data error: {message}")

    values = result.get("values")
    if not isinstance(values, list) or not values:
        raise Stage2Error(f"{ticker}: Twelve Data returned no time-series values")

    frame = pd.DataFrame(values).rename(
        columns={"datetime": "Date", "close": "Close", "volume": "Volume"}
    )
    history = _normalize_history(frame)
    if history.empty:
        raise Stage2Error(
            f"{ticker}: Twelve Data response lacks datetime, close, or volume"
        )
    return history


def download_twelve_data_batch(
    tickers: list[str],
    start: pd.Timestamp,
    end_inclusive: pd.Timestamp,
    api_key: str,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Download one batch once and return per-ticker data and errors."""
    request = Request(
        _twelve_data_url(tickers, start, end_inclusive, api_key),
        headers={"User-Agent": HTTP_USER_AGENT, "Accept": "application/json"},
    )

    try:
        with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            payload_text = response.read().decode("utf-8-sig", errors="replace")
    except HTTPError as exc:
        raise Stage2Error(f"Twelve Data returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise Stage2Error(f"Could not reach Twelve Data ({exc.reason})") from exc
    except TimeoutError as exc:
        raise Stage2Error("Twelve Data request timed out") from exc

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise Stage2Error("Twelve Data returned invalid JSON") from exc

    if not isinstance(payload, dict):
        raise Stage2Error("Twelve Data returned an unexpected response structure")

    # A one-symbol response contains meta/values/status directly. A batch
    # response is keyed by symbol. Normalize both forms to a symbol map.
    if len(tickers) == 1 and ("values" in payload or payload.get("status") == "error"):
        symbol_results: dict[str, object] = {tickers[0]: payload}
    else:
        symbol_results = {
            str(symbol).upper(): result
            for symbol, result in payload.items()
            if isinstance(result, dict)
        }

    downloaded: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    for ticker in tickers:
        result = symbol_results.get(ticker.upper())
        if result is None:
            failures.append(f"{ticker}: missing from Twelve Data batch response")
            continue
        try:
            downloaded[ticker] = _parse_twelve_data_series(ticker, result)
        except Stage2Error as exc:
            failures.append(str(exc))

    return downloaded, failures


def _merge_history(cached: pd.DataFrame, downloaded: pd.DataFrame) -> pd.DataFrame:
    """Merge cache and fresh data, preferring fresh duplicate dates."""
    combined = pd.concat([cached, downloaded]).sort_index()
    return combined.loc[~combined.index.duplicated(keep="last")]


def _write_cache(ticker: str, history: pd.DataFrame) -> None:
    """Write a ticker cache atomically as Date, Close, Volume."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker)
    temporary = path.with_suffix(".csv.tmp")

    output = history.reset_index().rename(columns={"index": "Date"})
    output = output.loc[:, CACHE_COLUMNS]
    output.to_csv(temporary, index=False, date_format="%Y-%m-%d")
    temporary.replace(path)


def download_market_data(
    tickers: list[str], start: pd.Timestamp, end_inclusive: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    """Load caches and batch-request every missing ticker at most once."""
    if not tickers:
        raise Stage2Error("No ETF tickers were supplied for data acquisition.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    market_data: dict[str, pd.DataFrame] = {}
    missing_tickers: list[str] = []

    for ticker in tickers:
        cached = load_cached_market_data(ticker)
        market_data[ticker] = cached
        if cache_covers_window(cached, start, end_inclusive):
            logging.info("Using cached data for %s.", ticker)
        else:
            missing_tickers.append(ticker)

    logging.info(
        "Cache satisfied %d of %d tickers; %d ticker(s) require download.",
        len(tickers) - len(missing_tickers),
        len(tickers),
        len(missing_tickers),
    )

    if not missing_tickers:
        return market_data
    if not TWELVE_DATA_API_KEY:
        raise Stage2Error(
            "TWELVE_DATA_API_KEY is not set. Create a Twelve Data API key, then "
            "set it before running. PowerShell example: "
            "$env:TWELVE_DATA_API_KEY='paste_your_key_here'"
        )

    batches = list(_ticker_batches(missing_tickers, SYMBOLS_PER_REQUEST))
    failures: list[str] = []
    for position, batch in enumerate(batches, start=1):
        if position > 1 and BATCH_PAUSE_SECONDS:
            logging.info(
                "Waiting %d seconds for the next API-credit window.",
                BATCH_PAUSE_SECONDS,
            )
            time.sleep(BATCH_PAUSE_SECONDS)

        logging.info(
            "Downloading Twelve Data batch %d/%d (%d ticker(s)).",
            position,
            len(batches),
            len(batch),
        )
        try:
            downloaded_by_ticker, batch_failures = download_twelve_data_batch(
                batch,
                start,
                end_inclusive,
                TWELVE_DATA_API_KEY,
            )
            failures.extend(batch_failures)
        except Stage2Error as exc:
            failures.append(f"batch {position} ({', '.join(batch)}): {exc}")
            continue

        for ticker, downloaded in downloaded_by_ticker.items():
            combined = _merge_history(market_data[ticker], downloaded)
            if not cache_covers_window(combined, start, end_inclusive):
                valid_days = len(_valid_window_rows(combined, start, end_inclusive))
                failures.append(
                    f"{ticker}: only {valid_days} valid/recent trading days after download"
                )
                continue
            market_data[ticker] = combined
            _write_cache(ticker, combined)

    if failures:
        raise Stage2Error(
            "Data acquisition failed for the ticker(s) below. Successful downloads "
            "were cached, so a later rerun will request only unresolved tickers:\n- "
            + "\n- ".join(failures)
        )

    return market_data


def calculate_adv(
    universe: pd.DataFrame,
    market_data: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end_inclusive: pd.Timestamp,
) -> pd.DataFrame:
    """Calculate mean daily Close * Volume for every candidate ETF."""
    rows: list[dict[str, object]] = []
    problems: list[str] = []

    for record in universe.itertuples(index=False):
        ticker = record.ticker
        valid = _valid_window_rows(
            market_data.get(ticker, pd.DataFrame()), start, end_inclusive
        )
        if len(valid) < MIN_VALID_TRADING_DAYS:
            problems.append(
                f"{ticker}: only {len(valid)} valid trading days; "
                f"minimum is {MIN_VALID_TRADING_DAYS}"
            )
            continue

        # Twelve Data's raw Close field (adjust=none) is used directly; no
        # adjusted-close or total-return series is requested or substituted.
        dollar_volume = valid["Close"] * valid["Volume"]
        adv = float(dollar_volume.mean())
        if not math.isfinite(adv) or adv <= 0:
            problems.append(f"{ticker}: ADV is not a finite positive value")
            continue

        rows.append(
            {
                "ticker": ticker,
                "etf_name": record.etf_name,
                "issuer": record.issuer,
                "gics_sector": record.gics_sector,
                "ADV": adv,
                "observation_period": (
                    f"{valid.index.min():%Y-%m-%d} to "
                    f"{valid.index.max():%Y-%m-%d} ({len(valid)} trading days)"
                ),
                "data_source": DATA_SOURCE,
            }
        )

    if problems:
        raise Stage2Error(
            "ADV validation failed. No output files were written:\n- "
            + "\n- ".join(problems)
        )

    result = pd.DataFrame(rows, columns=RANKING_COLUMNS)
    if len(result) != len(universe):
        raise Stage2Error(
            f"ADV was calculated for {len(result)} of {len(universe)} ETFs."
        )
    return result


def rank_sector_etfs(adv_data: pd.DataFrame) -> pd.DataFrame:
    """Sort each sector from highest to lowest ADV."""
    sector_order = {
        sector: position for position, sector in enumerate(EXPECTED_GICS_SECTORS)
    }
    ranked = adv_data.copy()
    ranked["_sector_order"] = ranked["gics_sector"].map(sector_order)
    ranked = ranked.sort_values(
        ["_sector_order", "ADV", "ticker"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return ranked.drop(columns="_sector_order").reset_index(drop=True)


def select_liquid_etfs(ranking: pd.DataFrame) -> pd.DataFrame:
    """Select the highest-ADV row in every sector without manual overrides."""
    # Alphabetical ticker order is only a deterministic tie-break for exactly
    # equal ADV values; no issuer or ETF family receives preference.
    selected = ranking.groupby("gics_sector", sort=False, as_index=False).head(1)
    return selected.loc[:, SELECTION_COLUMNS].reset_index(drop=True)


def validate_output(
    universe: pd.DataFrame, ranking: pd.DataFrame, selected: pd.DataFrame
) -> None:
    """Validate coverage, uniqueness, row counts, and the selection rule."""
    if tuple(ranking.columns) != RANKING_COLUMNS:
        raise Stage2Error("liquidity_ranking.csv columns do not match the specification.")
    if tuple(selected.columns) != SELECTION_COLUMNS:
        raise Stage2Error("selected_sector_etfs.csv columns do not match the specification.")
    if len(ranking) != len(universe):
        raise Stage2Error("Liquidity ranking does not contain every Stage 1 candidate ETF.")
    if ranking["ticker"].duplicated().any():
        raise Stage2Error("Liquidity ranking contains duplicate tickers.")
    if set(ranking["ticker"]) != set(universe["ticker"]):
        raise Stage2Error("Liquidity ranking ticker set differs from the Stage 1 universe.")
    if set(ranking["gics_sector"]) != set(EXPECTED_GICS_SECTORS):
        raise Stage2Error("Liquidity ranking does not represent all 11 GICS sectors.")

    if len(selected) != len(EXPECTED_GICS_SECTORS):
        raise Stage2Error("Selection must contain exactly 11 ETFs.")
    if selected["gics_sector"].duplicated().any():
        raise Stage2Error("Selection contains more than one ETF for a sector.")
    if selected["ticker"].duplicated().any():
        raise Stage2Error("Selection contains duplicate tickers.")
    if set(selected["gics_sector"]) != set(EXPECTED_GICS_SECTORS):
        raise Stage2Error("Selection does not contain exactly one ETF per GICS sector.")

    valid_adv = ranking["ADV"].map(
        lambda value: math.isfinite(float(value)) and float(value) > 0
    )
    if ranking["ADV"].isna().any() or not valid_adv.all():
        raise Stage2Error("Ranking contains missing, non-finite, or non-positive ADV values.")

    maximums = ranking.groupby("gics_sector")["ADV"].max()
    for row in selected.itertuples(index=False):
        if not math.isclose(
            float(row.ADV),
            float(maximums.loc[row.gics_sector]),
            rel_tol=1e-12,
            abs_tol=0.01,
        ):
            raise Stage2Error(
                f"{row.ticker} is not the highest-ADV ETF in {row.gics_sector}."
            )


def _write_csv_atomically(frame: pd.DataFrame, output_file: Path) -> None:
    """Write a CSV through a temporary file to avoid partial output."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.2f")
    temporary.replace(output_file)


def save_results(
    ranking: pd.DataFrame,
    selected: pd.DataFrame,
    ranking_file: Path = LIQUIDITY_RANKING_FILE,
    selected_file: Path = SELECTED_ETFS_FILE,
) -> None:
    """Save the two required Stage 2 output files."""
    _write_csv_atomically(ranking, ranking_file)
    _write_csv_atomically(selected, selected_file)
    logging.info("Saved full ranking to %s", ranking_file)
    logging.info("Saved sector selections to %s", selected_file)


def main() -> None:
    """Run the complete Stage 2 liquidity-selection workflow."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    universe = load_etf_universe()
    start, end_inclusive = resolve_observation_window()
    logging.info(
        "Using %s data requested from %s through %s.",
        DATA_SOURCE,
        start.strftime("%Y-%m-%d"),
        end_inclusive.strftime("%Y-%m-%d"),
    )

    market_data = download_market_data(
        universe["ticker"].tolist(), start, end_inclusive
    )
    adv_data = calculate_adv(universe, market_data, start, end_inclusive)
    ranking = rank_sector_etfs(adv_data)
    selected = select_liquid_etfs(ranking)
    validate_output(universe, ranking, selected)
    save_results(ranking, selected)

    logging.info(
        "Stage 2 complete: ranked %d ETFs and selected one ETF for each of %d sectors.",
        len(ranking),
        len(selected),
    )


if __name__ == "__main__":
    try:
        main()
    except Stage2Error as exc:
        logging.error("%s", exc)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        logging.error("Interrupted by user. Completed ticker downloads remain cached.")
        raise SystemExit(130)
