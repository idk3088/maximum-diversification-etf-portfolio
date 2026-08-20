"""Stage 3: select a correlation-estimation window by walk-forward validation.

The script downloads long-run adjusted daily close prices for the 11 ETFs in
``selected_sector_etfs.csv`` plus SPY. It computes daily total returns, then
tests several historical windows by comparing:

    correlation estimated from the trailing historical window
    versus
    correlation realized over the following forecast horizon

The optimal window is selected automatically using the lowest out-of-sample
RMSE. This stage does not optimize a portfolio or minimize correlation itself.

Data source
-----------
Twelve Data ``/time_series`` API with ``interval=1day`` and ``adjust=all``.
The returned close field is stored as AdjustedClose. An API key is required but
is read only from the ``TWELVE_DATA_API_KEY`` environment variable.

Local usage (PowerShell)
------------------------
    python -m pip install pandas numpy
    $env:TWELVE_DATA_API_KEY="paste_your_key_here"
    python stage3_correlation_window_selection.py

Downloaded data is cached separately from Stage 2 under
``data_cache/stage3_adjusted_prices``. Successful batches remain cached if a
later batch fails or the process is interrupted.
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
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install requirements with: "
        "python -m pip install pandas numpy"
    ) from exc


# ---------------------------------------------------------------------------
# User-configurable settings
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "selected_sector_etfs.csv"
CACHE_DIR = BASE_DIR / "data_cache" / "stage3_adjusted_prices"

DAILY_RETURNS_FILE = BASE_DIR / "daily_returns.csv"
WINDOW_RESULTS_FILE = BASE_DIR / "correlation_window_results.csv"
OPTIMAL_WINDOW_FILE = BASE_DIR / "optimal_correlation_window.csv"
PRICE_METADATA_FILE = BASE_DIR / "price_data_metadata.csv"
ROLLING_DETAILS_FILE = BASE_DIR / "rolling_forecast_details.csv"

HISTORY_START_DATE = "2015-01-01"
END_DATE: str | None = None

# Approximate trading-day equivalents required by the research specification.
# Dictionary order controls presentation order in the output files.
ESTIMATION_WINDOWS = {
    "3 months": 63,
    "6 months": 126,
    "1 year": 252,
    "2 years": 504,
    "3 years": 756,
}

# A quarterly forecast horizon and non-overlapping quarterly walk step.
FORECAST_HORIZON_DAYS = 63
WALK_FORWARD_STEP_DAYS = 63

BENCHMARK_TICKER = "SPY"
EXPECTED_SECTOR_COUNT = 11
CACHE_ALLOWED_BUSINESS_DAY_LAG = 1

# Twelve Data Basic currently allows 8 API credits per minute and the time
# series endpoint costs one credit per symbol. Defaults therefore use batches
# of 8 symbols and wait for the next quota window between batches.
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
TWELVE_DATA_API_URL = "https://api.twelvedata.com/time_series"
SYMBOLS_PER_REQUEST = 8
BATCH_PAUSE_SECONDS = 65
HTTP_TIMEOUT_SECONDS = 60
HTTP_USER_AGENT = "sector-etf-correlation-research/1.0"

DATA_SOURCE = "Twelve Data time_series API (adjust=all)"
RETRIEVAL_METHOD = (
    "HTTPS batch request to /time_series; interval=1day; adjust=all; local cache"
)

INPUT_COLUMNS = (
    "gics_sector",
    "ticker",
    "etf_name",
    "issuer",
    "ADV",
    "observation_period",
)
CACHE_COLUMNS = ("Date", "AdjustedClose")

WINDOW_RESULT_COLUMNS = (
    "window",
    "MAE",
    "RMSE",
    "number_of_tests",
    "forecast_error_std",
    "average_estimated_correlation",
    "average_realized_correlation",
)

OPTIMAL_WINDOW_COLUMNS = (
    "window",
    "MAE",
    "RMSE",
    "number_of_tests",
    "forecast_error_std",
    "average_estimated_correlation",
    "average_realized_correlation",
    "ranking_by_RMSE",
    "ranking_by_MAE",
    "selected_optimal_window",
)

METADATA_COLUMNS = (
    "ticker",
    "data_start_date",
    "data_end_date",
    "number_of_price_observations",
    "data_source",
    "retrieval_method",
)

# Ticker is retained as an additional audit key because each row represents one
# ETF-SPY forecast. The remaining fields are the requested detail-file schema.
ROLLING_DETAIL_COLUMNS = (
    "ticker",
    "estimation_window",
    "training_start_date",
    "training_end_date",
    "testing_start_date",
    "testing_end_date",
    "estimated_correlation",
    "realized_correlation",
    "forecast_error",
)


class Stage3Error(RuntimeError):
    """Raised when Stage 3 cannot safely produce valid outputs."""


def load_selected_etfs(input_file: Path = INPUT_FILE) -> pd.DataFrame:
    """Load and validate the 11 sector ETFs selected in Stage 2."""
    if not input_file.exists():
        raise Stage3Error(f"Stage 2 input file not found: {input_file}")

    try:
        selected = pd.read_csv(input_file, dtype=str)
    except Exception as exc:
        raise Stage3Error(f"Could not read {input_file}: {exc}") from exc

    missing_columns = sorted(set(INPUT_COLUMNS) - set(selected.columns))
    if missing_columns:
        raise Stage3Error(
            "Stage 2 input is missing required columns: "
            + ", ".join(missing_columns)
        )

    selected = selected.loc[:, INPUT_COLUMNS].copy()
    for column in INPUT_COLUMNS:
        selected[column] = selected[column].fillna("").str.strip()

    blank_columns = [column for column in INPUT_COLUMNS if selected[column].eq("").any()]
    if blank_columns:
        raise Stage3Error(
            "Stage 2 input contains blank values in required columns: "
            + ", ".join(blank_columns)
        )

    selected["ticker"] = selected["ticker"].str.upper()
    if len(selected) != EXPECTED_SECTOR_COUNT:
        raise Stage3Error(
            f"Expected {EXPECTED_SECTOR_COUNT} selected ETFs; found {len(selected)}."
        )
    if selected["ticker"].duplicated().any():
        duplicates = selected.loc[
            selected["ticker"].duplicated(keep=False), "ticker"
        ].unique()
        raise Stage3Error("Duplicate selected tickers: " + ", ".join(duplicates))
    if selected["gics_sector"].nunique() != EXPECTED_SECTOR_COUNT:
        raise Stage3Error("Stage 2 input must contain exactly one ETF per GICS sector.")
    if BENCHMARK_TICKER in set(selected["ticker"]):
        raise Stage3Error(
            f"{BENCHMARK_TICKER} cannot be both a selected sector ETF and the benchmark."
        )

    return selected


def _resolve_dates() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse and validate the configured historical date range."""
    try:
        start = pd.Timestamp(HISTORY_START_DATE).normalize()
    except Exception as exc:
        raise Stage3Error("HISTORY_START_DATE must use YYYY-MM-DD format.") from exc

    if END_DATE is None:
        end = pd.Timestamp(date.today()).normalize()
    else:
        try:
            end = pd.Timestamp(END_DATE).normalize()
        except Exception as exc:
            raise Stage3Error("END_DATE must use YYYY-MM-DD format.") from exc

    if start.tzinfo is not None:
        start = start.tz_localize(None)
    if end.tzinfo is not None:
        end = end.tz_localize(None)
    if start >= end:
        raise Stage3Error("HISTORY_START_DATE must be earlier than END_DATE.")

    if not ESTIMATION_WINDOWS:
        raise Stage3Error("ESTIMATION_WINDOWS cannot be empty.")
    if any(not isinstance(days, int) or days < 2 for days in ESTIMATION_WINDOWS.values()):
        raise Stage3Error("Every estimation window must contain at least two days.")
    if FORECAST_HORIZON_DAYS < 2:
        raise Stage3Error("FORECAST_HORIZON_DAYS must be at least two.")
    if WALK_FORWARD_STEP_DAYS <= 0:
        raise Stage3Error("WALK_FORWARD_STEP_DAYS must be positive.")
    if SYMBOLS_PER_REQUEST <= 0:
        raise Stage3Error("SYMBOLS_PER_REQUEST must be positive.")
    if BATCH_PAUSE_SECONDS < 0:
        raise Stage3Error("BATCH_PAUSE_SECONDS cannot be negative.")

    return start, end


def _ticker_batches(tickers: list[str], batch_size: int):
    """Yield deterministic ticker batches."""
    for position in range(0, len(tickers), batch_size):
        yield tickers[position : position + batch_size]


def _cache_price_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}.csv"


def _cache_metadata_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}.metadata.json"


def _normalize_adjusted_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize provider/cache data to Date-indexed AdjustedClose values."""
    empty = pd.DataFrame(columns=["AdjustedClose"])
    if frame is None or frame.empty:
        return empty

    normalized = frame.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    if not set(CACHE_COLUMNS).issubset(normalized.columns):
        return empty

    normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
    normalized["AdjustedClose"] = pd.to_numeric(
        normalized["AdjustedClose"], errors="coerce"
    )
    normalized = normalized.dropna(subset=["Date"])
    normalized = normalized.set_index("Date").loc[:, ["AdjustedClose"]]
    normalized.index = pd.DatetimeIndex(normalized.index).tz_localize(None).normalize()
    normalized = normalized.loc[~normalized.index.duplicated(keep="last")]
    normalized = normalized.sort_index()
    return normalized.loc[
        normalized["AdjustedClose"].notna() & (normalized["AdjustedClose"] > 0)
    ]


def _load_cached_prices(ticker: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load a ticker's cached adjusted prices and request metadata."""
    price_path = _cache_price_path(ticker)
    metadata_path = _cache_metadata_path(ticker)
    empty = pd.DataFrame(columns=["AdjustedClose"])

    if not price_path.exists() or not metadata_path.exists():
        return empty, {}

    try:
        prices = _normalize_adjusted_prices(pd.read_csv(price_path))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.warning("Ignoring unreadable Stage 3 cache for %s: %s", ticker, exc)
        return empty, {}

    if not isinstance(metadata, dict) or prices.empty:
        logging.warning("Ignoring incomplete Stage 3 cache for %s.", ticker)
        return empty, {}
    return prices, metadata


def _cache_covers_request(
    prices: pd.DataFrame,
    metadata: dict[str, str],
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
) -> bool:
    """Check requested range coverage while allowing post-start ETF inception."""
    if prices.empty or not metadata:
        return False

    try:
        cached_request_start = pd.Timestamp(metadata["requested_start"]).normalize()
        cached_request_end = pd.Timestamp(metadata["requested_end"]).normalize()
    except (KeyError, TypeError, ValueError):
        return False

    required_latest = (
        requested_end - pd.offsets.BDay(CACHE_ALLOWED_BUSINESS_DAY_LAG)
    ).normalize()
    return (
        cached_request_start <= requested_start
        and cached_request_end >= required_latest
        and prices.index.max() >= required_latest
    )


def cache_market_data(
    ticker: str,
    prices: pd.DataFrame,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
) -> None:
    """Atomically cache adjusted prices and their requested date range."""
    if prices.empty:
        raise Stage3Error(f"Refusing to cache empty price data for {ticker}.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    price_path = _cache_price_path(ticker)
    metadata_path = _cache_metadata_path(ticker)
    temporary_price = price_path.with_suffix(".csv.tmp")
    temporary_metadata = metadata_path.with_suffix(".json.tmp")

    output = prices.reset_index().rename(columns={"index": "Date"})
    output = output.loc[:, CACHE_COLUMNS]
    output.to_csv(temporary_price, index=False, date_format="%Y-%m-%d")

    metadata = {
        "ticker": ticker,
        "requested_start": requested_start.strftime("%Y-%m-%d"),
        "requested_end": requested_end.strftime("%Y-%m-%d"),
        "data_start": prices.index.min().strftime("%Y-%m-%d"),
        "data_end": prices.index.max().strftime("%Y-%m-%d"),
        "number_of_observations": int(len(prices)),
        "data_source": DATA_SOURCE,
    }
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    temporary_price.replace(price_path)
    temporary_metadata.replace(metadata_path)


def _twelve_data_url(
    tickers: list[str], start: pd.Timestamp, end: pd.Timestamp, api_key: str
) -> str:
    """Build one adjusted daily-price batch request."""
    query = urlencode(
        {
            "symbol": ",".join(tickers),
            "interval": "1day",
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "order": "asc",
            "adjust": "all",
            "apikey": api_key,
        }
    )
    return f"{TWELVE_DATA_API_URL}?{query}"


def _parse_twelve_data_prices(ticker: str, result: object) -> pd.DataFrame:
    """Parse one symbol from a Twelve Data response."""
    if not isinstance(result, dict):
        raise Stage3Error(f"{ticker}: malformed Twelve Data response")
    if result.get("status") == "error":
        message = str(result.get("message", "unknown API error"))
        raise Stage3Error(f"{ticker}: Twelve Data error: {message}")

    values = result.get("values")
    if not isinstance(values, list) or not values:
        raise Stage3Error(f"{ticker}: Twelve Data returned no adjusted prices")

    frame = pd.DataFrame(values).rename(
        columns={"datetime": "Date", "close": "AdjustedClose"}
    )
    prices = _normalize_adjusted_prices(frame)
    if prices.empty:
        raise Stage3Error(f"{ticker}: response lacks datetime or adjusted close")
    return prices


def _download_twelve_data_batch(
    tickers: list[str], start: pd.Timestamp, end: pd.Timestamp, api_key: str
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Request a batch once; return successful histories and per-symbol errors."""
    request = Request(
        _twelve_data_url(tickers, start, end, api_key),
        headers={"User-Agent": HTTP_USER_AGENT, "Accept": "application/json"},
    )

    try:
        with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            payload_text = response.read().decode("utf-8-sig", errors="replace")
    except HTTPError as exc:
        raise Stage3Error(f"Twelve Data returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise Stage3Error(f"Could not reach Twelve Data ({exc.reason})") from exc
    except TimeoutError as exc:
        raise Stage3Error("Twelve Data request timed out") from exc

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise Stage3Error("Twelve Data returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise Stage3Error("Twelve Data returned an unexpected response structure")

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
            downloaded[ticker] = _parse_twelve_data_prices(ticker, result)
        except Stage3Error as exc:
            failures.append(str(exc))
    return downloaded, failures


def _merge_prices(cached: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Merge cached and fresh prices, preferring fresh duplicate dates."""
    combined = pd.concat([cached, fresh]).sort_index()
    return combined.loc[~combined.index.duplicated(keep="last")]


def download_price_data(
    tickers: list[str], start: pd.Timestamp, end: pd.Timestamp
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load cache and download only missing/stale adjusted-price histories."""
    if not tickers:
        raise Stage3Error("No tickers were supplied for price acquisition.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    price_data: dict[str, pd.DataFrame] = {}
    cache_metadata: dict[str, dict[str, str]] = {}
    missing_tickers: list[str] = []

    for ticker in tickers:
        cached_prices, metadata = _load_cached_prices(ticker)
        price_data[ticker] = cached_prices
        cache_metadata[ticker] = metadata
        if _cache_covers_request(cached_prices, metadata, start, end):
            logging.info("Using cached adjusted prices for %s.", ticker)
        else:
            missing_tickers.append(ticker)

    logging.info(
        "Stage 3 cache satisfied %d of %d tickers; %d require download.",
        len(tickers) - len(missing_tickers),
        len(tickers),
        len(missing_tickers),
    )

    if missing_tickers and not TWELVE_DATA_API_KEY:
        raise Stage3Error(
            "TWELVE_DATA_API_KEY is not set. PowerShell example: "
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
            "Downloading adjusted-price batch %d/%d (%d ticker(s)).",
            position,
            len(batches),
            len(batch),
        )
        try:
            batch_prices, batch_failures = _download_twelve_data_batch(
                batch, start, end, TWELVE_DATA_API_KEY
            )
            failures.extend(batch_failures)
        except Stage3Error as exc:
            failures.append(f"batch {position} ({', '.join(batch)}): {exc}")
            continue

        for ticker, fresh in batch_prices.items():
            combined = _merge_prices(price_data[ticker], fresh)
            prior_metadata = cache_metadata.get(ticker, {})
            prior_requested_start = pd.Timestamp(
                prior_metadata.get("requested_start", start)
            ).normalize()
            combined_request_start = min(start, prior_requested_start)
            price_data[ticker] = combined
            cache_market_data(ticker, combined, combined_request_start, end)

            required_latest = (
                end - pd.offsets.BDay(CACHE_ALLOWED_BUSINESS_DAY_LAG)
            ).normalize()
            if combined.index.max() < required_latest:
                failures.append(
                    f"{ticker}: latest adjusted price is {combined.index.max():%Y-%m-%d}; "
                    f"expected at least {required_latest:%Y-%m-%d}"
                )

    if failures:
        raise Stage3Error(
            "Adjusted-price acquisition failed. Successful batches were cached, "
            "so a later rerun will request only unresolved tickers:\n- "
            + "\n- ".join(failures)
        )

    metadata_rows: list[dict[str, object]] = []
    for ticker in tickers:
        prices = price_data.get(ticker, pd.DataFrame())
        prices = prices.loc[(prices.index >= start) & (prices.index <= end)]
        if prices.empty:
            raise Stage3Error(f"No adjusted price observations are available for {ticker}.")
        metadata_rows.append(
            {
                "ticker": ticker,
                "data_start_date": prices.index.min().strftime("%Y-%m-%d"),
                "data_end_date": prices.index.max().strftime("%Y-%m-%d"),
                "number_of_price_observations": int(len(prices)),
                "data_source": DATA_SOURCE,
                "retrieval_method": RETRIEVAL_METHOD,
            }
        )

    metadata_frame = pd.DataFrame(metadata_rows, columns=METADATA_COLUMNS)
    return price_data, metadata_frame


def calculate_returns(
    price_data: dict[str, pd.DataFrame],
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Create an outer-aligned daily total-return dataset."""
    price_series: list[pd.Series] = []
    for ticker in tickers:
        history = price_data.get(ticker, pd.DataFrame())
        if history.empty or "AdjustedClose" not in history.columns:
            raise Stage3Error(f"Adjusted close data is missing for {ticker}.")
        series = history.loc[
            (history.index >= start) & (history.index <= end), "AdjustedClose"
        ].rename(ticker)
        price_series.append(series)

    adjusted_prices = pd.concat(price_series, axis=1, join="outer").sort_index()
    returns = adjusted_prices.pct_change(fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    returns.index.name = "Date"

    if returns.empty:
        raise Stage3Error("Daily return calculation produced no observations.")
    if tuple(returns.columns) != tuple(tickers):
        raise Stage3Error("Daily return columns differ from the requested ticker order.")
    if returns.index.duplicated().any():
        raise Stage3Error("Daily return data contains duplicate dates.")
    return returns


def calculate_rolling_correlation(
    daily_returns: pd.DataFrame,
    etf_tickers: list[str],
    benchmark_ticker: str = BENCHMARK_TICKER,
) -> pd.DataFrame:
    """Run an independent walk-forward experiment for every window."""
    if benchmark_ticker not in daily_returns.columns:
        raise Stage3Error(f"Benchmark return column {benchmark_ticker} is missing.")

    forecast_rows: list[dict[str, object]] = []
    insufficient: list[str] = []

    for ticker in etf_tickers:
        if ticker not in daily_returns.columns:
            insufficient.append(f"{ticker}: return column is missing")
            continue

        pair = daily_returns.loc[:, [ticker, benchmark_ticker]].dropna()
        for window_label, window_days in ESTIMATION_WINDOWS.items():
            required_pair_observations = window_days + FORECAST_HORIZON_DAYS
            if len(pair) < required_pair_observations:
                insufficient.append(
                    f"{ticker}, {window_label}: {len(pair)} paired returns; "
                    f"at least {required_pair_observations} are required"
                )
                continue

            # Each candidate window starts its own walk-forward sequence as
            # soon as its training history is available. Consequently, shorter
            # windows include earlier unseen testing periods, while every test
            # still follows immediately after its own training period.
            origin_positions = range(
                window_days,
                len(pair) - FORECAST_HORIZON_DAYS + 1,
                WALK_FORWARD_STEP_DAYS,
            )
            window_test_count = 0
            for origin in origin_positions:
                history = pair.iloc[origin - window_days : origin]
                future = pair.iloc[origin : origin + FORECAST_HORIZON_DAYS]

                estimated = float(history[ticker].corr(history[benchmark_ticker]))
                if not math.isfinite(estimated):
                    raise Stage3Error(
                        f"Estimated correlation is undefined for {ticker}, "
                        f"window {window_label}, origin {future.index.min():%Y-%m-%d}."
                    )

                realized = float(future[ticker].corr(future[benchmark_ticker]))
                if not math.isfinite(realized):
                    raise Stage3Error(
                        f"Realized correlation is undefined for {ticker}, "
                        f"window {window_label}, testing start "
                        f"{future.index.min():%Y-%m-%d}."
                    )

                error = estimated - realized
                forecast_rows.append(
                    {
                        "window": window_label,
                        "estimation_days": window_days,
                        "ticker": ticker,
                        "training_start_date": history.index.min(),
                        "training_end_date": history.index.max(),
                        "testing_start_date": future.index.min(),
                        "testing_end_date": future.index.max(),
                        "estimated_correlation": estimated,
                        "realized_correlation": realized,
                        "forecast_error": error,
                        "absolute_error": abs(error),
                        "squared_error": error**2,
                    }
                )
                window_test_count += 1

            if window_test_count == 0:
                insufficient.append(
                    f"{ticker}, {window_label}: no valid walk-forward origins"
                )

    if insufficient:
        raise Stage3Error(
            "Walk-forward validation cannot include every selected ETF:\n- "
            + "\n- ".join(insufficient)
        )
    if not forecast_rows:
        raise Stage3Error("Walk-forward validation generated no forecasts.")

    forecasts = pd.DataFrame(forecast_rows)

    # Validate the corrected design explicitly: candidate windows must not all
    # contain the identical collection of ETF/testing-period observations.
    testing_signatures: list[frozenset[tuple[object, ...]]] = []
    for window_label in ESTIMATION_WINDOWS:
        sample = forecasts.loc[forecasts["window"] == window_label]
        if sample.empty:
            raise Stage3Error(f"No rolling forecasts were produced for {window_label}.")
        signature = frozenset(
            sample.loc[
                :,
                [
                    "ticker",
                    "testing_start_date",
                    "testing_end_date",
                    "realized_correlation",
                ],
            ].itertuples(index=False, name=None)
        )
        testing_signatures.append(signature)

    if len(set(testing_signatures)) == 1:
        raise Stage3Error(
            "All estimation windows produced identical realized-correlation "
            "observations; independent walk-forward construction failed."
        )
    return forecasts


def calculate_forecast_error(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ETF-fold forecast errors into window-level metrics."""
    rows: list[dict[str, object]] = []
    for window_label in ESTIMATION_WINDOWS:
        sample = forecasts.loc[forecasts["window"] == window_label]
        if sample.empty:
            raise Stage3Error(f"No forecasts were produced for {window_label}.")

        mae = float(sample["absolute_error"].mean())
        rmse = float(np.sqrt(sample["squared_error"].mean()))
        error_std = float(sample["forecast_error"].std(ddof=1))
        rows.append(
            {
                "window": window_label,
                "MAE": mae,
                "RMSE": rmse,
                "number_of_tests": int(len(sample)),
                "forecast_error_std": error_std,
                "average_estimated_correlation": float(
                    sample["estimated_correlation"].mean()
                ),
                "average_realized_correlation": float(
                    sample["realized_correlation"].mean()
                ),
            }
        )

    results = pd.DataFrame(rows, columns=WINDOW_RESULT_COLUMNS)
    numeric_columns = [
        "MAE",
        "RMSE",
        "forecast_error_std",
        "average_estimated_correlation",
        "average_realized_correlation",
    ]
    if not np.isfinite(results[numeric_columns].to_numpy(dtype=float)).all():
        raise Stage3Error("Window comparison contains non-finite metrics.")
    return results


def compare_windows(window_results: pd.DataFrame) -> pd.DataFrame:
    """Validate the independently generated window-level results."""
    if tuple(window_results.columns) != WINDOW_RESULT_COLUMNS:
        raise Stage3Error("Correlation-window result columns are invalid.")
    if set(window_results["window"]) != set(ESTIMATION_WINDOWS):
        raise Stage3Error("Not all configured estimation windows were evaluated.")
    if (window_results["number_of_tests"] <= 0).any():
        raise Stage3Error("Every estimation window must have at least one rolling test.")
    if (window_results[["MAE", "RMSE"]] < 0).any().any():
        raise Stage3Error("MAE and RMSE cannot be negative.")

    order = {label: position for position, label in enumerate(ESTIMATION_WINDOWS)}
    compared = window_results.copy()
    compared["_order"] = compared["window"].map(order)
    return compared.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def select_optimal_window(window_results: pd.DataFrame) -> pd.DataFrame:
    """Rank all windows and mark the unique lowest-RMSE selection."""
    ranked = window_results.copy()
    ranked["ranking_by_RMSE"] = (
        ranked["RMSE"].rank(method="min", ascending=True).astype(int)
    )
    ranked["ranking_by_MAE"] = (
        ranked["MAE"].rank(method="min", ascending=True).astype(int)
    )

    # RMSE is primary. MAE and then shorter estimation length resolve an exact
    # numerical tie deterministically; they do not override a lower RMSE.
    ranked["_estimation_days"] = ranked["window"].map(ESTIMATION_WINDOWS)
    winner_index = ranked.sort_values(
        ["RMSE", "MAE", "_estimation_days"], kind="mergesort"
    ).index[0]
    ranked["selected_optimal_window"] = False
    ranked.loc[winner_index, "selected_optimal_window"] = True
    ranked = ranked.drop(columns="_estimation_days")

    ranked = ranked.sort_values(
        ["ranking_by_RMSE", "ranking_by_MAE", "window"], kind="mergesort"
    ).reset_index(drop=True)
    ranked = ranked.loc[:, OPTIMAL_WINDOW_COLUMNS]

    if int(ranked["selected_optimal_window"].sum()) != 1:
        raise Stage3Error("Exactly one optimal correlation window must be selected.")
    selected_rmse = float(ranked.loc[ranked["selected_optimal_window"], "RMSE"].iloc[0])
    if not math.isclose(selected_rmse, float(ranked["RMSE"].min()), abs_tol=1e-15):
        raise Stage3Error("Selected window does not have the minimum RMSE.")
    return ranked


def _write_csv_atomically(frame: pd.DataFrame, output_file: Path) -> None:
    """Write a CSV through a temporary file to avoid partial outputs."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.10f")
    temporary.replace(output_file)


def save_results(
    daily_returns: pd.DataFrame,
    rolling_forecasts: pd.DataFrame,
    window_results: pd.DataFrame,
    optimal_window: pd.DataFrame,
    price_metadata: pd.DataFrame,
) -> None:
    """Save all required outputs plus price-history metadata."""
    returns_output = daily_returns.reset_index()
    if returns_output.columns[0] != "Date":
        raise Stage3Error("daily_returns.csv must begin with a Date column.")
    if tuple(price_metadata.columns) != METADATA_COLUMNS:
        raise Stage3Error("Price metadata columns are invalid.")

    rolling_details = rolling_forecasts.rename(
        columns={"window": "estimation_window"}
    ).loc[:, ROLLING_DETAIL_COLUMNS]
    for date_column in (
        "training_start_date",
        "training_end_date",
        "testing_start_date",
        "testing_end_date",
    ):
        rolling_details[date_column] = pd.to_datetime(
            rolling_details[date_column]
        ).dt.strftime("%Y-%m-%d")

    if rolling_details.empty:
        raise Stage3Error("rolling_forecast_details.csv cannot be empty.")
    if rolling_details.isna().any().any():
        raise Stage3Error("Rolling forecast details contain missing values.")

    _write_csv_atomically(returns_output, DAILY_RETURNS_FILE)
    _write_csv_atomically(rolling_details, ROLLING_DETAILS_FILE)
    _write_csv_atomically(window_results, WINDOW_RESULTS_FILE)
    _write_csv_atomically(optimal_window, OPTIMAL_WINDOW_FILE)
    _write_csv_atomically(price_metadata, PRICE_METADATA_FILE)

    logging.info("Saved daily returns to %s", DAILY_RETURNS_FILE)
    logging.info("Saved rolling forecast details to %s", ROLLING_DETAILS_FILE)
    logging.info("Saved window results to %s", WINDOW_RESULTS_FILE)
    logging.info("Saved automatic window selection to %s", OPTIMAL_WINDOW_FILE)
    logging.info("Saved price-history metadata to %s", PRICE_METADATA_FILE)


def main() -> None:
    """Run Stage 3 from input validation through automatic window selection."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    selected = load_selected_etfs()
    start, end = _resolve_dates()
    etf_tickers = selected["ticker"].tolist()
    all_tickers = etf_tickers + [BENCHMARK_TICKER]

    logging.info(
        "Requesting adjusted daily prices from %s through %s for %d ETFs plus %s.",
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        len(etf_tickers),
        BENCHMARK_TICKER,
    )

    price_data, price_metadata = download_price_data(all_tickers, start, end)
    daily_returns = calculate_returns(price_data, all_tickers, start, end)
    forecasts = calculate_rolling_correlation(daily_returns, etf_tickers)
    window_results = calculate_forecast_error(forecasts)
    window_results = compare_windows(window_results)
    optimal_window = select_optimal_window(window_results)
    save_results(
        daily_returns,
        forecasts,
        window_results,
        optimal_window,
        price_metadata,
    )

    logging.info(
        "Stage 3 complete. The optimal window was selected automatically by "
        "minimum out-of-sample RMSE; inspect the generated CSV files for results."
    )


if __name__ == "__main__":
    try:
        main()
    except Stage3Error as exc:
        logging.error("%s", exc)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        logging.error("Interrupted by user. Completed downloads remain cached.")
        raise SystemExit(130)
