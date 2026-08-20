"""Prepare return and risk data for ETF Portfolio 4.0.

This script reads the ETF selection produced by Stage 1, downloads daily
adjusted price history from Tiingo, aligns the selected ETF series, and
creates the return, volatility, covariance, and data-summary files required by
later stages.

This stage performs data preparation only. It does not select ETFs, optimize a
portfolio, calculate portfolio weights, or implement a diversification-ratio
objective.

Before running, set the Tiingo API token in the environment:

    PowerShell:
        $env:TIINGO_API_TOKEN = "your_api_token"

    macOS/Linux:
        export TIINGO_API_TOKEN="your_api_token"

Then run this file with Python. See the Stage 2 README for full instructions.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing dependencies. Install them with: "
        "python -m pip install -r requirements.txt"
    ) from exc


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE2_DIR = SCRIPT_DIR.parent
PROJECT_DIR = STAGE2_DIR.parent

SELECTED_ETFS_FILE = (
    PROJECT_DIR
    / "Stage 1 - ETF Selection"
    / "data"
    / "selected_sector_etfs.csv"
)
RAW_DATA_DIR = STAGE2_DIR / "data" / "raw_tiingo"
OUTPUT_DIR = STAGE2_DIR / "outputs"


# ---------------------------------------------------------------------------
# User-configurable settings
# ---------------------------------------------------------------------------

TIINGO_API_TOKEN = os.getenv("TIINGO_API_TOKEN", "").strip()
TIINGO_API_BASE_URL = "https://api.tiingo.com/tiingo/daily"

# The EOD endpoint returns both raw close and adjClose. Tiingo documents
# adjClose as incorporating both split and dividend adjustments.
DOWNLOAD_START_DATE = "1990-01-01"
HTTP_TIMEOUT_SECONDS = 60
HTTP_USER_AGENT = "etf-portfolio-4.0-return-risk-preparation/1.0"

# Increase this value if required by the request limits of the user's Tiingo
# plan. Cached tickers are not requested again unless FORCE_REFRESH is True.
REQUEST_PAUSE_SECONDS = 2
FORCE_REFRESH = False

# Each selected ETF is expected to have at least this much calendar history.
# If a genuinely newer ETF is used later, the user may reduce the value after
# documenting the reason.
MINIMUM_HISTORY_YEARS = 3

# The full common history is used when this is None. Set an integer such as 250
# to use the latest N daily return observations. Raw downloaded history remains
# preserved in data/raw_tiingo regardless of this setting.
ESTIMATION_WINDOW_RETURNS: int | None = None

TRADING_DAYS_PER_YEAR = 252
SYMMETRY_RTOL = 1e-10
SYMMETRY_ATOL = 1e-12

REQUIRED_SELECTION_COLUMNS = ("ticker", "gics_sector")
RAW_PRICE_COLUMNS = ("Date", "Adjusted Close", "Close", "Volume")


class Stage2Error(RuntimeError):
    """Raised when Stage 2 input, data acquisition, or validation fails."""


def load_selected_etfs(
    input_file: Path = SELECTED_ETFS_FILE,
) -> pd.DataFrame:
    """Load and validate the Stage 1 ETF selection without modifying it."""
    if not input_file.exists():
        raise Stage2Error(f"Stage 1 selection file not found: {input_file}")

    try:
        selected = pd.read_csv(input_file, dtype=str)
    except Exception as exc:
        raise Stage2Error(f"Could not read Stage 1 selection: {exc}") from exc

    missing_columns = sorted(
        set(REQUIRED_SELECTION_COLUMNS) - set(selected.columns)
    )
    if missing_columns:
        raise Stage2Error(
            "Stage 1 selection is missing required columns: "
            + ", ".join(missing_columns)
        )

    selected = selected.copy()
    for column in REQUIRED_SELECTION_COLUMNS:
        selected[column] = selected[column].fillna("").str.strip()

    if selected.empty:
        raise Stage2Error("Stage 1 selection contains no ETFs.")
    if selected[list(REQUIRED_SELECTION_COLUMNS)].eq("").any().any():
        raise Stage2Error("Stage 1 selection contains a blank ticker or sector.")

    selected["ticker"] = selected["ticker"].str.upper()
    duplicates = sorted(
        selected.loc[selected["ticker"].duplicated(keep=False), "ticker"].unique()
    )
    if duplicates:
        raise Stage2Error(
            "Stage 1 selection contains duplicate tickers: "
            + ", ".join(duplicates)
        )

    return selected.reset_index(drop=True)


def _normalize_price_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize a provider or cache frame to the required price schema."""
    if frame is None or frame.empty:
        raise Stage2Error(f"{ticker}: price data is empty.")

    normalized = frame.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]

    required = {"Date", "Adjusted Close"}
    missing = sorted(required - set(normalized.columns))
    if missing:
        raise Stage2Error(
            f"{ticker}: price data is missing columns: {', '.join(missing)}"
        )

    for optional_column in ("Close", "Volume"):
        if optional_column not in normalized.columns:
            normalized[optional_column] = np.nan

    normalized = normalized.loc[:, RAW_PRICE_COLUMNS]
    normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
    for column in ("Adjusted Close", "Close", "Volume"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.dropna(subset=["Date", "Adjusted Close"])
    normalized = normalized.loc[
        np.isfinite(normalized["Adjusted Close"])
        & (normalized["Adjusted Close"] > 0)
    ]
    normalized = normalized.sort_values("Date", kind="mergesort")
    normalized = normalized.drop_duplicates(subset="Date", keep="last")
    normalized = normalized.reset_index(drop=True)

    if normalized.empty:
        raise Stage2Error(f"{ticker}: no valid positive adjusted prices remain.")
    return normalized


def _parse_tiingo_response(payload: object, ticker: str) -> pd.DataFrame:
    """Parse one Tiingo end-of-day historical-price JSON response."""
    if isinstance(payload, dict):
        message = payload.get("detail") or payload.get("message") or str(payload)
        raise Stage2Error(f"{ticker}: Tiingo response: {message}")
    if not isinstance(payload, list) or not payload:
        raise Stage2Error(f"{ticker}: Tiingo returned no historical prices.")

    records: list[dict[str, object]] = []
    for values in payload:
        if not isinstance(values, dict):
            continue
        records.append(
            {
                "Date": values.get("date"),
                "Adjusted Close": values.get("adjClose"),
                "Close": values.get("close"),
                "Volume": values.get("volume"),
            }
        )

    return _normalize_price_frame(pd.DataFrame(records), ticker)


def _download_tiingo_ticker(ticker: str, api_token: str) -> pd.DataFrame:
    """Download one ticker from Tiingo's adjusted end-of-day endpoint."""
    query = urlencode(
        {
            "startDate": DOWNLOAD_START_DATE,
            "format": "json",
            "resampleFreq": "daily",
        }
    )
    request = Request(
        f"{TIINGO_API_BASE_URL}/{quote(ticker, safe='')}/prices?{query}",
        headers={
            "User-Agent": HTTP_USER_AGENT,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Token {api_token}",
        },
    )

    try:
        with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            response_text = response.read().decode("utf-8-sig", errors="replace")
    except HTTPError as exc:
        raise Stage2Error(
            f"{ticker}: Tiingo returned HTTP status {exc.code}."
        ) from exc
    except URLError as exc:
        raise Stage2Error(
            f"{ticker}: could not reach Tiingo ({exc.reason})."
        ) from exc
    except TimeoutError as exc:
        raise Stage2Error(f"{ticker}: Tiingo request timed out.") from exc

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise Stage2Error(
            f"{ticker}: Tiingo returned invalid JSON."
        ) from exc

    return _parse_tiingo_response(payload, ticker)


def _write_csv_atomically(
    frame: pd.DataFrame,
    output_file: Path,
    *,
    index: bool,
    index_label: str | None = None,
) -> None:
    """Write a CSV through a temporary file to avoid partial output."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    frame.to_csv(
        temporary,
        index=index,
        index_label=index_label,
        float_format="%.10f",
        date_format="%Y-%m-%d",
    )
    temporary.replace(output_file)


def download_price_data(
    selected_etfs: pd.DataFrame,
    api_token: str | None = None,
    raw_data_dir: Path = RAW_DATA_DIR,
    force_refresh: bool = FORCE_REFRESH,
) -> dict[str, pd.DataFrame]:
    """Load cached prices or download every selected ETF from Tiingo."""
    tickers = selected_etfs["ticker"].tolist()
    if not tickers:
        raise Stage2Error("No selected ETF tickers were supplied.")

    resolved_api_token = (api_token or TIINGO_API_TOKEN).strip()
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    price_data: dict[str, pd.DataFrame] = {}
    downloaded_count = 0

    for ticker in tickers:
        cache_file = raw_data_dir / f"{ticker}.csv"

        if cache_file.exists() and not force_refresh:
            try:
                price_data[ticker] = _normalize_price_frame(
                    pd.read_csv(cache_file), ticker
                )
            except Exception as exc:
                raise Stage2Error(
                    f"{ticker}: cached price file is invalid: {exc}"
                ) from exc
            logging.info("Using cached adjusted prices for %s.", ticker)
            continue

        if not resolved_api_token:
            raise Stage2Error(
                "TIINGO_API_TOKEN is not set and at least one ticker "
                "requires download. Set the environment variable before running."
            )

        if downloaded_count > 0 and REQUEST_PAUSE_SECONDS > 0:
            logging.info(
                "Waiting %d seconds before the next provider request.",
                REQUEST_PAUSE_SECONDS,
            )
            time.sleep(REQUEST_PAUSE_SECONDS)

        logging.info("Downloading daily adjusted prices for %s.", ticker)
        frame = _download_tiingo_ticker(ticker, resolved_api_token)
        _write_csv_atomically(frame, cache_file, index=False)
        price_data[ticker] = frame
        downloaded_count += 1

    missing_tickers = [ticker for ticker in tickers if ticker not in price_data]
    if missing_tickers:
        raise Stage2Error(
            "No price data was obtained for selected ETFs: "
            + ", ".join(missing_tickers)
        )
    if set(price_data) != set(tickers):
        raise Stage2Error("Downloaded ticker set differs from the Stage 1 selection.")

    return price_data


def clean_price_data(
    price_data: dict[str, pd.DataFrame],
    selected_etfs: pd.DataFrame,
    estimation_window_returns: int | None = ESTIMATION_WINDOW_RETURNS,
) -> pd.DataFrame:
    """Clean adjusted prices and align all selected ETFs to common dates."""
    tickers = selected_etfs["ticker"].tolist()

    if estimation_window_returns is not None:
        if (
            not isinstance(estimation_window_returns, int)
            or isinstance(estimation_window_returns, bool)
            or estimation_window_returns < 2
        ):
            raise Stage2Error(
                "ESTIMATION_WINDOW_RETURNS must be None or an integer of at "
                "least 2."
            )

    cleaned_series: list[pd.Series] = []
    for ticker in tickers:
        if ticker not in price_data:
            raise Stage2Error(f"{ticker}: selected ETF has no price data.")

        frame = _normalize_price_frame(price_data[ticker], ticker)
        history_start = frame["Date"].min()
        history_end = frame["Date"].max()
        required_start = history_end - pd.DateOffset(years=MINIMUM_HISTORY_YEARS)
        if history_start > required_start:
            raise Stage2Error(
                f"{ticker}: available history spans {history_start:%Y-%m-%d} "
                f"to {history_end:%Y-%m-%d}, less than the configured "
                f"{MINIMUM_HISTORY_YEARS} years."
            )

        series = frame.set_index("Date")["Adjusted Close"].rename(ticker)
        cleaned_series.append(series)

    # An inner join retains only common trading dates. dropna is explicit so
    # no missing value can flow into returns or covariance calculations.
    aligned = pd.concat(cleaned_series, axis=1, join="inner")
    aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    aligned = aligned.sort_index(kind="mergesort")
    aligned = aligned.loc[~aligned.index.duplicated(keep="last")]
    aligned = aligned.loc[:, tickers]

    if estimation_window_returns is not None:
        required_price_rows = estimation_window_returns + 1
        if len(aligned) < required_price_rows:
            raise Stage2Error(
                "Common adjusted-price history has only "
                f"{len(aligned)} rows; {required_price_rows} are required to "
                f"calculate {estimation_window_returns} return observations."
            )
        aligned = aligned.tail(required_price_rows)

    if aligned.empty:
        raise Stage2Error("No common trading dates remain after price alignment.")
    if aligned.shape[1] != len(tickers):
        raise Stage2Error("Aligned price matrix has an incorrect column count.")
    if list(aligned.columns) != tickers:
        raise Stage2Error("Aligned price columns differ from selected ETF order.")
    if aligned.isna().any().any():
        raise Stage2Error("Missing values remain in aligned adjusted prices.")
    if not np.isfinite(aligned.to_numpy(dtype=float)).all():
        raise Stage2Error("Aligned adjusted prices contain non-finite values.")
    if (aligned <= 0).any().any():
        raise Stage2Error("Aligned adjusted prices must all be positive.")
    if len(aligned) < 3:
        raise Stage2Error(
            "At least three common price observations are required for returns "
            "and sample covariance."
        )

    aligned.index.name = "Date"
    return aligned


def calculate_daily_returns(aligned_adjusted_prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple daily returns as Price_t / Price_(t-1) - 1."""
    returns = aligned_adjusted_prices.pct_change(fill_method=None).iloc[1:].copy()

    expected_shape = (
        len(aligned_adjusted_prices) - 1,
        aligned_adjusted_prices.shape[1],
    )
    if returns.shape != expected_shape:
        raise Stage2Error(
            f"Return matrix has shape {returns.shape}; expected {expected_shape}."
        )
    if list(returns.columns) != list(aligned_adjusted_prices.columns):
        raise Stage2Error("Return columns differ from aligned price columns.")
    if returns.empty:
        raise Stage2Error("Daily return matrix is empty.")
    if returns.isna().any().any():
        raise Stage2Error("Missing values remain in the daily return matrix.")
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise Stage2Error("Daily return matrix contains non-finite values.")

    returns.index.name = "Date"
    return returns


def calculate_annualized_volatility(
    daily_returns: pd.DataFrame,
    selected_etfs: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate sample daily volatility and annualize it by sqrt(252)."""
    tickers = selected_etfs["ticker"].tolist()
    if list(daily_returns.columns) != tickers:
        raise Stage2Error(
            "Daily return columns differ from the Stage 1 selected tickers."
        )

    annualized = daily_returns.std(axis=0, ddof=1) * math.sqrt(
        TRADING_DAYS_PER_YEAR
    )
    sector_by_ticker = selected_etfs.set_index("ticker")["gics_sector"]

    result = pd.DataFrame(
        {
            "ticker": tickers,
            "sector": [sector_by_ticker.loc[ticker] for ticker in tickers],
            "annualized_volatility": [annualized.loc[ticker] for ticker in tickers],
        }
    )

    values = result["annualized_volatility"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise Stage2Error("Annualized volatility contains non-finite values.")
    if (values <= 0).any():
        raise Stage2Error("Annualized volatility must be positive for every ETF.")
    return result


def calculate_covariance_matrix(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate the annualized sample covariance matrix as daily cov * 252."""
    covariance = daily_returns.cov(ddof=1) * TRADING_DAYS_PER_YEAR
    tickers = daily_returns.columns.tolist()
    covariance = covariance.loc[tickers, tickers]
    covariance.index.name = "ticker"

    expected_shape = (len(tickers), len(tickers))
    if covariance.shape != expected_shape:
        raise Stage2Error(
            f"Covariance matrix has shape {covariance.shape}; "
            f"expected {expected_shape}."
        )

    values = covariance.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise Stage2Error("Covariance matrix contains non-finite values.")
    if not np.allclose(
        values,
        values.T,
        rtol=SYMMETRY_RTOL,
        atol=SYMMETRY_ATOL,
    ):
        raise Stage2Error("Covariance matrix is not symmetric.")
    if (np.diag(values) <= 0).any():
        raise Stage2Error("Covariance matrix diagonal values must be positive.")

    return covariance


def _build_data_summary(
    aligned_adjusted_prices: pd.DataFrame,
    selected_etfs: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize the common cleaned history used for each selected ETF."""
    tickers = selected_etfs["ticker"].tolist()
    sector_by_ticker = selected_etfs.set_index("ticker")["gics_sector"]
    start_date = aligned_adjusted_prices.index.min().strftime("%Y-%m-%d")
    end_date = aligned_adjusted_prices.index.max().strftime("%Y-%m-%d")

    return pd.DataFrame(
        {
            "ticker": tickers,
            "sector": [sector_by_ticker.loc[ticker] for ticker in tickers],
            "start_date": start_date,
            "end_date": end_date,
            "number_of_observations": len(aligned_adjusted_prices),
        }
    )


def save_outputs(
    aligned_adjusted_prices: pd.DataFrame,
    daily_returns: pd.DataFrame,
    annualized_volatility: pd.DataFrame,
    covariance_matrix: pd.DataFrame,
    selected_etfs: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Save the five validated Stage 2 output files."""
    data_summary = _build_data_summary(aligned_adjusted_prices, selected_etfs)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_atomically(
        aligned_adjusted_prices,
        output_dir / "aligned_adjusted_prices.csv",
        index=True,
        index_label="Date",
    )
    _write_csv_atomically(
        daily_returns,
        output_dir / "daily_returns.csv",
        index=True,
        index_label="Date",
    )
    _write_csv_atomically(
        annualized_volatility,
        output_dir / "annualized_volatility.csv",
        index=False,
    )
    _write_csv_atomically(
        covariance_matrix,
        output_dir / "covariance_matrix.csv",
        index=True,
        index_label="ticker",
    )
    _write_csv_atomically(
        data_summary,
        output_dir / "data_summary.csv",
        index=False,
    )


def main() -> None:
    """Run the complete Stage 2 return-and-risk data preparation workflow."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    selected_etfs = load_selected_etfs()
    price_data = download_price_data(selected_etfs)
    aligned_adjusted_prices = clean_price_data(price_data, selected_etfs)
    daily_returns = calculate_daily_returns(aligned_adjusted_prices)
    annualized_volatility = calculate_annualized_volatility(
        daily_returns, selected_etfs
    )
    covariance_matrix = calculate_covariance_matrix(daily_returns)
    save_outputs(
        aligned_adjusted_prices,
        daily_returns,
        annualized_volatility,
        covariance_matrix,
        selected_etfs,
    )

    logging.info(
        "Stage 2 complete: prepared %d ETFs across %d daily return observations.",
        len(selected_etfs),
        len(daily_returns),
    )
    logging.info("No portfolio optimization or weight calculation was performed.")


if __name__ == "__main__":
    try:
        main()
    except Stage2Error as exc:
        logging.error("%s", exc)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        logging.error("Interrupted by user. Completed ticker downloads remain cached.")
        raise SystemExit(130)
