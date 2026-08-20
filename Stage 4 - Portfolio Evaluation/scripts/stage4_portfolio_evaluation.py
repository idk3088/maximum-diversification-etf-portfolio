"""Evaluate the completed Maximum Diversification Portfolio.

This script reads fixed MDP weights from Stage 3 and daily ETF returns from
Stage 2. It calculates portfolio daily returns as the weighted sum of ETF daily
returns before compounding them into wealth curves. It compares the MDP with an
equal-weight sector ETF portfolio and SPY over one common evaluation period.

No ETF selection, weight optimization, or Stage 2/3 input modification occurs.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
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
STAGE4_DIR = SCRIPT_DIR.parent
PROJECT_DIR = STAGE4_DIR.parent
STAGE2_OUTPUT_DIR = (
    PROJECT_DIR / "Stage 2 - Return and Risk Data Preparation" / "outputs"
)
STAGE3_OUTPUT_DIR = (
    PROJECT_DIR
    / "Stage 3 - Maximum Diversification Portfolio Construction"
    / "outputs"
)

DAILY_RETURNS_FILE = STAGE2_OUTPUT_DIR / "daily_returns.csv"
VOLATILITY_FILE = STAGE2_OUTPUT_DIR / "annualized_volatility.csv"
COVARIANCE_FILE = STAGE2_OUTPUT_DIR / "covariance_matrix.csv"
MDP_WEIGHTS_FILE = STAGE3_OUTPUT_DIR / "optimal_mdp_weights.csv"

SPY_CACHE_FILE = STAGE4_DIR / "data" / "raw_tiingo" / "SPY.csv"
OUTPUT_DIR = STAGE4_DIR / "outputs"


# ---------------------------------------------------------------------------
# User-configurable settings
# ---------------------------------------------------------------------------

TIINGO_API_TOKEN = os.getenv("TIINGO_API_TOKEN", "").strip()
TIINGO_API_URL = "https://api.tiingo.com/tiingo/daily/SPY/prices"
HTTP_TIMEOUT_SECONDS = 60
HTTP_USER_AGENT = "etf-portfolio-4.0-evaluation/1.0"
SPY_DOWNLOAD_BUFFER_CALENDAR_DAYS = 10
FORCE_SPY_REFRESH = False

TRADING_DAYS_PER_YEAR = 252
INITIAL_PORTFOLIO_VALUE = 100.0
WEIGHT_TOLERANCE = 1e-8
SYMMETRY_RTOL = 1e-10
SYMMETRY_ATOL = 1e-12

REQUIRED_WEIGHT_COLUMNS = ("ticker", "sector", "weight")
REQUIRED_VOLATILITY_COLUMNS = (
    "ticker",
    "sector",
    "annualized_volatility",
)
SPY_CACHE_COLUMNS = ("Date", "Adjusted Close", "Close", "Volume")


class Stage4Error(RuntimeError):
    """Raised when Stage 4 input, data acquisition, or validation fails."""


def load_weights(weights_file: Path = MDP_WEIGHTS_FILE) -> pd.DataFrame:
    """Load and validate the completed Stage 3 weights without changing them."""
    if not weights_file.exists():
        raise Stage4Error(f"Stage 3 weight file not found: {weights_file}")

    try:
        weights = pd.read_csv(weights_file)
    except Exception as exc:
        raise Stage4Error(f"Could not read Stage 3 weights: {exc}") from exc

    missing_columns = sorted(set(REQUIRED_WEIGHT_COLUMNS) - set(weights.columns))
    if missing_columns:
        raise Stage4Error(
            "Stage 3 weights are missing columns: " + ", ".join(missing_columns)
        )

    weights = weights.loc[:, REQUIRED_WEIGHT_COLUMNS].copy()
    weights["ticker"] = (
        weights["ticker"].fillna("").astype(str).str.strip().str.upper()
    )
    weights["sector"] = weights["sector"].fillna("").astype(str).str.strip()
    weights["weight"] = pd.to_numeric(weights["weight"], errors="coerce")

    if weights.empty:
        raise Stage4Error("Stage 3 weight file contains no ETFs.")
    if weights[["ticker", "sector"]].eq("").any().any():
        raise Stage4Error("Stage 3 weights contain a blank ticker or sector.")
    if weights["ticker"].duplicated().any():
        raise Stage4Error("Stage 3 weights contain duplicate tickers.")
    if not np.isfinite(weights["weight"].to_numpy(dtype=float)).all():
        raise Stage4Error("Stage 3 weights contain non-finite values.")
    if (weights["weight"] < 0).any():
        raise Stage4Error("Stage 3 weights contain negative values.")
    if not np.isclose(
        weights["weight"].sum(),
        1.0,
        rtol=0.0,
        atol=WEIGHT_TOLERANCE,
    ):
        raise Stage4Error("Stage 3 MDP weights do not sum approximately to one.")

    return weights.reset_index(drop=True)


def load_returns(
    returns_file: Path = DAILY_RETURNS_FILE,
    expected_tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Load and validate the Stage 2 daily ETF return matrix."""
    if not returns_file.exists():
        raise Stage4Error(f"Stage 2 daily return file not found: {returns_file}")

    try:
        returns = pd.read_csv(returns_file)
    except Exception as exc:
        raise Stage4Error(f"Could not read Stage 2 daily returns: {exc}") from exc

    if "Date" not in returns.columns:
        raise Stage4Error("Stage 2 daily returns must contain a Date column.")

    returns["Date"] = pd.to_datetime(returns["Date"], errors="coerce")
    if returns["Date"].isna().any():
        raise Stage4Error("Stage 2 daily returns contain invalid dates.")
    if returns["Date"].duplicated().any():
        raise Stage4Error("Stage 2 daily returns contain duplicate dates.")

    returns = returns.set_index("Date").sort_index(kind="mergesort")
    returns.index.name = "Date"
    returns.columns = [str(column).strip().upper() for column in returns.columns]
    if returns.columns.duplicated().any():
        raise Stage4Error("Stage 2 daily returns contain duplicate ticker columns.")

    returns = returns.apply(pd.to_numeric, errors="coerce")
    if returns.empty:
        raise Stage4Error("Stage 2 daily return matrix is empty.")
    if returns.isna().any().any():
        raise Stage4Error("Stage 2 daily return matrix contains missing values.")
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise Stage4Error("Stage 2 daily return matrix contains non-finite values.")

    if expected_tickers is not None:
        if set(returns.columns) != set(expected_tickers):
            missing = sorted(set(expected_tickers) - set(returns.columns))
            extra = sorted(set(returns.columns) - set(expected_tickers))
            details: list[str] = []
            if missing:
                details.append("missing tickers: " + ", ".join(missing))
            if extra:
                details.append("unexpected tickers: " + ", ".join(extra))
            raise Stage4Error(
                "Stage 2 return universe differs from Stage 3 weights ("
                + "; ".join(details)
                + ")."
            )
        returns = returns.loc[:, expected_tickers]

    return returns


def _normalize_spy_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize Tiingo or cached SPY prices to the Stage 4 cache schema."""
    if frame is None or frame.empty:
        raise Stage4Error("SPY adjusted-price data is empty.")

    normalized = frame.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    required = {"Date", "Adjusted Close"}
    missing = sorted(required - set(normalized.columns))
    if missing:
        raise Stage4Error(
            "SPY data is missing required columns: " + ", ".join(missing)
        )

    for optional_column in ("Close", "Volume"):
        if optional_column not in normalized.columns:
            normalized[optional_column] = np.nan

    normalized = normalized.loc[:, SPY_CACHE_COLUMNS]
    # Tiingo JSON timestamps include a UTC offset, while Stage 2 CSV dates are
    # timezone-naive trading dates. Parse every provider/cache value through
    # UTC, then remove the timezone and time component before comparisons and
    # alignment. This also handles the date-only values written to the cache.
    normalized["Date"] = (
        pd.to_datetime(normalized["Date"], errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
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
        raise Stage4Error("No valid positive SPY adjusted prices remain.")
    return normalized


def _download_spy_prices(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    api_token: str,
) -> pd.DataFrame:
    """Download SPY adjusted daily prices from Tiingo's EOD endpoint."""
    request_start = start_date - pd.Timedelta(
        days=SPY_DOWNLOAD_BUFFER_CALENDAR_DAYS
    )
    query = urlencode(
        {
            "startDate": request_start.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d"),
            "format": "json",
            "resampleFreq": "daily",
        }
    )
    request = Request(
        f"{TIINGO_API_URL}?{query}",
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
        raise Stage4Error(f"Tiingo returned HTTP status {exc.code} for SPY.") from exc
    except URLError as exc:
        raise Stage4Error(f"Could not reach Tiingo ({exc.reason}).") from exc
    except TimeoutError as exc:
        raise Stage4Error("Tiingo SPY request timed out.") from exc

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise Stage4Error("Tiingo returned invalid JSON for SPY.") from exc

    if isinstance(payload, dict):
        message = payload.get("detail") or payload.get("message") or str(payload)
        raise Stage4Error(f"Tiingo SPY response: {message}")
    if not isinstance(payload, list) or not payload:
        raise Stage4Error("Tiingo returned no historical SPY prices.")

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
    return _normalize_spy_prices(pd.DataFrame(records))


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


def load_spy_data(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    api_token: str | None = None,
    cache_file: Path = SPY_CACHE_FILE,
    force_refresh: bool = FORCE_SPY_REFRESH,
) -> pd.Series:
    """Load cached SPY data or download Tiingo adjusted close, then return returns."""
    start_date = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date).normalize()
    if start_date >= end_date:
        raise Stage4Error("The requested SPY evaluation period is invalid.")

    prices: pd.DataFrame | None = None
    if cache_file.exists() and not force_refresh:
        try:
            cached = _normalize_spy_prices(pd.read_csv(cache_file))
        except Exception as exc:
            raise Stage4Error(f"Cached SPY data is invalid: {exc}") from exc

        has_prior_price = cached["Date"].min() < start_date
        covers_end = cached["Date"].max() >= end_date
        if has_prior_price and covers_end:
            prices = cached
            logging.info("Using cached Tiingo adjusted prices for SPY.")

    if prices is None:
        resolved_token = (api_token or TIINGO_API_TOKEN).strip()
        if not resolved_token:
            raise Stage4Error(
                "TIINGO_API_TOKEN is not set and the SPY cache does not cover "
                "the evaluation period. Set the environment variable before running."
            )
        logging.info("Downloading SPY adjusted prices from Tiingo.")
        prices = _download_spy_prices(start_date, end_date, resolved_token)
        _write_csv_atomically(prices, cache_file, index=False)

    spy_prices = prices.set_index("Date")["Adjusted Close"].sort_index()
    spy_returns = spy_prices.pct_change(fill_method=None).dropna()
    spy_returns = spy_returns.loc[
        (spy_returns.index >= start_date) & (spy_returns.index <= end_date)
    ]
    spy_returns.name = "SPY daily return"
    spy_returns.index.name = "Date"

    if spy_returns.empty:
        raise Stage4Error("No SPY daily returns remain in the evaluation period.")
    if spy_returns.isna().any():
        raise Stage4Error("SPY daily returns contain missing values.")
    if not np.isfinite(spy_returns.to_numpy(dtype=float)).all():
        raise Stage4Error("SPY daily returns contain non-finite values.")
    return spy_returns


def calculate_portfolio_daily_returns(
    daily_returns: pd.DataFrame,
    weights: np.ndarray,
    name: str,
) -> pd.Series:
    """Calculate daily weighted returns before any cumulative compounding."""
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (daily_returns.shape[1],):
        raise Stage4Error(f"{name}: weight dimensions do not match ETF returns.")
    if not np.isfinite(weights).all():
        raise Stage4Error(f"{name}: weights contain non-finite values.")

    weighted_returns = daily_returns.to_numpy(dtype=float) @ weights
    result = pd.Series(weighted_returns, index=daily_returns.index, name=name)

    # This explicit equivalence check guards against calculating individual
    # cumulative ETF returns and weighting those cumulative values afterward.
    expected = np.sum(
        daily_returns.to_numpy(dtype=float) * weights.reshape(1, -1), axis=1
    )
    if not np.allclose(result.to_numpy(), expected, rtol=1e-12, atol=1e-14):
        raise Stage4Error(f"{name}: daily weighted-return validation failed.")
    if result.isna().any() or not np.isfinite(result.to_numpy()).all():
        raise Stage4Error(f"{name}: portfolio returns contain invalid values.")
    return result


def calculate_wealth_curve(
    daily_returns: pd.Series,
    starting_value: float = INITIAL_PORTFOLIO_VALUE,
) -> pd.Series:
    """Compound daily portfolio returns from a value of 100."""
    if not math.isfinite(starting_value) or starting_value <= 0:
        raise Stage4Error("Starting portfolio value must be finite and positive.")
    if daily_returns.empty:
        raise Stage4Error("Cannot calculate a wealth curve from empty returns.")
    if (daily_returns <= -1.0).any():
        raise Stage4Error("A daily return is less than or equal to -100%.")

    wealth = starting_value * (1.0 + daily_returns).cumprod()
    if wealth.isna().any() or not np.isfinite(wealth.to_numpy()).all():
        raise Stage4Error("Wealth curve contains invalid values.")
    return wealth


def calculate_cumulative_return(
    wealth_curve: pd.Series,
    starting_value: float = INITIAL_PORTFOLIO_VALUE,
) -> float:
    """Calculate final wealth divided by initial wealth minus one."""
    result = float(wealth_curve.iloc[-1] / starting_value - 1.0)
    if not math.isfinite(result):
        raise Stage4Error("Cumulative return is not finite.")
    return result


def calculate_annualized_return(daily_returns: pd.Series) -> float:
    """Annualize compounded growth using 252 divided by observation count."""
    number_of_days = len(daily_returns)
    if number_of_days <= 0:
        raise Stage4Error("Cannot annualize an empty daily return series.")

    growth_factor = float((1.0 + daily_returns).prod())
    if not math.isfinite(growth_factor) or growth_factor <= 0:
        raise Stage4Error("Compounded growth factor must be finite and positive.")
    result = growth_factor ** (TRADING_DAYS_PER_YEAR / number_of_days) - 1.0
    if not math.isfinite(result):
        raise Stage4Error("Annualized return is not finite.")
    return float(result)


def calculate_volatility(daily_returns: pd.Series) -> float:
    """Calculate sample daily volatility annualized by sqrt(252)."""
    if len(daily_returns) < 2:
        raise Stage4Error("At least two daily returns are required for volatility.")
    result = float(
        daily_returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)
    )
    if not math.isfinite(result) or result <= 0:
        raise Stage4Error("Annualized volatility must be finite and positive.")
    return result


def calculate_sharpe_ratio(
    annualized_return: float,
    annualized_volatility: float,
) -> float:
    """Calculate Sharpe ratio with the specified zero risk-free rate."""
    if not math.isfinite(annualized_volatility) or annualized_volatility <= 0:
        raise Stage4Error("Sharpe-ratio volatility must be finite and positive.")
    result = annualized_return / annualized_volatility
    if not math.isfinite(result):
        raise Stage4Error("Sharpe ratio is not finite.")
    return float(result)


def calculate_spy_correlation(
    portfolio_returns: pd.Series,
    spy_returns: pd.Series,
) -> float:
    """Calculate daily return correlation with SPY over identical dates."""
    if not portfolio_returns.index.equals(spy_returns.index):
        raise Stage4Error("Portfolio and SPY dates are not aligned.")
    result = float(portfolio_returns.corr(spy_returns))
    if not math.isfinite(result):
        raise Stage4Error("SPY correlation is not finite.")
    return result


def calculate_max_drawdown(wealth_curve: pd.Series) -> float:
    """Calculate the minimum wealth-to-running-peak drawdown."""
    running_peak = wealth_curve.cummax()
    drawdown = (wealth_curve - running_peak) / running_peak
    result = float(drawdown.min())
    if not math.isfinite(result):
        raise Stage4Error("Maximum drawdown is not finite.")
    return result


def _load_risk_inputs(
    tickers: list[str],
    volatility_file: Path = VOLATILITY_FILE,
    covariance_file: Path = COVARIANCE_FILE,
) -> tuple[np.ndarray, np.ndarray]:
    """Load Stage 2 sigma and covariance solely for the achieved MDP DR."""
    if not volatility_file.exists() or not covariance_file.exists():
        raise Stage4Error("Stage 2 volatility or covariance input is missing.")

    volatility = pd.read_csv(volatility_file)
    missing_columns = sorted(
        set(REQUIRED_VOLATILITY_COLUMNS) - set(volatility.columns)
    )
    if missing_columns:
        raise Stage4Error(
            "Stage 2 volatility is missing columns: " + ", ".join(missing_columns)
        )
    volatility["ticker"] = (
        volatility["ticker"].fillna("").astype(str).str.strip().str.upper()
    )
    volatility["annualized_volatility"] = pd.to_numeric(
        volatility["annualized_volatility"], errors="coerce"
    )
    if volatility["ticker"].duplicated().any():
        raise Stage4Error("Stage 2 volatility contains duplicate tickers.")
    if set(volatility["ticker"]) != set(tickers):
        raise Stage4Error("Stage 2 volatility tickers differ from MDP weights.")
    volatility = volatility.set_index("ticker").loc[tickers]
    sigma = volatility["annualized_volatility"].to_numpy(dtype=float)
    if not np.isfinite(sigma).all() or (sigma <= 0).any():
        raise Stage4Error("Stage 2 annualized volatility values are invalid.")

    covariance_frame = pd.read_csv(covariance_file)
    if "ticker" not in covariance_frame.columns:
        raise Stage4Error("Stage 2 covariance must contain a ticker column.")
    covariance_frame["ticker"] = (
        covariance_frame["ticker"].fillna("").astype(str).str.strip().str.upper()
    )
    covariance_frame = covariance_frame.set_index("ticker")
    covariance_frame.columns = [
        str(column).strip().upper() for column in covariance_frame.columns
    ]
    if set(covariance_frame.index) != set(tickers) or set(
        covariance_frame.columns
    ) != set(tickers):
        raise Stage4Error("Stage 2 covariance tickers differ from MDP weights.")
    covariance_frame = covariance_frame.loc[tickers, tickers]
    covariance_frame = covariance_frame.apply(pd.to_numeric, errors="coerce")
    covariance = covariance_frame.to_numpy(dtype=float)
    if not np.isfinite(covariance).all():
        raise Stage4Error("Stage 2 covariance contains invalid values.")
    if not np.allclose(
        covariance,
        covariance.T,
        rtol=SYMMETRY_RTOL,
        atol=SYMMETRY_ATOL,
    ):
        raise Stage4Error("Stage 2 covariance is not symmetric.")
    if (np.diag(covariance) <= 0).any():
        raise Stage4Error("Stage 2 covariance diagonal must be positive.")
    return sigma, covariance


def calculate_diversification_ratio(
    weights: np.ndarray,
    sigma: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Calculate achieved DR from fixed weights and existing Stage 2 risk data."""
    weights = np.asarray(weights, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if weights.shape != sigma.shape:
        raise Stage4Error("Weight and volatility dimensions do not match.")
    if covariance.shape != (len(weights), len(weights)):
        raise Stage4Error("Covariance dimensions do not match the weights.")

    numerator = float(weights @ sigma)
    variance = float(weights @ covariance @ weights)
    if not math.isfinite(numerator) or not math.isfinite(variance) or variance <= 0:
        raise Stage4Error("Diversification-ratio inputs produce invalid values.")
    result = numerator / math.sqrt(variance)
    if not math.isfinite(result):
        raise Stage4Error("Diversification ratio is not finite.")
    return float(result)


def _calculate_performance_row(
    portfolio_name: str,
    daily_returns: pd.Series,
    wealth_curve: pd.Series,
    spy_returns: pd.Series,
) -> dict[str, object]:
    """Calculate all required performance statistics for one portfolio."""
    annualized_return = calculate_annualized_return(daily_returns)
    annualized_volatility = calculate_volatility(daily_returns)
    return {
        "Portfolio": portfolio_name,
        "Cumulative Return": calculate_cumulative_return(wealth_curve),
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": calculate_sharpe_ratio(
            annualized_return, annualized_volatility
        ),
        "SPY Correlation": calculate_spy_correlation(
            daily_returns, spy_returns
        ),
        "Maximum Drawdown": calculate_max_drawdown(wealth_curve),
    }


def generate_report(
    performance_summary: pd.DataFrame,
    allocation_summary: pd.DataFrame,
    diagnostics: dict[str, float | int],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    number_of_observations: int,
) -> str:
    """Generate a factual Markdown methodology and metric summary."""
    sector_allocations = (
        allocation_summary.groupby("sector", sort=True)["weight"].sum()
    )
    sector_lines = [
        f"| {sector} | {weight:.10f} |"
        for sector, weight in sector_allocations.items()
    ]

    return "\n".join(
        [
            "# ETF Portfolio 4.0 - Stage 4 Evaluation Summary",
            "",
            "## Methodology",
            "",
            "The evaluation uses the fixed Stage 3 MDP weights without "
            "re-optimization. On each common trading date, ETF daily returns "
            "are multiplied by their fixed weights and summed before the "
            "portfolio return series is compounded.",
            "",
            f"Evaluation period: {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}",
            f"Common daily observations: {number_of_observations}",
            "",
            "## Benchmark definitions",
            "",
            "- MDP Portfolio: fixed weights from Stage 3.",
            "- Equal Weight Portfolio: equal weight across the same selected ETFs.",
            "- SPY: Tiingo adjusted-close daily returns over the common period.",
            "",
            "## Formulas",
            "",
            "- Daily portfolio return: `r_p,t = sum(w_i * r_i,t)`.",
            "- Wealth: `V_t = V_(t-1) * (1 + r_p,t)`, starting from 100.",
            "- Cumulative return: `V_T / V_0 - 1`.",
            "- Annualized return: `(V_T / V_0)^(252/T) - 1`.",
            "- Annualized volatility: `std(daily returns) * sqrt(252)`.",
            "- Sharpe ratio: annualized return divided by annualized volatility; "
            "risk-free rate is zero.",
            "- SPY correlation: correlation of aligned daily returns.",
            "- Maximum drawdown: minimum of `(V_t - Peak_t) / Peak_t`.",
            "- Diversification ratio: `(w^T sigma) / sqrt(w^T Sigma w)`.",
            "",
            "## Metrics calculated",
            "",
            "The CSV outputs contain cumulative return, annualized return, "
            "annualized volatility, Sharpe ratio, SPY correlation, and maximum "
            "drawdown for all three portfolios.",
            "",
            "## MDP diagnostics",
            "",
            f"- Number of active ETFs: {diagnostics['number_of_active_etfs']}",
            f"- Maximum ETF weight: {diagnostics['maximum_weight']:.10f}",
            f"- Minimum ETF weight: {diagnostics['minimum_weight']:.10f}",
            f"- Achieved diversification ratio: "
            f"{diagnostics['diversification_ratio']:.10f}",
            "",
            "## MDP sector allocation",
            "",
            "| Sector | Weight |",
            "|---|---:|",
            *sector_lines,
            "",
            "This report records calculations and methodology only; it does not "
            "interpret the results.",
            "",
        ]
    )


def save_outputs(
    performance_summary: pd.DataFrame,
    return_series: pd.DataFrame,
    value_series: pd.DataFrame,
    allocation_summary: pd.DataFrame,
    report_text: str,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Save the five validated Stage 4 outputs."""
    expected_rows = ["MDP Portfolio", "Equal Weight Portfolio", "SPY"]
    if performance_summary["Portfolio"].tolist() != expected_rows:
        raise Stage4Error("Performance summary portfolio rows are incorrect.")
    if not return_series.index.equals(value_series.index):
        raise Stage4Error("Return and value output dates are not identical.")
    if return_series.isna().any().any() or value_series.isna().any().any():
        raise Stage4Error("Missing values remain in Stage 4 output series.")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_atomically(
        performance_summary,
        output_dir / "portfolio_performance_summary.csv",
        index=False,
    )
    _write_csv_atomically(
        return_series,
        output_dir / "portfolio_return_series.csv",
        index=True,
        index_label="Date",
    )
    _write_csv_atomically(
        value_series,
        output_dir / "portfolio_value_series.csv",
        index=True,
        index_label="Date",
    )
    _write_csv_atomically(
        allocation_summary,
        output_dir / "mdp_allocation_summary.csv",
        index=False,
    )

    report_file = output_dir / "evaluation_summary.md"
    temporary_report = report_file.with_suffix(".md.tmp")
    temporary_report.write_text(report_text, encoding="utf-8")
    temporary_report.replace(report_file)


def main() -> None:
    """Run the fixed-weight Stage 4 evaluation workflow."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    weights_frame = load_weights()
    tickers = weights_frame["ticker"].tolist()
    mdp_weights = weights_frame["weight"].to_numpy(dtype=float)
    etf_returns = load_returns(expected_tickers=tickers)
    spy_returns = load_spy_data(etf_returns.index.min(), etf_returns.index.max())

    aligned = etf_returns.join(spy_returns, how="inner")
    aligned = aligned.dropna(how="any")
    if aligned.empty:
        raise Stage4Error("No common ETF and SPY return dates remain.")
    if aligned.isna().any().any():
        raise Stage4Error("Missing values remain after ETF/SPY date alignment.")

    aligned_etf_returns = aligned.loc[:, tickers]
    aligned_spy_returns = aligned["SPY daily return"].copy()
    if not aligned_etf_returns.index.equals(aligned_spy_returns.index):
        raise Stage4Error("ETF and SPY daily return dates are not aligned.")

    equal_weights = np.full(len(tickers), 1.0 / len(tickers), dtype=float)
    mdp_returns = calculate_portfolio_daily_returns(
        aligned_etf_returns, mdp_weights, "MDP daily return"
    )
    equal_returns = calculate_portfolio_daily_returns(
        aligned_etf_returns, equal_weights, "Equal Weight daily return"
    )

    return_series = pd.concat(
        [mdp_returns, equal_returns, aligned_spy_returns], axis=1
    )
    return_series.index.name = "Date"
    if return_series.isna().any().any():
        raise Stage4Error("Missing values remain in portfolio return series.")

    mdp_value = calculate_wealth_curve(mdp_returns).rename(
        "MDP portfolio value"
    )
    equal_value = calculate_wealth_curve(equal_returns).rename(
        "Equal Weight portfolio value"
    )
    spy_value = calculate_wealth_curve(aligned_spy_returns).rename("SPY value")
    value_series = pd.concat([mdp_value, equal_value, spy_value], axis=1)
    value_series.index.name = "Date"

    performance_summary = pd.DataFrame(
        [
            _calculate_performance_row(
                "MDP Portfolio", mdp_returns, mdp_value, aligned_spy_returns
            ),
            _calculate_performance_row(
                "Equal Weight Portfolio",
                equal_returns,
                equal_value,
                aligned_spy_returns,
            ),
            _calculate_performance_row(
                "SPY", aligned_spy_returns, spy_value, aligned_spy_returns
            ),
        ]
    )

    sigma, covariance = _load_risk_inputs(tickers)
    diversification_ratio = calculate_diversification_ratio(
        mdp_weights, sigma, covariance
    )
    diagnostics: dict[str, float | int] = {
        "number_of_active_etfs": int(np.count_nonzero(mdp_weights > 0.0)),
        "maximum_weight": float(np.max(mdp_weights)),
        "minimum_weight": float(np.min(mdp_weights)),
        "diversification_ratio": diversification_ratio,
    }

    allocation_summary = weights_frame.loc[:, REQUIRED_WEIGHT_COLUMNS].copy()
    report_text = generate_report(
        performance_summary,
        allocation_summary,
        diagnostics,
        return_series.index.min(),
        return_series.index.max(),
        len(return_series),
    )
    save_outputs(
        performance_summary,
        return_series,
        value_series,
        allocation_summary,
        report_text,
    )

    logging.info(
        "Stage 4 complete: evaluated MDP, equal weight, and SPY over %d dates.",
        len(return_series),
    )
    logging.info("Stage 3 weights were used directly and were not optimized again.")


if __name__ == "__main__":
    try:
        main()
    except Stage4Error as exc:
        logging.error("%s", exc)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        logging.error("Interrupted by user. No partial evaluation is considered final.")
        raise SystemExit(130)
