"""Stage 4: constrained minimum portfolio-SPY correlation optimization.

Inputs
------
* selected_sector_etfs.csv
* daily_returns.csv
* optimal_correlation_window.csv

The Stage 3-selected estimation window determines how many of the latest
complete daily return observations are used. SciPy SLSQP directly minimizes:

    correlation(ETF_Returns @ weights, SPY_Returns)

subject to:

    sum(weights) = 1
    0 <= weight_i <= 0.30

The 30% maximum is a configurable investment constraint. Return, volatility,
Sharpe ratio, drawdown, and concentration measures are calculated only after
optimization and never affect the objective.

Local usage
-----------
    python -m pip install pandas numpy scipy
    python stage4_minimum_correlation_portfolio.py

All numerical results are generated locally when this script is run.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from scipy.optimize import OptimizeResult, minimize
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install requirements with: "
        "python -m pip install pandas numpy scipy"
    ) from exc


# ---------------------------------------------------------------------------
# User-configurable settings
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
SELECTED_ETFS_FILE = BASE_DIR / "selected_sector_etfs.csv"
DAILY_RETURNS_FILE = BASE_DIR / "daily_returns.csv"
OPTIMAL_WINDOW_FILE = BASE_DIR / "optimal_correlation_window.csv"

WEIGHTS_OUTPUT_FILE = BASE_DIR / "optimal_portfolio_weights.csv"
PERFORMANCE_OUTPUT_FILE = BASE_DIR / "portfolio_performance_summary.csv"
OPTIMIZATION_DETAILS_FILE = BASE_DIR / "optimization_details.csv"
CONCENTRATION_OUTPUT_FILE = BASE_DIR / "portfolio_concentration_metrics.csv"

BENCHMARK_TICKER = "SPY"
EXPECTED_SECTOR_COUNT = 11
TRADING_DAYS_PER_YEAR = 252
ANNUAL_RISK_FREE_RATE = 0.0

MAX_ETF_WEIGHT = 0.30
ACTIVE_WEIGHT_THRESHOLD = 1e-6

NUMBER_OF_RANDOM_INITIALIZATIONS = 30
RANDOM_SEED = 20260808
SLSQP_MAX_ITERATIONS = 2_000
SLSQP_FTOL = 1e-12
WEIGHT_TOLERANCE = 1e-8
CORRELATION_TOLERANCE = 1e-10

WINDOW_TO_TRADING_DAYS = {
    "3 months": 63,
    "6 months": 126,
    "1 year": 252,
    "2 years": 504,
    "3 years": 756,
}

SELECTED_ETF_COLUMNS = (
    "gics_sector",
    "ticker",
    "etf_name",
    "issuer",
    "ADV",
    "observation_period",
)

WEIGHT_COLUMNS = (
    "ticker",
    "gics_sector",
    "weight",
)

PERFORMANCE_COLUMNS = (
    "portfolio_SPY_correlation",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "estimation_window",
)

# resulting_weights is retained because every initialization must record its
# complete solution, in addition to the explicitly requested diagnostic fields.
OPTIMIZATION_DETAIL_COLUMNS = (
    "initialization_method",
    "optimization_success",
    "final_correlation",
    "iterations",
    "constraint_satisfied",
    "resulting_weights",
)

CONCENTRATION_COLUMNS = (
    "maximum_weight",
    "HHI",
    "number_of_active_ETFs",
)


class Stage4Error(RuntimeError):
    """Raised when Stage 4 cannot safely produce valid portfolio outputs."""


@dataclass(frozen=True)
class EstimationPeriod:
    """Selected window label, length, dates, and complete return observations."""

    window_label: str
    trading_days: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    returns: pd.DataFrame

    @property
    def description(self) -> str:
        return (
            f"{self.window_label} ({self.trading_days} trading days; "
            f"{self.start_date:%Y-%m-%d} to {self.end_date:%Y-%m-%d})"
        )


def _parse_boolean(value: object) -> bool:
    """Parse common CSV boolean representations without truthy-string errors."""
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n", ""}:
        return False
    raise Stage4Error(f"Unrecognized boolean value: {value!r}")


def _load_selected_universe(
    selected_file: Path = SELECTED_ETFS_FILE,
) -> pd.DataFrame:
    """Load and validate the 11 sector ETFs selected in Stage 2."""
    if not selected_file.exists():
        raise Stage4Error(f"Stage 2 input not found: {selected_file}")
    try:
        selected = pd.read_csv(selected_file, dtype=str)
    except Exception as exc:
        raise Stage4Error(f"Could not read {selected_file}: {exc}") from exc

    missing = sorted(set(SELECTED_ETF_COLUMNS) - set(selected.columns))
    if missing:
        raise Stage4Error(
            "selected_sector_etfs.csv is missing columns: " + ", ".join(missing)
        )

    selected = selected.loc[:, SELECTED_ETF_COLUMNS].copy()
    for column in SELECTED_ETF_COLUMNS:
        selected[column] = selected[column].fillna("").str.strip()
    if selected.eq("").any().any():
        blank_columns = selected.columns[selected.eq("").any()].tolist()
        raise Stage4Error(
            "selected_sector_etfs.csv has blank values in: "
            + ", ".join(blank_columns)
        )

    selected["ticker"] = selected["ticker"].str.upper()
    if len(selected) != EXPECTED_SECTOR_COUNT:
        raise Stage4Error(
            f"Expected {EXPECTED_SECTOR_COUNT} selected ETFs; found {len(selected)}."
        )
    if selected["ticker"].duplicated().any():
        raise Stage4Error("Selected ETF tickers must be unique.")
    if selected["gics_sector"].nunique() != EXPECTED_SECTOR_COUNT:
        raise Stage4Error("There must be exactly one selected ETF per GICS sector.")
    if BENCHMARK_TICKER in set(selected["ticker"]):
        raise Stage4Error(f"{BENCHMARK_TICKER} cannot also be a sector ETF.")
    return selected


def _load_selected_window(
    optimal_window_file: Path = OPTIMAL_WINDOW_FILE,
) -> tuple[str, int]:
    """Read the single Stage 3-selected window and convert it to trading days."""
    if not optimal_window_file.exists():
        raise Stage4Error(f"Stage 3 window file not found: {optimal_window_file}")
    try:
        windows = pd.read_csv(optimal_window_file, dtype=str)
    except Exception as exc:
        raise Stage4Error(f"Could not read {optimal_window_file}: {exc}") from exc

    required = {"window", "selected_optimal_window"}
    missing = sorted(required - set(windows.columns))
    if missing:
        raise Stage4Error(
            "optimal_correlation_window.csv is missing columns: "
            + ", ".join(missing)
        )

    selected_flags = windows["selected_optimal_window"].map(_parse_boolean)
    selected_rows = windows.loc[selected_flags]
    if len(selected_rows) != 1:
        raise Stage4Error(
            "optimal_correlation_window.csv must mark exactly one selected window."
        )

    window_label = str(selected_rows.iloc[0]["window"]).strip()
    if window_label not in WINDOW_TO_TRADING_DAYS:
        supported = ", ".join(WINDOW_TO_TRADING_DAYS)
        raise Stage4Error(
            f"Unsupported selected window {window_label!r}. Supported labels: {supported}"
        )
    return window_label, WINDOW_TO_TRADING_DAYS[window_label]


def load_returns(returns_file: Path = DAILY_RETURNS_FILE) -> pd.DataFrame:
    """Load the Stage 3 daily return matrix without choosing dates yet."""
    if not returns_file.exists():
        raise Stage4Error(f"Stage 3 return file not found: {returns_file}")
    try:
        returns = pd.read_csv(returns_file)
    except Exception as exc:
        raise Stage4Error(f"Could not read {returns_file}: {exc}") from exc

    if "Date" not in returns.columns:
        raise Stage4Error("daily_returns.csv must contain a Date column.")
    returns["Date"] = pd.to_datetime(returns["Date"], errors="coerce")
    if returns["Date"].isna().any():
        raise Stage4Error("daily_returns.csv contains invalid dates.")
    if returns["Date"].duplicated().any():
        raise Stage4Error("daily_returns.csv contains duplicate dates.")

    returns = returns.set_index("Date").sort_index()
    returns.index = pd.DatetimeIndex(returns.index).tz_localize(None).normalize()
    for column in returns.columns:
        returns[column] = pd.to_numeric(returns[column], errors="coerce")
    returns = returns.replace([np.inf, -np.inf], np.nan)
    if returns.empty:
        raise Stage4Error("daily_returns.csv contains no return observations.")
    return returns


def select_estimation_period(
    all_returns: pd.DataFrame,
    etf_tickers: list[str],
    benchmark_ticker: str,
    window_label: str,
    trading_days: int,
) -> EstimationPeriod:
    """Select the latest complete observations required by the Stage 3 window."""
    required_columns = etf_tickers + [benchmark_ticker]
    missing_columns = sorted(set(required_columns) - set(all_returns.columns))
    if missing_columns:
        raise Stage4Error(
            "daily_returns.csv is missing selected ETF/benchmark columns: "
            + ", ".join(missing_columns)
        )
    if trading_days < 2:
        raise Stage4Error("The estimation window must contain at least two days.")

    complete_returns = all_returns.loc[:, required_columns].dropna(how="any")
    if len(complete_returns) < trading_days:
        raise Stage4Error(
            f"Selected Stage 3 window requires {trading_days} complete observations; "
            f"only {len(complete_returns)} are available."
        )

    selected_returns = complete_returns.tail(trading_days).copy()
    if len(selected_returns) != trading_days:
        raise Stage4Error("Estimation-period row count does not match the selected window.")
    if selected_returns.index.duplicated().any():
        raise Stage4Error("Estimation period contains duplicate dates.")

    standard_deviations = selected_returns.std(ddof=1)
    zero_variance = standard_deviations[
        (~np.isfinite(standard_deviations)) | (standard_deviations <= 0)
    ].index.tolist()
    if zero_variance:
        raise Stage4Error(
            "Correlation is undefined for zero-variance columns: "
            + ", ".join(zero_variance)
        )

    return EstimationPeriod(
        window_label=window_label,
        trading_days=trading_days,
        start_date=selected_returns.index.min(),
        end_date=selected_returns.index.max(),
        returns=selected_returns,
    )


def portfolio_return(etf_returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Calculate the weighted daily ETF portfolio-return series."""
    return np.asarray(etf_returns, dtype=float) @ np.asarray(weights, dtype=float)


def correlation_objective(
    weights: np.ndarray,
    etf_returns: np.ndarray,
    spy_returns: np.ndarray,
) -> float:
    """Directly calculate portfolio-SPY correlation for SLSQP minimization."""
    portfolio = portfolio_return(etf_returns, weights)
    if not np.isfinite(portfolio).all() or not np.isfinite(spy_returns).all():
        return 2.0
    if np.std(portfolio, ddof=1) <= 0 or np.std(spy_returns, ddof=1) <= 0:
        return 2.0
    correlation = float(np.corrcoef(portfolio, spy_returns)[0, 1])
    return correlation if math.isfinite(correlation) else 2.0


def _fully_invested_constraint(weights: np.ndarray) -> float:
    """Equality constraint supplied to SLSQP."""
    return float(np.sum(weights) - 1.0)


def _weights_are_feasible(weights: np.ndarray) -> bool:
    """Check full investment, long-only weights, and the configured maximum."""
    weights = np.asarray(weights, dtype=float)
    return bool(
        weights.ndim == 1
        and len(weights) > 0
        and np.isfinite(weights).all()
        and weights.min() >= -WEIGHT_TOLERANCE
        and weights.max() <= MAX_ETF_WEIGHT + WEIGHT_TOLERANCE
        and math.isclose(float(weights.sum()), 1.0, abs_tol=WEIGHT_TOLERANCE)
    )


def validate_constraints(weights: np.ndarray) -> None:
    """Raise a clear error if any final portfolio constraint is violated."""
    weights = np.asarray(weights, dtype=float)
    problems: list[str] = []
    if weights.ndim != 1 or len(weights) == 0:
        problems.append("weights must be a nonempty one-dimensional vector")
    elif not np.isfinite(weights).all():
        problems.append("weights contain non-finite values")
    else:
        if not math.isclose(float(weights.sum()), 1.0, abs_tol=WEIGHT_TOLERANCE):
            problems.append(f"weights sum to {weights.sum():.12f}, not 1")
        negative = np.flatnonzero(weights < -WEIGHT_TOLERANCE)
        if len(negative):
            problems.append(f"{len(negative)} weight(s) are negative")
        above_cap = np.flatnonzero(weights > MAX_ETF_WEIGHT + WEIGHT_TOLERANCE)
        if len(above_cap):
            problems.append(
                f"{len(above_cap)} weight(s) exceed the {MAX_ETF_WEIGHT:.0%} maximum"
            )
    if problems:
        raise Stage4Error("Portfolio constraint violation: " + "; ".join(problems))


def _validate_problem_feasibility(number_of_assets: int) -> None:
    """Ensure a fully invested portfolio can exist under the maximum weight."""
    if number_of_assets < 1:
        raise Stage4Error("At least one ETF is required.")
    if not 0.0 < MAX_ETF_WEIGHT <= 1.0:
        raise Stage4Error("MAX_ETF_WEIGHT must be greater than 0 and at most 1.")
    if number_of_assets * MAX_ETF_WEIGHT < 1.0 - WEIGHT_TOLERANCE:
        raise Stage4Error("The maximum ETF weight is infeasible for the ETF count.")


def _random_valid_weights(
    rng: np.random.Generator,
    number_of_assets: int,
    max_attempts: int = 100_000,
) -> np.ndarray:
    """Sample a reproducible long-only portfolio that respects the 30% cap."""
    for _ in range(max_attempts):
        candidate = rng.dirichlet(np.ones(number_of_assets))
        if _weights_are_feasible(candidate):
            return candidate
    raise Stage4Error("Could not generate a random portfolio below the maximum weight.")


def run_slsqp_optimization(
    etf_returns: np.ndarray,
    spy_returns: np.ndarray,
    initial_weights: np.ndarray,
) -> OptimizeResult:
    """Run one direct minimum-correlation SLSQP optimization attempt."""
    number_of_assets = etf_returns.shape[1]
    initial_weights = np.asarray(initial_weights, dtype=float)
    if initial_weights.shape != (number_of_assets,):
        raise Stage4Error("Initial-weight vector has an invalid shape.")
    if not _weights_are_feasible(initial_weights):
        raise Stage4Error("Initial weights violate portfolio constraints.")

    constraints = ({"type": "eq", "fun": _fully_invested_constraint},)
    bounds = [(0.0, MAX_ETF_WEIGHT)] * number_of_assets
    return minimize(
        correlation_objective,
        x0=initial_weights,
        args=(etf_returns, spy_returns),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": SLSQP_MAX_ITERATIONS,
            "ftol": SLSQP_FTOL,
            "disp": False,
        },
    )


def _clean_solver_weights(weights: np.ndarray) -> np.ndarray:
    """Remove solver-level numerical noise without materially changing weights."""
    cleaned = np.asarray(weights, dtype=float).copy()
    if not _weights_are_feasible(cleaned):
        raise Stage4Error("SLSQP returned an infeasible weight vector.")
    cleaned = np.clip(cleaned, 0.0, MAX_ETF_WEIGHT)
    residual = 1.0 - float(cleaned.sum())
    if residual >= 0:
        target = int(np.argmax(MAX_ETF_WEIGHT - cleaned))
    else:
        target = int(np.argmax(cleaned))
    cleaned[target] += residual
    validate_constraints(cleaned)
    return cleaned


def run_multiple_initializations(
    etf_returns: np.ndarray,
    spy_returns: np.ndarray,
    etf_tickers: list[str],
    random_seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, float, pd.DataFrame]:
    """Run equal/random starts and retain the lowest successful correlation."""
    etf_returns = np.asarray(etf_returns, dtype=float)
    spy_returns = np.asarray(spy_returns, dtype=float)
    if etf_returns.ndim != 2:
        raise Stage4Error("ETF returns must be a two-dimensional matrix.")
    number_of_assets = etf_returns.shape[1]
    _validate_problem_feasibility(number_of_assets)
    if len(etf_tickers) != number_of_assets:
        raise Stage4Error("ETF ticker count does not match the return matrix.")
    if spy_returns.ndim != 1 or len(spy_returns) != len(etf_returns):
        raise Stage4Error("SPY returns do not align with ETF returns.")
    if not np.isfinite(etf_returns).all() or not np.isfinite(spy_returns).all():
        raise Stage4Error("Optimization returns must be finite.")
    if NUMBER_OF_RANDOM_INITIALIZATIONS < 0:
        raise Stage4Error("NUMBER_OF_RANDOM_INITIALIZATIONS cannot be negative.")

    equal_weights = np.full(number_of_assets, 1.0 / number_of_assets)
    if not _weights_are_feasible(equal_weights):
        raise Stage4Error("Equal-weight initialization violates the maximum weight.")

    rng = np.random.default_rng(random_seed)
    initializations: list[tuple[str, np.ndarray]] = [
        ("equal_weight", equal_weights)
    ]
    for attempt in range(1, NUMBER_OF_RANDOM_INITIALIZATIONS + 1):
        initializations.append(
            (
                f"random_dirichlet_{attempt:03d}",
                _random_valid_weights(rng, number_of_assets),
            )
        )

    detail_rows: list[dict[str, object]] = []
    successful: list[tuple[float, np.ndarray, str]] = []

    for method_name, initial_weights in initializations:
        try:
            result = run_slsqp_optimization(
                etf_returns, spy_returns, initial_weights
            )
            candidate = np.asarray(result.x, dtype=float)
            constraint_satisfied = _weights_are_feasible(candidate)
            final_correlation = correlation_objective(
                candidate, etf_returns, spy_returns
            )
            success = bool(
                result.success
                and constraint_satisfied
                and math.isfinite(final_correlation)
                and -1.0 - CORRELATION_TOLERANCE
                <= final_correlation
                <= 1.0 + CORRELATION_TOLERANCE
            )
            iterations = int(getattr(result, "nit", 0))
        except Exception as exc:
            logging.warning("Initialization %s failed: %s", method_name, exc)
            candidate = np.full(number_of_assets, np.nan)
            constraint_satisfied = False
            final_correlation = float("nan")
            success = False
            iterations = 0

        resulting_weights = (
            json.dumps(
                {
                    ticker: round(float(weight), 12)
                    for ticker, weight in zip(etf_tickers, candidate)
                },
                separators=(",", ":"),
            )
            if np.isfinite(candidate).all()
            else ""
        )
        detail_rows.append(
            {
                "initialization_method": method_name,
                "optimization_success": success,
                "final_correlation": final_correlation,
                "iterations": iterations,
                "constraint_satisfied": constraint_satisfied,
                "resulting_weights": resulting_weights,
            }
        )
        if success:
            successful.append((final_correlation, candidate.copy(), method_name))

    details = pd.DataFrame(detail_rows, columns=OPTIMIZATION_DETAIL_COLUMNS)
    if not successful:
        raise Stage4Error(
            "No SLSQP initialization produced a successful feasible solution."
        )

    successful.sort(key=lambda item: (item[0], item[2]))
    _, best_weights, _ = successful[0]
    best_weights = _clean_solver_weights(best_weights)
    best_correlation = correlation_objective(
        best_weights, etf_returns, spy_returns
    )
    if not math.isfinite(best_correlation) or not -1.0 <= best_correlation <= 1.0:
        raise Stage4Error("The selected portfolio correlation is invalid.")
    return best_weights, best_correlation, details


def calculate_portfolio_metrics(
    portfolio_returns: np.ndarray,
    spy_returns: np.ndarray,
    weights: np.ndarray,
    optimized_correlation: float,
    estimation_window: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate post-optimization performance and concentration measures."""
    portfolio_returns = np.asarray(portfolio_returns, dtype=float)
    spy_returns = np.asarray(spy_returns, dtype=float)
    weights = np.asarray(weights, dtype=float)
    validate_constraints(weights)
    if len(portfolio_returns) != len(spy_returns) or len(portfolio_returns) < 2:
        raise Stage4Error("Portfolio and SPY returns must have equal valid lengths.")
    if not np.isfinite(portfolio_returns).all() or not np.isfinite(spy_returns).all():
        raise Stage4Error("Portfolio metrics received non-finite returns.")
    if (portfolio_returns <= -1.0).any():
        raise Stage4Error("A return at or below -100% prevents compounding.")

    realized_correlation = float(np.corrcoef(portfolio_returns, spy_returns)[0, 1])
    if not math.isclose(
        realized_correlation,
        optimized_correlation,
        rel_tol=CORRELATION_TOLERANCE,
        abs_tol=CORRELATION_TOLERANCE,
    ):
        raise Stage4Error("Evaluation correlation differs from the optimized objective.")

    growth = float(np.prod(1.0 + portfolio_returns))
    annualized_return = growth ** (TRADING_DAYS_PER_YEAR / len(portfolio_returns)) - 1.0
    annualized_volatility = float(
        np.std(portfolio_returns, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)
    )
    if annualized_volatility <= 0:
        raise Stage4Error("Portfolio volatility must be positive.")
    sharpe_ratio = (
        annualized_return - ANNUAL_RISK_FREE_RATE
    ) / annualized_volatility

    wealth = np.concatenate(([1.0], np.cumprod(1.0 + portfolio_returns)))
    running_maximum = np.maximum.accumulate(wealth)
    maximum_drawdown = float(np.min(wealth / running_maximum - 1.0))

    performance = pd.DataFrame(
        [
            {
                "portfolio_SPY_correlation": realized_correlation,
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_volatility,
                "sharpe_ratio": sharpe_ratio,
                "maximum_drawdown": maximum_drawdown,
                "estimation_window": estimation_window,
            }
        ],
        columns=PERFORMANCE_COLUMNS,
    )
    concentration = pd.DataFrame(
        [
            {
                "maximum_weight": float(np.max(weights)),
                "HHI": float(np.dot(weights, weights)),
                "number_of_active_ETFs": int(
                    np.count_nonzero(weights > ACTIVE_WEIGHT_THRESHOLD)
                ),
            }
        ],
        columns=CONCENTRATION_COLUMNS,
    )
    numeric_values = [
        realized_correlation,
        annualized_return,
        annualized_volatility,
        sharpe_ratio,
        maximum_drawdown,
        float(concentration.iloc[0]["maximum_weight"]),
        float(concentration.iloc[0]["HHI"]),
    ]
    if not all(math.isfinite(value) for value in numeric_values):
        raise Stage4Error("Portfolio output metrics contain non-finite values.")
    return performance, concentration


def _write_csv_atomically(frame: pd.DataFrame, output_file: Path) -> None:
    """Write through a temporary file to avoid leaving partial CSV content."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.10f")
    temporary.replace(output_file)


def save_results(
    portfolio_weights: pd.DataFrame,
    performance_summary: pd.DataFrame,
    optimization_details: pd.DataFrame,
    concentration_metrics: pd.DataFrame,
) -> None:
    """Validate every output before saving any final Stage 4 CSV."""
    if tuple(portfolio_weights.columns) != WEIGHT_COLUMNS:
        raise Stage4Error("optimal_portfolio_weights.csv columns are invalid.")
    if tuple(performance_summary.columns) != PERFORMANCE_COLUMNS:
        raise Stage4Error("portfolio_performance_summary.csv columns are invalid.")
    if tuple(optimization_details.columns) != OPTIMIZATION_DETAIL_COLUMNS:
        raise Stage4Error("optimization_details.csv columns are invalid.")
    if tuple(concentration_metrics.columns) != CONCENTRATION_COLUMNS:
        raise Stage4Error("portfolio_concentration_metrics.csv columns are invalid.")

    if len(portfolio_weights) != EXPECTED_SECTOR_COUNT:
        raise Stage4Error("Final weights must contain all 11 sector ETFs.")
    if portfolio_weights["ticker"].duplicated().any():
        raise Stage4Error("Final portfolio contains duplicate tickers.")
    if portfolio_weights["gics_sector"].nunique() != EXPECTED_SECTOR_COUNT:
        raise Stage4Error("Final portfolio must contain all 11 unique sectors.")
    final_weights = portfolio_weights["weight"].to_numpy(dtype=float)
    validate_constraints(final_weights)

    if not optimization_details["optimization_success"].any():
        raise Stage4Error("Optimization details contain no successful run.")
    successful_details = optimization_details.loc[
        optimization_details["optimization_success"]
    ]
    if not successful_details["constraint_satisfied"].all():
        raise Stage4Error("A successful optimization run violates constraints.")
    if successful_details["resulting_weights"].eq("").any():
        raise Stage4Error("A successful optimization run is missing its weights.")

    expected_concentration = {
        "maximum_weight": float(np.max(final_weights)),
        "HHI": float(np.dot(final_weights, final_weights)),
        "number_of_active_ETFs": int(
            np.count_nonzero(final_weights > ACTIVE_WEIGHT_THRESHOLD)
        ),
    }
    for column, expected_value in expected_concentration.items():
        actual = float(concentration_metrics.iloc[0][column])
        if not math.isclose(actual, float(expected_value), abs_tol=WEIGHT_TOLERANCE):
            raise Stage4Error(f"Concentration metric {column} does not match weights.")

    _write_csv_atomically(portfolio_weights, WEIGHTS_OUTPUT_FILE)
    _write_csv_atomically(performance_summary, PERFORMANCE_OUTPUT_FILE)
    _write_csv_atomically(optimization_details, OPTIMIZATION_DETAILS_FILE)
    _write_csv_atomically(concentration_metrics, CONCENTRATION_OUTPUT_FILE)
    logging.info("Saved portfolio weights to %s", WEIGHTS_OUTPUT_FILE)
    logging.info("Saved performance summary to %s", PERFORMANCE_OUTPUT_FILE)
    logging.info("Saved optimization details to %s", OPTIMIZATION_DETAILS_FILE)
    logging.info("Saved concentration metrics to %s", CONCENTRATION_OUTPUT_FILE)


def main() -> None:
    """Run the complete constrained minimum-correlation Stage 4 workflow."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    selected = _load_selected_universe()
    all_returns = load_returns()
    window_label, trading_days = _load_selected_window()
    etf_tickers = selected["ticker"].tolist()
    period = select_estimation_period(
        all_returns,
        etf_tickers,
        BENCHMARK_TICKER,
        window_label,
        trading_days,
    )
    logging.info("Using Stage 3-selected estimation window: %s", period.description)

    etf_matrix = period.returns.loc[:, etf_tickers].to_numpy(dtype=float)
    spy_vector = period.returns[BENCHMARK_TICKER].to_numpy(dtype=float)
    best_weights, best_correlation, optimization_details = (
        run_multiple_initializations(
            etf_matrix,
            spy_vector,
            etf_tickers,
            RANDOM_SEED,
        )
    )
    validate_constraints(best_weights)

    portfolio_returns = portfolio_return(etf_matrix, best_weights)
    performance_summary, concentration_metrics = calculate_portfolio_metrics(
        portfolio_returns,
        spy_vector,
        best_weights,
        best_correlation,
        period.description,
    )
    portfolio_weights = selected.loc[:, ["ticker", "gics_sector"]].copy()
    portfolio_weights["weight"] = best_weights
    portfolio_weights = portfolio_weights.loc[:, WEIGHT_COLUMNS]

    save_results(
        portfolio_weights,
        performance_summary,
        optimization_details,
        concentration_metrics,
    )
    logging.info(
        "Stage 4 complete. Inspect the generated CSV files for calculated results."
    )


if __name__ == "__main__":
    try:
        main()
    except Stage4Error as exc:
        logging.error("%s", exc)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        logging.error("Interrupted by user; no further optimization was performed.")
        raise SystemExit(130)
