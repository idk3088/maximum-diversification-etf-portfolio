"""Construct the long-only Maximum Diversification Portfolio.

This script implements the diversification-ratio methodology described by
Choueifaty and Coignard (2008). It reads the annualized volatility vector and
annualized covariance matrix already produced by Stage 2. It does not download
data, recalculate risk estimates, backtest, or evaluate unrelated objectives.

The practical portfolio is found by maximizing

    DR(w) = (w.T @ sigma) / sqrt(w.T @ covariance @ w)

subject only to full investment and long-only weights. SciPy's SLSQP optimizer
is run from equal weights and multiple reproducible random valid starts. The
successful feasible solution with the highest diversification ratio is saved.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize
except ImportError as exc:
    raise SystemExit(
        "Missing dependencies. Install them with: "
        "python -m pip install -r requirements.txt"
    ) from exc


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent
PROJECT_DIR = STAGE3_DIR.parent
STAGE2_OUTPUT_DIR = (
    PROJECT_DIR / "Stage 2 - Return and Risk Data Preparation" / "outputs"
)

VOLATILITY_FILE = STAGE2_OUTPUT_DIR / "annualized_volatility.csv"
COVARIANCE_FILE = STAGE2_OUTPUT_DIR / "covariance_matrix.csv"
OUTPUT_DIR = STAGE3_DIR / "outputs"


# ---------------------------------------------------------------------------
# User-configurable numerical settings
# ---------------------------------------------------------------------------

NUMBER_OF_RANDOM_INITIALIZATIONS = 25
RANDOM_SEED = 42
SLSQP_MAX_ITERATIONS = 2_000
SLSQP_FTOL = 1e-12

WEIGHT_TOLERANCE = 1e-8
SYMMETRY_RTOL = 1e-10
SYMMETRY_ATOL = 1e-12
DR_COMPARISON_TOLERANCE = 1e-7
ACTIVE_WEIGHT_TOLERANCE = 1e-8

REQUIRED_VOLATILITY_COLUMNS = (
    "ticker",
    "sector",
    "annualized_volatility",
)


class Stage3Error(RuntimeError):
    """Raised when Stage 3 input, optimization, or validation fails."""


def load_inputs(
    volatility_file: Path = VOLATILITY_FILE,
    covariance_file: Path = COVARIANCE_FILE,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load and validate Stage 2 volatility and covariance outputs."""
    if not volatility_file.exists():
        raise Stage3Error(f"Stage 2 volatility file not found: {volatility_file}")
    if not covariance_file.exists():
        raise Stage3Error(f"Stage 2 covariance file not found: {covariance_file}")

    try:
        volatility = pd.read_csv(volatility_file)
    except Exception as exc:
        raise Stage3Error(f"Could not read annualized volatility: {exc}") from exc

    missing_columns = sorted(
        set(REQUIRED_VOLATILITY_COLUMNS) - set(volatility.columns)
    )
    if missing_columns:
        raise Stage3Error(
            "Annualized volatility file is missing columns: "
            + ", ".join(missing_columns)
        )

    volatility = volatility.loc[:, REQUIRED_VOLATILITY_COLUMNS].copy()
    volatility["ticker"] = (
        volatility["ticker"].fillna("").astype(str).str.strip().str.upper()
    )
    volatility["sector"] = (
        volatility["sector"].fillna("").astype(str).str.strip()
    )
    volatility["annualized_volatility"] = pd.to_numeric(
        volatility["annualized_volatility"], errors="coerce"
    )

    if volatility.empty:
        raise Stage3Error("Annualized volatility file contains no ETFs.")
    if volatility[["ticker", "sector"]].eq("").any().any():
        raise Stage3Error("Annualized volatility contains a blank ticker or sector.")
    if volatility["ticker"].duplicated().any():
        duplicates = sorted(
            volatility.loc[
                volatility["ticker"].duplicated(keep=False), "ticker"
            ].unique()
        )
        raise Stage3Error(
            "Annualized volatility contains duplicate tickers: "
            + ", ".join(duplicates)
        )

    sigma = volatility["annualized_volatility"].to_numpy(dtype=float)
    if not np.isfinite(sigma).all():
        raise Stage3Error("Annualized volatility contains non-finite values.")
    if (sigma <= 0).any():
        raise Stage3Error("Annualized volatility values must all be positive.")

    try:
        covariance_frame = pd.read_csv(covariance_file)
    except Exception as exc:
        raise Stage3Error(f"Could not read covariance matrix: {exc}") from exc

    if "ticker" not in covariance_frame.columns:
        raise Stage3Error(
            "Covariance matrix must contain a first column named 'ticker'."
        )

    covariance_frame["ticker"] = (
        covariance_frame["ticker"].fillna("").astype(str).str.strip().str.upper()
    )
    covariance_frame = covariance_frame.set_index("ticker")
    covariance_frame.index.name = "ticker"
    covariance_frame.columns = [
        str(column).strip().upper() for column in covariance_frame.columns
    ]

    if covariance_frame.index.has_duplicates:
        raise Stage3Error("Covariance matrix contains duplicate row tickers.")
    if covariance_frame.columns.duplicated().any():
        raise Stage3Error("Covariance matrix contains duplicate column tickers.")

    tickers = volatility["ticker"].tolist()
    ticker_set = set(tickers)
    row_set = set(covariance_frame.index)
    column_set = set(covariance_frame.columns)
    if row_set != ticker_set or column_set != ticker_set:
        missing_rows = sorted(ticker_set - row_set)
        extra_rows = sorted(row_set - ticker_set)
        missing_columns = sorted(ticker_set - column_set)
        extra_columns = sorted(column_set - ticker_set)
        details: list[str] = []
        if missing_rows:
            details.append("missing rows: " + ", ".join(missing_rows))
        if extra_rows:
            details.append("extra rows: " + ", ".join(extra_rows))
        if missing_columns:
            details.append("missing columns: " + ", ".join(missing_columns))
        if extra_columns:
            details.append("extra columns: " + ", ".join(extra_columns))
        raise Stage3Error(
            "Covariance tickers differ from the volatility universe ("
            + "; ".join(details)
            + ")."
        )

    covariance_frame = covariance_frame.loc[tickers, tickers]
    covariance_frame = covariance_frame.apply(pd.to_numeric, errors="coerce")
    covariance = covariance_frame.to_numpy(dtype=float)

    expected_shape = (len(tickers), len(tickers))
    if covariance.shape != expected_shape:
        raise Stage3Error(
            f"Covariance matrix has shape {covariance.shape}; "
            f"expected {expected_shape}."
        )
    if not np.isfinite(covariance).all():
        raise Stage3Error("Covariance matrix contains non-finite values.")
    if not np.allclose(
        covariance,
        covariance.T,
        rtol=SYMMETRY_RTOL,
        atol=SYMMETRY_ATOL,
    ):
        raise Stage3Error("Covariance matrix is not symmetric.")
    if (np.diag(covariance) <= 0).any():
        raise Stage3Error("Covariance matrix diagonal values must be positive.")

    return volatility, sigma, covariance


def calculate_diversification_ratio(
    weights: np.ndarray,
    sigma: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Calculate DR(w) = (w.T sigma) / sqrt(w.T covariance w)."""
    weights = np.asarray(weights, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    covariance = np.asarray(covariance, dtype=float)

    if weights.ndim != 1 or sigma.ndim != 1:
        raise Stage3Error("Weights and volatility must be one-dimensional vectors.")
    if weights.shape != sigma.shape:
        raise Stage3Error("Weight and volatility vector dimensions do not match.")
    if covariance.shape != (len(weights), len(weights)):
        raise Stage3Error("Covariance dimensions do not match the weight vector.")
    if not np.isfinite(weights).all():
        raise Stage3Error("Weight vector contains non-finite values.")

    numerator = float(weights @ sigma)
    portfolio_variance = float(weights @ covariance @ weights)
    if not math.isfinite(numerator):
        raise Stage3Error("Diversification-ratio numerator is not finite.")
    if not math.isfinite(portfolio_variance) or portfolio_variance <= 0:
        raise Stage3Error(
            "Portfolio variance must be finite and positive to calculate DR."
        )

    diversification_ratio = numerator / math.sqrt(portfolio_variance)
    if not math.isfinite(diversification_ratio):
        raise Stage3Error("Diversification ratio is not finite.")
    return float(diversification_ratio)


def calculate_analytical_solution(
    sigma: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Calculate the unconstrained benchmark proportional to Sigma^-1 sigma."""
    try:
        inverse_covariance_times_sigma = np.linalg.solve(covariance, sigma)
    except np.linalg.LinAlgError as exc:
        raise Stage3Error(
            "Covariance matrix is singular; the analytical benchmark cannot be "
            "calculated without changing the supplied Stage 2 estimates."
        ) from exc

    normalization = float(inverse_covariance_times_sigma.sum())
    if not math.isfinite(normalization) or abs(normalization) <= 1e-14:
        raise Stage3Error(
            "Analytical solution normalization is zero or non-finite."
        )

    weights = inverse_covariance_times_sigma / normalization
    if not np.isfinite(weights).all():
        raise Stage3Error("Analytical weights contain non-finite values.")
    if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=WEIGHT_TOLERANCE):
        raise Stage3Error("Analytical weights do not sum to one.")

    diversification_ratio = calculate_diversification_ratio(
        weights, sigma, covariance
    )
    return weights, diversification_ratio


def validate_constraints(
    weights: np.ndarray,
    tolerance: float = WEIGHT_TOLERANCE,
) -> bool:
    """Return True when weights are finite, fully invested, and long-only."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        return False
    if not np.isfinite(weights).all():
        return False

    sum_satisfied = np.isclose(
        weights.sum(), 1.0, rtol=0.0, atol=tolerance
    )
    # SLSQP is given explicit [0, +infinity) bounds, so the final validation
    # enforces the requested long-only condition directly rather than clipping
    # or manually adjusting an optimizer result.
    long_only_satisfied = bool(np.all(weights >= 0.0))
    return bool(sum_satisfied and long_only_satisfied)


def run_slsqp_optimization(
    initial_weights: np.ndarray,
    sigma: np.ndarray,
    covariance: np.ndarray,
    initialization_method: str,
) -> dict[str, object]:
    """Run one long-only, fully invested SLSQP optimization."""
    initial_weights = np.asarray(initial_weights, dtype=float)
    number_of_assets = len(sigma)

    if initial_weights.shape != (number_of_assets,):
        raise Stage3Error(
            f"{initialization_method}: initial weight dimensions are invalid."
        )
    if not validate_constraints(initial_weights):
        raise Stage3Error(
            f"{initialization_method}: initial weights are not valid long-only "
            "fully invested weights."
        )

    def negative_diversification_ratio(weights: np.ndarray) -> float:
        try:
            return -calculate_diversification_ratio(weights, sigma, covariance)
        except Stage3Error:
            return 1e20

    result = minimize(
        fun=negative_diversification_ratio,
        x0=initial_weights,
        method="SLSQP",
        bounds=[(0.0, None)] * number_of_assets,
        constraints={
            "type": "eq",
            "fun": lambda weights: float(np.sum(weights) - 1.0),
        },
        options={
            "maxiter": SLSQP_MAX_ITERATIONS,
            "ftol": SLSQP_FTOL,
            "disp": False,
        },
    )

    final_weights = np.asarray(result.x, dtype=float)
    constraint_satisfied = validate_constraints(final_weights)
    try:
        final_dr = calculate_diversification_ratio(
            final_weights, sigma, covariance
        )
    except Stage3Error:
        final_dr = float("nan")

    return {
        "initialization_method": initialization_method,
        "success": bool(result.success),
        "iterations": int(getattr(result, "nit", 0)),
        "final_DR": final_dr,
        "constraint_satisfied": constraint_satisfied,
        "weights": final_weights,
    }


def run_multiple_initializations(
    sigma: np.ndarray,
    covariance: np.ndarray,
    tickers: list[str],
) -> tuple[np.ndarray, float, pd.DataFrame]:
    """Run equal-weight and random-start SLSQP, then select the best solution."""
    if NUMBER_OF_RANDOM_INITIALIZATIONS < 0:
        raise Stage3Error("NUMBER_OF_RANDOM_INITIALIZATIONS cannot be negative.")

    number_of_assets = len(sigma)
    if number_of_assets != len(tickers):
        raise Stage3Error("Ticker count does not match the risk-input dimensions.")

    initializations: list[tuple[str, np.ndarray]] = [
        (
            "equal_weight",
            np.full(number_of_assets, 1.0 / number_of_assets, dtype=float),
        )
    ]

    random_generator = np.random.default_rng(RANDOM_SEED)
    for position in range(1, NUMBER_OF_RANDOM_INITIALIZATIONS + 1):
        random_weights = random_generator.dirichlet(np.ones(number_of_assets))
        initializations.append((f"random_{position:02d}", random_weights))

    runs: list[dict[str, object]] = []
    for initialization_method, initial_weights in initializations:
        run = run_slsqp_optimization(
            initial_weights,
            sigma,
            covariance,
            initialization_method,
        )
        runs.append(run)
        logging.info(
            "%s: success=%s, feasible=%s, iterations=%d, DR=%.10f",
            initialization_method,
            run["success"],
            run["constraint_satisfied"],
            run["iterations"],
            run["final_DR"],
        )

    eligible_runs = [
        run
        for run in runs
        if run["success"]
        and run["constraint_satisfied"]
        and math.isfinite(float(run["final_DR"]))
    ]
    if not eligible_runs:
        raise Stage3Error(
            "No SLSQP run produced a successful feasible finite solution."
        )

    best_run = max(eligible_runs, key=lambda run: float(run["final_DR"]))
    best_weights = np.asarray(best_run["weights"], dtype=float)
    best_dr = float(best_run["final_DR"])

    detail_rows: list[dict[str, object]] = []
    for run in runs:
        row: dict[str, object] = {
            "initialization_method": run["initialization_method"],
            "success": run["success"],
            "iterations": run["iterations"],
            "final_DR": run["final_DR"],
            "constraint_satisfied": run["constraint_satisfied"],
        }
        run_weights = np.asarray(run["weights"], dtype=float)
        for ticker, weight in zip(tickers, run_weights, strict=True):
            row[f"weight_{ticker}"] = weight
        detail_rows.append(row)

    details = pd.DataFrame(detail_rows)
    return best_weights, best_dr, details


def _write_csv_atomically(
    frame: pd.DataFrame,
    output_file: Path,
) -> None:
    """Write a CSV through a temporary file to avoid partial output."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.10f")
    temporary.replace(output_file)


def save_results(
    volatility: pd.DataFrame,
    optimal_weights: np.ndarray,
    analytical_weights: np.ndarray,
    optimization_details: pd.DataFrame,
    constrained_dr: float,
    analytical_dr: float,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Validate and save all four Stage 3 result files."""
    tickers = volatility["ticker"].tolist()
    sectors = volatility["sector"].tolist()

    if optimal_weights.shape != (len(tickers),):
        raise Stage3Error("Optimal weight dimensions do not match the ETF universe.")
    if analytical_weights.shape != (len(tickers),):
        raise Stage3Error(
            "Analytical weight dimensions do not match the ETF universe."
        )
    if not validate_constraints(optimal_weights):
        raise Stage3Error("Final constrained weights violate portfolio constraints.")
    if not np.isclose(
        analytical_weights.sum(), 1.0, rtol=0.0, atol=WEIGHT_TOLERANCE
    ):
        raise Stage3Error("Analytical benchmark weights do not sum to one.")
    if not math.isfinite(constrained_dr) or not math.isfinite(analytical_dr):
        raise Stage3Error("Final diversification-ratio comparison is not finite.")
    if constrained_dr > analytical_dr + DR_COMPARISON_TOLERANCE:
        raise Stage3Error(
            "Constrained DR exceeds the analytical unconstrained benchmark "
            "beyond numerical tolerance."
        )

    optimal_output = pd.DataFrame(
        {
            "ticker": tickers,
            "sector": sectors,
            "weight": optimal_weights,
        }
    )
    analytical_output = pd.DataFrame(
        {
            "ticker": tickers,
            "sector": sectors,
            "weight": analytical_weights,
        }
    )
    summary = pd.DataFrame(
        [
            {
                "final_diversification_ratio": constrained_dr,
                "analytical_diversification_ratio": analytical_dr,
                "number_of_active_etfs": int(
                    np.count_nonzero(optimal_weights > ACTIVE_WEIGHT_TOLERANCE)
                ),
                "maximum_weight": float(np.max(optimal_weights)),
                "minimum_weight": float(np.min(optimal_weights)),
            }
        ]
    )

    _write_csv_atomically(
        optimal_output, output_dir / "optimal_mdp_weights.csv"
    )
    _write_csv_atomically(
        analytical_output, output_dir / "analytical_solution_weights.csv"
    )
    _write_csv_atomically(
        optimization_details, output_dir / "optimization_details.csv"
    )
    _write_csv_atomically(summary, output_dir / "mdp_summary.csv")


def main() -> None:
    """Run the analytical benchmark and multi-start long-only MDP workflow."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    volatility, sigma, covariance = load_inputs()
    tickers = volatility["ticker"].tolist()

    analytical_weights, analytical_dr = calculate_analytical_solution(
        sigma, covariance
    )
    optimal_weights, constrained_dr, optimization_details = (
        run_multiple_initializations(sigma, covariance, tickers)
    )
    save_results(
        volatility,
        optimal_weights,
        analytical_weights,
        optimization_details,
        constrained_dr,
        analytical_dr,
    )

    logging.info(
        "Stage 3 complete: selected the best feasible solution from %d SLSQP runs.",
        len(optimization_details),
    )
    logging.info("No backtest or unrelated portfolio objective was calculated.")


if __name__ == "__main__":
    try:
        main()
    except Stage3Error as exc:
        logging.error("%s", exc)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        logging.error("Interrupted by user. No partial result is considered final.")
        raise SystemExit(130)
