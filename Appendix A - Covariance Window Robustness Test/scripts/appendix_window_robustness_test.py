"""Test MDP robustness across covariance estimation windows.

This appendix is separate from the ETF Portfolio 4.0 main pipeline. It uses the
existing Stage 1 ETF universe, Stage 2 daily returns, and Stage 4 SPY return
series. For each requested trailing estimation window, it estimates annualized
volatility and covariance, applies the same multi-start long-only SLSQP method
and numerical settings as Stage 3, and evaluates the resulting fixed weights
over one identical common return period.

The script does not select a preferred window, replace Stage 3 weights, modify
the main model, download data, or interpret the results.
"""

from __future__ import annotations

import math
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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
APPENDIX_DIR = SCRIPT_DIR.parent
PROJECT_DIR = APPENDIX_DIR.parent

SELECTED_ETFS_FILE = (
    PROJECT_DIR
    / "Stage 1 - ETF Selection"
    / "data"
    / "selected_sector_etfs.csv"
)
DAILY_RETURNS_FILE = (
    PROJECT_DIR
    / "Stage 2 - Return and Risk Data Preparation"
    / "outputs"
    / "daily_returns.csv"
)
STAGE4_RETURN_SERIES_FILE = (
    PROJECT_DIR
    / "Stage 4 - Portfolio Evaluation"
    / "outputs"
    / "portfolio_return_series.csv"
)
OUTPUT_DIR = APPENDIX_DIR / "outputs"


# ---------------------------------------------------------------------------
# Fixed research design and Stage 3 optimization settings
# ---------------------------------------------------------------------------

WINDOW_SPECIFICATIONS: tuple[tuple[str, int | None], ...] = (
    ("3 months (63 trading days)", 63),
    ("6 months (126 trading days)", 126),
    ("1 year (252 trading days)", 252),
    ("2 years (504 trading days)", 504),
    ("3 years (756 trading days)", 756),
    ("Full available history", None),
)

# These settings match Stage 3 - Maximum Diversification Portfolio
# Construction exactly.
NUMBER_OF_RANDOM_INITIALIZATIONS = 25
RANDOM_SEED = 42
SLSQP_MAX_ITERATIONS = 2_000
SLSQP_FTOL = 1e-12

TRADING_DAYS_PER_YEAR = 252
INITIAL_PORTFOLIO_VALUE = 100.0
WEIGHT_TOLERANCE = 1e-8
SYMMETRY_RTOL = 1e-10
SYMMETRY_ATOL = 1e-12

SPY_RETURN_COLUMN = "SPY daily return"

WINDOW_PLOT_LABELS = (
    "3M\n63d",
    "6M\n126d",
    "1Y\n252d",
    "2Y\n504d",
    "3Y\n756d",
    "Full\nhistory",
)


class AppendixError(RuntimeError):
    """Raised when appendix input, optimization, or validation fails."""


def _read_dated_returns(input_file: Path, description: str) -> pd.DataFrame:
    """Read a CSV return matrix with a unique, sorted Date index."""
    if not input_file.exists():
        raise AppendixError(f"{description} not found: {input_file}")
    try:
        frame = pd.read_csv(input_file)
    except Exception as exc:
        raise AppendixError(f"Could not read {description}: {exc}") from exc

    if "Date" not in frame.columns:
        raise AppendixError(f"{description} must contain a Date column.")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    if frame["Date"].isna().any():
        raise AppendixError(f"{description} contains invalid dates.")
    if frame["Date"].duplicated().any():
        raise AppendixError(f"{description} contains duplicate dates.")

    frame = frame.set_index("Date").sort_index(kind="mergesort")
    frame.index.name = "Date"
    return frame


def load_returns(
    selected_etfs_file: Path = SELECTED_ETFS_FILE,
    daily_returns_file: Path = DAILY_RETURNS_FILE,
    stage4_return_file: Path = STAGE4_RETURN_SERIES_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the fixed ETF universe, Stage 2 returns, and existing SPY returns."""
    if not selected_etfs_file.exists():
        raise AppendixError(f"Stage 1 ETF selection not found: {selected_etfs_file}")
    try:
        selected = pd.read_csv(selected_etfs_file)
    except Exception as exc:
        raise AppendixError(f"Could not read Stage 1 ETF selection: {exc}") from exc

    required_selection_columns = {"ticker", "gics_sector"}
    missing_selection_columns = sorted(
        required_selection_columns - set(selected.columns)
    )
    if missing_selection_columns:
        raise AppendixError(
            "Stage 1 selection is missing columns: "
            + ", ".join(missing_selection_columns)
        )

    metadata = selected.loc[:, ["ticker", "gics_sector"]].copy()
    metadata["ticker"] = (
        metadata["ticker"].fillna("").astype(str).str.strip().str.upper()
    )
    metadata["gics_sector"] = (
        metadata["gics_sector"].fillna("").astype(str).str.strip()
    )
    metadata = metadata.rename(columns={"gics_sector": "sector"})
    if metadata.empty or metadata[["ticker", "sector"]].eq("").any().any():
        raise AppendixError("Stage 1 selection contains invalid ticker metadata.")
    if metadata["ticker"].duplicated().any():
        raise AppendixError("Stage 1 selection contains duplicate tickers.")

    tickers = metadata["ticker"].tolist()
    full_returns = _read_dated_returns(
        daily_returns_file, "Stage 2 daily ETF returns"
    )
    full_returns.columns = [
        str(column).strip().upper() for column in full_returns.columns
    ]
    if full_returns.columns.duplicated().any():
        raise AppendixError("Stage 2 daily returns contain duplicate tickers.")
    if set(full_returns.columns) != set(tickers):
        raise AppendixError(
            "Stage 2 return universe differs from the Stage 1 ETF selection."
        )
    full_returns = full_returns.loc[:, tickers]
    full_returns = full_returns.apply(pd.to_numeric, errors="coerce")
    if full_returns.empty or full_returns.isna().any().any():
        raise AppendixError("Stage 2 daily ETF returns are empty or missing values.")
    if not np.isfinite(full_returns.to_numpy(dtype=float)).all():
        raise AppendixError("Stage 2 daily ETF returns contain non-finite values.")

    stage4_returns = _read_dated_returns(
        stage4_return_file, "Stage 4 portfolio return series"
    )
    if SPY_RETURN_COLUMN not in stage4_returns.columns:
        raise AppendixError(
            f"Stage 4 return series is missing '{SPY_RETURN_COLUMN}'."
        )
    spy_returns = pd.to_numeric(
        stage4_returns[SPY_RETURN_COLUMN], errors="coerce"
    ).rename(SPY_RETURN_COLUMN)
    if spy_returns.empty or spy_returns.isna().any():
        raise AppendixError("Stage 4 SPY return series is empty or missing values.")
    if not np.isfinite(spy_returns.to_numpy(dtype=float)).all():
        raise AppendixError("Stage 4 SPY returns contain non-finite values.")

    evaluation_data = full_returns.join(spy_returns, how="inner").dropna(how="any")
    if evaluation_data.empty:
        raise AppendixError("No common ETF and SPY evaluation dates remain.")
    if evaluation_data.isna().any().any():
        raise AppendixError("Missing values remain in the common evaluation data.")
    evaluation_returns = evaluation_data.loc[:, tickers]
    evaluation_spy_returns = evaluation_data[SPY_RETURN_COLUMN].copy()
    if not evaluation_returns.index.equals(evaluation_spy_returns.index):
        raise AppendixError("ETF and SPY evaluation dates are not aligned.")

    return metadata, full_returns, evaluation_returns, evaluation_spy_returns


def create_window_samples(
    full_returns: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Create the five requested trailing samples plus full available history."""
    finite_windows = [
        observations
        for _, observations in WINDOW_SPECIFICATIONS
        if observations is not None
    ]
    maximum_required = max(finite_windows)
    if len(full_returns) < maximum_required:
        raise AppendixError(
            f"Stage 2 has {len(full_returns)} observations; at least "
            f"{maximum_required} are required for the 3-year window."
        )

    samples: dict[str, pd.DataFrame] = {}
    reference_columns = full_returns.columns.tolist()
    for label, observations in WINDOW_SPECIFICATIONS:
        sample = (
            full_returns.copy()
            if observations is None
            else full_returns.tail(observations).copy()
        )
        expected_rows = len(full_returns) if observations is None else observations
        if len(sample) != expected_rows:
            raise AppendixError(f"{label}: sample row count is incorrect.")
        if sample.columns.tolist() != reference_columns:
            raise AppendixError(f"{label}: ETF universe or ordering changed.")
        if sample.isna().any().any():
            raise AppendixError(f"{label}: estimation sample contains missing values.")
        samples[label] = sample
    return samples


def estimate_volatility(window_returns: pd.DataFrame) -> np.ndarray:
    """Estimate annualized sample volatility from one return window."""
    sigma = (
        window_returns.std(axis=0, ddof=1).to_numpy(dtype=float)
        * math.sqrt(TRADING_DAYS_PER_YEAR)
    )
    if not np.isfinite(sigma).all() or (sigma <= 0).any():
        raise AppendixError("Estimated annualized volatility is invalid.")
    return sigma


def estimate_covariance(window_returns: pd.DataFrame) -> np.ndarray:
    """Estimate annualized sample covariance from one return window."""
    covariance = (
        window_returns.cov(ddof=1).to_numpy(dtype=float)
        * TRADING_DAYS_PER_YEAR
    )
    expected_shape = (window_returns.shape[1], window_returns.shape[1])
    if covariance.shape != expected_shape:
        raise AppendixError("Estimated covariance dimensions are incorrect.")
    if not np.isfinite(covariance).all():
        raise AppendixError("Estimated covariance contains non-finite values.")
    if not np.allclose(
        covariance,
        covariance.T,
        rtol=SYMMETRY_RTOL,
        atol=SYMMETRY_ATOL,
    ):
        raise AppendixError("Estimated covariance is not symmetric.")
    if (np.diag(covariance) <= 0).any():
        raise AppendixError("Estimated covariance diagonal must be positive.")
    return covariance


def _calculate_diversification_ratio(
    weights: np.ndarray,
    sigma: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Calculate DR(w) for one weight and risk-estimate combination."""
    weights = np.asarray(weights, dtype=float)
    numerator = float(weights @ sigma)
    variance = float(weights @ covariance @ weights)
    if not math.isfinite(numerator) or not math.isfinite(variance) or variance <= 0:
        raise AppendixError("Diversification-ratio inputs produce invalid values.")
    diversification_ratio = numerator / math.sqrt(variance)
    if not math.isfinite(diversification_ratio):
        raise AppendixError("Diversification ratio is not finite.")
    return float(diversification_ratio)


def _constraints_satisfied(weights: np.ndarray) -> bool:
    """Check finite, fully invested, strictly nonnegative weights."""
    weights = np.asarray(weights, dtype=float)
    return bool(
        weights.ndim == 1
        and weights.size > 0
        and np.isfinite(weights).all()
        and np.all(weights >= 0.0)
        and np.isclose(
            weights.sum(), 1.0, rtol=0.0, atol=WEIGHT_TOLERANCE
        )
    )


def run_mdp_optimization(
    sigma: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Run the same equal-plus-random-start long-only SLSQP method as Stage 3."""
    number_of_assets = len(sigma)
    if covariance.shape != (number_of_assets, number_of_assets):
        raise AppendixError("Risk-input dimensions do not match.")

    def negative_diversification_ratio(weights: np.ndarray) -> float:
        try:
            return -_calculate_diversification_ratio(weights, sigma, covariance)
        except AppendixError:
            return 1e20

    initializations: list[np.ndarray] = [
        np.full(number_of_assets, 1.0 / number_of_assets, dtype=float)
    ]
    random_generator = np.random.default_rng(RANDOM_SEED)
    for _ in range(NUMBER_OF_RANDOM_INITIALIZATIONS):
        initializations.append(
            random_generator.dirichlet(np.ones(number_of_assets))
        )

    eligible_solutions: list[tuple[np.ndarray, float]] = []
    for initial_weights in initializations:
        if not _constraints_satisfied(initial_weights):
            raise AppendixError("An SLSQP initialization is not feasible.")
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
        if not result.success or not _constraints_satisfied(final_weights):
            continue
        final_dr = _calculate_diversification_ratio(
            final_weights, sigma, covariance
        )
        eligible_solutions.append((final_weights, final_dr))

    if not eligible_solutions:
        raise AppendixError(
            "No SLSQP initialization produced a successful feasible solution."
        )

    best_weights, best_dr = max(eligible_solutions, key=lambda item: item[1])
    if not _constraints_satisfied(best_weights):
        raise AppendixError("Selected MDP solution violates portfolio constraints.")
    return best_weights, float(best_dr)


def calculate_performance_metrics(
    weights: np.ndarray,
    evaluation_returns: pd.DataFrame,
    spy_returns: pd.Series,
) -> dict[str, float]:
    """Apply the Stage 4 methodology over the common fixed evaluation period."""
    if not evaluation_returns.index.equals(spy_returns.index):
        raise AppendixError("ETF and SPY evaluation dates are not identical.")
    if weights.shape != (evaluation_returns.shape[1],):
        raise AppendixError("Weight dimensions do not match evaluation returns.")
    if not _constraints_satisfied(weights):
        raise AppendixError("Performance weights violate portfolio constraints.")

    portfolio_returns = pd.Series(
        evaluation_returns.to_numpy(dtype=float) @ weights,
        index=evaluation_returns.index,
        name="portfolio_return",
    )
    if portfolio_returns.isna().any() or not np.isfinite(
        portfolio_returns.to_numpy(dtype=float)
    ).all():
        raise AppendixError("Portfolio evaluation returns are invalid.")
    if (portfolio_returns <= -1.0).any():
        raise AppendixError("A portfolio daily return is at or below -100%.")

    number_of_days = len(portfolio_returns)
    if number_of_days < 2:
        raise AppendixError("Evaluation requires at least two daily observations.")
    growth_factor = float((1.0 + portfolio_returns).prod())
    if not math.isfinite(growth_factor) or growth_factor <= 0:
        raise AppendixError("Portfolio compounded growth factor is invalid.")

    annualized_return = (
        growth_factor ** (TRADING_DAYS_PER_YEAR / number_of_days) - 1.0
    )
    annualized_volatility = float(
        portfolio_returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)
    )
    if not math.isfinite(annualized_volatility) or annualized_volatility <= 0:
        raise AppendixError("Portfolio annualized volatility is invalid.")
    sharpe_ratio = annualized_return / annualized_volatility
    spy_correlation = float(portfolio_returns.corr(spy_returns))

    wealth = INITIAL_PORTFOLIO_VALUE * (1.0 + portfolio_returns).cumprod()
    running_peak = wealth.cummax()
    maximum_drawdown = float(((wealth - running_peak) / running_peak).min())

    values = np.array(
        [
            annualized_return,
            annualized_volatility,
            sharpe_ratio,
            spy_correlation,
            maximum_drawdown,
        ],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise AppendixError("A performance metric is non-finite.")

    return {
        "annualized_return": float(annualized_return),
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "spy_correlation": spy_correlation,
        "maximum_drawdown": maximum_drawdown,
    }


def compare_weights(weights_by_window: pd.DataFrame) -> pd.DataFrame:
    """Calculate every pairwise L1 distance between window portfolios."""
    if weights_by_window.empty:
        raise AppendixError("Window weight matrix is empty.")
    if weights_by_window.isna().any().any():
        raise AppendixError("Window weight matrix contains missing values.")

    values = weights_by_window.to_numpy(dtype=float)
    distances = np.empty((len(values), len(values)), dtype=float)
    for row_position, row_weights in enumerate(values):
        for column_position, column_weights in enumerate(values):
            distances[row_position, column_position] = float(
                np.abs(row_weights - column_weights).sum()
            )

    stability = pd.DataFrame(
        distances,
        index=weights_by_window.index,
        columns=weights_by_window.index,
    )
    stability.index.name = "estimation_window"
    if not np.allclose(stability.to_numpy(), stability.to_numpy().T):
        raise AppendixError("Weight-stability matrix is not symmetric.")
    if not np.allclose(np.diag(stability.to_numpy()), 0.0):
        raise AppendixError("Weight-stability matrix diagonal is not zero.")
    return stability


def generate_plots(
    comparison_summary: pd.DataFrame,
    weights_by_window: pd.DataFrame,
) -> dict[str, plt.Figure]:
    """Create the three requested appendix comparison figures."""
    if len(comparison_summary) != len(WINDOW_SPECIFICATIONS):
        raise AppendixError("Comparison summary does not contain all windows.")
    if len(weights_by_window) != len(WINDOW_SPECIFICATIONS):
        raise AppendixError("Weight matrix does not contain all windows.")

    positions = np.arange(len(WINDOW_SPECIFICATIONS))

    dr_figure, dr_axis = plt.subplots(figsize=(10, 5.5))
    dr_axis.plot(
        positions,
        comparison_summary["diversification_ratio"],
        marker="o",
        color="#1f5a94",
        linewidth=1.6,
    )
    dr_axis.set_xticks(positions, WINDOW_PLOT_LABELS)
    dr_axis.set_xlabel("Covariance estimation window")
    dr_axis.set_ylabel("Diversification ratio")
    dr_axis.set_title("Diversification Ratio by Covariance Estimation Window")
    dr_axis.grid(True, alpha=0.25)
    dr_figure.tight_layout()

    correlation_figure, correlation_axis = plt.subplots(figsize=(10, 5.5))
    correlation_axis.plot(
        positions,
        comparison_summary["spy_correlation"],
        marker="o",
        color="#8c3b2a",
        linewidth=1.6,
    )
    correlation_axis.set_xticks(positions, WINDOW_PLOT_LABELS)
    correlation_axis.set_xlabel("Covariance estimation window")
    correlation_axis.set_ylabel("SPY correlation")
    correlation_axis.set_ylim(-1.0, 1.0)
    correlation_axis.set_title("Portfolio Correlation with SPY by Estimation Window")
    correlation_axis.grid(True, alpha=0.25)
    correlation_figure.tight_layout()

    weights_figure, weights_axis = plt.subplots(figsize=(12, 6.5))
    for ticker in weights_by_window.columns:
        weights_axis.plot(
            positions,
            weights_by_window[ticker],
            marker="o",
            linewidth=1.2,
            label=ticker,
        )
    weights_axis.set_xticks(positions, WINDOW_PLOT_LABELS)
    weights_axis.set_xlabel("Covariance estimation window")
    weights_axis.set_ylabel("Portfolio weight")
    weights_axis.set_title("Portfolio Weight Changes Across Covariance Windows")
    weights_axis.grid(True, alpha=0.25)
    weights_axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    weights_figure.tight_layout()

    return {
        "diversification_ratio_by_window.png": dr_figure,
        "spy_correlation_by_window.png": correlation_figure,
        "weight_changes_across_windows.png": weights_figure,
    }


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
    )
    temporary.replace(output_file)


def save_results(
    portfolio_weights: pd.DataFrame,
    comparison_summary: pd.DataFrame,
    weight_stability: pd.DataFrame,
    figures: dict[str, plt.Figure],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Save all appendix comparison tables and visualizations."""
    expected_weight_columns = ["ticker", "sector", "weight", "estimation_window"]
    expected_summary_columns = [
        "estimation_window",
        "diversification_ratio",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "spy_correlation",
        "maximum_drawdown",
    ]
    if portfolio_weights.columns.tolist() != expected_weight_columns:
        raise AppendixError("Portfolio weight output columns are incorrect.")
    if comparison_summary.columns.tolist() != expected_summary_columns:
        raise AppendixError("Window comparison output columns are incorrect.")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_atomically(
        portfolio_weights,
        output_dir / "window_portfolio_weights.csv",
        index=False,
    )
    _write_csv_atomically(
        comparison_summary,
        output_dir / "window_comparison_summary.csv",
        index=False,
    )
    _write_csv_atomically(
        weight_stability,
        output_dir / "weight_stability_matrix.csv",
        index=True,
        index_label="estimation_window",
    )

    expected_figure_names = {
        "diversification_ratio_by_window.png",
        "spy_correlation_by_window.png",
        "weight_changes_across_windows.png",
    }
    if set(figures) != expected_figure_names:
        raise AppendixError("Appendix figure set is incorrect.")
    for filename, figure in figures.items():
        output_file = output_dir / filename
        temporary = output_file.with_suffix(".png.tmp")
        figure.savefig(temporary, format="png", dpi=160, bbox_inches="tight")
        temporary.replace(output_file)
        plt.close(figure)


def main() -> None:
    """Run the covariance-window robustness appendix."""
    metadata, full_returns, evaluation_returns, spy_returns = load_returns()
    samples = create_window_samples(full_returns)
    tickers = metadata["ticker"].tolist()
    sector_by_ticker = metadata.set_index("ticker")["sector"]

    weight_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    weights_by_window_data: dict[str, np.ndarray] = {}

    for window_label, _ in WINDOW_SPECIFICATIONS:
        sample = samples[window_label]
        sigma = estimate_volatility(sample)
        covariance = estimate_covariance(sample)
        weights, diversification_ratio = run_mdp_optimization(sigma, covariance)
        if not _constraints_satisfied(weights):
            raise AppendixError(f"{window_label}: final constraints are not satisfied.")

        metrics = calculate_performance_metrics(
            weights, evaluation_returns, spy_returns
        )
        weights_by_window_data[window_label] = weights

        for ticker, weight in zip(tickers, weights, strict=True):
            weight_rows.append(
                {
                    "ticker": ticker,
                    "sector": sector_by_ticker.loc[ticker],
                    "weight": weight,
                    "estimation_window": window_label,
                }
            )
        summary_rows.append(
            {
                "estimation_window": window_label,
                "diversification_ratio": diversification_ratio,
                **metrics,
            }
        )

    portfolio_weights = pd.DataFrame(weight_rows)
    comparison_summary = pd.DataFrame(summary_rows)
    weights_by_window = pd.DataFrame.from_dict(
        weights_by_window_data,
        orient="index",
        columns=tickers,
    )
    weights_by_window.index.name = "estimation_window"

    if list(weights_by_window.index) != [
        label for label, _ in WINDOW_SPECIFICATIONS
    ]:
        raise AppendixError("Window ordering changed before comparison.")
    if weights_by_window.columns.tolist() != tickers:
        raise AppendixError("ETF universe changed before weight comparison.")

    weight_stability = compare_weights(weights_by_window)
    figures = generate_plots(comparison_summary, weights_by_window)
    save_results(
        portfolio_weights,
        comparison_summary,
        weight_stability,
        figures,
    )

    print(
        "Appendix robustness test complete for "
        f"{len(WINDOW_SPECIFICATIONS)} covariance estimation windows."
    )
    print("The main Stage 3 portfolio was not modified or replaced.")


if __name__ == "__main__":
    try:
        main()
    except AppendixError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    except KeyboardInterrupt:
        raise SystemExit("Interrupted by user.")
