"""Calculate 252-day rolling correlation between the fixed MDP and SPY.

This additional Stage 4 module analyzes the time-varying daily-return
relationship between the completed Maximum Diversification Portfolio and SPY.
It reads the existing Stage 4 portfolio return series and verifies the MDP
series against Stage 2 ETF daily returns and the unchanged Stage 3 weights.

No optimization, ETF selection, benchmark change, or new backtest is performed.
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
except ImportError as exc:
    raise SystemExit(
        "Missing dependencies. Install them with: "
        "python -m pip install -r requirements.txt"
    ) from exc


# ---------------------------------------------------------------------------
# Paths and fixed methodology settings
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE4_DIR = SCRIPT_DIR.parent
PROJECT_DIR = STAGE4_DIR.parent

STAGE2_DAILY_RETURNS_FILE = (
    PROJECT_DIR
    / "Stage 2 - Return and Risk Data Preparation"
    / "outputs"
    / "daily_returns.csv"
)
STAGE3_WEIGHTS_FILE = (
    PROJECT_DIR
    / "Stage 3 - Maximum Diversification Portfolio Construction"
    / "outputs"
    / "optimal_mdp_weights.csv"
)
STAGE4_RETURN_SERIES_FILE = STAGE4_DIR / "outputs" / "portfolio_return_series.csv"
OUTPUT_DIR = STAGE4_DIR / "outputs"

ROLLING_WINDOW = 252
WEIGHT_TOLERANCE = 1e-8
RETURN_VALIDATION_RTOL = 1e-8
RETURN_VALIDATION_ATOL = 1e-9
CORRELATION_BOUND_TOLERANCE = 1e-12

MDP_RETURN_COLUMN = "MDP daily return"
SPY_RETURN_COLUMN = "SPY daily return"
ROLLING_CORRELATION_COLUMN = "MDP_SPY_Rolling_Correlation"


class RollingCorrelationError(RuntimeError):
    """Raised when rolling-correlation input or validation fails."""


def _read_dated_csv(input_file: Path, description: str) -> pd.DataFrame:
    """Read a CSV with a unique, valid, sorted Date index."""
    if not input_file.exists():
        raise RollingCorrelationError(f"{description} not found: {input_file}")

    try:
        frame = pd.read_csv(input_file)
    except Exception as exc:
        raise RollingCorrelationError(f"Could not read {description}: {exc}") from exc

    if "Date" not in frame.columns:
        raise RollingCorrelationError(f"{description} must contain a Date column.")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    if frame["Date"].isna().any():
        raise RollingCorrelationError(f"{description} contains invalid dates.")
    if frame["Date"].duplicated().any():
        raise RollingCorrelationError(f"{description} contains duplicate dates.")

    frame = frame.set_index("Date").sort_index(kind="mergesort")
    frame.index.name = "Date"
    return frame


def load_returns(
    stage4_return_file: Path = STAGE4_RETURN_SERIES_FILE,
    stage2_return_file: Path = STAGE2_DAILY_RETURNS_FILE,
    stage3_weights_file: Path = STAGE3_WEIGHTS_FILE,
) -> tuple[pd.Series, pd.Series]:
    """Load MDP/SPY returns and verify MDP against fixed Stage 2/3 inputs."""
    stage4_returns = _read_dated_csv(
        stage4_return_file, "Stage 4 portfolio return series"
    )
    missing_stage4_columns = sorted(
        {MDP_RETURN_COLUMN, SPY_RETURN_COLUMN} - set(stage4_returns.columns)
    )
    if missing_stage4_columns:
        raise RollingCorrelationError(
            "Stage 4 portfolio return series is missing columns: "
            + ", ".join(missing_stage4_columns)
        )

    selected_returns = stage4_returns.loc[
        :, [MDP_RETURN_COLUMN, SPY_RETURN_COLUMN]
    ].apply(pd.to_numeric, errors="coerce")
    if selected_returns.empty:
        raise RollingCorrelationError("Stage 4 MDP/SPY return series is empty.")
    if selected_returns.isna().any().any():
        raise RollingCorrelationError(
            "Stage 4 MDP or SPY return series contains missing values."
        )
    if not np.isfinite(selected_returns.to_numpy(dtype=float)).all():
        raise RollingCorrelationError(
            "Stage 4 MDP or SPY return series contains non-finite values."
        )

    if not stage3_weights_file.exists():
        raise RollingCorrelationError(
            f"Stage 3 MDP weights not found: {stage3_weights_file}"
        )
    try:
        weights = pd.read_csv(stage3_weights_file)
    except Exception as exc:
        raise RollingCorrelationError(
            f"Could not read Stage 3 MDP weights: {exc}"
        ) from exc

    required_weight_columns = {"ticker", "weight"}
    missing_weight_columns = sorted(
        required_weight_columns - set(weights.columns)
    )
    if missing_weight_columns:
        raise RollingCorrelationError(
            "Stage 3 weights are missing columns: "
            + ", ".join(missing_weight_columns)
        )
    weights = weights.loc[:, ["ticker", "weight"]].copy()
    weights["ticker"] = (
        weights["ticker"].fillna("").astype(str).str.strip().str.upper()
    )
    weights["weight"] = pd.to_numeric(weights["weight"], errors="coerce")
    if weights.empty or weights["ticker"].eq("").any():
        raise RollingCorrelationError("Stage 3 weights contain no valid tickers.")
    if weights["ticker"].duplicated().any():
        raise RollingCorrelationError("Stage 3 weights contain duplicate tickers.")
    if not np.isfinite(weights["weight"].to_numpy(dtype=float)).all():
        raise RollingCorrelationError("Stage 3 weights contain invalid values.")
    if (weights["weight"] < 0).any():
        raise RollingCorrelationError("Stage 3 weights contain negative values.")
    if not np.isclose(
        weights["weight"].sum(),
        1.0,
        rtol=0.0,
        atol=WEIGHT_TOLERANCE,
    ):
        raise RollingCorrelationError(
            "Stage 3 MDP weights do not sum approximately to one."
        )

    stage2_returns = _read_dated_csv(
        stage2_return_file, "Stage 2 daily ETF returns"
    )
    stage2_returns.columns = [
        str(column).strip().upper() for column in stage2_returns.columns
    ]
    stage2_returns = stage2_returns.apply(pd.to_numeric, errors="coerce")
    tickers = weights["ticker"].tolist()
    if set(stage2_returns.columns) != set(tickers):
        raise RollingCorrelationError(
            "Stage 2 return tickers differ from the fixed Stage 3 weights."
        )
    stage2_returns = stage2_returns.loc[:, tickers]
    if stage2_returns.isna().any().any() or not np.isfinite(
        stage2_returns.to_numpy(dtype=float)
    ).all():
        raise RollingCorrelationError("Stage 2 ETF returns contain invalid values.")

    validation_returns = stage2_returns.reindex(selected_returns.index)
    if validation_returns.isna().any().any():
        raise RollingCorrelationError(
            "Stage 2 ETF returns do not cover every Stage 4 evaluation date."
        )

    fixed_weights = weights["weight"].to_numpy(dtype=float)
    expected_mdp_returns = validation_returns.to_numpy(dtype=float) @ fixed_weights
    if not np.allclose(
        selected_returns[MDP_RETURN_COLUMN].to_numpy(dtype=float),
        expected_mdp_returns,
        rtol=RETURN_VALIDATION_RTOL,
        atol=RETURN_VALIDATION_ATOL,
    ):
        raise RollingCorrelationError(
            "Stage 4 MDP returns do not match daily Stage 2 returns weighted by "
            "the fixed Stage 3 portfolio."
        )

    mdp_returns = selected_returns[MDP_RETURN_COLUMN].copy()
    spy_returns = selected_returns[SPY_RETURN_COLUMN].copy()
    return mdp_returns, spy_returns


def align_dates(
    mdp_returns: pd.Series,
    spy_returns: pd.Series,
) -> pd.DataFrame:
    """Align MDP and SPY daily returns to identical dates and remove missing rows."""
    aligned = pd.concat(
        [
            mdp_returns.rename(MDP_RETURN_COLUMN),
            spy_returns.rename(SPY_RETURN_COLUMN),
        ],
        axis=1,
        join="inner",
    )
    aligned = aligned.dropna(how="any").sort_index(kind="mergesort")
    aligned = aligned.loc[~aligned.index.duplicated(keep="last")]
    aligned.index.name = "Date"

    if aligned.empty:
        raise RollingCorrelationError("No aligned MDP and SPY dates remain.")
    if aligned.isna().any().any():
        raise RollingCorrelationError("Missing values remain after date alignment.")
    if not np.isfinite(aligned.to_numpy(dtype=float)).all():
        raise RollingCorrelationError(
            "Aligned MDP/SPY returns contain non-finite values."
        )
    if not aligned[MDP_RETURN_COLUMN].index.equals(
        aligned[SPY_RETURN_COLUMN].index
    ):
        raise RollingCorrelationError("MDP and SPY dates are not aligned.")
    return aligned


def calculate_rolling_correlation(
    aligned_returns: pd.DataFrame,
    window: int = ROLLING_WINDOW,
) -> pd.Series:
    """Calculate the required 252-trading-day rolling MDP/SPY correlation."""
    if window != 252:
        raise RollingCorrelationError(
            f"Rolling window must be exactly 252 trading days, not {window}."
        )
    if len(aligned_returns) < window:
        raise RollingCorrelationError(
            f"At least {window} aligned daily observations are required."
        )

    rolling_correlation = (
        aligned_returns[MDP_RETURN_COLUMN]
        .rolling(window=window, min_periods=window)
        .corr(aligned_returns[SPY_RETURN_COLUMN])
        .dropna()
    )
    rolling_correlation.name = ROLLING_CORRELATION_COLUMN
    rolling_correlation.index.name = "Date"

    expected_observations = len(aligned_returns) - window + 1
    if len(rolling_correlation) != expected_observations:
        raise RollingCorrelationError(
            "Rolling-correlation observation count is inconsistent with a "
            "252-day window."
        )
    values = rolling_correlation.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RollingCorrelationError(
            "Rolling correlation contains non-finite values."
        )
    if (
        (values < -1.0 - CORRELATION_BOUND_TOLERANCE).any()
        or (values > 1.0 + CORRELATION_BOUND_TOLERANCE).any()
    ):
        raise RollingCorrelationError(
            "Rolling correlation contains values outside [-1, 1]."
        )

    # Remove only floating-point spillover beyond the mathematical bounds.
    rolling_correlation = rolling_correlation.clip(lower=-1.0, upper=1.0)
    return rolling_correlation


def calculate_summary_statistics(
    rolling_correlation: pd.Series,
) -> pd.DataFrame:
    """Calculate mean, maximum, minimum, and sample standard deviation."""
    if len(rolling_correlation) < 2:
        raise RollingCorrelationError(
            "At least two rolling correlations are required for summary statistics."
        )

    summary = pd.DataFrame(
        {
            "metric": [
                "average_rolling_correlation",
                "maximum_rolling_correlation",
                "minimum_rolling_correlation",
                "standard_deviation_rolling_correlation",
            ],
            "value": [
                rolling_correlation.mean(),
                rolling_correlation.max(),
                rolling_correlation.min(),
                rolling_correlation.std(ddof=1),
            ],
        }
    )
    if not np.isfinite(summary["value"].to_numpy(dtype=float)).all():
        raise RollingCorrelationError(
            "Rolling-correlation summary contains non-finite values."
        )
    return summary


def plot_rolling_correlation(
    rolling_correlation: pd.Series,
) -> plt.Figure:
    """Create the required dated rolling-correlation line plot."""
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(
        rolling_correlation.index,
        rolling_correlation.to_numpy(dtype=float),
        color="#1f5a94",
        linewidth=1.4,
    )
    axis.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--")
    axis.set_title("252-Day Rolling Correlation: MDP Portfolio vs SPY")
    axis.set_xlabel("Date")
    axis.set_ylabel("Rolling correlation")
    axis.set_ylim(-1.0, 1.0)
    axis.grid(True, alpha=0.25)
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure


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


def save_outputs(
    rolling_correlation: pd.Series,
    summary_statistics: pd.DataFrame,
    figure: plt.Figure,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Save the rolling series, summary statistics, and PNG visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_atomically(
        rolling_correlation.to_frame(),
        output_dir / "rolling_correlation_mdp_spy.csv",
        index=True,
        index_label="Date",
    )
    _write_csv_atomically(
        summary_statistics,
        output_dir / "rolling_correlation_summary.csv",
        index=False,
    )

    plot_file = output_dir / "rolling_correlation_mdp_spy.png"
    temporary_plot = plot_file.with_suffix(".png.tmp")
    figure.savefig(temporary_plot, format="png", dpi=160, bbox_inches="tight")
    temporary_plot.replace(plot_file)
    plt.close(figure)


def main() -> None:
    """Run the additional fixed-weight rolling-correlation analysis."""
    mdp_returns, spy_returns = load_returns()
    aligned_returns = align_dates(mdp_returns, spy_returns)
    rolling_correlation = calculate_rolling_correlation(
        aligned_returns, window=ROLLING_WINDOW
    )
    summary_statistics = calculate_summary_statistics(rolling_correlation)
    figure = plot_rolling_correlation(rolling_correlation)
    save_outputs(rolling_correlation, summary_statistics, figure)

    print(
        "Rolling-correlation analysis complete. "
        f"Saved {len(rolling_correlation)} valid 252-day observations."
    )
    print("Stage 3 weights were validated and were not modified or re-optimized.")


if __name__ == "__main__":
    try:
        main()
    except RollingCorrelationError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    except KeyboardInterrupt:
        raise SystemExit("Interrupted by user.")
