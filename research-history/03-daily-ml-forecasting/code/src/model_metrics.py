"""Out-of-sample regression metrics for Step 4 model comparison."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def directional_accuracy(actual: pd.Series, predicted: pd.Series) -> float:
    """Exact sign match; a zero is correct only when both values are zero."""
    if len(actual) == 0:
        return math.nan
    return float(np.mean(np.sign(actual.to_numpy()) == np.sign(predicted.to_numpy())))


def prediction_correlation(actual: pd.Series, predicted: pd.Series) -> tuple[float, str]:
    """Pearson correlation, preserving undefined constant-series cases as NaN."""
    if len(actual) < 2:
        return math.nan, "fewer_than_two_predictions"
    if actual.nunique(dropna=True) <= 1:
        return math.nan, "actual_return_is_constant"
    if predicted.nunique(dropna=True) <= 1:
        return math.nan, "predicted_return_is_constant"
    return float(actual.corr(predicted)), ""


def calculate_model_metrics(predictions: pd.DataFrame) -> dict[str, object]:
    """Aggregate outer-fold predictions without using any training metric."""
    successful = predictions.loc[
        predictions["prediction_status"].eq("success")
    ].copy()
    if successful.empty:
        return {
            "outer_fold_count": 0,
            "valid_prediction_count": 0,
            "mean_validation_MSE": math.nan,
            "fold_MSE_std": math.nan,
            "standard_error_MSE": math.nan,
            "MAE": math.nan,
            "OOS_R2": math.nan,
            "directional_accuracy": math.nan,
            "prediction_correlation": math.nan,
            "metric_notes": "no_successful_outer_predictions",
        }
    actual = pd.to_numeric(successful["actual_return"], errors="coerce")
    predicted = pd.to_numeric(successful["predicted_return"], errors="coerce")
    benchmark = pd.to_numeric(successful["benchmark_prediction"], errors="coerce")
    fold_mse = successful.groupby("outer_fold_id", sort=True).apply(
        lambda group: float(
            np.mean(
                (
                    pd.to_numeric(group["actual_return"], errors="coerce")
                    - pd.to_numeric(group["predicted_return"], errors="coerce")
                )
                ** 2
            )
        ),
        include_groups=False,
    )
    fold_count = int(len(fold_mse))
    fold_std = float(fold_mse.std(ddof=1)) if fold_count >= 2 else math.nan
    standard_error = fold_std / math.sqrt(fold_count) if fold_count >= 2 else math.nan
    model_sse = float(np.sum((actual - predicted) ** 2))
    benchmark_sse = float(np.sum((actual - benchmark) ** 2))
    oos_r2 = 1.0 - model_sse / benchmark_sse if benchmark_sse > 0 else math.nan
    correlation, correlation_note = prediction_correlation(actual, predicted)
    return {
        "outer_fold_count": fold_count,
        "valid_prediction_count": int(len(successful)),
        "mean_validation_MSE": float(fold_mse.mean()),
        "fold_MSE_std": fold_std,
        "standard_error_MSE": standard_error,
        "MAE": float(np.mean(np.abs(actual - predicted))),
        "OOS_R2": oos_r2,
        "directional_accuracy": directional_accuracy(actual, predicted),
        "prediction_correlation": correlation,
        "metric_notes": correlation_note,
    }
