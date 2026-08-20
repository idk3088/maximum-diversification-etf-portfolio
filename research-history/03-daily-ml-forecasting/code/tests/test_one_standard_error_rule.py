"""Deterministic tests for minimum-MSE and one-standard-error selection."""

from __future__ import annotations

import math

import pandas as pd

from src.model_selection import apply_one_standard_error_rule


COMPLEXITY = ["OLS", "Ridge", "Lasso", "RandomForest", "XGBoost"]


def metric_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": COMPLEXITY,
            "mean_validation_MSE": [1.08, 1.06, 1.04, 1.02, 1.00],
            "standard_error_MSE": [0.02, 0.02, 0.03, 0.05, 0.10],
            "outer_fold_count": [3] * 5,
            "model_status": ["ok"] * 5,
            # Auxiliary metrics deliberately favor other models.
            "MAE": [0.1, 0.2, 0.3, 0.4, 0.5],
            "OOS_R2": [0.9, 0.8, 0.7, 0.6, 0.5],
            "directional_accuracy": [1.0, 0.9, 0.8, 0.7, 0.6],
            "prediction_correlation": [0.9, 0.8, 0.7, 0.6, 0.5],
        }
    )


def test_minimum_mse_is_the_only_primary_selection_metric() -> None:
    metrics, selection = apply_one_standard_error_rule(
        metric_frame(), complexity_order=COMPLEXITY, enabled=False,
        insufficient_folds_fallback="minimum_mse",
    )
    assert selection["minimum_MSE_model"] == "XGBoost"
    assert metrics.loc[metrics["selected_by_minimum_MSE"], "model"].item() == "XGBoost"


def test_one_se_threshold_and_simplest_candidate_are_correct() -> None:
    metrics, selection = apply_one_standard_error_rule(
        metric_frame(), complexity_order=COMPLEXITY, enabled=True,
        insufficient_folds_fallback="minimum_mse",
    )
    assert selection["one_SE_threshold"] == 1.10
    assert metrics.loc[metrics["within_one_SE"], "model"].tolist() == COMPLEXITY
    assert selection["final_selected_model"] == "OLS"


def test_complexity_order_is_fixed_and_not_result_dependent() -> None:
    metrics, _ = apply_one_standard_error_rule(
        metric_frame(), complexity_order=COMPLEXITY, enabled=True,
        insufficient_folds_fallback="minimum_mse",
    )
    assert metrics.sort_values("complexity_rank")["model"].tolist() == COMPLEXITY


def test_missing_standard_error_falls_back_to_minimum_mse() -> None:
    frame = metric_frame()
    frame.loc[frame["model"].eq("XGBoost"), "standard_error_MSE"] = math.nan
    _, selection = apply_one_standard_error_rule(
        frame, complexity_order=COMPLEXITY, enabled=True,
        insufficient_folds_fallback="minimum_mse",
    )
    assert selection["final_selected_model"] == "XGBoost"
    assert "fell back" in selection["selection_notes"]
