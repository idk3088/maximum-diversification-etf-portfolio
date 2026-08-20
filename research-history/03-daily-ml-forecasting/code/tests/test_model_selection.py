"""Small synthetic tests for Step 4 nested model comparison."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.model_factory import build_model_pipeline
from src.model_metrics import calculate_model_metrics, prediction_correlation
from src.model_selection import (
    detect_feature_columns,
    evaluate_ticker_selection_date,
)
from src.purged_time_series_split import available_rows_as_of
from tests.test_purged_time_series_split import small_config, synthetic_model_dataset


def logger() -> logging.Logger:
    result = logging.getLogger("step4-test")
    result.addHandler(logging.NullHandler())
    return result


def test_target_dates_and_prices_are_excluded_from_features() -> None:
    dataset = synthetic_model_dataset(rows=300)
    assert detect_feature_columns(dataset) == ["feature_one", "feature_two"]


def test_five_models_share_identical_outer_validation_dates() -> None:
    dataset = synthetic_model_dataset(rows=500)
    selection_date = pd.Timestamp(dataset["feature_date"].max())
    result = evaluate_ticker_selection_date(
        dataset,
        ticker="AAA",
        selection_date=selection_date,
        feature_columns=detect_feature_columns(dataset),
        config=small_config(),
        debug=True,
        logger=logger(),
    )
    assert set(result.model_metrics["model"]) == {
        "OLS", "Ridge", "Lasso", "RandomForest", "XGBoost"
    }
    grouped = result.fold_predictions.groupby(["model", "outer_fold_id"])[
        "feature_date"
    ].apply(tuple)
    for fold_id in sorted(result.fold_predictions["outer_fold_id"].unique()):
        dates = [grouped.loc[(model, fold_id)] for model in result.model_metrics["model"]]
        assert all(candidate == dates[0] for candidate in dates[1:])


def test_ols_generates_a_fixed_parameter_record() -> None:
    dataset = synthetic_model_dataset(rows=500)
    result = evaluate_ticker_selection_date(
        dataset,
        ticker="AAA",
        selection_date=pd.Timestamp(dataset["feature_date"].max()),
        feature_columns=detect_feature_columns(dataset),
        config=small_config(),
        debug=True,
        logger=logger(),
    )
    ols = result.hyperparameter_search[result.hyperparameter_search["model"].eq("OLS")]
    assert not ols.empty
    assert ols["hyperparameters_json"].eq("{}").all()
    assert ols.groupby("outer_fold_id")["selected_for_outer_refit"].sum().eq(1).all()


def test_oos_r2_uses_fold_specific_training_benchmark_predictions() -> None:
    predictions = pd.DataFrame(
        {
            "outer_fold_id": [0, 0, 1, 1],
            "actual_return": [1.0, 2.0, 3.0, 4.0],
            "predicted_return": [1.1, 1.9, 3.1, 3.9],
            "benchmark_prediction": [0.5, 0.5, 2.5, 2.5],
            "prediction_status": ["success"] * 4,
        }
    )
    metrics = calculate_model_metrics(predictions)
    expected = 1 - 0.04 / ((0.5**2) + (1.5**2) + (0.5**2) + (1.5**2))
    assert metrics["OOS_R2"] == pytest.approx(expected)


def test_undefined_correlation_is_nan_not_zero() -> None:
    correlation, reason = prediction_correlation(
        pd.Series([1.0, 2.0, 3.0]), pd.Series([0.5, 0.5, 0.5])
    )
    assert np.isnan(correlation)
    assert reason == "predicted_return_is_constant"


def test_random_seed_reproduces_random_forest_predictions() -> None:
    dataset = synthetic_model_dataset(rows=220)
    complete = dataset[dataset["target_available"]]
    features = ["feature_one", "feature_two"]
    parameters = small_config()["step4_debug_parameters"]["RandomForest"]
    first = build_model_pipeline(
        "RandomForest", parameters, imputer_strategy="median",
        random_seed=42, n_jobs=1, lasso_max_iter=2000,
    )
    second = build_model_pipeline(
        "RandomForest", parameters, imputer_strategy="median",
        random_seed=42, n_jobs=1, lasso_max_iter=2000,
    )
    first.fit(complete[features], complete["target_21d_return"])
    second.fit(complete[features], complete["target_21d_return"])
    np.testing.assert_allclose(first.predict(complete[features]), second.predict(complete[features]))


def test_ticker_available_rows_never_mix_products() -> None:
    dataset = synthetic_model_dataset(("AAA", "BBB"), rows=300)
    available = available_rows_as_of(
        dataset,
        ticker="BBB",
        selection_date=dataset["feature_date"].max(),
        training_years=5,
    )
    assert set(available["ticker"]) == {"BBB"}
