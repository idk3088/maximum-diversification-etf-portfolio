"""Synthetic-data tests for label-end-aware walk-forward folds."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.purged_time_series_split import (
    PurgedWalkForwardSplitter,
    actual_month_end_selection_dates,
    available_rows_as_of,
)


def synthetic_model_dataset(
    tickers: tuple[str, ...] = ("AAA",), rows: int = 500, horizon: int = 21
) -> pd.DataFrame:
    frames = []
    dates = pd.bdate_range("2022-01-03", periods=rows)
    for ticker_number, ticker in enumerate(tickers):
        feature_one = np.sin(np.arange(rows) / 20) + ticker_number
        feature_two = np.cos(np.arange(rows) / 15)
        target = 0.02 * feature_one - 0.01 * feature_two
        label_end = pd.Series(dates).shift(-horizon)
        available = label_end.notna()
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "feature_date": dates,
                    "label_start_date": dates,
                    "label_end_date": label_end,
                    "current_adjusted_close": 100 + np.arange(rows),
                    "future_adjusted_close": 101 + np.arange(rows),
                    "target_21d_return": pd.Series(target).where(available),
                    "target_available": available,
                    "feature_one": feature_one,
                    "feature_two": feature_two,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def small_config() -> dict[str, object]:
    return {
        "model_training_years": 5,
        "prediction_horizon_days": 21,
        "purge_days": 21,
        "walk_forward_mode": "rolling",
        "outer_validation_days": 10,
        "outer_fold_count": 3,
        "minimum_outer_folds": 3,
        "inner_validation_days": 8,
        "inner_fold_count": 2,
        "minimum_inner_folds": 2,
        "minimum_outer_train_rows": 100,
        "minimum_inner_train_rows": 50,
        "embargo_days": 0,
        "imputer_strategy": "median",
        "inner_insufficient_fallback": "default_parameters",
        "one_standard_error_rule": True,
        "one_se_insufficient_folds_fallback": "minimum_mse",
        "direction_zero_rule": "exact_sign_match",
        "model_complexity_order": ["OLS", "Ridge", "Lasso", "RandomForest", "XGBoost"],
        "candidate_models": ["ols", "ridge", "lasso", "random_forest", "xgboost"],
        "random_seed": 42,
        "n_jobs": 1,
        "lasso_max_iter": 2000,
        "ridge_alpha_grid": [0.1, 1.0],
        "lasso_alpha_grid": [0.0001, 0.001],
        "random_forest_grid": {
            "n_estimators": [10], "max_depth": [3],
            "min_samples_leaf": [5], "max_features": ["sqrt"],
        },
        "xgboost_grid": {
            "n_estimators": [10], "max_depth": [2], "learning_rate": [0.1],
            "subsample": [0.8], "colsample_bytree": [0.8], "reg_lambda": [1.0],
        },
        "step4_debug_parameters": {
            "Ridge": {"alpha": 1.0},
            "Lasso": {"alpha": 0.001},
            "RandomForest": {
                "n_estimators": 10, "max_depth": 3,
                "min_samples_leaf": 5, "max_features": "sqrt",
            },
            "XGBoost": {
                "n_estimators": 10, "max_depth": 2, "learning_rate": 0.1,
                "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
            },
        },
    }


def test_training_dates_and_label_intervals_precede_validation() -> None:
    dataset = synthetic_model_dataset(rows=400)
    complete = dataset[dataset["target_available"]].reset_index(drop=True)
    splitter = PurgedWalkForwardSplitter(
        n_splits=3, validation_days=10, min_train_rows=100,
        mode="rolling", max_train_rows=len(complete),
    )
    folds = list(splitter.split(complete))
    assert len(folds) == 3
    for fold in folds:
        train = complete.iloc[fold.train_indices]
        validation = complete.iloc[fold.validation_indices]
        validation_start = validation["feature_date"].min()
        assert train["feature_date"].max() < validation_start
        assert train["label_end_date"].max() < validation_start
        assert fold.purge_valid
        assert fold.purged_row_count > 0


def test_selection_date_excludes_unobserved_labels_and_final_tail() -> None:
    dataset = synthetic_model_dataset(rows=300)
    selection_date = dataset.loc[250, "feature_date"]
    available = available_rows_as_of(
        dataset, ticker="AAA", selection_date=selection_date, training_years=5
    )
    assert available["label_end_date"].le(selection_date).all()
    assert available["target_available"].all()
    assert available["target_21d_return"].notna().all()
    assert not set(dataset.tail(21)["feature_date"]).intersection(
        set(available["feature_date"])
    )


def test_inner_split_uses_the_same_label_end_purge() -> None:
    dataset = synthetic_model_dataset(rows=320)
    complete = dataset[dataset["target_available"]].reset_index(drop=True)
    splitter = PurgedWalkForwardSplitter(
        n_splits=2, validation_days=8, min_train_rows=50,
        mode="rolling", max_train_rows=len(complete),
    )
    for fold in splitter.split(complete):
        train = complete.iloc[fold.train_indices]
        validation_start = complete.iloc[fold.validation_indices]["feature_date"].min()
        assert train["label_end_date"].lt(validation_start).all()


def test_selection_dates_use_observed_month_end_trading_days() -> None:
    dates = pd.Series(pd.bdate_range("2024-01-01", "2024-06-30"))
    selected = actual_month_end_selection_dates(
        dates, start_date="2024-01-31", end_date="2024-06-30"
    )
    assert selected[-1] == pd.Timestamp("2024-06-28")
    assert all(date in set(dates) for date in selected)


def test_insufficient_train_rows_are_explicitly_flagged() -> None:
    dataset = synthetic_model_dataset(rows=80)
    complete = dataset[dataset["target_available"]].reset_index(drop=True)
    splitter = PurgedWalkForwardSplitter(
        n_splits=2, validation_days=10, min_train_rows=100,
        mode="rolling", max_train_rows=len(complete),
    )
    assert all(fold.fold_status == "insufficient_train_data" for fold in splitter.split(complete))
