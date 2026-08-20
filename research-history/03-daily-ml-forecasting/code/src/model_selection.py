"""Nested purged walk-forward evaluation and per-ETF monthly model selection."""

from __future__ import annotations

import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.model_factory import (
    MODEL_NAMES,
    build_model_pipeline,
    configured_model_names,
    default_parameters,
    parameter_grid,
)
from src.model_metrics import calculate_model_metrics
from src.purged_time_series_split import (
    PurgedWalkForwardSplitter,
    actual_month_end_selection_dates,
    available_rows_as_of,
    fold_definition_record,
)


NON_FEATURE_COLUMNS = {
    "ticker",
    "feature_date",
    "label_start_date",
    "label_end_date",
    "target_21d_return",
    "target_available",
    "current_adjusted_close",
    "future_adjusted_close",
}

FOLD_DEFINITION_COLUMNS = [
    "ticker", "selection_date", "outer_fold_id", "train_feature_start",
    "train_feature_end", "train_label_end_max", "validation_feature_start",
    "validation_feature_end", "validation_label_end_max",
    "train_row_count_before_purge", "train_row_count_after_purge",
    "purged_row_count", "validation_row_count", "purge_valid", "fold_status",
]
PREDICTION_COLUMNS = [
    "ticker", "selection_date", "model", "outer_fold_id", "feature_date",
    "label_end_date", "actual_return", "predicted_return",
    "benchmark_prediction", "squared_error", "absolute_error",
    "actual_direction", "predicted_direction", "hyperparameter_set_id",
    "prediction_status",
]
SEARCH_COLUMNS = [
    "ticker", "selection_date", "model", "outer_fold_id",
    "hyperparameter_set_id", "hyperparameters_json", "inner_fold_count",
    "mean_inner_MSE", "std_inner_MSE", "selected_for_outer_refit",
    "search_status", "failure_reason",
]
METRIC_COLUMNS = [
    "ticker", "selection_date", "model", "outer_fold_count",
    "valid_prediction_count", "mean_validation_MSE", "fold_MSE_std",
    "standard_error_MSE", "MAE", "OOS_R2", "directional_accuracy",
    "prediction_correlation", "selected_by_minimum_MSE", "within_one_SE",
    "selected_after_one_SE_rule", "complexity_rank", "model_status",
    "metric_notes",
]
SELECTED_COLUMNS = [
    "ticker", "selection_date", "minimum_MSE_model", "minimum_MSE",
    "one_SE_threshold", "final_selected_model",
    "final_selected_hyperparameters", "valid_outer_fold_count",
    "selection_status", "selection_notes",
]
FAILURE_COLUMNS = [
    "ticker", "selection_date", "model", "stage", "fold_id", "error_type",
    "error_message", "fallback_used", "final_status",
]


@dataclass(frozen=True)
class Step4SelectionResult:
    fold_definitions: pd.DataFrame
    fold_predictions: pd.DataFrame
    hyperparameter_search: pd.DataFrame
    model_metrics: pd.DataFrame
    selected_models: pd.DataFrame
    model_failures: pd.DataFrame


def detect_feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Identify numeric features while rejecting all target/date semantics."""
    forbidden_tokens = ("target", "future", "label")
    features = [
        column
        for column in dataset.columns
        if column not in NON_FEATURE_COLUMNS
        and not any(token in column.casefold() for token in forbidden_tokens)
    ]
    if not features:
        raise ValueError("No feature columns were detected")
    nonnumeric = [column for column in features if not pd.api.types.is_numeric_dtype(dataset[column])]
    if nonnumeric:
        raise ValueError(f"Non-numeric feature columns detected: {nonnumeric}")
    return features


def _failure(
    *,
    ticker: str,
    selection_date: pd.Timestamp,
    model: str,
    stage: str,
    fold_id: int | str,
    error: BaseException | str,
    fallback_used: bool,
    final_status: str,
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "selection_date": selection_date.date().isoformat(),
        "model": model,
        "stage": stage,
        "fold_id": fold_id,
        "error_type": type(error).__name__ if isinstance(error, BaseException) else "RuleFailure",
        "error_message": str(error),
        "fallback_used": fallback_used,
        "final_status": final_status,
    }


def _failed_prediction_records(
    validation: pd.DataFrame,
    *,
    ticker: str,
    selection_date: pd.Timestamp,
    model_name: str,
    fold_id: int,
    benchmark: float,
    parameter_id: str,
) -> list[dict[str, object]]:
    """Keep shared validation dates visible when a model fold fails."""
    return [
        {
            "ticker": ticker,
            "selection_date": selection_date.date().isoformat(),
            "model": model_name,
            "outer_fold_id": fold_id,
            "feature_date": pd.Timestamp(row["feature_date"]).date().isoformat(),
            "label_end_date": pd.Timestamp(row["label_end_date"]).date().isoformat(),
            "actual_return": float(row["target_21d_return"]),
            "predicted_return": math.nan,
            "benchmark_prediction": benchmark,
            "squared_error": math.nan,
            "absolute_error": math.nan,
            "actual_direction": int(np.sign(float(row["target_21d_return"]))),
            "predicted_direction": math.nan,
            "hyperparameter_set_id": parameter_id,
            "prediction_status": "failed",
        }
        for _, row in validation.iterrows()
    ]


def _build_splitter(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    inner: bool,
) -> PurgedWalkForwardSplitter:
    return PurgedWalkForwardSplitter(
        n_splits=int(config["inner_fold_count"] if inner else config["outer_fold_count"]),
        validation_days=int(
            config["inner_validation_days"] if inner else config["outer_validation_days"]
        ),
        min_train_rows=int(
            config["minimum_inner_train_rows"]
            if inner
            else config["minimum_outer_train_rows"]
        ),
        mode=str(config["walk_forward_mode"]),
        max_train_rows=max(len(frame), 1),
        embargo_days=int(config.get("embargo_days", 0)),
    )


def _parameter_set_id(model: str, index: int) -> str:
    return f"{model.lower()}_{index:03d}"


def _inner_parameter_search(
    outer_train: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    ticker: str,
    selection_date: pd.Timestamp,
    model_name: str,
    outer_fold_id: int,
    config: Mapping[str, Any],
    debug: bool,
) -> tuple[dict[str, Any] | None, str, list[dict[str, object]], str]:
    """Tune exclusively inside outer training data and return audit rows."""
    inner_folds = [
        fold for fold in _build_splitter(outer_train, config, inner=True).split(outer_train)
        if fold.fold_status == "ok"
    ]
    minimum_inner = int(config["minimum_inner_folds"])
    if len(inner_folds) < minimum_inner:
        if config["inner_insufficient_fallback"] != "default_parameters":
            return None, "", [], "insufficient_inner_data"
        parameters = default_parameters(model_name, config)
        parameter_id = _parameter_set_id(model_name, 0)
        row = {
            "ticker": ticker,
            "selection_date": selection_date.date().isoformat(),
            "model": model_name,
            "outer_fold_id": outer_fold_id,
            "hyperparameter_set_id": parameter_id,
            "hyperparameters_json": json.dumps(parameters, sort_keys=True),
            "inner_fold_count": len(inner_folds),
            "mean_inner_MSE": math.nan,
            "std_inner_MSE": math.nan,
            "selected_for_outer_refit": True,
            "search_status": "fallback_default_parameters",
            "failure_reason": "insufficient_inner_data",
        }
        return parameters, parameter_id, [row], "fallback_default_parameters"

    rows: list[dict[str, object]] = []
    parameter_sets = parameter_grid(model_name, config, debug=debug)
    for parameter_index, parameters in enumerate(parameter_sets):
        parameter_id = _parameter_set_id(model_name, parameter_index)
        fold_errors: list[float] = []
        failure_reason = ""
        for inner_fold in inner_folds:
            inner_train = outer_train.iloc[inner_fold.train_indices]
            inner_validation = outer_train.iloc[inner_fold.validation_indices]
            try:
                pipeline = build_model_pipeline(
                    model_name,
                    parameters,
                    imputer_strategy=str(config["imputer_strategy"]),
                    random_seed=int(config["random_seed"]),
                    n_jobs=int(config["n_jobs"]),
                    lasso_max_iter=int(config["lasso_max_iter"]),
                )
                pipeline.fit(
                    inner_train.loc[:, feature_columns],
                    inner_train["target_21d_return"],
                )
                predicted = pipeline.predict(inner_validation.loc[:, feature_columns])
                if len(predicted) != len(inner_validation) or not np.isfinite(predicted).all():
                    raise ValueError("Model did not predict every inner validation row")
                fold_errors.append(
                    float(mean_squared_error(inner_validation["target_21d_return"], predicted))
                )
            except Exception as exc:  # failures are recorded, never silently dropped
                failure_reason = f"{type(exc).__name__}: {exc}"
                break
        success = not failure_reason and len(fold_errors) == len(inner_folds)
        rows.append(
            {
                "ticker": ticker,
                "selection_date": selection_date.date().isoformat(),
                "model": model_name,
                "outer_fold_id": outer_fold_id,
                "hyperparameter_set_id": parameter_id,
                "hyperparameters_json": json.dumps(parameters, sort_keys=True),
                "inner_fold_count": len(fold_errors),
                "mean_inner_MSE": float(np.mean(fold_errors)) if success else math.nan,
                "std_inner_MSE": (
                    float(np.std(fold_errors, ddof=1)) if success and len(fold_errors) >= 2 else math.nan
                ),
                "selected_for_outer_refit": False,
                "search_status": "success" if success else "failed",
                "failure_reason": failure_reason,
            }
        )
    successful_indices = [
        index for index, row in enumerate(rows) if row["search_status"] == "success"
    ]
    if not successful_indices:
        return None, "", rows, "all_parameter_sets_failed"
    selected_index = min(
        successful_indices,
        key=lambda index: (float(rows[index]["mean_inner_MSE"]), index),
    )
    rows[selected_index]["selected_for_outer_refit"] = True
    return (
        dict(parameter_sets[selected_index]),
        str(rows[selected_index]["hyperparameter_set_id"]),
        rows,
        "success",
    )


def apply_one_standard_error_rule(
    metrics: pd.DataFrame,
    *,
    complexity_order: Sequence[str],
    enabled: bool,
    insufficient_folds_fallback: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select minimum MSE, then the simplest model inside the best model's SE."""
    result = metrics.copy()
    ranks = {model: rank for rank, model in enumerate(complexity_order, start=1)}
    result["complexity_rank"] = result["model"].map(ranks)
    result["selected_by_minimum_MSE"] = False
    result["within_one_SE"] = False
    result["selected_after_one_SE_rule"] = False
    eligible = result.loc[
        result["model_status"].eq("ok") & result["mean_validation_MSE"].notna()
    ].copy()
    if eligible.empty:
        return result, {
            "selection_status": "failed",
            "selection_notes": "no model has the required successful outer folds",
        }
    best_index = eligible.sort_values(
        ["mean_validation_MSE", "complexity_rank", "model"]
    ).index[0]
    result.loc[best_index, "selected_by_minimum_MSE"] = True
    best = result.loc[best_index]
    best_se = float(best["standard_error_MSE"])
    if not enabled:
        threshold = float(best["mean_validation_MSE"])
        notes = "one-standard-error rule disabled"
    elif not np.isfinite(best_se):
        if insufficient_folds_fallback != "minimum_mse":
            raise ValueError("Unsupported one-SE insufficient-fold fallback")
        threshold = float(best["mean_validation_MSE"])
        notes = "standard error unavailable; fell back to minimum MSE"
    else:
        threshold = float(best["mean_validation_MSE"]) + best_se
        notes = "one-standard-error rule applied using best model fold MSE SE"
    within = result["model_status"].eq("ok") & result["mean_validation_MSE"].le(threshold)
    result.loc[within, "within_one_SE"] = True
    final_index = result.loc[within].sort_values(
        ["complexity_rank", "mean_validation_MSE", "model"]
    ).index[0]
    result.loc[final_index, "selected_after_one_SE_rule"] = True
    return result, {
        "minimum_MSE_model": best["model"],
        "minimum_MSE": float(best["mean_validation_MSE"]),
        "one_SE_threshold": threshold,
        "final_selected_model": result.loc[final_index, "model"],
        "valid_outer_fold_count": int(result.loc[final_index, "outer_fold_count"]),
        "selection_status": "ok",
        "selection_notes": notes,
    }


def evaluate_ticker_selection_date(
    dataset: pd.DataFrame,
    *,
    ticker: str,
    selection_date: pd.Timestamp,
    feature_columns: Sequence[str],
    config: Mapping[str, Any],
    debug: bool,
    logger: logging.Logger,
) -> Step4SelectionResult:
    """Evaluate five models independently for one ETF and one month-end."""
    available = available_rows_as_of(
        dataset,
        ticker=ticker,
        selection_date=selection_date,
        training_years=int(config["model_training_years"]),
    )
    folds = list(_build_splitter(available, config, inner=False).split(available))
    fold_records = [
        fold_definition_record(
            available, fold, ticker=ticker, selection_date=selection_date
        )
        for fold in folds
    ]
    valid_folds = [fold for fold in folds if fold.fold_status == "ok"]
    failures: list[dict[str, object]] = []
    if len(valid_folds) < int(config["minimum_outer_folds"]):
        failures.append(
            _failure(
                ticker=ticker,
                selection_date=selection_date,
                model="ALL",
                stage="outer_split",
                fold_id="all",
                error="insufficient outer folds after label-date purge",
                fallback_used=False,
                final_status="insufficient_data",
            )
        )
        return Step4SelectionResult(
            pd.DataFrame(fold_records, columns=FOLD_DEFINITION_COLUMNS),
            pd.DataFrame(columns=PREDICTION_COLUMNS),
            pd.DataFrame(columns=SEARCH_COLUMNS),
            pd.DataFrame(columns=METRIC_COLUMNS),
            pd.DataFrame(
                [{
                    "ticker": ticker,
                    "selection_date": selection_date.date().isoformat(),
                    "selection_status": "insufficient_data",
                    "selection_notes": "minimum outer folds not available",
                }],
                columns=SELECTED_COLUMNS,
            ),
            pd.DataFrame(failures, columns=FAILURE_COLUMNS),
        )

    predictions: list[dict[str, object]] = []
    searches: list[dict[str, object]] = []
    chosen_parameters: dict[tuple[str, int], dict[str, Any]] = {}
    for model_name in configured_model_names(config):
        model_started = time.perf_counter()
        logger.info("Model start | %s | %s | %s", ticker, selection_date.date(), model_name)
        for fold in valid_folds:
            outer_train = available.iloc[fold.train_indices]
            outer_validation = available.iloc[fold.validation_indices]
            parameters, parameter_id, search_rows, search_status = _inner_parameter_search(
                outer_train,
                feature_columns,
                ticker=ticker,
                selection_date=selection_date,
                model_name=model_name,
                outer_fold_id=fold.fold_id,
                config=config,
                debug=debug,
            )
            searches.extend(search_rows)
            if parameters is None:
                predictions.extend(
                    _failed_prediction_records(
                        outer_validation,
                        ticker=ticker,
                        selection_date=selection_date,
                        model_name=model_name,
                        fold_id=fold.fold_id,
                        benchmark=float(outer_train["target_21d_return"].mean()),
                        parameter_id="",
                    )
                )
                failures.append(
                    _failure(
                        ticker=ticker,
                        selection_date=selection_date,
                        model=model_name,
                        stage="inner_search",
                        fold_id=fold.fold_id,
                        error=search_status,
                        fallback_used=False,
                        final_status="failed",
                    )
                )
                continue
            chosen_parameters[(model_name, fold.fold_id)] = parameters
            try:
                pipeline = build_model_pipeline(
                    model_name,
                    parameters,
                    imputer_strategy=str(config["imputer_strategy"]),
                    random_seed=int(config["random_seed"]),
                    n_jobs=int(config["n_jobs"]),
                    lasso_max_iter=int(config["lasso_max_iter"]),
                )
                pipeline.fit(
                    outer_train.loc[:, feature_columns],
                    outer_train["target_21d_return"],
                )
                predicted = pipeline.predict(outer_validation.loc[:, feature_columns])
                if len(predicted) != len(outer_validation) or not np.isfinite(predicted).all():
                    raise ValueError("Model did not predict every shared outer validation row")
                benchmark = float(outer_train["target_21d_return"].mean())
                for row_position, (_, row) in enumerate(outer_validation.iterrows()):
                    actual = float(row["target_21d_return"])
                    prediction = float(predicted[row_position])
                    predictions.append(
                        {
                            "ticker": ticker,
                            "selection_date": selection_date.date().isoformat(),
                            "model": model_name,
                            "outer_fold_id": fold.fold_id,
                            "feature_date": pd.Timestamp(row["feature_date"]).date().isoformat(),
                            "label_end_date": pd.Timestamp(row["label_end_date"]).date().isoformat(),
                            "actual_return": actual,
                            "predicted_return": prediction,
                            "benchmark_prediction": benchmark,
                            "squared_error": (actual - prediction) ** 2,
                            "absolute_error": abs(actual - prediction),
                            "actual_direction": int(np.sign(actual)),
                            "predicted_direction": int(np.sign(prediction)),
                            "hyperparameter_set_id": parameter_id,
                            "prediction_status": "success",
                        }
                    )
            except Exception as exc:
                predictions.extend(
                    _failed_prediction_records(
                        outer_validation,
                        ticker=ticker,
                        selection_date=selection_date,
                        model_name=model_name,
                        fold_id=fold.fold_id,
                        benchmark=float(outer_train["target_21d_return"].mean()),
                        parameter_id=parameter_id,
                    )
                )
                failures.append(
                    _failure(
                        ticker=ticker,
                        selection_date=selection_date,
                        model=model_name,
                        stage="outer_refit_or_predict",
                        fold_id=fold.fold_id,
                        error=exc,
                        fallback_used=search_status.startswith("fallback"),
                        final_status="failed",
                    )
                )
        logger.info(
            "Model finish | %s | %s | %s | elapsed_seconds=%.3f",
            ticker,
            selection_date.date(),
            model_name,
            time.perf_counter() - model_started,
        )

    prediction_frame = pd.DataFrame(predictions, columns=PREDICTION_COLUMNS)
    metric_rows: list[dict[str, object]] = []
    for model_name in MODEL_NAMES:
        model_predictions = prediction_frame.loc[prediction_frame["model"].eq(model_name)]
        metrics = calculate_model_metrics(model_predictions)
        status = (
            "ok"
            if int(metrics["outer_fold_count"]) >= int(config["minimum_outer_folds"])
            else "failed_or_insufficient_outer_folds"
        )
        metric_rows.append(
            {
                "ticker": ticker,
                "selection_date": selection_date.date().isoformat(),
                "model": model_name,
                **metrics,
                "selected_by_minimum_MSE": False,
                "within_one_SE": False,
                "selected_after_one_SE_rule": False,
                "complexity_rank": 0,
                "model_status": status,
            }
        )
    metric_frame, selection = apply_one_standard_error_rule(
        pd.DataFrame(metric_rows),
        complexity_order=config["model_complexity_order"],
        enabled=bool(config["one_standard_error_rule"]),
        insufficient_folds_fallback=str(config["one_se_insufficient_folds_fallback"]),
    )
    final_model = selection.get("final_selected_model")
    final_parameters = {
        str(fold_id): parameters
        for (model, fold_id), parameters in chosen_parameters.items()
        if model == final_model
    }
    selected_record = {
        "ticker": ticker,
        "selection_date": selection_date.date().isoformat(),
        **selection,
        "final_selected_hyperparameters": json.dumps(final_parameters, sort_keys=True),
    }
    return Step4SelectionResult(
        pd.DataFrame(fold_records, columns=FOLD_DEFINITION_COLUMNS),
        prediction_frame,
        pd.DataFrame(searches, columns=SEARCH_COLUMNS),
        metric_frame.reindex(columns=METRIC_COLUMNS),
        pd.DataFrame([selected_record], columns=SELECTED_COLUMNS),
        pd.DataFrame(failures, columns=FAILURE_COLUMNS),
    )


def run_monthly_model_selection(
    dataset: pd.DataFrame,
    *,
    tickers: Sequence[str],
    start_date: str,
    end_date: str,
    config: Mapping[str, Any],
    debug: bool,
    logger: logging.Logger,
) -> Step4SelectionResult:
    """Run independent ticker-month selections and concatenate audit outputs."""
    feature_columns = detect_feature_columns(dataset)
    tasks: list[tuple[str, pd.Timestamp]] = []
    for ticker in tickers:
        ticker_dates = dataset.loc[
            dataset["ticker"].astype(str).str.upper().eq(ticker.upper()), "feature_date"
        ]
        selection_dates = actual_month_end_selection_dates(
            ticker_dates, start_date=start_date, end_date=end_date
        )
        if debug:
            selection_dates = selection_dates[:2]
        for selection_date in selection_dates:
            tasks.append((ticker, selection_date))

    def evaluate(task: tuple[str, pd.Timestamp]) -> Step4SelectionResult:
        ticker, selection_date = task
        logger.info("Ticker-month start | %s | %s", ticker, selection_date.date())
        result = evaluate_ticker_selection_date(
            dataset,
            ticker=ticker,
            selection_date=selection_date,
            feature_columns=feature_columns,
            config=config,
            debug=debug,
            logger=logger,
        )
        logger.info("Ticker-month finish | %s | %s", ticker, selection_date.date())
        return result

    if bool(config.get("parallelize_across_tickers", False)):
        if int(config["n_jobs"]) != 1:
            raise ValueError("ticker parallelism requires model n_jobs=1")
        with ThreadPoolExecutor(max_workers=int(config["ticker_workers"])) as executor:
            all_results = list(executor.map(evaluate, tasks))
    else:
        all_results = [evaluate(task) for task in tasks]
    if not all_results:
        raise ValueError("No actual trading month-end selection dates were found")

    def combine(attribute: str, columns: Sequence[str]) -> pd.DataFrame:
        frames = [getattr(result, attribute) for result in all_results]
        nonempty = [frame for frame in frames if not frame.empty]
        return (
            pd.concat(nonempty, ignore_index=True).reindex(columns=columns)
            if nonempty
            else pd.DataFrame(columns=columns)
        )

    return Step4SelectionResult(
        combine("fold_definitions", FOLD_DEFINITION_COLUMNS),
        combine("fold_predictions", PREDICTION_COLUMNS),
        combine("hyperparameter_search", SEARCH_COLUMNS),
        combine("model_metrics", METRIC_COLUMNS),
        combine("selected_models", SELECTED_COLUMNS),
        combine("model_failures", FAILURE_COLUMNS),
    )
