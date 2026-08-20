"""Static and synthetic safeguards against Step 4 leakage and silent failure."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.model_selection as selection_module
from src.model_factory import build_model_pipeline
from src.model_selection import (
    _inner_parameter_search,
    detect_feature_columns,
    evaluate_ticker_selection_date,
    run_monthly_model_selection,
)
from src.paths import PROJECT_ROOT
from tests.test_model_selection import logger
from tests.test_purged_time_series_split import small_config, synthetic_model_dataset


def test_source_forbids_random_split_shuffle_and_full_sample_preprocessing() -> None:
    source = (PROJECT_ROOT / "src" / "model_selection.py").read_text(encoding="utf-8")
    for forbidden in ("train_test_split", "KFold(", "shuffle=True", "pipeline.fit(dataset"):
        assert forbidden not in source
    assert "pipeline.fit(\n                    inner_train" in source
    assert "pipeline.fit(\n                    outer_train" in source


def test_pipelines_are_fresh_and_unfitted_before_fold_training() -> None:
    pipeline = build_model_pipeline(
        "Ridge", {"alpha": 1.0}, imputer_strategy="median",
        random_seed=42, n_jobs=1, lasso_max_iter=2000,
    )
    assert not hasattr(pipeline.named_steps["imputer"], "statistics_")
    assert not hasattr(pipeline.named_steps["scaler"], "mean_")


def test_inner_search_api_cannot_receive_outer_validation() -> None:
    parameters = set(inspect.signature(_inner_parameter_search).parameters)
    assert "outer_train" in parameters
    assert "outer_validation" not in parameters


class MeanPipeline:
    def fit(self, features: pd.DataFrame, target: pd.Series) -> "MeanPipeline":
        self.mean_ = float(target.mean())
        self.fit_rows_ = len(features)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.mean_)


def test_model_failure_is_recorded_without_changing_other_validation_rows(monkeypatch) -> None:
    def fake_factory(model_name, *args, **kwargs):
        if model_name == "Lasso":
            raise RuntimeError("intentional test failure")
        return MeanPipeline()

    monkeypatch.setattr(selection_module, "build_model_pipeline", fake_factory)
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
    assert result.model_failures["model"].eq("Lasso").any()
    remaining = result.fold_predictions[result.fold_predictions["model"].ne("Lasso")]
    grouped = remaining.groupby(["model", "outer_fold_id"])["feature_date"].apply(tuple)
    for fold_id in sorted(remaining["outer_fold_id"].unique()):
        dates = [grouped.loc[(model, fold_id)] for model in sorted(remaining["model"].unique())]
        assert all(candidate == dates[0] for candidate in dates[1:])


def test_runner_reads_no_existing_step4_result_and_protects_prior_outputs() -> None:
    source = (PROJECT_ROOT / "scripts" / "run_step4_model_selection.py").read_text(
        encoding="utf-8"
    )
    assert "read_csv(" not in source
    assert source.count("read_parquet(") == 1
    for prior_output in (
        "features_sample.parquet", "targets_sample.parquet", "feature_audit.csv",
        "monthly_sector_representatives.csv",
    ):
        assert prior_output not in source


def test_output_guard_requires_explicit_overwrite(tmp_path: Path) -> None:
    from scripts.run_step4_model_selection import _guard_outputs

    existing = tmp_path / "model_metrics_debug.csv"
    existing.touch()
    outputs = {"model_metrics": existing, "log": tmp_path / "step4.log"}
    with pytest.raises(FileExistsError, match="Nothing was overwritten"):
        _guard_outputs(outputs, overwrite=False)
    _guard_outputs(outputs, overwrite=True)


def test_single_ticker_single_month_execution_is_supported(monkeypatch) -> None:
    monkeypatch.setattr(
        selection_module, "build_model_pipeline", lambda *args, **kwargs: MeanPipeline()
    )
    dataset = synthetic_model_dataset(rows=500)
    last_date = pd.Timestamp(dataset["feature_date"].max())
    result = run_monthly_model_selection(
        dataset,
        tickers=["AAA"],
        start_date=last_date.date().isoformat(),
        end_date=last_date.date().isoformat(),
        config=small_config(),
        debug=True,
        logger=logger(),
    )
    assert len(result.selected_models) == 1
    assert set(result.selected_models["ticker"]) == {"AAA"}
