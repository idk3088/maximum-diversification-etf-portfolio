"""Run Step 4 nested purged model selection from the Step 3 sample dataset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logging_config import configure_logging  # noqa: E402
from src.model_selection import run_monthly_model_selection  # noqa: E402
from src.paths import step4_output_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--output-suffix", default="sample")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    with resolved.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a mapping")
    required = {
        "model_training_years", "prediction_horizon_days", "purge_days",
        "walk_forward_mode", "outer_validation_days", "outer_fold_count",
        "minimum_outer_folds", "inner_validation_days", "inner_fold_count",
        "minimum_inner_folds", "minimum_outer_train_rows",
        "minimum_inner_train_rows", "imputer_strategy", "random_seed",
        "one_standard_error_rule", "model_complexity_order",
        "step4_sample_tickers", "step4_test_start_date", "step4_test_end_date",
        "step4_input_dataset", "overwrite_existing_step4_outputs", "n_jobs",
        "parallelize_across_tickers", "ticker_workers",
        "candidate_models", "ridge_alpha_grid", "lasso_alpha_grid",
        "random_forest_grid", "xgboost_grid", "step4_debug_parameters",
        "inner_insufficient_fallback", "one_se_insufficient_folds_fallback",
        "lasso_max_iter", "direction_zero_rule", "embargo_days",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Step 4 config is missing: {sorted(missing)}")
    if int(config["prediction_horizon_days"]) != int(config["purge_days"]):
        raise ValueError("prediction_horizon_days and purge_days must match")
    if int(config["n_jobs"]) == 0:
        raise ValueError("n_jobs cannot be zero")
    if bool(config["parallelize_across_tickers"]) and int(config["n_jobs"]) != 1:
        raise ValueError(
            "When ticker-level parallelism is enabled, model n_jobs must remain 1"
        )
    if config["direction_zero_rule"] != "exact_sign_match":
        raise ValueError("Only direction_zero_rule=exact_sign_match is implemented")
    if int(config["outer_fold_count"]) < int(config["minimum_outer_folds"]):
        raise ValueError("outer_fold_count cannot be below minimum_outer_folds")
    if int(config["inner_fold_count"]) < int(config["minimum_inner_folds"]):
        raise ValueError("inner_fold_count cannot be below minimum_inner_folds")
    return config


def _normalize_tickers(values: list[str]) -> list[str]:
    tickers = [str(value).strip().upper() for value in values if str(value).strip()]
    if not tickers or len(tickers) != len(set(tickers)):
        raise ValueError("Tickers must be nonempty and unique")
    return tickers


def _guard_outputs(outputs: dict[str, Path], overwrite: bool) -> None:
    existing = [path for name, path in outputs.items() if name != "log" and path.exists()]
    if existing and not overwrite:
        rendered = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Step 4 outputs already exist. Nothing was overwritten. Use a new "
            "--output-suffix or explicitly pass --overwrite:\n" + rendered
        )


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    outputs = step4_output_paths(args.output_suffix)
    _guard_outputs(outputs, overwrite=args.overwrite)
    logger = configure_logging(log_file=outputs["log"])

    tickers = _normalize_tickers(args.tickers or config["step4_sample_tickers"])
    if args.debug and args.tickers is None:
        tickers = tickers[:1]
    start_date = args.start_date or str(config["step4_test_start_date"])
    end_date = args.end_date or str(config["step4_test_end_date"])
    input_path = PROJECT_ROOT / str(config["step4_input_dataset"])
    if not input_path.exists():
        raise FileNotFoundError(f"Step 3 model dataset not found: {input_path}")

    started = time.perf_counter()
    logger.info("Step 4 started | debug=%s", args.debug)
    logger.info("Loading read-only Step 3 model dataset | %s", input_path)
    dataset = pd.read_parquet(input_path)
    logger.info("Dataset loaded | requested_tickers=%s", ",".join(tickers))
    logger.info(
        "Nested validation config | outer_folds=%s | inner_folds=%s | purge_by=label_end_date | ticker_parallel=%s | n_jobs=%s",
        config["outer_fold_count"],
        config["inner_fold_count"],
        config["parallelize_across_tickers"],
        config["n_jobs"],
    )
    result = run_monthly_model_selection(
        dataset,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        config=config,
        debug=args.debug,
        logger=logger,
    )

    logger.info("Writing Step 4 outputs atomically")
    frames = {
        "fold_definitions": result.fold_definitions,
        "fold_predictions": result.fold_predictions,
        "hyperparameter_search": result.hyperparameter_search,
        "model_metrics": result.model_metrics,
        "selected_models": result.selected_models,
        "model_failures": result.model_failures,
    }
    for name, frame in frames.items():
        _atomic_csv(frame, outputs[name])
    summary = {
        "step": 4,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "debug": args.debug,
        "tickers": tickers,
        "requested_start_date": start_date,
        "requested_end_date": end_date,
        "output_suffix": args.output_suffix,
        "row_counts": {name: len(frame) for name, frame in frames.items()},
        "selection_status_counts": result.selected_models["selection_status"].value_counts(
            dropna=False
        ).to_dict(),
        "elapsed_seconds": time.perf_counter() - started,
        "n_jobs": int(config["n_jobs"]),
        "network_access_used": False,
        "step4_results_used_as_input": False,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    _atomic_json(summary, outputs["run_summary"])
    logger.info("Step 4 completed successfully | elapsed_seconds=%.3f", time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
