"""Offline, repeatable Step 3 sample feature/target build."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_step3_prices  # noqa: E402
from src.feature_audit import (  # noqa: E402
    audit_features,
    audit_target_alignment,
    build_manual_feature_checks,
)
from src.feature_engineering import FeatureConfig, build_features  # noqa: E402
from src.logging_config import configure_logging  # noqa: E402
from src.paths import RAW_DATA_DIR, step3_output_paths  # noqa: E402
from src.target_builder import build_targets, merge_features_and_targets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--tickers", nargs="+", help="Override configured sample tickers")
    parser.add_argument("--output-suffix", default="sample")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacement of Step 3 outputs for this suffix",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    with resolved.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a mapping")
    required = {
        "step3_sample_tickers",
        "prediction_horizon_days",
        "feature_return_windows",
        "feature_sma_windows",
        "feature_volatility_windows",
        "feature_drawdown_windows",
        "rsi_window",
        "adv_window_days",
        "random_seed",
        "annualize_feature_volatility",
        "overwrite_existing_step3_outputs",
        "feature_extreme_abs_thresholds",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Step 3 config is missing: {sorted(missing)}")
    if config["random_seed"] is None:
        raise ValueError("random_seed must be an integer for reproducible checks")
    return config


def _normalize_tickers(values: list[str]) -> list[str]:
    tickers = [str(value).strip().upper() for value in values if str(value).strip()]
    if not tickers:
        raise ValueError("At least one ticker is required")
    if len(tickers) != len(set(tickers)):
        raise ValueError("Ticker list contains duplicates")
    return tickers


def _guard_outputs(outputs: dict[str, Path], overwrite: bool) -> None:
    protected = [path for name, path in outputs.items() if name != "log"]
    existing = [path for path in protected if path.exists()]
    if existing and not overwrite:
        rendered = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Step 3 output files already exist. Nothing was overwritten. "
            "Use a new --output-suffix or explicitly pass --overwrite:\n" + rendered
        )


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


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
    outputs = step3_output_paths(args.output_suffix)
    logger = configure_logging(log_file=outputs["log"])
    tickers = _normalize_tickers(args.tickers or config["step3_sample_tickers"])
    _guard_outputs(outputs, overwrite=args.overwrite)

    logger.info("Step 3 started: loading configuration")
    logger.info("Loading isolated local OHLCV caches for: %s", ", ".join(tickers))
    prices_by_ticker = load_step3_prices(tickers, raw_dir=RAW_DATA_DIR)
    logger.info("Input validation complete: dates, duplicates, fields, ticker isolation")

    feature_config = FeatureConfig.from_mapping(config)
    horizon = int(config["prediction_horizon_days"])
    feature_results = {}
    feature_frames = []
    target_frames = []
    for ticker in tickers:
        logger.info("Building backward-looking features for %s", ticker)
        result = build_features(prices_by_ticker[ticker], ticker, feature_config)
        feature_results[ticker] = result
        feature_frames.append(result.features)
        logger.info("Building exact %s-trading-day targets for %s", horizon, ticker)
        target_frames.append(build_targets(prices_by_ticker[ticker], ticker, horizon))

    features = pd.concat(feature_frames, ignore_index=True)
    targets = pd.concat(target_frames, ignore_index=True)
    logger.info("Aligning features and targets on ticker + feature_date")
    model_dataset = merge_features_and_targets(features, targets)

    logger.info("Auditing feature missingness, extremes, and leakage indicators")
    source_max_dates = {
        ticker: pd.to_datetime(frame["Date"], errors="coerce").max()
        for ticker, frame in prices_by_ticker.items()
    }
    feature_audit = audit_features(
        feature_results,
        extreme_abs_thresholds=config["feature_extreme_abs_thresholds"],
        source_max_dates=source_max_dates,
    )
    logger.info("Auditing target trading-day alignment")
    target_audit = audit_target_alignment(targets, horizon)
    logger.info("Sampling reproducible manual feature checks")
    manual_checks = build_manual_feature_checks(
        prices_by_ticker,
        features,
        targets,
        horizon_days=horizon,
        random_seed=int(config["random_seed"]),
        sample_size=5,
    )

    logger.info("Writing Step 3 outputs atomically")
    _atomic_parquet(features, outputs["features"])
    _atomic_parquet(targets, outputs["targets"])
    _atomic_parquet(model_dataset, outputs["model_dataset"])
    _atomic_csv(feature_audit, outputs["feature_audit"])
    _atomic_csv(target_audit, outputs["target_alignment_audit"])
    _atomic_csv(manual_checks, outputs["manual_feature_checks"])
    summary = {
        "step": 3,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "tickers": tickers,
        "prediction_horizon_days": horizon,
        "output_suffix": args.output_suffix,
        "feature_rows": len(features),
        "target_rows": len(targets),
        "model_dataset_rows": len(model_dataset),
        "feature_audit_rows": len(feature_audit),
        "target_alignment_invalid_rows": int((~target_audit["alignment_valid"]).sum()),
        "manual_check_failures": int(manual_checks["check_status"].ne("PASS").sum()),
        "outputs": {name: str(path) for name, path in outputs.items()},
        "network_access_used": False,
        "imputation_or_standardization_used": False,
    }
    _atomic_json(summary, outputs["run_summary"])
    logger.info("Step 3 completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
