"""Centralized project path definitions.

All project code should import paths from this module instead of embedding
machine-specific absolute paths.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

PROJECT_ROOT_ENV = "ETF_PORTFOLIO_ROOT"


def _project_root() -> Path:
    """Return the configured root, or infer it from this source file."""
    override = os.getenv(PROJECT_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = _project_root()
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
METADATA_SNAPSHOT_DIR = METADATA_DIR / "source_snapshots"
METADATA_ARCHIVE_DIR = METADATA_DIR / "archive"
ETF_METADATA_PATH = METADATA_DIR / "etf_metadata.csv"
ETF_LIFECYCLE_PATH = METADATA_DIR / "etf_lifecycle.csv"
CLEAN_US_EQUITY_ETFS_PATH = METADATA_DIR / "clean_us_equity_etfs.csv"
EXCLUDED_ETPS_PATH = METADATA_DIR / "excluded_etps.csv"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_ARCHIVE_DIR = RESULTS_DIR / "archive"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
SELECTED_MODELS_DIR = RESULTS_DIR / "selected_models"
COVARIANCE_VALIDATION_DIR = RESULTS_DIR / "covariance_validation"
WEIGHTS_DIR = RESULTS_DIR / "weights"
PERFORMANCE_DIR = RESULTS_DIR / "performance"
CHARTS_DIR = RESULTS_DIR / "charts"
DATA_AUDIT_PATH = RESULTS_DIR / "data_audit.csv"
DATA_AUDIT_V2_PATH = RESULTS_DIR / "data_audit_v2.csv"
RESULTS_METADATA_DIR = RESULTS_DIR / "metadata"
RESULTS_DATA_QUALITY_DIR = RESULTS_DIR / "data_quality"
TICKER_REUSE_CONFLICTS_PATH = RESULTS_METADATA_DIR / "ticker_reuse_conflicts.csv"
VENDOR_CONFLICTS_PATH = RESULTS_METADATA_DIR / "vendor_conflicts.csv"
METADATA_NEEDS_REVIEW_PATH = RESULTS_METADATA_DIR / "metadata_needs_review.csv"
PRICE_ANOMALY_RESOLUTION_PATH = (
    RESULTS_DATA_QUALITY_DIR / "price_anomaly_resolution.csv"
)
CLEAN_UNIVERSE_DETAIL_PATH = RESULTS_METADATA_DIR / "clean_universe_detail.csv"
CURRENT_SECTOR_COVERAGE_PATH = RESULTS_METADATA_DIR / "current_sector_coverage.csv"
MONTHLY_SECTOR_COVERAGE_AUDIT_PATH = (
    RESULTS_METADATA_DIR / "monthly_sector_coverage_audit.csv"
)
PRIORITY_REVIEW_CANDIDATES_PATH = (
    RESULTS_METADATA_DIR / "priority_review_candidates.csv"
)
VENDOR_CONFLICTS_RESOLVED_V2_PATH = (
    RESULTS_METADATA_DIR / "vendor_conflicts_resolved_v2.csv"
)
STEP2_OFFICIAL_EVIDENCE_PATH = (
    RAW_DATA_DIR / "metadata" / "step2_sector_official_evidence.csv"
)
CLEAN_SECTOR_ETF_CANDIDATES_PATH = METADATA_DIR / "clean_sector_etf_candidates.csv"
MONTHLY_ELIGIBLE_UNIVERSE_PATH = RESULTS_DIR / "monthly_eligible_universe.csv"
MONTHLY_SECTOR_REPRESENTATIVES_PATH = RESULTS_DIR / "monthly_sector_representatives.csv"
CLASSIFICATION_REVIEW_PATH = RESULTS_DIR / "classification_review.csv"
REJECTED_SECTOR_CANDIDATES_PATH = RESULTS_DIR / "rejected_sector_candidates.csv"
MONTHLY_SECTOR_CANDIDATE_COVERAGE_PATH = (
    RESULTS_DIR / "monthly_sector_candidate_coverage.csv"
)
STEP2_AUTOMATED_CHECKS_PATH = RESULTS_DIR / "step2_automated_checks.csv"
LOGS_DIR = PROJECT_ROOT / "logs"
STEP3_LOG_PATH = LOGS_DIR / "step3_feature_build.log"
MODEL_VALIDATION_DIR = RESULTS_DIR / "model_validation"
STEP4_LOG_PATH = LOGS_DIR / "step4_model_selection.log"

PROJECT_DIRECTORIES = (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    METADATA_DIR,
    METADATA_SNAPSHOT_DIR,
    METADATA_ARCHIVE_DIR,
    NOTEBOOKS_DIR,
    PREDICTIONS_DIR,
    SELECTED_MODELS_DIR,
    COVARIANCE_VALIDATION_DIR,
    WEIGHTS_DIR,
    PERFORMANCE_DIR,
    CHARTS_DIR,
    LOGS_DIR,
    RESULTS_ARCHIVE_DIR,
    RESULTS_METADATA_DIR,
    RESULTS_DATA_QUALITY_DIR,
    MODEL_VALIDATION_DIR,
)


def ensure_project_directories() -> None:
    """Create the standard project directories if any are missing."""
    for directory in PROJECT_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


def from_project_root(*parts: str) -> Path:
    """Build a path relative to the project root without hard-coding it."""
    return PROJECT_ROOT.joinpath(*parts)


def step3_output_paths(suffix: str = "sample") -> dict[str, Path]:
    """Return isolated Step 3 output paths for a safe filename suffix."""
    normalized = suffix.strip().lower()
    if not normalized or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", normalized):
        raise ValueError("output suffix must contain only letters, numbers, _ or -")
    default = normalized == "sample"
    result_suffix = "" if default else f"_{normalized}"
    return {
        "features": PROCESSED_DATA_DIR / f"features_{normalized}.parquet",
        "targets": PROCESSED_DATA_DIR / f"targets_{normalized}.parquet",
        "model_dataset": PROCESSED_DATA_DIR / f"model_dataset_{normalized}.parquet",
        "feature_audit": RESULTS_DIR / f"feature_audit{result_suffix}.csv",
        "target_alignment_audit": RESULTS_DIR
        / f"target_alignment_audit{result_suffix}.csv",
        "manual_feature_checks": RESULTS_DIR
        / f"manual_feature_checks{result_suffix}.csv",
        "run_summary": RESULTS_DIR / f"step3_run_summary{result_suffix}.json",
        "log": STEP3_LOG_PATH,
    }


def step4_output_paths(suffix: str = "sample") -> dict[str, Path]:
    """Return Step 4-only outputs without touching earlier stage files."""
    normalized = suffix.strip().lower()
    if not normalized or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", normalized):
        raise ValueError("output suffix must contain only letters, numbers, _ or -")
    file_suffix = "" if normalized == "sample" else f"_{normalized}"
    return {
        "fold_definitions": MODEL_VALIDATION_DIR / f"fold_definitions{file_suffix}.csv",
        "fold_predictions": MODEL_VALIDATION_DIR / f"fold_predictions{file_suffix}.csv",
        "hyperparameter_search": MODEL_VALIDATION_DIR
        / f"hyperparameter_search{file_suffix}.csv",
        "model_metrics": MODEL_VALIDATION_DIR / f"model_metrics{file_suffix}.csv",
        "selected_models": MODEL_VALIDATION_DIR / f"selected_models{file_suffix}.csv",
        "model_failures": MODEL_VALIDATION_DIR / f"model_failures{file_suffix}.csv",
        "run_summary": MODEL_VALIDATION_DIR / f"step4_run_summary{file_suffix}.json",
        "log": STEP4_LOG_PATH,
    }
