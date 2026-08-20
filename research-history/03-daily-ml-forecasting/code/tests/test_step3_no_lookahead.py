"""Static and synthetic-data leakage safeguards for Step 3."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import load_step3_prices, validate_step3_price_frame
from src.feature_audit import build_manual_feature_checks
from src.feature_engineering import FeatureConfig, build_features, feature_columns
from src.paths import PROJECT_ROOT
from src.target_builder import build_targets
from tests.test_feature_engineering import synthetic_prices


FEATURE_SOURCE = PROJECT_ROOT / "src" / "feature_engineering.py"
RUNNER_SOURCE = PROJECT_ROOT / "scripts" / "run_step3_feature_build.py"


def test_feature_source_has_no_negative_shift_or_centered_rolling() -> None:
    source = FEATURE_SOURCE.read_text(encoding="utf-8")
    assert re.search(r"shift\s*\(\s*-", source) is None
    assert re.search(r"center\s*=\s*True", source, flags=re.IGNORECASE) is None


def test_future_perturbation_cannot_change_features_on_or_before_cutoff() -> None:
    original = synthetic_prices()
    modified = original.copy()
    cutoff = 180
    modified.loc[cutoff + 1 :, ["Open", "High", "Low", "Close", "Adjusted Close", "Volume"]] *= 7
    first = build_features(original, "TEST", FeatureConfig()).features
    second = build_features(modified, "TEST", FeatureConfig()).features
    pd.testing.assert_frame_equal(first.loc[:cutoff], second.loc[:cutoff])


def test_no_target_column_or_global_missing_value_fill_enters_x() -> None:
    result = build_features(synthetic_prices(), "TEST", FeatureConfig()).features
    names = feature_columns(result)
    assert all("target" not in name.casefold() for name in names)
    assert result[names].isna().any().any()
    source = FEATURE_SOURCE.read_text(encoding="utf-8")
    for forbidden in ("SimpleImputer", "StandardScaler", ".bfill(", ".ffill(", ".interpolate("):
        assert forbidden not in source


def test_ticker_windows_are_isolated() -> None:
    first_prices = synthetic_prices(ticker="AAA")
    second_prices = synthetic_prices(ticker="BBB")
    second_prices["Adjusted Close"] *= 5
    first_alone = build_features(first_prices, "AAA", FeatureConfig()).features
    build_features(second_prices, "BBB", FeatureConfig())
    first_again = build_features(first_prices, "AAA", FeatureConfig()).features
    pd.testing.assert_frame_equal(first_alone, first_again)


def test_same_seed_produces_identical_manual_checks() -> None:
    prices = synthetic_prices()
    features = build_features(prices, "TEST", FeatureConfig()).features
    targets = build_targets(prices, "TEST", 21)
    first = build_manual_feature_checks(
        {"TEST": prices}, features, targets, horizon_days=21, random_seed=42
    )
    second = build_manual_feature_checks(
        {"TEST": prices}, features, targets, horizon_days=21, random_seed=42
    )
    pd.testing.assert_frame_equal(first, second)


def test_loader_rejects_unsorted_duplicates_and_uses_only_raw_cache(tmp_path: Path) -> None:
    prices = synthetic_prices(40).drop(columns="ticker")
    bad = pd.concat([prices.iloc[[1]], prices.iloc[[0]], prices.iloc[[1]]], ignore_index=True)
    with pytest.raises(ValueError):
        validate_step3_price_frame(bad, "TEST")
    prices.to_csv(tmp_path / "TEST.csv", index=False)
    loaded = load_step3_prices(["TEST"], raw_dir=tmp_path)
    assert set(loaded) == {"TEST"}
    assert loaded["TEST"]["ticker"].eq("TEST").all()


def test_runner_has_no_step3_result_reads_and_protects_prior_stage_outputs() -> None:
    source = RUNNER_SOURCE.read_text(encoding="utf-8")
    assert "read_parquet" not in source
    assert "monthly_sector_representatives.csv" not in source
    assert "monthly_eligible_universe.csv" not in source
    for protected in ("data_audit_v2.csv", "clean_sector_etf_candidates.csv"):
        assert protected not in source


def test_output_guard_requires_explicit_overwrite(tmp_path: Path) -> None:
    from scripts.run_step3_feature_build import _guard_outputs

    existing = tmp_path / "features_debug.parquet"
    existing.touch()
    outputs = {"features": existing, "log": tmp_path / "step3.log"}
    with pytest.raises(FileExistsError, match="Nothing was overwritten"):
        _guard_outputs(outputs, overwrite=False)
    _guard_outputs(outputs, overwrite=True)
