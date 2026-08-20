"""Offline target and feature-date alignment tests for Step 3."""

from __future__ import annotations

import pandas as pd
import pytest

from src.feature_audit import audit_target_alignment
from src.feature_engineering import FeatureConfig, build_features
from src.target_builder import build_targets, merge_features_and_targets
from tests.test_feature_engineering import synthetic_prices


def test_target_uses_exact_21st_future_trading_row_not_calendar_days() -> None:
    prices = synthetic_prices(80)
    targets = build_targets(prices, "TEST", 21)
    expected = prices["Adjusted Close"].iloc[21] / prices["Adjusted Close"].iloc[0] - 1
    assert targets.loc[0, "target_21d_return"] == pytest.approx(expected)
    assert targets.loc[0, "label_end_date"] == prices.loc[21, "Date"]
    assert targets.loc[0, "label_end_date"] != prices.loc[0, "Date"] + pd.Timedelta(days=21)


def test_final_21_rows_remain_but_have_no_available_target() -> None:
    targets = build_targets(synthetic_prices(80), "TEST", 21)
    assert len(targets) == 80
    assert not targets.tail(21)["target_available"].any()
    assert targets.tail(21)["target_21d_return"].isna().all()
    assert targets.iloc[:-21]["target_available"].all()


def test_label_end_is_later_and_alignment_distance_is_21() -> None:
    targets = build_targets(synthetic_prices(80), "TEST", 21)
    audit = audit_target_alignment(targets, 21)
    available = audit[audit["target_available"]]
    assert available["trading_day_distance"].eq(21).all()
    assert pd.to_datetime(available["label_end_date"]).gt(
        pd.to_datetime(available["feature_date"])
    ).all()
    assert audit["alignment_valid"].all()


def test_features_and_targets_merge_one_to_one_without_dropping_tail() -> None:
    prices = synthetic_prices(330)
    features = build_features(prices, "TEST", FeatureConfig()).features
    targets = build_targets(prices, "TEST", 21)
    merged = merge_features_and_targets(features, targets)
    assert len(merged) == len(prices)
    assert not merged.duplicated(["ticker", "feature_date"]).any()
    assert not merged.tail(21)["target_available"].any()
    feature_names = set(features.columns)
    assert "target_21d_return" not in feature_names
    assert "future_adjusted_close" not in feature_names
