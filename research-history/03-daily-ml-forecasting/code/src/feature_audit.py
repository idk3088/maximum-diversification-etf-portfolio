"""Audits for Step 3 features, targets, and reproducible manual checks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.feature_engineering import (
    FEATURE_DEFINITIONS,
    FeatureBuildResult,
    feature_columns,
)


FEATURE_AUDIT_COLUMNS = [
    "ticker",
    "feature_name",
    "feature_category",
    "lookback_days",
    "first_valid_date",
    "total_rows",
    "missing_count",
    "missing_rate",
    "expected_missing_count",
    "source_missing_count",
    "calculation_invalid_count",
    "constant_feature",
    "extreme_value_count",
    "suspected_lookahead",
    "formula_description",
]

TARGET_AUDIT_COLUMNS = [
    "ticker",
    "feature_date",
    "label_start_date",
    "label_end_date",
    "trading_day_distance",
    "current_adjusted_close",
    "future_adjusted_close",
    "target_21d_return",
    "target_available",
    "alignment_valid",
    "error_message",
]

MANUAL_CHECK_COLUMNS = [
    "ticker",
    "feature_date",
    "adjusted_close_t",
    "adjusted_close_t_minus_21",
    "adjusted_close_t_plus_21",
    "calculated_return_21d_feature",
    "calculated_target_21d",
    "feature_source_start_date",
    "label_end_date",
    "source_row_positions",
    "check_status",
]


def scan_feature_source_for_lookahead(source_path: Path | None = None) -> list[str]:
    """Return static red flags from the feature implementation source."""
    path = source_path or Path(__file__).with_name("feature_engineering.py")
    source = path.read_text(encoding="utf-8")
    flags: list[str] = []
    if re.search(r"shift\s*\(\s*-", source):
        flags.append("negative_shift_in_feature_source")
    if re.search(r"center\s*=\s*True", source, flags=re.IGNORECASE):
        flags.append("centered_rolling_in_feature_source")
    forbidden_fit_terms = ("StandardScaler", "SimpleImputer", "bfill(", "ffill(")
    for term in forbidden_fit_terms:
        if term in source:
            flags.append(f"forbidden_preprocessing_term:{term}")
    return flags


def audit_features(
    results: Mapping[str, FeatureBuildResult],
    *,
    extreme_abs_thresholds: Mapping[str, float],
    source_max_dates: Mapping[str, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Create per-ticker, per-feature missingness and leakage audit rows."""
    static_flags = scan_feature_source_for_lookahead()
    rows: list[dict[str, object]] = []
    for ticker, result in results.items():
        features = result.features
        feature_dates = pd.to_datetime(features["feature_date"], errors="coerce")
        date_violation = False
        if source_max_dates and ticker in source_max_dates:
            date_violation = feature_dates.gt(pd.Timestamp(source_max_dates[ticker])).any()
        for name in feature_columns(features):
            if name not in FEATURE_DEFINITIONS:
                raise ValueError(f"Missing feature definition for {name}")
            category, lookback, formula = FEATURE_DEFINITIONS[name]
            threshold = float(extreme_abs_thresholds[category])
            values = pd.to_numeric(features[name], errors="coerce")
            reasons = result.missing_reasons[name]
            first_valid = feature_dates.loc[values.notna()].min()
            name_violation = any(
                token in name.casefold() for token in ("target", "future", "label_")
            )
            rows.append(
                {
                    "ticker": ticker,
                    "feature_name": name,
                    "feature_category": category,
                    "lookback_days": lookback,
                    "first_valid_date": (
                        first_valid.date().isoformat()
                        if pd.notna(first_valid)
                        else "unknown"
                    ),
                    "total_rows": len(values),
                    "missing_count": int(values.isna().sum()),
                    "missing_rate": float(values.isna().mean()),
                    "expected_missing_count": int(reasons.eq("expected_missing").sum()),
                    "source_missing_count": int(reasons.eq("source_missing").sum()),
                    "calculation_invalid_count": int(
                        reasons.eq("calculation_invalid").sum()
                    ),
                    "constant_feature": bool(
                        values.notna().any() and values.dropna().nunique() <= 1
                    ),
                    "extreme_value_count": int(values.abs().gt(threshold).sum()),
                    "suspected_lookahead": bool(
                        static_flags or date_violation or name_violation
                    ),
                    "formula_description": formula,
                }
            )
    return pd.DataFrame(rows, columns=FEATURE_AUDIT_COLUMNS)


def audit_target_alignment(targets: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Verify that every available label uses the exact future trading row."""
    rows: list[pd.DataFrame] = []
    for ticker, group in targets.groupby("ticker", sort=True):
        ordered = group.sort_values("feature_date").reset_index(drop=True).copy()
        feature_dates = pd.to_datetime(ordered["feature_date"], errors="coerce")
        label_start = pd.to_datetime(ordered["label_start_date"], errors="coerce")
        label_end = pd.to_datetime(ordered["label_end_date"], errors="coerce")
        expected_end = feature_dates.shift(-horizon_days)
        position = {date: index for index, date in enumerate(feature_dates)}
        distance = pd.Series(
            [
                position.get(end, np.nan) - index if pd.notna(end) else np.nan
                for index, end in enumerate(label_end)
            ],
            dtype="Float64",
        )
        available = ordered["target_available"].astype(bool)
        structural = label_start.eq(feature_dates) & (
            label_end.eq(expected_end) | (label_end.isna() & expected_end.isna())
        )
        formula = (
            pd.to_numeric(ordered["future_adjusted_close"], errors="coerce")
            / pd.to_numeric(ordered["current_adjusted_close"], errors="coerce")
            - 1.0
        )
        formula_valid = np.isclose(
            formula,
            pd.to_numeric(ordered["target_21d_return"], errors="coerce"),
            equal_nan=False,
        )
        available_valid = (
            distance.eq(horizon_days)
            & label_end.gt(feature_dates)
            & pd.Series(formula_valid, index=ordered.index)
        )
        alignment_valid = structural & (~available | available_valid)
        error = pd.Series("", index=ordered.index, dtype="string")
        error.loc[~structural] = "label dates do not match the exact future row"
        error.loc[available & ~available_valid] = (
            "available target does not have the configured trading-day distance or formula"
        )
        ordered["trading_day_distance"] = distance.astype("Int64")
        ordered["alignment_valid"] = alignment_valid.astype(bool)
        ordered["error_message"] = error
        rows.append(ordered[TARGET_AUDIT_COLUMNS])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=TARGET_AUDIT_COLUMNS)


def build_manual_feature_checks(
    prices_by_ticker: Mapping[str, pd.DataFrame],
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    horizon_days: int,
    random_seed: int,
    sample_size: int = 5,
) -> pd.DataFrame:
    """Sample reproducible dates and expose source prices for hand checking."""
    if sample_size < 5:
        raise ValueError("Manual checks require at least five dates per ticker")
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    for ticker in sorted(prices_by_ticker):
        prices = prices_by_ticker[ticker].reset_index(drop=True)
        dates = pd.to_datetime(prices["Date"], errors="coerce").dt.normalize()
        adjusted = pd.to_numeric(prices["Adjusted Close"], errors="coerce")
        ticker_features = features.loc[features["ticker"].eq(ticker)].set_index(
            "feature_date"
        )
        ticker_targets = targets.loc[targets["ticker"].eq(ticker)].set_index(
            "feature_date"
        )
        candidates = [
            position
            for position in range(horizon_days, len(prices) - horizon_days)
            if adjusted.iloc[[position - horizon_days, position, position + horizon_days]]
            .notna()
            .all()
            and (adjusted.iloc[[position - horizon_days, position, position + horizon_days]] > 0).all()
        ]
        if len(candidates) < sample_size:
            raise ValueError(f"{ticker}: fewer than {sample_size} valid manual-check dates")
        selected_positions = sorted(rng.choice(candidates, size=sample_size, replace=False))
        for position in selected_positions:
            feature_date = dates.iloc[position]
            before = adjusted.iloc[position - horizon_days]
            current = adjusted.iloc[position]
            after = adjusted.iloc[position + horizon_days]
            calculated_feature = current / before - 1.0
            calculated_target = after / current - 1.0
            stored_feature = ticker_features.loc[
                feature_date, f"return_{horizon_days}d"
            ]
            stored_target = ticker_targets.loc[feature_date, "target_21d_return"]
            check_ok = bool(
                np.isclose(calculated_feature, stored_feature)
                and np.isclose(calculated_target, stored_target)
            )
            rows.append(
                {
                    "ticker": ticker,
                    "feature_date": feature_date,
                    "adjusted_close_t": current,
                    "adjusted_close_t_minus_21": before,
                    "adjusted_close_t_plus_21": after,
                    "calculated_return_21d_feature": calculated_feature,
                    "calculated_target_21d": calculated_target,
                    "feature_source_start_date": dates.iloc[position - horizon_days],
                    "label_end_date": dates.iloc[position + horizon_days],
                    "source_row_positions": (
                        f"t-21={position-horizon_days};t={position};t+21={position+horizon_days}"
                    ),
                    "check_status": "PASS" if check_ok else "FAIL",
                }
            )
    return pd.DataFrame(rows, columns=MANUAL_CHECK_COLUMNS)
