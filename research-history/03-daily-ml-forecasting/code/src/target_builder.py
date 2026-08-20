"""Forward 21-trading-day return targets and feature/target alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import PRICE_COLUMNS
from src.feature_engineering import feature_columns


TARGET_COLUMNS = [
    "ticker",
    "feature_date",
    "label_start_date",
    "label_end_date",
    "current_adjusted_close",
    "future_adjusted_close",
    "target_21d_return",
    "target_available",
]


def build_targets(
    prices: pd.DataFrame,
    ticker: str,
    horizon_days: int = 21,
) -> pd.DataFrame:
    """Build a direct return target from the exact future trading-row offset.

    This is the only project module allowed to look forward.  Rows without the
    full horizon remain present with ``target_available=False``.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    missing = set(PRICE_COLUMNS) - set(prices.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    if "ticker" in prices.columns:
        input_tickers = set(prices["ticker"].dropna().astype(str).str.upper())
        if input_tickers and input_tickers != {ticker.upper()}:
            raise ValueError("build_targets accepts exactly one ticker per call")

    frame = prices.loc[:, PRICE_COLUMNS].copy().reset_index(drop=True)
    dates = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    if dates.isna().any() or not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError("Prices must have unique, strictly increasing dates")
    adjusted = pd.to_numeric(frame["Adjusted Close"], errors="coerce")

    future_adjusted = adjusted.shift(-horizon_days)
    label_end = dates.shift(-horizon_days)
    available = (
        adjusted.notna()
        & future_adjusted.notna()
        & adjusted.gt(0)
        & future_adjusted.gt(0)
        & label_end.notna()
    )
    target = future_adjusted / adjusted - 1.0
    target = target.where(available)

    result = pd.DataFrame(
        {
            "ticker": ticker.upper(),
            "feature_date": dates,
            "label_start_date": dates,
            "label_end_date": label_end,
            "current_adjusted_close": adjusted,
            "future_adjusted_close": future_adjusted,
            "target_21d_return": target,
            "target_available": available.astype(bool),
        },
        columns=TARGET_COLUMNS,
    )
    if len(result) >= horizon_days and result.tail(horizon_days)["target_available"].any():
        raise AssertionError("The final horizon rows cannot have available targets")
    return result


def merge_features_and_targets(
    features: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """One-to-one merge on ticker and feature_date without dropping tail rows."""
    model_features = feature_columns(features)
    target_like = {
        column
        for column in model_features
        if any(token in column.casefold() for token in ("target", "future", "label_"))
    }
    if target_like:
        raise ValueError(f"Target-like columns cannot be features: {sorted(target_like)}")
    if features.duplicated(["ticker", "feature_date"]).any():
        raise ValueError("Feature keys are not unique")
    if targets.duplicated(["ticker", "feature_date"]).any():
        raise ValueError("Target keys are not unique")
    merged = features.merge(
        targets,
        on=["ticker", "feature_date"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(merged) != len(features):
        raise AssertionError("Feature/target merge changed the feature row count")
    if merged["target_available"].isna().any():
        raise ValueError("Some feature rows did not match a target row")
    merged["target_available"] = merged["target_available"].astype(bool)
    numeric_targets = merged.loc[merged["target_available"], "target_21d_return"]
    if not np.isfinite(numeric_targets).all():
        raise ValueError("Available targets must be finite")
    return merged
