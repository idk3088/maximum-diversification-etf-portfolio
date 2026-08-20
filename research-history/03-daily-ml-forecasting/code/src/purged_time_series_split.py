"""Chronological walk-forward folds purged by observed label end dates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


REQUIRED_TIME_COLUMNS = {
    "ticker",
    "feature_date",
    "label_end_date",
    "target_21d_return",
    "target_available",
}


@dataclass(frozen=True)
class PurgedFold:
    """Positional indices and audit metadata for one chronological fold."""

    fold_id: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_row_count_before_purge: int
    train_row_count_after_purge: int
    purged_row_count: int
    purge_valid: bool
    fold_status: str


def validate_temporal_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a date-normalized copy after strict single-ticker validation."""
    missing = REQUIRED_TIME_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing temporal columns: {sorted(missing)}")
    validated = frame.copy()
    validated["feature_date"] = pd.to_datetime(
        validated["feature_date"], errors="coerce"
    ).dt.normalize()
    validated["label_end_date"] = pd.to_datetime(
        validated["label_end_date"], errors="coerce"
    ).dt.normalize()
    if validated["feature_date"].isna().any():
        raise ValueError("feature_date contains invalid values")
    if validated["ticker"].nunique(dropna=False) != 1:
        raise ValueError("A purged splitter frame must contain exactly one ticker")
    if not validated["feature_date"].is_monotonic_increasing:
        raise ValueError("feature_date must be increasing")
    if validated["feature_date"].duplicated().any():
        raise ValueError("feature_date must be unique within ticker")
    return validated.reset_index(drop=True)


def available_rows_as_of(
    dataset: pd.DataFrame,
    *,
    ticker: str,
    selection_date: str | pd.Timestamp,
    training_years: int,
) -> pd.DataFrame:
    """Apply the point-in-time label-availability and trailing-window rules."""
    if training_years <= 0:
        raise ValueError("training_years must be positive")
    selection = pd.Timestamp(selection_date).normalize()
    lower_bound = selection - pd.DateOffset(years=training_years)
    frame = dataset.loc[dataset["ticker"].astype(str).str.upper().eq(ticker.upper())].copy()
    frame["feature_date"] = pd.to_datetime(frame["feature_date"], errors="coerce").dt.normalize()
    frame["label_end_date"] = pd.to_datetime(frame["label_end_date"], errors="coerce").dt.normalize()
    available = frame.loc[
        frame["feature_date"].between(lower_bound, selection, inclusive="both")
        & frame["label_end_date"].le(selection)
        & frame["target_available"].astype(bool)
        & pd.to_numeric(frame["target_21d_return"], errors="coerce").notna()
    ].copy()
    return validate_temporal_frame(available.sort_values("feature_date"))


def actual_month_end_selection_dates(
    trading_dates: pd.Series,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> list[pd.Timestamp]:
    """Choose the last observed trading date in each requested calendar month."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("start_date must not be later than end_date")
    dates = pd.DatetimeIndex(pd.to_datetime(trading_dates, errors="coerce").dropna().unique())
    dates = dates.sort_values()
    selections: list[pd.Timestamp] = []
    for period in pd.period_range(start=start.to_period("M"), end=end.to_period("M"), freq="M"):
        candidates = dates[(dates.to_period("M") == period) & (dates <= end)]
        if len(candidates):
            selections.append(pd.Timestamp(candidates.max()).normalize())
    return selections


class PurgedWalkForwardSplitter:
    """Build shared chronological folds with label-interval purge.

    ``validation_days`` and ``embargo_days`` count observed trading rows.  The
    overlap guard itself never relies on a fixed row deletion: training rows
    are retained only when ``label_end_date < validation_start_date``.
    """

    def __init__(
        self,
        *,
        n_splits: int,
        validation_days: int,
        min_train_rows: int,
        mode: str = "rolling",
        max_train_rows: int | None = None,
        embargo_days: int = 0,
    ) -> None:
        if n_splits <= 0 or validation_days <= 0 or min_train_rows <= 0:
            raise ValueError("n_splits, validation_days and min_train_rows must be positive")
        if mode not in {"rolling", "expanding"}:
            raise ValueError("mode must be rolling or expanding")
        if mode == "rolling" and (max_train_rows is None or max_train_rows <= 0):
            raise ValueError("rolling mode requires positive max_train_rows")
        if embargo_days < 0:
            raise ValueError("embargo_days cannot be negative")
        self.n_splits = n_splits
        self.validation_days = validation_days
        self.min_train_rows = min_train_rows
        self.mode = mode
        self.max_train_rows = max_train_rows
        self.embargo_days = embargo_days

    def split(self, frame: pd.DataFrame) -> Iterator[PurgedFold]:
        data = validate_temporal_frame(frame)
        validation_total = self.n_splits * self.validation_days
        first_validation = len(data) - validation_total
        if first_validation <= 0:
            return
        embargoed_positions: set[int] = set()
        for fold_id in range(self.n_splits):
            validation_start_pos = first_validation + fold_id * self.validation_days
            validation_end_pos = min(
                validation_start_pos + self.validation_days, len(data)
            )
            validation_indices = np.arange(validation_start_pos, validation_end_pos)
            candidate_positions = np.arange(0, validation_start_pos)
            if embargoed_positions:
                candidate_positions = np.array(
                    [position for position in candidate_positions if position not in embargoed_positions],
                    dtype=int,
                )
            if self.mode == "rolling" and self.max_train_rows is not None:
                candidate_positions = candidate_positions[-self.max_train_rows :]

            validation_start_date = data.loc[validation_start_pos, "feature_date"]
            candidate_label_end = data.loc[candidate_positions, "label_end_date"]
            train_indices = candidate_positions[
                candidate_label_end.lt(validation_start_date).to_numpy()
            ]
            purge_valid = bool(
                len(train_indices) == 0
                or data.loc[train_indices, "label_end_date"].lt(validation_start_date).all()
            )
            status = (
                "ok"
                if purge_valid and len(train_indices) >= self.min_train_rows
                else "insufficient_train_data"
            )
            yield PurgedFold(
                fold_id=fold_id,
                train_indices=train_indices,
                validation_indices=validation_indices,
                train_row_count_before_purge=len(candidate_positions),
                train_row_count_after_purge=len(train_indices),
                purged_row_count=len(candidate_positions) - len(train_indices),
                purge_valid=purge_valid,
                fold_status=status,
            )

            if self.embargo_days:
                embargoed_positions.update(
                    range(
                        validation_end_pos,
                        min(validation_end_pos + self.embargo_days, len(data)),
                    )
                )


def fold_definition_record(
    frame: pd.DataFrame,
    fold: PurgedFold,
    *,
    ticker: str,
    selection_date: pd.Timestamp,
) -> dict[str, object]:
    """Convert one fold to the required auditable output schema."""
    train = frame.iloc[fold.train_indices]
    validation = frame.iloc[fold.validation_indices]

    def date_value(series: pd.Series, operation: str) -> str:
        if series.empty:
            return "unknown"
        value = getattr(pd.to_datetime(series), operation)()
        return value.date().isoformat() if pd.notna(value) else "unknown"

    return {
        "ticker": ticker,
        "selection_date": selection_date.date().isoformat(),
        "outer_fold_id": fold.fold_id,
        "train_feature_start": date_value(train["feature_date"], "min"),
        "train_feature_end": date_value(train["feature_date"], "max"),
        "train_label_end_max": date_value(train["label_end_date"], "max"),
        "validation_feature_start": date_value(validation["feature_date"], "min"),
        "validation_feature_end": date_value(validation["feature_date"], "max"),
        "validation_label_end_max": date_value(validation["label_end_date"], "max"),
        "train_row_count_before_purge": fold.train_row_count_before_purge,
        "train_row_count_after_purge": fold.train_row_count_after_purge,
        "purged_row_count": fold.purged_row_count,
        "validation_row_count": len(validation),
        "purge_valid": fold.purge_valid,
        "fold_status": fold.fold_status,
    }
