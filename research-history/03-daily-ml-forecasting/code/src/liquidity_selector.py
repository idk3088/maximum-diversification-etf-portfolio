"""Trailing-dollar-ADV calculations and deterministic sector selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AdvResult:
    """A point-in-time ADV calculation and its validation metadata."""

    adv20: float | None
    window_start: pd.Timestamp | None
    window_end: pd.Timestamp | None
    valid_observations: int
    price_data_valid: bool
    reason: str


def calculate_adv_as_of(
    prices: pd.DataFrame,
    selection_date: pd.Timestamp,
    window_days: int = 20,
) -> AdvResult:
    """Calculate mean(Close * Volume) using only rows through selection_date."""
    frame = prices.copy()
    if frame.empty:
        return AdvResult(None, None, None, 0, False, "no_cached_prices")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame["Volume"] = pd.to_numeric(frame["Volume"], errors="coerce")
    frame = frame[frame["Date"].le(selection_date)].sort_values("Date")
    if frame["Date"].duplicated().any():
        return AdvResult(None, None, None, 0, False, "duplicate_price_dates")
    window = frame.tail(window_days)
    if len(window) < window_days:
        return AdvResult(None, None, None, len(window), False, "fewer_than_20_observations")
    if window["Date"].iloc[-1] != selection_date:
        return AdvResult(
            None,
            window["Date"].iloc[0],
            window["Date"].iloc[-1],
            len(window),
            False,
            "no_valid_price_on_selection_date",
        )
    valid = (
        window["Close"].notna()
        & window["Volume"].notna()
        & window["Close"].gt(0)
        & window["Volume"].ge(0)
    )
    if not valid.all():
        return AdvResult(
            None,
            window["Date"].iloc[0],
            window["Date"].iloc[-1],
            int(valid.sum()),
            False,
            "invalid_close_or_volume_in_adv_window",
        )
    dollar_volume = window["Close"] * window["Volume"]
    adv = float(np.mean(dollar_volume.to_numpy(dtype=float)))
    return AdvResult(
        adv,
        window["Date"].iloc[0],
        window["Date"].iloc[-1],
        window_days,
        True,
        "",
    )


def select_highest_adv(candidates: pd.DataFrame) -> tuple[pd.Series, bool, str]:
    """Select the highest ADV candidate with the documented tie-break order."""
    if candidates.empty:
        raise ValueError("Cannot select from an empty candidate set")
    ordered = candidates.sort_values(
        ["ADV20", "adv_valid_observations", "available_history_months", "ticker"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    top_adv = ordered["ADV20"].iloc[0]
    tied = ordered[np.isclose(ordered["ADV20"], top_adv, rtol=0.0, atol=0.0)]
    if len(tied) == 1:
        return ordered.iloc[0], False, ""
    top = tied.iloc[0]
    reason = (
        "ADV20 tied exactly; selected by valid ADV observations, then longer "
        "listing history, then ticker alphabetical order"
    )
    return top, True, reason
