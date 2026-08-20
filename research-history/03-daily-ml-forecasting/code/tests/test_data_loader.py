"""Offline tests for cache merging, field normalization, and audit checks."""

from __future__ import annotations

import pandas as pd

from src.data_loader import (
    DownloadResult,
    PRICE_COLUMNS,
    _merge_prices,
    audit_cached_prices,
)


def _prices(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "Open": 10.0,
            "High": 11.0,
            "Low": 9.0,
            "Close": 10.5,
            "Adjusted Close": 10.5,
            "Volume": 100,
        },
        columns=PRICE_COLUMNS,
    )


def test_merge_prices_deduplicates_and_sorts() -> None:
    cached = _prices(["2024-01-03", "2024-01-02"])
    incoming = _prices(["2024-01-03", "2024-01-04"])
    merged = _merge_prices(cached, incoming)
    assert merged["Date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
    ]


def test_audit_flags_short_history_and_bad_values(tmp_path) -> None:
    prices = _prices(["2024-01-02", "2024-01-03"])
    prices.loc[0, "Adjusted Close"] = 0
    prices.loc[1, "Volume"] = -1
    prices.assign(Date=prices["Date"].dt.strftime("%Y-%m-%d")).to_csv(
        tmp_path / "TEST.csv", index=False
    )
    metadata = pd.DataFrame(
        {
            "ticker": ["TEST"],
            "fund_name": ["Test ETF"],
            "listing_date": ["unknown"],
            "delisting_date": ["unknown"],
            "metadata_status": ["needs_review"],
            "point_in_time_coverage": ["unknown"],
        }
    )
    audit = audit_cached_prices(
        metadata,
        [DownloadResult("TEST", "downloaded")],
        raw_dir=tmp_path,
    ).iloc[0]
    assert not bool(audit["sufficient_60_month_history"])
    assert not bool(audit["adjusted_close_positive"])
    assert not bool(audit["volume_nonnegative"])


def test_audit_flags_long_consecutive_missing_block(tmp_path) -> None:
    prices = _prices(pd.bdate_range("2024-01-02", periods=10).strftime("%Y-%m-%d").tolist())
    prices.loc[2:8, "Adjusted Close"] = None
    prices.assign(Date=prices["Date"].dt.strftime("%Y-%m-%d")).to_csv(
        tmp_path / "GAP.csv", index=False
    )
    metadata = pd.DataFrame(
        {
            "ticker": ["GAP"],
            "fund_name": ["Gap ETF"],
            "listing_date": ["unknown"],
            "delisting_date": ["unknown"],
            "metadata_status": ["needs_review"],
            "point_in_time_coverage": ["unknown"],
        }
    )
    audit = audit_cached_prices(
        metadata,
        [DownloadResult("GAP", "downloaded")],
        raw_dir=tmp_path,
    ).iloc[0]
    assert bool(audit["large_missing_gap"])
    assert int(audit["max_consecutive_missing_rows"]) == 7
