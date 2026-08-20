"""Offline tests for point-in-time lifecycle handling and unknown metadata."""

from __future__ import annotations

import pandas as pd

from src.universe_builder import eligible_lifecycles_as_of


def test_unknown_listing_date_is_never_assumed_eligible() -> None:
    metadata = pd.DataFrame(
        {
            "ticker": ["KNOWN", "UNKNOWN"],
            "listing_date": ["2015-01-02", "unknown"],
            "delisting_date": ["unknown", "unknown"],
        }
    )
    eligible = eligible_lifecycles_as_of(metadata, "2020-01-02")
    assert eligible["ticker"].tolist() == ["KNOWN"]


def test_delisted_ticker_is_only_eligible_before_delisting() -> None:
    metadata = pd.DataFrame(
        {
            "ticker": ["OLD"],
            "listing_date": ["2012-01-03"],
            "delisting_date": ["2019-06-30"],
        }
    )
    assert len(eligible_lifecycles_as_of(metadata, "2018-12-31")) == 1
    assert eligible_lifecycles_as_of(metadata, "2020-01-02").empty
