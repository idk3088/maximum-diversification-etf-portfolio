"""Regression tests for the Step 2 point-in-time sector universe outputs."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data" / "metadata" / "etf_metadata.csv"
CANDIDATES = ROOT / "data" / "metadata" / "clean_sector_etf_candidates.csv"
MONTHLY = ROOT / "results" / "monthly_eligible_universe.csv"
REPRESENTATIVES = ROOT / "results" / "monthly_sector_representatives.csv"
COVERAGE = ROOT / "results" / "monthly_sector_candidate_coverage.csv"
CHECKS = ROOT / "results" / "step2_automated_checks.csv"


CURRENT_SECTORS = {
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Health Care",
    "Financials",
    "Information Technology",
    "Communication Services",
    "Utilities",
    "Real Estate",
}


def test_metadata_and_clean_candidate_contract() -> None:
    metadata = pd.read_csv(METADATA, low_memory=False)
    candidates = pd.read_csv(CANDIDATES, low_memory=False)
    required = {
        "lifecycle_id",
        "ticker",
        "listing_date",
        "delisting_date",
        "instrument_type",
        "asset_class",
        "geographic_scope",
        "classification_level",
        "sector_or_industry",
        "classification_start",
        "classification_end",
        "leveraged",
        "inverse",
        "single_stock",
        "etn",
        "thematic",
        "source",
        "classification_evidence",
        "review_status",
    }
    assert required <= set(metadata.columns)
    assert len(metadata) == 8207
    assert len(candidates) == 63
    assert set(candidates["sector_or_industry"]) == CURRENT_SECTORS
    assert (candidates["instrument_type"] == "ETF").all()
    assert (candidates["asset_class"] == "equity").all()
    assert (candidates["geographic_scope"] == "US domestic").all()
    assert (candidates["classification_level"] == "sector").all()
    for column in ["leveraged", "inverse", "single_stock", "etn", "thematic"]:
        assert ~candidates[column].astype(bool).any()


def test_monthly_universe_is_complete_and_point_in_time() -> None:
    monthly = pd.read_csv(MONTHLY, low_memory=False)
    assert len(monthly) == 7560
    assert monthly["selection_date"].nunique() == 120
    assert monthly.groupby("selection_date")["ticker"].nunique().eq(63).all()
    assert not monthly.duplicated(["selection_date", "ticker"]).any()

    eligible = monthly[monthly["eligible"].astype(bool)].copy()
    assert (eligible["available_history_months"] >= 60).all()
    assert pd.to_datetime(eligible["listing_date"]).le(
        pd.to_datetime(eligible["selection_date"])
    ).all()
    assert pd.to_datetime(eligible["ADV_window_end"]).le(
        pd.to_datetime(eligible["selection_date"])
    ).all()
    assert pd.to_datetime(eligible["effective_trade_date"]).gt(
        pd.to_datetime(eligible["selection_date"])
    ).all()


def test_representatives_are_maximum_adv_and_have_full_coverage() -> None:
    monthly = pd.read_csv(MONTHLY, low_memory=False)
    reps = pd.read_csv(REPRESENTATIVES, low_memory=False)
    coverage = pd.read_csv(COVERAGE, low_memory=False)
    eligible = monthly[monthly["eligible"].astype(bool)]
    maximum = eligible.groupby(
        ["selection_date", "historical_sector_name"], as_index=False
    )["ADV20"].max()
    joined = reps.merge(
        maximum,
        left_on=["selection_date", "classification_name"],
        right_on=["selection_date", "historical_sector_name"],
        suffixes=("", "_max"),
        validate="one_to_one",
    )
    assert len(reps) == 1320
    assert not reps.duplicated(["selection_date", "classification_name"]).any()
    assert np.allclose(joined["ADV20"], joined["ADV20_max"], rtol=0.0, atol=0.0)
    assert coverage["coverage_available"].astype(bool).all()
    assert coverage.groupby("selection_date")["classification_name"].nunique().eq(11).all()


def test_gics_transitions_are_not_backcast() -> None:
    coverage = pd.read_csv(COVERAGE, low_memory=False)
    august_2016 = set(
        coverage.loc[coverage["selection_date"] == "2016-08-31", "classification_name"]
    )
    august_2018 = set(
        coverage.loc[coverage["selection_date"] == "2018-08-31", "classification_name"]
    )
    september_2018 = set(
        coverage.loc[coverage["selection_date"] == "2018-09-28", "classification_name"]
    )
    assert "Real Estate" in august_2016
    assert "Telecommunication Services" in august_2018
    assert "Communication Services" not in august_2018
    # The 2018-09-28 selection becomes tradable on 2018-10-01, when the new
    # Communication Services classification is effective.
    assert "Communication Services" in september_2018
    assert "Telecommunication Services" not in september_2018


def test_known_non_sector_products_are_excluded_and_all_checks_pass() -> None:
    candidates = pd.read_csv(CANDIDATES, low_memory=False)
    checks = pd.read_csv(CHECKS)
    prohibited_examples = {"IYZ", "XBI", "XSD", "SOXX", "SMH", "FRI"}
    assert prohibited_examples.isdisjoint(set(candidates["ticker"]))
    assert len(checks) == 17
    assert (checks["status"] == "PASS").all()
