"""Integrity checks for the generated clean-universe coverage audit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "metadata"
GICS_SECTORS = {
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


def test_clean_detail_does_not_force_broad_funds_into_sectors() -> None:
    detail = pd.read_csv(OUTPUT_DIR / "clean_universe_detail.csv")
    assert len(detail) == 22
    assert detail["ticker"].is_unique
    assert set(detail["broad_market_or_sector"]) == {"broad_market"}
    assert set(detail["likely_sector"]) == {"not_applicable_broad_market"}
    assert detail["evidence_source"].str.startswith("https://").all()


def test_current_and_monthly_coverage_have_complete_grids() -> None:
    current = pd.read_csv(OUTPUT_DIR / "current_sector_coverage.csv")
    monthly = pd.read_csv(OUTPUT_DIR / "monthly_sector_coverage_audit.csv")
    assert set(current["sector"]) == GICS_SECTORS
    assert current["clean_candidate_count"].eq(0).all()
    assert not monthly.duplicated(["month_end", "sector"]).any()
    assert monthly.groupby("month_end")["sector"].nunique().eq(11).all()
    assert set(monthly["sector"]) == GICS_SECTORS
    assert monthly["available_sector_count_in_month"].eq(0).all()


def test_priority_review_is_narrow_and_one_per_missing_sector() -> None:
    priority = pd.read_csv(OUTPUT_DIR / "priority_review_candidates.csv")
    assert len(priority) <= 200
    assert set(priority["target_sector"]) == GICS_SECTORS
    assert priority.groupby("target_sector").size().le(20).all()
    assert priority["ticker"].is_unique
    assert priority["vendor_conflict_type"].eq("none").all()
    assert priority["official_source_available"].astype(str).str.lower().eq("true").all()
    assert priority["first_eligible_month_end"].ne("unknown").all()


def test_resolved_vendor_conflicts_remain_conservative() -> None:
    conflicts = pd.read_csv(OUTPUT_DIR / "vendor_conflicts_resolved_v2.csv")
    resolved = conflicts[conflicts["resolved_v2"].astype(str).str.lower().eq("true")]
    prohibited = {"ticker_reuse_overlapping", "entity_type_conflict", "unresolved_lifecycle_conflict"}
    assert not set(resolved["resolution_category"]) & prohibited
    nonoverlap = resolved[resolved["resolution_category"].eq("ticker_reuse_non_overlapping")]
    assert nonoverlap["usable_price_start_date"].ne("unknown").all()
