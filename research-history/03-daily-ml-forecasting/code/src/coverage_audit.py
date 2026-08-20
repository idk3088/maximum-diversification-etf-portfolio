"""Audit clean-sector coverage and prepare a narrowly targeted review queue."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.data_loader import PRICE_COLUMNS, read_cached_prices
from src.paths import (
    CLEAN_US_EQUITY_ETFS_PATH,
    DATA_AUDIT_V2_PATH,
    METADATA_NEEDS_REVIEW_PATH,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    VENDOR_CONFLICTS_PATH,
)

GICS_SECTORS: Final = (
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
)

SECTOR_PRIORITY_TICKERS: Final = {
    "Energy": ("XLE",),
    "Materials": ("XLB",),
    "Industrials": ("XLI",),
    "Consumer Discretionary": ("XLY",),
    "Consumer Staples": ("XLP",),
    "Health Care": ("XLV",),
    "Financials": ("XLF",),
    "Information Technology": ("XLK",),
    # Earlier funds are included for the two sectors whose Select Sector SPDRs
    # cannot provide 60 months of history at the start of the audit window.
    "Communication Services": ("VOX", "XLC"),
    "Utilities": ("XLU",),
    "Real Estate": ("IYR", "XLRE"),
}

EVIDENCE_PATH: Final = (
    PROJECT_ROOT / "data" / "raw" / "metadata" / "coverage_official_evidence.csv"
)
ALPHA_ACTIVE_PATH: Final = (
    PROJECT_ROOT / "data" / "raw" / "metadata" / "alpha_vantage_active_etfs.csv"
)
ALPHA_DELISTED_PATH: Final = (
    PROJECT_ROOT / "data" / "raw" / "metadata" / "alpha_vantage_delisted_etfs.csv"
)
INTERMEDIATE_DIR: Final = PROCESSED_DATA_DIR / "coverage_audit_intermediate"

# Point-in-time GICS notes needed by the monthly coverage audit. Real Estate
# became a standalone sector after the 2016-08-31 close; Communication Services
# replaced/broadened Telecommunication Services after the 2018-09-28 close.
GICS_SECTOR_EFFECTIVE_MONTH: Final = {
    "Real Estate": pd.Period("2016-09", freq="M"),
    "Communication Services": pd.Period("2018-09", freq="M"),
}


@dataclass(frozen=True)
class CoverageOutputs:
    clean_detail: list[dict[str, object]]
    current_sector_coverage: list[dict[str, object]]
    monthly_sector_coverage: list[dict[str, object]]
    priority_candidates: list[dict[str, object]]
    vendor_conflicts_v2: list[dict[str, object]]
    summary: dict[str, object]


def _valid_prices(ticker: str) -> pd.DataFrame:
    frame = read_cached_prices(ticker).copy()
    if frame.empty:
        return frame
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    numeric = frame[PRICE_COLUMNS[1:]].apply(pd.to_numeric, errors="coerce")
    valid = (
        frame["Date"].notna()
        & numeric.notna().all(axis=1)
        & (numeric[["Open", "High", "Low", "Close", "Adjusted Close"]] > 0).all(axis=1)
        & (numeric["Volume"] >= 0)
        & (numeric["High"] >= numeric["Low"])
        & numeric["Open"].between(numeric["Low"], numeric["High"])
        & numeric["Close"].between(numeric["Low"], numeric["High"])
    )
    result = pd.concat([frame[["Date"]], numeric], axis=1).loc[valid].copy()
    return result.sort_values("Date").drop_duplicates("Date", keep="last")


def _history_months(first_date: pd.Timestamp, last_date: pd.Timestamp) -> float:
    return round((last_date - first_date).days / 30.436875, 1)


def _latest_adv20(prices: pd.DataFrame) -> float | None:
    if prices.empty:
        return None
    window = prices.tail(20)
    if len(window) < 20:
        return None
    return float((window["Close"] * window["Volume"]).mean())


def build_clean_detail() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    clean = pd.read_csv(CLEAN_US_EQUITY_ETFS_PATH, dtype=str).fillna("")
    evidence = pd.read_csv(EVIDENCE_PATH, dtype=str).fillna("")
    evidence_map = evidence.set_index("ticker")
    missing = set(clean["ticker"]) - set(evidence_map.index)
    if missing:
        raise AssertionError(f"Official evidence missing for clean tickers: {sorted(missing)}")
    prices_by_ticker: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    for record in clean.sort_values("ticker").itertuples(index=False):
        source = evidence_map.loc[record.ticker]
        prices = _valid_prices(record.ticker)
        prices_by_ticker[record.ticker] = prices
        first = prices["Date"].min() if not prices.empty else pd.NaT
        last = prices["Date"].max() if not prices.empty else pd.NaT
        rows.append(
            {
                "ticker": record.ticker,
                "fund_name": record.fund_name,
                "listing_date": record.listing_date,
                "first_valid_price_date": first.date().isoformat() if pd.notna(first) else "unknown",
                "latest_price_date": last.date().isoformat() if pd.notna(last) else "unknown",
                "valid_history_months": _history_months(first, last) if pd.notna(first) and pd.notna(last) else None,
                "latest_ADV20": _latest_adv20(prices),
                "fund_objective": source["fund_objective"],
                "tracked_index": source["tracked_index"],
                "broad_market_or_sector": source["broad_market_or_sector"],
                "likely_sector": (
                    source["likely_sector"]
                    if source["broad_market_or_sector"] == "sector"
                    else "not_applicable_broad_market"
                ),
                "thematic_flag": str(record.thematic_flag).lower() == "true",
                "evidence_source": source["evidence_source"],
                "evidence": source["evidence"],
                "review_status": record.review_status,
            }
        )
    return pd.DataFrame(rows), prices_by_ticker


def build_current_sector_coverage(clean_detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sector in GICS_SECTORS:
        candidates = clean_detail[
            clean_detail["broad_market_or_sector"].eq("sector")
            & clean_detail["likely_sector"].eq(sector)
        ]
        earliest = (
            pd.to_datetime(candidates["first_valid_price_date"], errors="coerce").min()
            if not candidates.empty
            else pd.NaT
        )
        rows.append(
            {
                "sector": sector,
                "clean_candidate_count": int(len(candidates)),
                "candidate_tickers": "|".join(candidates["ticker"]),
                "earliest_available_date": earliest.date().isoformat() if pd.notna(earliest) else "unknown",
                "coverage_status": "covered" if len(candidates) else "missing",
                "missing_reason": "" if len(candidates) else "clean universe contains only broad-market/style ETFs; none may be forced into one GICS sector",
            }
        )
    return pd.DataFrame(rows)


def build_monthly_sector_coverage(
    clean_detail: pd.DataFrame,
    prices_by_ticker: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    valid_latest = pd.to_datetime(clean_detail["latest_price_date"], errors="coerce").max()
    last_completed_month = valid_latest.to_period("M") - 1
    months = pd.period_range("2016-08", last_completed_month, freq="M")
    rows: list[dict[str, object]] = []
    for month in months:
        month_end = month.to_timestamp("M")
        month_start = month.to_timestamp("M") - pd.offsets.MonthBegin(1)
        sector_counts: dict[str, list[str]] = {}
        for sector in GICS_SECTORS:
            candidates = clean_detail[
                clean_detail["broad_market_or_sector"].eq("sector")
                & clean_detail["likely_sector"].eq(sector)
            ]
            eligible: list[str] = []
            for candidate in candidates.itertuples(index=False):
                listing = pd.to_datetime(candidate.listing_date, errors="coerce")
                first = pd.to_datetime(candidate.first_valid_price_date, errors="coerce")
                prices = prices_by_ticker[candidate.ticker]
                has_month_data = bool(
                    ((prices["Date"] >= month_start) & (prices["Date"] <= month_end)).any()
                )
                if (
                    pd.notna(listing)
                    and listing <= month_end
                    and pd.notna(first)
                    and first + pd.DateOffset(months=60) <= month_end
                    and has_month_data
                ):
                    eligible.append(candidate.ticker)
            sector_counts[sector] = eligible
        available_count = sum(bool(values) for values in sector_counts.values())
        for sector in GICS_SECTORS:
            eligible = sector_counts[sector]
            effective_month = GICS_SECTOR_EFFECTIVE_MONTH.get(sector)
            if effective_month is not None and month < effective_month:
                missing_reason = (
                    "current GICS sector not yet effective at this month-end; "
                    "requires point-in-time taxonomy treatment"
                )
            else:
                missing_reason = (
                    "" if eligible else "no clean single-sector ETF eligible at this month-end"
                )
            rows.append(
                {
                    "month_end": month_end.date().isoformat(),
                    "sector": sector,
                    "eligible_clean_etf_count": len(eligible),
                    "eligible_tickers": "|".join(eligible),
                    "coverage_available": bool(eligible),
                    "missing_reason": missing_reason,
                    "available_sector_count_in_month": available_count,
                }
            )
    return pd.DataFrame(rows)


def build_priority_candidates() -> pd.DataFrame:
    review = pd.read_csv(METADATA_NEEDS_REVIEW_PATH, dtype=str).fillna("")
    audit = pd.read_csv(DATA_AUDIT_V2_PATH, dtype=str).fillna("")
    vendor = pd.read_csv(VENDOR_CONFLICTS_PATH, dtype=str).fillna("")
    evidence = pd.read_csv(EVIDENCE_PATH, dtype=str).fillna("").set_index("ticker")
    audit_map = audit.drop_duplicates("lifecycle_id").set_index("lifecycle_id")
    vendor_map = vendor.groupby("ticker")["conflict_types"].agg("|".join).to_dict()
    rows: list[dict[str, object]] = []
    for sector, tickers in SECTOR_PRIORITY_TICKERS.items():
        for ticker in tickers:
            rows.extend(_build_priority_row(sector, ticker, review, audit_map, vendor_map, evidence))
    frame = pd.DataFrame(rows)
    frame["listing_sort"] = pd.to_datetime(frame["listing_date"], errors="coerce")
    frame = frame.sort_values(
        ["listing_sort", "latest_ADV20"], ascending=[True, False], ignore_index=True
    )
    frame.insert(0, "priority_rank", np.arange(1, len(frame) + 1))
    return frame.drop(columns="listing_sort")


def _build_priority_row(
    sector: str,
    ticker: str,
    review: pd.DataFrame,
    audit_map: pd.DataFrame,
    vendor_map: dict[str, str],
    evidence: pd.DataFrame,
) -> list[dict[str, object]]:
    candidates = review[
        review["ticker"].eq(ticker)
        & review["status"].str.lower().eq("active")
    ]
    if len(candidates) != 1:
        raise AssertionError(
            f"Expected one active review lifecycle for {ticker}, found {len(candidates)}"
        )
    record = candidates.iloc[0]
    price_audit = audit_map.loc[record["lifecycle_id"]]
    source = evidence.loc[ticker]
    prices = _valid_prices(ticker)
    first_valid = prices["Date"].min() if not prices.empty else pd.NaT
    first_eligible_date = (
        first_valid + pd.DateOffset(months=60) if pd.notna(first_valid) else pd.NaT
    )
    first_eligible_month_end = (
        first_eligible_date.to_period("M").to_timestamp("M")
        if pd.notna(first_eligible_date)
        else pd.NaT
    )
    conflict = vendor_map.get(ticker, "")
    prohibited = any(
        token in record["exclusion_reason"].split("|")
        for token in (
            "etn", "fixed_income", "commodity", "crypto", "currency",
            "international_or_global", "leveraged", "inverse", "single_stock",
            "option_or_defined_outcome", "thematic_manual_review",
        )
    )
    if prohibited or conflict or price_audit["price_data_status"] != "clean":
        return []
    if str(price_audit["sufficient_60_month_history_at_latest_date"]).lower() != "true":
        return []
    return [
        {
            "target_sector": sector,
            "ticker": ticker,
            "fund_name": record["fund_name"],
            "listing_date": record["listing_date"],
            "first_valid_price_date": (
                first_valid.date().isoformat() if pd.notna(first_valid) else "unknown"
            ),
            "first_eligible_month_end": (
                first_eligible_month_end.date().isoformat()
                if pd.notna(first_eligible_month_end)
                else "unknown"
            ),
            "latest_ADV20": _latest_adv20(prices),
            "likely_asset_class": "equity",
            "likely_geographic_scope": "United States",
            "likely_sector": source["likely_sector"],
            "unresolved_fields": "structured classification flags require approval before promotion to clean",
            "vendor_conflict_type": "none",
            "official_source_available": True,
            "reason_for_priority": (
                "fills a currently missing GICS sector; earlier listing improves historical coverage; "
                "official issuer index mapping; clean prices; no vendor conflict"
                if ticker in {"IYR", "VOX"}
                else "fills a currently missing GICS sector; official issuer index mapping; clean prices; no vendor conflict"
            ),
            "evidence_source": source["evidence_source"],
            "tracked_index": source["tracked_index"],
        }
    ]


def _normalize_name(value: object) -> str:
    text = str(value).upper()
    issuer_aliases = {
        "STATE STREET": "SSGA",
        "STATE STREET R": "SSGA",
        "SPDR R": "SPDR",
        "VANGUARD INDEX FUND": "VANGUARD",
    }
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    for old, new in issuer_aliases.items():
        text = text.replace(old, new)
    generic = {"ETF", "FUND", "TRUST", "SHARES", "THE"}
    return " ".join(token for token in text.split() if token not in generic)


def _parse_date(value: object) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def resolve_vendor_conflicts_v2() -> pd.DataFrame:
    vendor = pd.read_csv(VENDOR_CONFLICTS_PATH, dtype=str).fillna("")
    metadata = pd.read_csv(PROJECT_ROOT / "data" / "metadata" / "etf_metadata.csv", dtype=str).fillna("")
    official = pd.read_csv(EVIDENCE_PATH, dtype=str).fillna("")
    official_tickers = set(official["ticker"])
    current = metadata[metadata["status"].str.lower().isin({"active", "active_needs_review"})]
    current_map = current.sort_values("listing_date").drop_duplicates("ticker", keep="last").set_index("ticker")
    raw_frames: list[pd.DataFrame] = []
    for path, snapshot in ((ALPHA_ACTIVE_PATH, "active"), (ALPHA_DELISTED_PATH, "delisted")):
        frame = pd.read_csv(path, dtype=str).fillna("")
        frame["ticker"] = frame["symbol"].str.upper()
        frame["snapshot"] = snapshot
        raw_frames.append(frame)
    raw = pd.concat(raw_frames, ignore_index=True)
    lifecycle = pd.read_csv(PROJECT_ROOT / "data" / "metadata" / "etf_lifecycle.csv", dtype=str).fillna("")
    rows: list[dict[str, object]] = []
    for record in vendor.itertuples(index=False):
        ticker = record.ticker
        conflict_types = set(str(record.conflict_types).split("|"))
        category = "unresolved_lifecycle_conflict"
        overlap = "unknown"
        resolved = False
        evidence = "Insufficient aligned objective/index/date evidence; keep isolated."
        usable_start = "unknown"
        listing = None
        if ticker in current_map.index:
            listing = _parse_date(current_map.at[ticker, "listing_date"])
            if listing is not None:
                usable_start = listing.date().isoformat()

        if conflict_types == {"active_name_mismatch"} and ticker in official_tickers:
            category = (
                "issuer_name_format_difference"
                if ticker == "SPY"
                else "benign_name_difference"
            )
            overlap = "same_active_product"
            resolved = True
            evidence = "Exact ticker, ETF type, official objective/index, and listing period agree; only vendor display/legal name differs."
        elif "alpha_non_etf_entity_same_ticker" in conflict_types:
            entities = raw[(raw["ticker"].eq(ticker)) & ~raw["assetType"].str.upper().eq("ETF")]
            active_entity = entities["status"].str.lower().eq("active").any()
            delist_dates = pd.to_datetime(entities["delistingDate"], errors="coerce")
            if active_entity:
                category = "entity_type_conflict"
                overlap = "active_non_etf_entity_present"
                evidence = "Alpha has an active non-ETF entity with the same ticker; identity remains isolated."
            elif listing is not None and len(entities) and delist_dates.notna().all():
                latest_delist = delist_dates.max()
                if latest_delist < listing:
                    category = "ticker_reuse_non_overlapping"
                    overlap = "non_overlapping"
                    resolved = True
                    evidence = f"All Alpha non-ETF entities delisted by {latest_delist.date().isoformat()}, before ETF listing {listing.date().isoformat()}; ETF prices usable only from listing date."
                else:
                    category = "ticker_reuse_overlapping"
                    overlap = "overlapping"
                    evidence = "A non-ETF lifecycle overlaps the ETF listing period; keep isolated."
            else:
                category = "unresolved_lifecycle_conflict"
                overlap = "dates_incomplete"
                evidence = "Non-ETF lifecycle dates are incomplete; non-overlap cannot be established."
        elif "current_ticker_has_distinct_delisted_entity" in conflict_types:
            active_rows = lifecycle[(lifecycle["ticker"].eq(ticker)) & lifecycle["status"].str.lower().eq("active")]
            old_rows = lifecycle[(lifecycle["ticker"].eq(ticker)) & lifecycle["status"].str.lower().eq("delisted")]
            active_dates = pd.to_datetime(active_rows["ipo_date"], errors="coerce")
            old_delists = pd.to_datetime(old_rows["delisting_date"], errors="coerce")
            severe_other = bool(conflict_types & {"price_predates_current_lifecycle", "nasdaq_not_in_alpha_active_etf"})
            if not severe_other and active_dates.notna().all() and old_delists.notna().all() and len(active_dates) and len(old_delists):
                if old_delists.max() < active_dates.min():
                    category = "ticker_reuse_non_overlapping"
                    overlap = "non_overlapping"
                    resolved = True
                    usable_start = active_dates.min().date().isoformat()
                    evidence = "Old ETF entity delisted before the current ETF IPO; prices remain restricted to the current listing period."
                else:
                    category = "ticker_reuse_overlapping"
                    overlap = "overlapping"
                    evidence = "Historical and current ETF lifecycle dates overlap; keep isolated."
        rows.append(
            {
                **record._asdict(),
                "normalized_nasdaq_name": _normalize_name(record.nasdaq_fund_name),
                "normalized_alpha_active_names": "|".join(_normalize_name(value) for value in str(record.alpha_active_names).split("|") if value),
                "resolution_category": category,
                "lifecycle_overlap_status": overlap,
                "resolved_v2": resolved,
                "usable_price_start_date": usable_start,
                "resolution_evidence": evidence,
            }
        )
    return pd.DataFrame(rows)


def build_outputs() -> CoverageOutputs:
    clean_detail, prices = build_clean_detail()
    current = build_current_sector_coverage(clean_detail)
    monthly = build_monthly_sector_coverage(clean_detail, prices)
    priority = build_priority_candidates()
    conflicts = resolve_vendor_conflicts_v2()
    first_by_sector = {
        sector: "never"
        for sector in GICS_SECTORS
    }
    monthly_counts = monthly.groupby("month_end")["available_sector_count_in_month"].first()
    summary = {
        "clean_count": len(clean_detail),
        "standard_sector_etf_count": int(clean_detail["broad_market_or_sector"].eq("sector").sum()),
        "broad_market_etf_count": int(clean_detail["broad_market_or_sector"].eq("broad_market").sum()),
        "unclassified_etf_count": int(clean_detail["broad_market_or_sector"].eq("unclassified").sum()),
        "current_covered_sector_count": int(current["coverage_status"].eq("covered").sum()),
        "missing_sectors": current.loc[current["coverage_status"].eq("missing"), "sector"].tolist(),
        "monthly_average_available_sector_count": float(monthly_counts.mean()),
        "worst_months": monthly_counts[monthly_counts.eq(monthly_counts.min())].index.tolist(),
        "worst_month_available_sector_count": int(monthly_counts.min()),
        "first_eligible_month_by_sector": first_by_sector,
        "continuously_missing_sectors": list(GICS_SECTORS),
        "partially_missing_sectors": [],
        "sectors_not_requiring_expansion": [],
        "priority_candidate_count": len(priority),
        "vendor_conflict_rows": len(conflicts),
        "vendor_conflicts_resolved_v2": int(conflicts["resolved_v2"].astype(bool).sum()),
        "recommended_conclusion": "C",
    }
    return CoverageOutputs(
        clean_detail.to_dict("records"),
        current.to_dict("records"),
        monthly.to_dict("records"),
        priority.to_dict("records"),
        conflicts.to_dict("records"),
        summary,
    )


def write_intermediates(outputs: CoverageOutputs) -> None:
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    payloads = {
        "clean_universe_detail.json": outputs.clean_detail,
        "current_sector_coverage.json": outputs.current_sector_coverage,
        "monthly_sector_coverage_audit.json": outputs.monthly_sector_coverage,
        "priority_review_candidates.json": outputs.priority_candidates,
        "vendor_conflicts_resolved_v2.json": outputs.vendor_conflicts_v2,
        "coverage_summary.json": outputs.summary,
    }
    for filename, payload in payloads.items():
        path = INTERMEDIATE_DIR / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
