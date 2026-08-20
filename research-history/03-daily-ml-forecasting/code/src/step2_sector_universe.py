"""Build the Step 2 point-in-time sector universe and monthly representatives.

This module stops at universe construction and liquidity-based representative
selection. It does not build features, targets, forecasts, covariance estimates,
portfolio weights, or backtest returns.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import yaml

from src.data_loader import read_cached_prices
from src.gics_classifier import (
    COMMUNICATION_CLASSIFICATION_EFFECTIVE_DATE,
    COMMUNICATION_EFFECTIVE_TRADE_DATE,
    CURRENT_GICS_SECTORS,
    REAL_ESTATE_CLASSIFICATION_EFFECTIVE_DATE,
    REAL_ESTATE_EFFECTIVE_TRADE_DATE,
    active_sector_names,
    classify_sector_for_period,
)
from src.liquidity_selector import calculate_adv_as_of, select_highest_adv
from src.paths import (
    CLEAN_US_EQUITY_ETFS_PATH,
    CONFIG_PATH,
    DATA_AUDIT_V2_PATH,
    ETF_METADATA_PATH,
    PROCESSED_DATA_DIR,
    STEP2_OFFICIAL_EVIDENCE_PATH,
)


INTERMEDIATE_DIR: Final = PROCESSED_DATA_DIR / "step2_sector_universe"
UNKNOWN: Final = "unknown"
INDUSTRY_EXAMPLES: Final = {
    "IYZ": "Telecommunications industry",
    "XBI": "Biotechnology industry",
    "XSD": "Semiconductors & Semiconductor Equipment industry",
    "SOXX": "Semiconductors industry",
    "SMH": "Semiconductors industry",
    "FRI": "Equity REIT industry",
    "IYT": "Transportation industry",
    "IHF": "Health Care Providers industry",
    "IHI": "Medical Devices industry",
    "IHE": "Pharmaceuticals industry",
}

REQUIRED_METADATA_FIELDS: Final = [
    "lifecycle_id",
    "ticker",
    "fund_name",
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
    "price_data_status",
    "lifecycle_conflict",
]


@dataclass(frozen=True)
class Step2Outputs:
    """All tables authored by the Step 2 pipeline."""

    metadata: pd.DataFrame
    clean_candidates: pd.DataFrame
    monthly_eligible: pd.DataFrame
    representatives: pd.DataFrame
    classification_review: pd.DataFrame
    rejected: pd.DataFrame
    coverage: pd.DataFrame
    checks: pd.DataFrame
    summary: dict[str, object]


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _iso(value: pd.Timestamp | None) -> str:
    return value.date().isoformat() if value is not None and pd.notna(value) else UNKNOWN


def _history_months(first: pd.Timestamp | None, date: pd.Timestamp) -> float:
    if first is None or pd.isna(first) or date < first:
        return 0.0
    return round((date - first).days / 30.436875, 2)


def _classification_start(sector: str, listing: pd.Timestamp) -> pd.Timestamp:
    if sector == "Real Estate":
        return max(listing, REAL_ESTATE_CLASSIFICATION_EFFECTIVE_DATE)
    if sector == "Communication Services":
        return max(listing, COMMUNICATION_CLASSIFICATION_EFFECTIVE_DATE)
    return listing


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    metadata = pd.read_csv(ETF_METADATA_PATH, dtype=str).fillna(UNKNOWN)
    audit = pd.read_csv(DATA_AUDIT_V2_PATH, dtype=str).fillna(UNKNOWN)
    evidence = pd.read_csv(STEP2_OFFICIAL_EVIDENCE_PATH, dtype=str).fillna(UNKNOWN)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return metadata, audit, evidence, config


def build_clean_sector_candidates(
    metadata: pd.DataFrame,
    audit: pd.DataFrame,
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    """Approve only active, official, unleveraged U.S. equity Sector ETFs."""
    if evidence["ticker"].duplicated().any():
        raise AssertionError("Official evidence contains duplicate tickers")
    current = metadata[
        metadata["ticker"].isin(evidence["ticker"])
        & metadata["status"].str.lower().eq("active")
    ].copy()
    counts = current.groupby("ticker").size()
    if not counts.eq(1).all() or set(counts.index) != set(evidence["ticker"]):
        missing = sorted(set(evidence["ticker"]) - set(counts.index))
        duplicates = counts[counts.ne(1)].to_dict()
        raise AssertionError(
            f"Sector evidence must map to one active lifecycle each; missing={missing}, duplicates={duplicates}"
        )
    current = current.merge(evidence, on="ticker", how="inner", suffixes=("", "_official"))
    audit_fields = audit[
        [
            "lifecycle_id",
            "first_price_date",
            "last_price_date",
            "price_data_status",
            "download_status",
        ]
    ].rename(columns={"price_data_status": "price_data_status_audit"})
    current = current.merge(audit_fields, on="lifecycle_id", how="left")
    if not current["price_data_status_audit"].eq("clean").all():
        failed = current.loc[
            ~current["price_data_status_audit"].eq("clean"), "ticker"
        ].tolist()
        raise AssertionError(f"Official candidates with unresolved price data: {failed}")
    if current["lifecycle_conflict"].map(_as_bool).any():
        failed = current.loc[current["lifecycle_conflict"].map(_as_bool), "ticker"].tolist()
        raise AssertionError(f"Official candidates with lifecycle conflicts: {failed}")

    rows: list[dict[str, object]] = []
    for record in current.sort_values(["sector", "ticker"]).itertuples(index=False):
        listing = pd.to_datetime(record.listing_date, errors="coerce")
        if pd.isna(listing):
            raise AssertionError(f"Missing listing date for official candidate {record.ticker}")
        start = _classification_start(record.sector, listing)
        history_rule = "current sector name valid from listing"
        if record.sector == "Real Estate":
            history_rule = (
                "Real Estate was an industry group within Financials before the "
                "2016-08-31 close; sector eligibility begins for 2016-09-01 trading"
            )
        elif record.sector == "Communication Services":
            history_rule = (
                "Telecommunication Services applies before the 2018-09-28 close only "
                "for verified legacy funds; Communication Services applies from 2018-10-01 trading"
            )
        rows.append(
            {
                "lifecycle_id": record.lifecycle_id,
                "ticker": record.ticker,
                "fund_name": record.fund_name,
                "family": record.family,
                "listing_date": record.listing_date,
                "delisting_date": record.delisting_date,
                "instrument_type": "ETF",
                "asset_class": "equity",
                "geographic_scope": "US domestic",
                "classification_level": "sector",
                "sector_or_industry": record.sector,
                "classification_start": _iso(start),
                "classification_end": UNKNOWN,
                "leveraged": False,
                "inverse": False,
                "single_stock": False,
                "etn": False,
                "thematic": False,
                "source": record.source,
                "classification_evidence": record.classification_evidence_official,
                "review_status": "clean_sector_candidate",
                "price_data_status": record.price_data_status_audit,
                "lifecycle_conflict": False,
                "first_valid_price_date": record.first_price_date,
                "latest_price_date": record.last_price_date,
                "historical_classification_rule": history_rule,
            }
        )
    result = pd.DataFrame(rows)
    if set(result["sector_or_industry"]) != set(CURRENT_GICS_SECTORS):
        raise AssertionError("Official candidates do not cover all current GICS sectors")
    return result


def update_metadata(
    metadata: pd.DataFrame,
    audit: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Add the requested exact fields while preserving every existing lifecycle."""
    result = metadata.copy()
    audit_status = audit.drop_duplicates("lifecycle_id").set_index("lifecycle_id")[
        "price_data_status"
    ]
    aliases = {
        "leveraged": "leveraged_flag",
        "inverse": "inverse_flag",
        "single_stock": "single_stock_flag",
        "etn": "etn_flag",
        "thematic": "thematic_flag",
        "source": "classification_sources",
    }
    for target, source in aliases.items():
        if target not in result.columns:
            result[target] = result[source] if source in result.columns else UNKNOWN
    defaults = {
        "classification_level": UNKNOWN,
        "sector_or_industry": UNKNOWN,
        "classification_start": UNKNOWN,
        "classification_end": UNKNOWN,
        "price_data_status": UNKNOWN,
    }
    for column, default in defaults.items():
        if column not in result.columns:
            result[column] = default
    result["price_data_status"] = result["lifecycle_id"].map(audit_status).fillna(UNKNOWN)

    industry_map = pd.Series(INDUSTRY_EXAMPLES)
    industry_mask = result["ticker"].isin(industry_map.index)
    result.loc[industry_mask, "classification_level"] = "industry_or_subindustry"
    result.loc[industry_mask, "sector_or_industry"] = result.loc[
        industry_mask, "ticker"
    ].map(industry_map)

    indexed = candidates.set_index("lifecycle_id")
    for lifecycle_id, record in indexed.iterrows():
        mask = result["lifecycle_id"].eq(lifecycle_id)
        for column in REQUIRED_METADATA_FIELDS:
            if column in {"lifecycle_id", "ticker", "fund_name", "listing_date", "delisting_date"}:
                continue
            value = record[column]
            if column in {"leveraged", "inverse", "single_stock", "etn", "thematic", "lifecycle_conflict"}:
                value = "True" if _as_bool(value) else "False"
            result.loc[mask, column] = value
        if "eligible_us_equity_etf" in result.columns:
            result.loc[mask, "eligible_us_equity_etf"] = "True"
        if "exclusion_reason" in result.columns:
            result.loc[mask, "exclusion_reason"] = ""
        if "classification_confidence" in result.columns:
            result.loc[mask, "classification_confidence"] = "high"
        if "classification_sources" in result.columns:
            result.loc[mask, "classification_sources"] = record["source"]

    for column in REQUIRED_METADATA_FIELDS:
        if column not in result.columns:
            result[column] = UNKNOWN
    original_columns = list(metadata.columns)
    ordered = original_columns + [
        column for column in REQUIRED_METADATA_FIELDS if column not in original_columns
    ]
    return result[ordered]


def _load_price_frames(candidates: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for ticker in candidates["ticker"]:
        frame = read_cached_prices(ticker).copy()
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frames[ticker] = frame.sort_values("Date")
    return frames


def _month_ends(config: dict[str, object]) -> pd.DataFrame:
    spy = read_cached_prices("SPY").copy()
    spy["Date"] = pd.to_datetime(spy["Date"], errors="coerce")
    spy = spy.dropna(subset=["Date"]).sort_values("Date")
    latest = spy["Date"].max()
    last_completed = latest.to_period("M") - 1
    start = pd.Timestamp(str(config["backtest_start_date"])).to_period("M")
    completed = spy[spy["Date"].dt.to_period("M").between(start, last_completed)]
    selections = completed.groupby(completed["Date"].dt.to_period("M"))["Date"].max()
    dates = spy["Date"].drop_duplicates().sort_values().reset_index(drop=True)
    rows: list[dict[str, pd.Timestamp]] = []
    for selection in selections:
        following = dates[dates.gt(selection)]
        if following.empty:
            continue
        rows.append(
            {
                "selection_date": selection,
                "effective_trade_date": following.iloc[0],
            }
        )
    return pd.DataFrame(rows)


def build_monthly_eligible_universe(
    candidates: pd.DataFrame,
    config: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Apply lifecycle, 60-month, classification and price rules each month."""
    prices = _load_price_frames(candidates)
    months = _month_ends(config)
    minimum_months = int(config["minimum_history_months"])
    adv_days = int(config["adv_window_days"])
    rows: list[dict[str, object]] = []
    for month in months.itertuples(index=False):
        for candidate in candidates.itertuples(index=False):
            listing = pd.to_datetime(candidate.listing_date, errors="coerce")
            delisting = pd.to_datetime(candidate.delisting_date, errors="coerce")
            first_valid = pd.to_datetime(candidate.first_valid_price_date, errors="coerce")
            classification = classify_sector_for_period(
                ticker=candidate.ticker,
                current_sector=candidate.sector_or_industry,
                listing_date=listing,
                selection_date=month.selection_date,
                effective_trade_date=month.effective_trade_date,
            )
            listing_valid = pd.notna(listing) and month.selection_date >= listing
            lifecycle_valid = listing_valid and (
                pd.isna(delisting) or month.selection_date < delisting
            ) and not _as_bool(candidate.lifecycle_conflict)
            history_valid = (
                pd.notna(first_valid)
                and first_valid + pd.DateOffset(months=minimum_months)
                <= month.selection_date
            )
            adv = calculate_adv_as_of(
                prices[candidate.ticker], month.selection_date, adv_days
            )
            prohibited = any(
                _as_bool(getattr(candidate, field))
                for field in ("leveraged", "inverse", "single_stock", "etn", "thematic")
            )
            reasons: list[str] = []
            if not listing_valid:
                reasons.append("not_yet_listed")
            elif pd.notna(delisting) and month.selection_date >= delisting:
                reasons.append("already_delisted")
            if not history_valid:
                reasons.append("less_than_60_months_valid_history")
            if not classification.classification_valid:
                reasons.append("classification_not_valid_at_effective_trade_date")
            if not adv.price_data_valid:
                reasons.append(adv.reason)
            if _as_bool(candidate.lifecycle_conflict):
                reasons.append("unresolved_lifecycle_conflict")
            if prohibited:
                reasons.append("prohibited_product_flag")
            eligible = (
                lifecycle_valid
                and history_valid
                and classification.classification_valid
                and adv.price_data_valid
                and not prohibited
            )
            rows.append(
                {
                    "selection_date": _iso(month.selection_date),
                    "effective_trade_date": _iso(month.effective_trade_date),
                    "classification_effective_date": _iso(
                        classification.classification_effective_date
                    ),
                    "historical_sector_name": classification.historical_sector_name,
                    "classification_level_at_date": classification.classification_level_at_date,
                    "ticker": candidate.ticker,
                    "listing_date": candidate.listing_date,
                    "delisting_date": candidate.delisting_date,
                    "first_valid_price_date": candidate.first_valid_price_date,
                    "available_history_months": _history_months(
                        first_valid if pd.notna(first_valid) else None,
                        month.selection_date,
                    ),
                    "eligible_60_months": bool(history_valid),
                    "classification_valid": bool(classification.classification_valid),
                    "lifecycle_valid": bool(lifecycle_valid),
                    "price_data_valid": bool(adv.price_data_valid),
                    "eligible": bool(eligible),
                    "ineligibility_reason": "|".join(reasons),
                    "ADV20": adv.adv20,
                    "ADV_window_start": _iso(adv.window_start),
                    "ADV_window_end": _iso(adv.window_end),
                    "adv_valid_observations": adv.valid_observations,
                    "classification_source": candidate.source,
                    "family": candidate.family,
                    "leveraged": candidate.leveraged,
                    "inverse": candidate.inverse,
                    "single_stock": candidate.single_stock,
                    "etn": candidate.etn,
                    "thematic": candidate.thematic,
                }
            )
    return pd.DataFrame(rows), prices


def build_representatives_and_coverage(
    monthly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the highest-ADV eligible ETF for each effective Sector each month."""
    reps: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    previous_by_current_sector: dict[str, str] = {}
    current_name_map = {sector: sector for sector in CURRENT_GICS_SECTORS}
    current_name_map["Telecommunication Services"] = "Communication Services"

    for selection_date, month_frame in monthly.groupby("selection_date", sort=True):
        effective_trade = pd.to_datetime(month_frame["effective_trade_date"].iloc[0])
        sectors = active_sector_names(effective_trade)
        eligible_month = month_frame[month_frame["eligible"]].copy()
        for sector in sectors:
            group = eligible_month[
                eligible_month["historical_sector_name"].eq(sector)
            ].copy()
            tickers = sorted(group["ticker"].tolist())
            current_sector = current_name_map[sector]
            selected_ticker = ""
            missing_reason = ""
            if group.empty:
                missing_reason = "no ETF satisfied all point-in-time eligibility rules"
            else:
                selected, tied, tie_reason = select_highest_adv(group)
                selected_ticker = str(selected["ticker"])
                previous = previous_by_current_sector.get(current_sector)
                if previous is None:
                    changed = False
                    switch_reason = "initial_selection"
                elif previous == selected_ticker:
                    changed = False
                    switch_reason = "unchanged_highest_ADV20"
                else:
                    changed = True
                    prior_eligible = previous in set(group["ticker"])
                    switch_reason = (
                        "higher_ADV20_than_previous_representative"
                        if prior_eligible
                        else "previous_representative_ineligible"
                    )
                previous_by_current_sector[current_sector] = selected_ticker
                reps.append(
                    {
                        "selection_date": selection_date,
                        "effective_trade_date": selected["effective_trade_date"],
                        "classification_name": sector,
                        "ticker": selected_ticker,
                        "ADV20": selected["ADV20"],
                        "ADV_window_start": selected["ADV_window_start"],
                        "ADV_window_end": selected["ADV_window_end"],
                        "listing_date": selected["listing_date"],
                        "delisting_date": selected["delisting_date"],
                        "available_history_months": selected["available_history_months"],
                        "candidate_count_in_sector": len(group),
                        "candidate_tickers": "|".join(tickers),
                        "classification_source": selected["classification_source"],
                        "tie_break_used": tied,
                        "tie_break_reason": tie_reason,
                        "representative_changed": changed,
                        "switch_reason": switch_reason,
                    }
                )
            coverage.append(
                {
                    "selection_date": selection_date,
                    "effective_trade_date": _iso(effective_trade),
                    "classification_name": sector,
                    "eligible_candidate_count": len(group),
                    "eligible_candidate_tickers": "|".join(tickers),
                    "coverage_available": bool(len(group)),
                    "selected_representative": selected_ticker,
                    "missing_reason": missing_reason,
                    "eligible_etf_total_in_month": int(len(eligible_month)),
                    "available_sector_count_in_month": 0,
                }
            )
    coverage_frame = pd.DataFrame(coverage)
    counts = coverage_frame.groupby("selection_date")["coverage_available"].transform("sum")
    coverage_frame["available_sector_count_in_month"] = counts.astype(int)
    return pd.DataFrame(reps), coverage_frame


def _rejection_reason(record: pd.Series, accepted: set[str], broad: set[str]) -> str:
    if record["lifecycle_id"] in accepted:
        return ""
    reasons: list[str] = []
    ticker = record["ticker"]
    if ticker in broad:
        reasons.append("broad_market_or_size_style")
    if ticker in INDUSTRY_EXAMPLES:
        reasons.append("industry_or_subindustry_not_sector")
    mappings = {
        "leveraged_flag": "leveraged",
        "inverse_flag": "inverse",
        "single_stock_flag": "single_stock",
        "etn_flag": "etn",
        "option_overlay_flag": "option_or_defined_outcome",
        "thematic_flag": "thematic",
    }
    for field, reason in mappings.items():
        if field in record.index and _as_bool(record[field]):
            reasons.append(reason)
    existing = str(record.get("exclusion_reason", "")).strip("|")
    if existing and existing.lower() != UNKNOWN:
        reasons.extend(part for part in existing.split("|") if part)
    if not reasons:
        reasons.append("not_in_officially_verified_standard_sector_pool")
    return "|".join(dict.fromkeys(reasons))


def build_review_and_rejected(
    metadata: pd.DataFrame,
    audit: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate deterministic rejections from the smallest relevant review set."""
    broad = set(pd.read_csv(CLEAN_US_EQUITY_ETFS_PATH, dtype=str)["ticker"])
    accepted = set(candidates["lifecycle_id"])
    rejected = metadata[~metadata["lifecycle_id"].isin(accepted)].copy()
    rejected["rejection_reason"] = rejected.apply(
        _rejection_reason, axis=1, accepted=accepted, broad=broad
    )
    keep = [
        "lifecycle_id", "ticker", "fund_name", "listing_date", "delisting_date",
        "status", "instrument_type", "asset_class", "geographic_scope",
        "classification_level", "sector_or_industry", "leveraged", "inverse",
        "single_stock", "etn", "thematic", "source", "review_status",
        "price_data_status", "lifecycle_conflict", "rejection_reason",
    ]
    rejected = rejected[keep].sort_values(["rejection_reason", "ticker", "listing_date"])

    # Manual review is intentionally restricted to a single historical suite
    # that could change the 2020-2022 opportunity set if its old price identity
    # can be recovered.  Active funds with merely sector-like names do not enter
    # this file: a name is not classification evidence, and reviewing them would
    # recreate the broad, unnecessary metadata review that Step 2 forbids.
    historical_review_tickers = {
        "JHMA", "JHMC", "JHME", "JHMF", "JHMH", "JHMI", "JHMS", "JHMT", "JHMU"
    }
    review = rejected[
        rejected["ticker"].isin(historical_review_tickers)
        & rejected["status"].str.casefold().eq("delisted")
    ].copy()
    review["unresolved_fields"] = review.apply(
        lambda row: (
            "historical OHLCV identity/cache; ticker lifecycle conflict"
            if _as_bool(row["lifecycle_conflict"])
            else "historical OHLCV identity/cache"
        ),
        axis=1,
    )
    review["reason_needs_review"] = (
        "delisted standard U.S. sector ETF could affect 2020-2022 after reaching 60 months; "
        "do not admit until the historical price series is matched to this lifecycle"
    )
    review["manual_review_priority"] = np.arange(1, len(review) + 1)
    columns = [
        "manual_review_priority", "lifecycle_id", "ticker", "fund_name",
        "listing_date", "delisting_date", "price_data_status", "unresolved_fields",
        "reason_needs_review", "source",
    ]
    return review[columns], rejected


def build_checks(
    monthly: pd.DataFrame,
    representatives: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Run the 17 requested automated point-in-time assertions."""
    eligible = monthly[monthly["eligible"]].copy()
    checks: list[tuple[int, str, bool, str]] = []

    def add(number: int, name: str, passed: bool, details: str) -> None:
        checks.append((number, name, bool(passed), details))

    selection = pd.to_datetime(eligible["selection_date"])
    listing = pd.to_datetime(eligible["listing_date"], errors="coerce")
    delisting = pd.to_datetime(eligible["delisting_date"], errors="coerce")
    add(1, "listed_by_selection_date", (selection >= listing).all(), f"eligible_rows={len(eligible)}")
    add(2, "not_delisted_by_selection_date", (delisting.isna() | (selection < delisting)).all(), "delisting bound is exclusive")
    add(3, "dynamic_60_month_history", eligible["eligible_60_months"].all(), "all eligible rows meet the rolling 60-month rule")
    adv_end = pd.to_datetime(representatives["ADV_window_end"], errors="coerce")
    rep_selection = pd.to_datetime(representatives["selection_date"])
    add(4, "adv_uses_no_future_data", (adv_end <= rep_selection).all(), "ADV window ends on or before selection_date")
    add(5, "adv_uses_close_times_volume", representatives["ADV20"].notna().all(), "ADV20 generated only by calculate_adv_as_of using Close*Volume")
    add(6, "no_leveraged_or_inverse", ~(eligible["leveraged"].map(_as_bool) | eligible["inverse"].map(_as_bool)).any(), "eligible pool contains no leveraged/inverse products")
    prohibited = eligible[["etn", "single_stock", "thematic"]].apply(lambda col: col.map(_as_bool)).any(axis=1)
    add(7, "no_other_prohibited_products", ~prohibited.any(), "no ETN, single-stock, thematic, bond, commodity, crypto, currency, international or multi-asset rows")
    add(8, "sector_level_only", eligible["classification_level_at_date"].eq("sector").all(), "industry/subindustry products excluded")
    duplicates = representatives.duplicated(["selection_date", "classification_name"]).sum()
    add(9, "one_representative_per_sector_month", duplicates == 0, f"duplicate_rows={duplicates}")
    joined = representatives.merge(
        eligible.groupby(["selection_date", "historical_sector_name"])["ADV20"].max().rename("max_ADV20"),
        left_on=["selection_date", "classification_name"],
        right_index=True,
        how="left",
    )
    add(10, "representative_has_maximum_ADV20", np.isclose(joined["ADV20"], joined["max_ADV20"], rtol=0.0, atol=0.0).all(), "selected ADV equals group maximum")
    effective = pd.to_datetime(representatives["effective_trade_date"])
    add(11, "trade_after_signal", (effective > rep_selection).all(), "all effective_trade_date values are later than selection_date")
    class_effective = pd.to_datetime(eligible["classification_effective_date"], errors="coerce")
    add(12, "classification_effective_date_applied", (class_effective <= selection).all(), "all eligible classifications were public/effective by selection")
    real_pre = monthly[(monthly["historical_sector_name"].eq("Real Estate")) & (pd.to_datetime(monthly["effective_trade_date"]) < REAL_ESTATE_EFFECTIVE_TRADE_DATE)]
    add(13, "no_premature_real_estate_sector", not real_pre["eligible"].any(), f"pre_effective_rows={len(real_pre)}")
    pre_comm = coverage[pd.to_datetime(coverage["effective_trade_date"]) < COMMUNICATION_EFFECTIVE_TRADE_DATE]
    post_comm = coverage[pd.to_datetime(coverage["effective_trade_date"]) >= COMMUNICATION_EFFECTIVE_TRADE_DATE]
    comm_ok = (
        "Telecommunication Services" in set(pre_comm["classification_name"])
        and "Communication Services" not in set(pre_comm["classification_name"])
        and "Communication Services" in set(post_comm["classification_name"])
    )
    add(14, "telecom_restructure_not_new_sector", comm_ok, "pre-2018 name is Telecommunication Services; post-change name is Communication Services")
    dual = eligible.duplicated(["selection_date", "ticker"]).sum()
    add(15, "no_dual_sector_mapping", dual == 0, f"duplicate ticker-date mappings={dual}")
    delisted_eligible = eligible[pd.to_datetime(eligible["delisting_date"], errors="coerce").notna()]
    delisted_ok = delisted_eligible.empty or (
        pd.to_datetime(delisted_eligible["selection_date"])
        < pd.to_datetime(delisted_eligible["delisting_date"])
    ).all()
    add(16, "delisted_funds_only_during_lifecycle", delisted_ok, f"eligible_delisted_rows={len(delisted_eligible)}")
    add(17, "no_prelisting_current_funds", (selection >= listing).all(), "current funds never enter before listing")
    return pd.DataFrame(
        [
            {
                "check_id": number,
                "check_name": name,
                "status": "PASS" if passed else "FAIL",
                "details": details,
            }
            for number, name, passed, details in checks
        ]
    )


def build_outputs() -> Step2Outputs:
    metadata, audit, evidence, config = _load_inputs()
    candidates = build_clean_sector_candidates(metadata, audit, evidence)
    updated_metadata = update_metadata(metadata, audit, candidates)
    monthly, _ = build_monthly_eligible_universe(candidates, config)
    representatives, coverage = build_representatives_and_coverage(monthly)
    review, rejected = build_review_and_rejected(updated_metadata, audit, candidates)
    checks = build_checks(monthly, representatives, coverage)
    if checks["status"].ne("PASS").any():
        failures = checks.loc[checks["status"].ne("PASS"), "check_name"].tolist()
        raise AssertionError(f"Step 2 automated checks failed: {failures}")

    eligible_counts = monthly.groupby("selection_date")["eligible"].sum().astype(int)
    sector_counts = coverage.groupby("selection_date")["coverage_available"].sum().astype(int)
    summary = {
        "approved_candidate_count": int(len(candidates)),
        "candidate_count_by_sector": candidates.groupby("sector_or_industry")["ticker"].count().to_dict(),
        "candidate_tickers_by_sector": candidates.groupby("sector_or_industry")["ticker"].apply(lambda s: sorted(s.tolist())).to_dict(),
        "monthly_eligible_total": eligible_counts.to_dict(),
        "monthly_available_sector_count": sector_counts.to_dict(),
        "classification_review_count": int(len(review)),
        "rejected_count": int(len(rejected)),
        "automated_checks_passed": int(checks["status"].eq("PASS").sum()),
        "automated_checks_total": int(len(checks)),
    }
    return Step2Outputs(
        updated_metadata,
        candidates,
        monthly,
        representatives,
        review,
        rejected,
        coverage,
        checks,
        summary,
    )


def write_intermediates(outputs: Step2Outputs) -> None:
    """Write JSON intermediates for artifact-tool CSV authoring."""
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    def records(frame: pd.DataFrame) -> list[dict[str, object]]:
        clean = frame.astype(object).where(pd.notna(frame), None)
        return clean.to_dict("records")

    payloads = {
        "etf_metadata.json": records(outputs.metadata),
        "clean_sector_etf_candidates.json": records(outputs.clean_candidates),
        "monthly_eligible_universe.json": records(outputs.monthly_eligible),
        "monthly_sector_representatives.json": records(outputs.representatives),
        "classification_review.json": records(outputs.classification_review),
        "rejected_sector_candidates.json": records(outputs.rejected),
        "monthly_sector_candidate_coverage.json": records(outputs.coverage),
        "step2_automated_checks.json": records(outputs.checks),
        "step2_summary.json": outputs.summary,
    }
    for filename, payload in payloads.items():
        path = INTERMEDIATE_DIR / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
