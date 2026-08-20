"""Rebuild auditable ETF lifecycles and a conservative U.S. equity ETF universe."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Final

import pandas as pd

from src.paths import (
    CLEAN_US_EQUITY_ETFS_PATH,
    DATA_AUDIT_PATH,
    ETF_LIFECYCLE_PATH,
    ETF_METADATA_PATH,
    EXCLUDED_ETPS_PATH,
    METADATA_NEEDS_REVIEW_PATH,
    PROJECT_ROOT,
    TICKER_REUSE_CONFLICTS_PATH,
    VENDOR_CONFLICTS_PATH,
)

RAW_ALPHA_DIR: Final = PROJECT_ROOT / "data" / "raw" / "metadata"
ALPHA_ACTIVE_PATH: Final = RAW_ALPHA_DIR / "alpha_vantage_active_etfs.csv"
ALPHA_DELISTED_PATH: Final = RAW_ALPHA_DIR / "alpha_vantage_delisted_etfs.csv"
OFFICIAL_VERIFICATION_PATH: Final = RAW_ALPHA_DIR / "official_fund_verification.csv"
NASDAQ_CURRENT_SNAPSHOT_PATH: Final = (
    RAW_ALPHA_DIR / "nasdaq_current_etp_snapshot.csv"
)

LIFECYCLE_COLUMNS: Final = [
    "lifecycle_id",
    "ticker",
    "fund_name",
    "exchange",
    "asset_type_from_source",
    "ipo_date",
    "delisting_date",
    "status",
    "source",
    "source_download_date",
    "entity_key",
]

METADATA_V2_COLUMNS: Final = [
    "lifecycle_id",
    "ticker",
    "fund_name",
    "exchange",
    "listing_date",
    "delisting_date",
    "status",
    "instrument_type",
    "asset_class",
    "geographic_scope",
    "leveraged_flag",
    "inverse_flag",
    "single_stock_flag",
    "etn_flag",
    "option_overlay_flag",
    "thematic_flag",
    "classification_confidence",
    "classification_sources",
    "classification_evidence",
    "review_status",
    "exclusion_reason",
    "eligible_us_equity_etf",
    "lifecycle_conflict",
    "price_identity_match",
]

EXCLUSION_PATTERNS: Final = {
    "etn": r"\bETN(?:S)?\b|EXCHANGE[- ]TRADED NOTE|NOTES DUE",
    "fixed_income": (
        r"\bBOND\b|TREASURY|MUNICIPAL|T[- ]?BILL|FIXED INCOME|CLO\b|"
        r"MORTGAGE|\bMBS\b|CORPORATE DEBT|HIGH YIELD|SENIOR LOAN|"
        r"FLOATING RATE|AGGREGATE BOND"
    ),
    "commodity": (
        r"\bGOLD\b|\bSILVER\b|COMMODIT|CRUDE OIL|NATURAL GAS|\bCOPPER\b|"
        r"URANIUM|PLATINUM|PALLADIUM|AGRICULTUR"
    ),
    "crypto": r"BITCOIN|ETHEREUM|CRYPTO|BLOCKCHAIN.*INCOME|DIGITAL ASSET",
    "currency": r"CURRENCY|DOLLAR INDEX|EURO|YEN|SWISS FRANC|FOREX",
    "international_or_global": (
        r"EMERGING MARKET|INTERNATIONAL|GLOBAL|EUROPE|CHINA|JAPAN|INDIA|"
        r"BRAZIL|KOREA|TAIWAN|MEXICO|CANADA|AUSTRALIA|GERMANY|FRANCE|"
        r"UNITED KINGDOM|LATIN AMERICA|\bEAFE\b|EX[- ]US|WORLD|ASIA"
    ),
    "leveraged": r"(?<!\w)[+-]?[23]X(?!\w)|LEVERAGED|ULTRAPRO|ULTRASHORT|DAILY BULL",
    "inverse": r"INVERSE|DAILY BEAR|\bSHORT\b.*(?:DAILY|ETF|ETN)|-[123]X",
    "preferred_stock": r"PREFERRED (?:STOCK|SECURIT|ETF|INCOME)",
    "convertible_bond": r"CONVERTIBLE (?:BOND|SECURIT)",
    "mlp": r"\bMLP\b|MASTER LIMITED PARTNERSHIP",
    "multi_asset": r"MULTI[- ]ASSET|ASSET ALLOCATION|RISK PARITY|BALANCED ETF",
    "volatility": r"\bVIX\b|VOLATILITY (?:FUTURES|INDEX|ETF|ETN)",
    "option_or_defined_outcome": (
        r"COVERED CALL|OPTION INCOME|PREMIUM INCOME|BUYWRITE|BUY[- ]WRITE|"
        r"BUFFER|DEFINED OUTCOME|AUTOCALLABLE|COLLAR|YIELDBOOST|MAX INCOME"
    ),
}

THEMATIC_PATTERN: Final = (
    r"ARTIFICIAL INTELLIGENCE|\bAI\b|ROBOT|CYBER|CLEAN ENERGY|GENOMIC|"
    r"SPACE|METAVERSE|CANNABIS|ESPORTS|FINTECH|NEXT GEN|INNOVATION|"
    r"QUANTUM|HYDROGEN|CLOUD COMPUTING|DIGITAL ECONOMY"
)


@dataclass(frozen=True)
class MetadataRebuildResult:
    lifecycle: pd.DataFrame
    metadata: pd.DataFrame
    clean: pd.DataFrame
    excluded: pd.DataFrame
    needs_review: pd.DataFrame
    ticker_reuse_conflicts: pd.DataFrame
    vendor_conflicts: pd.DataFrame


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _text(value: object, default: str = "unknown") -> str:
    if value is None or pd.isna(value):
        return default
    cleaned = str(value).strip()
    if not cleaned or cleaned.lower() in {"nan", "nat", "none", "null"}:
        return default
    return cleaned


def _date(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    return "unknown" if pd.isna(parsed) else parsed.date().isoformat()


def _normalized_name(value: object) -> str:
    text = re.sub(r"[^A-Z0-9]+", " ", _text(value).upper()).strip()
    removable = {"ETF", "FUND", "TRUST", "SHARES", "THE"}
    return " ".join(token for token in text.split() if token not in removable)


def _entity_key(row: pd.Series) -> str:
    parts = [
        _text(row["ticker"]).upper(),
        _normalized_name(row["fund_name"]),
        _text(row["ipo_date"]),
        _text(row["delisting_date"]),
    ]
    return "|".join(parts)


def _lifecycle_id(entity_key: str) -> str:
    return "lc_" + hashlib.sha256(entity_key.encode("utf-8")).hexdigest()[:20]


def load_alpha_lifecycles() -> pd.DataFrame:
    """Load both raw Alpha listing tables and retain every ETF lifecycle row."""
    frames: list[pd.DataFrame] = []
    for path, requested_status in (
        (ALPHA_ACTIVE_PATH, "active"),
        (ALPHA_DELISTED_PATH, "delisted"),
    ):
        source = pd.read_csv(path, dtype=str)
        required = {
            "symbol",
            "name",
            "exchange",
            "assetType",
            "ipoDate",
            "delistingDate",
            "status",
        }
        if not required.issubset(source.columns):
            raise ValueError(f"{path.name} missing columns: {sorted(required-set(source.columns))}")
        source = source[source["assetType"].fillna("").str.upper().eq("ETF")].copy()
        normalized = pd.DataFrame(
            {
                "ticker": source["symbol"].map(_text).str.upper(),
                "fund_name": source["name"].map(_text),
                "exchange": source["exchange"].map(_text),
                "asset_type_from_source": source["assetType"].map(_text),
                "ipo_date": source["ipoDate"].map(_date),
                "delisting_date": source["delistingDate"].map(_date),
                "status": source["status"].map(_text),
                "source": f"alpha_vantage_listing_status_{requested_status}",
                "source_download_date": date.today().isoformat(),
            }
        )
        normalized.loc[normalized["status"].eq("unknown"), "status"] = requested_status
        frames.append(normalized)

    lifecycle = pd.concat(frames, ignore_index=True)
    lifecycle["entity_key"] = lifecycle.apply(_entity_key, axis=1)
    lifecycle["lifecycle_id"] = lifecycle["entity_key"].map(_lifecycle_id)
    lifecycle = lifecycle.drop_duplicates("lifecycle_id", keep="first")
    return lifecycle[LIFECYCLE_COLUMNS].sort_values(
        ["ticker", "ipo_date", "delisting_date", "fund_name"], ignore_index=True
    )


def add_nasdaq_only_current_lifecycles(
    lifecycle: pd.DataFrame, current_nasdaq: pd.DataFrame
) -> pd.DataFrame:
    """Represent current Nasdaq products absent from Alpha active as distinct rows.

    These records never inherit an Alpha delisted entity's dates. Unknown IPO dates
    force manual review and prevent accidental attribution of old prices to a newly
    reused ticker.
    """
    active_tickers = set(
        lifecycle.loc[lifecycle["status"].str.lower().eq("active"), "ticker"]
    )
    additions: list[dict[str, str]] = []
    for record in current_nasdaq.itertuples(index=False):
        ticker = _text(record.ticker).upper()
        if ticker in active_tickers:
            continue
        row = {
            "ticker": ticker,
            "fund_name": _text(record.fund_name),
            "exchange": _text(getattr(record, "exchange", "unknown")),
            "asset_type_from_source": _text(
                getattr(record, "asset_type", "ETF")
            ),
            "ipo_date": "unknown",
            "delisting_date": "unknown",
            "status": "active_needs_review",
            "source": "nasdaq_trader_current_snapshot_only",
            "source_download_date": _text(
                getattr(record, "source_as_of", date.today().isoformat())
            ),
        }
        row["entity_key"] = _entity_key(pd.Series(row))
        row["lifecycle_id"] = _lifecycle_id(row["entity_key"])
        additions.append(row)
    if additions:
        lifecycle = pd.concat(
            [lifecycle, pd.DataFrame(additions, columns=LIFECYCLE_COLUMNS)],
            ignore_index=True,
        )
    return lifecycle.drop_duplicates("lifecycle_id").sort_values(
        ["ticker", "ipo_date", "delisting_date", "fund_name"], ignore_index=True
    )


def _ticker_reuse_table(lifecycle: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker, group in lifecycle.groupby("ticker", sort=True):
        names = sorted(group["fund_name"].map(_normalized_name).unique())
        if len(group) <= 1 or len(names) <= 1:
            continue
        rows.append(
            {
                "ticker": ticker,
                "lifecycle_record_count": len(group),
                "distinct_normalized_names": len(names),
                "lifecycle_ids": "|".join(group["lifecycle_id"]),
                "fund_names": "|".join(group["fund_name"]),
                "ipo_dates": "|".join(group["ipo_date"]),
                "delisting_dates": "|".join(group["delisting_date"]),
                "conflict_type": "ticker_reuse_or_name_conflict",
                "resolution_status": "needs_review",
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "lifecycle_record_count",
            "distinct_normalized_names",
            "lifecycle_ids",
            "fund_names",
            "ipo_dates",
            "delisting_dates",
            "conflict_type",
            "resolution_status",
        ],
    )


def _name_similarity(left: object, right: object) -> float:
    return SequenceMatcher(None, _normalized_name(left), _normalized_name(right)).ratio()


def _vendor_conflict_table(
    lifecycle: pd.DataFrame,
    current_nasdaq: pd.DataFrame,
    audit: pd.DataFrame,
    reuse_tickers: set[str],
) -> pd.DataFrame:
    active = lifecycle[lifecycle["status"].str.lower().eq("active")]
    delisted = lifecycle[lifecycle["status"].str.lower().eq("delisted")]
    active_groups = {ticker: group for ticker, group in active.groupby("ticker")}
    delisted_groups = {ticker: group for ticker, group in delisted.groupby("ticker")}
    audit_map = audit.drop_duplicates("ticker").set_index("ticker")
    rows: list[dict[str, object]] = []

    for record in current_nasdaq.itertuples(index=False):
        ticker = str(record.ticker).upper()
        current_name = _text(record.fund_name)
        current_active = active_groups.get(ticker)
        conflict_types: list[str] = []
        details: list[str] = []
        if current_active is None:
            conflict_types.append("nasdaq_not_in_alpha_active_etf")
        else:
            best_similarity = max(
                _name_similarity(current_name, name) for name in current_active["fund_name"]
            )
            if best_similarity < 0.72:
                conflict_types.append("active_name_mismatch")
                details.append(f"best_name_similarity={best_similarity:.3f}")
            if ticker in audit_map.index:
                first_price = pd.to_datetime(
                    audit_map.at[ticker, "first_price_date"], errors="coerce"
                )
                known_ipos = pd.to_datetime(
                    current_active["ipo_date"], errors="coerce", format="mixed"
                ).dropna()
                if pd.notna(first_price) and not known_ipos.empty:
                    latest_ipo = known_ipos.max()
                    if first_price < latest_ipo - pd.Timedelta(days=31):
                        conflict_types.append("price_predates_current_lifecycle")
                        details.append(
                            f"first_price={first_price.date()};latest_active_ipo={latest_ipo.date()}"
                        )
        if ticker in delisted_groups:
            current_norm = _normalized_name(current_name)
            old_names = set(delisted_groups[ticker]["fund_name"].map(_normalized_name))
            if any(name != current_norm for name in old_names):
                conflict_types.append("current_ticker_has_distinct_delisted_entity")
        if ticker in reuse_tickers:
            conflict_types.append("alpha_lifecycle_name_conflict")
        if conflict_types:
            rows.append(
                {
                    "ticker": ticker,
                    "conflict_types": "|".join(sorted(set(conflict_types))),
                    "nasdaq_fund_name": current_name,
                    "alpha_active_names": (
                        "|".join(current_active["fund_name"])
                        if current_active is not None
                        else ""
                    ),
                    "alpha_delisted_names": (
                        "|".join(delisted_groups[ticker]["fund_name"])
                        if ticker in delisted_groups
                        else ""
                    ),
                    "details": "|".join(details),
                    "resolution_status": "needs_review",
                }
            )

    nasdaq_tickers = set(current_nasdaq["ticker"].str.upper())
    for ticker, group in active.groupby("ticker", sort=True):
        if ticker not in nasdaq_tickers:
            rows.append(
                {
                    "ticker": ticker,
                    "conflict_types": "alpha_active_not_in_nasdaq_snapshot",
                    "nasdaq_fund_name": "",
                    "alpha_active_names": "|".join(group["fund_name"]),
                    "alpha_delisted_names": "",
                    "details": "",
                    "resolution_status": "needs_review",
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "conflict_types",
            "nasdaq_fund_name",
            "alpha_active_names",
            "alpha_delisted_names",
            "details",
            "resolution_status",
        ],
    ).drop_duplicates(["ticker", "conflict_types"])


def _alpha_non_etf_conflicts(current_nasdaq: pd.DataFrame) -> pd.DataFrame:
    """Flag current ETP tickers also used by Alpha non-ETF entities."""
    current_map = current_nasdaq.drop_duplicates("ticker").set_index("ticker")
    current_tickers = set(current_map.index.str.upper())
    frames: list[pd.DataFrame] = []
    for path, snapshot in (
        (ALPHA_ACTIVE_PATH, "active"),
        (ALPHA_DELISTED_PATH, "delisted"),
    ):
        source = pd.read_csv(path, dtype=str).fillna("")
        source["ticker"] = source["symbol"].str.upper()
        source = source[
            source["ticker"].isin(current_tickers)
            & ~source["assetType"].str.upper().eq("ETF")
        ].copy()
        source["snapshot"] = snapshot
        frames.append(source)
    matches = pd.concat(frames, ignore_index=True)
    rows: list[dict[str, str]] = []
    for ticker, group in matches.groupby("ticker", sort=True):
        active_names = group.loc[group["snapshot"].eq("active"), "name"]
        delisted_names = group.loc[group["snapshot"].eq("delisted"), "name"]
        details = sorted(
            {
                f"{row.snapshot}:{_text(row.assetType)}:{_text(row.status)}"
                for row in group.itertuples(index=False)
            }
        )
        rows.append(
            {
                "ticker": ticker,
                "conflict_types": "alpha_non_etf_entity_same_ticker",
                "nasdaq_fund_name": _text(current_map.at[ticker, "fund_name"]),
                "alpha_active_names": "|".join(map(_text, active_names)),
                "alpha_delisted_names": "|".join(map(_text, delisted_names)),
                "details": "|".join(details),
                "resolution_status": "needs_review",
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "conflict_types",
            "nasdaq_fund_name",
            "alpha_active_names",
            "alpha_delisted_names",
            "details",
            "resolution_status",
        ],
    )


def _preliminary_classification(name: str) -> dict[str, object]:
    upper = name.upper()
    reasons = [
        reason
        for reason, pattern in EXCLUSION_PATTERNS.items()
        if re.search(pattern, upper, flags=re.IGNORECASE)
    ]
    instrument_type = "ETN" if "etn" in reasons else "ETF"
    if "fixed_income" in reasons or "convertible_bond" in reasons:
        asset_class = "fixed_income"
    elif "commodity" in reasons:
        asset_class = "commodity"
    elif "crypto" in reasons:
        asset_class = "crypto"
    elif "currency" in reasons:
        asset_class = "currency"
    elif "multi_asset" in reasons:
        asset_class = "multi_asset"
    elif "mlp" in reasons:
        asset_class = "mlp"
    elif re.search(r"\bEQUITY\b|\bSTOCK\b|S&P|RUSSELL|NASDAQ", upper):
        asset_class = "equity_preliminary"
    else:
        asset_class = "unknown"

    if "international_or_global" in reasons:
        geography = "non_us_or_global"
    elif re.search(r"\bU\.?S\.?\b|UNITED STATES|S&P 500|RUSSELL", upper):
        geography = "United States_preliminary"
    else:
        geography = "unknown"

    return {
        "instrument_type": instrument_type,
        "asset_class": asset_class,
        "geographic_scope": geography,
        "leveraged_flag": True if "leveraged" in reasons else "unknown",
        "inverse_flag": True if "inverse" in reasons else "unknown",
        "single_stock_flag": (
            True
            if re.search(r"SINGLE[- ]STOCK|DAILY (?:BULL|BEAR)|[23]X (?:LONG|SHORT)", upper)
            else "unknown"
        ),
        "etn_flag": instrument_type == "ETN",
        "option_overlay_flag": (
            True if "option_or_defined_outcome" in reasons else "unknown"
        ),
        "thematic_flag": bool(re.search(THEMATIC_PATTERN, upper, flags=re.IGNORECASE)),
        "exclusion_reasons": reasons,
    }


def _bool_text(value: object) -> bool | str:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return "unknown"


def _audit_bool(value: object, default: bool = False) -> bool:
    """Read booleans safely from either typed or CSV string audit fields."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return default


def _build_metadata(
    lifecycle: pd.DataFrame,
    current_nasdaq: pd.DataFrame,
    official: pd.DataFrame,
    audit: pd.DataFrame,
    reuse_tickers: set[str],
    vendor_conflict_tickers: set[str],
) -> pd.DataFrame:
    nasdaq_map = current_nasdaq.drop_duplicates("ticker").set_index("ticker")
    official_map = official.drop_duplicates("ticker").set_index("ticker")
    audit_map = audit.drop_duplicates("ticker").set_index("ticker")
    rows: list[dict[str, object]] = []

    for record in lifecycle.itertuples(index=False):
        ticker = record.ticker
        preliminary = _preliminary_classification(record.fund_name)
        sources = [record.source]
        evidence = ["Alpha Vantage assetType=ETF lifecycle record"]
        lifecycle_conflict = ticker in reuse_tickers or ticker in vendor_conflict_tickers
        has_current_nasdaq_match = False
        if ticker in nasdaq_map.index and str(record.status).lower() == "active":
            similarity = _name_similarity(record.fund_name, nasdaq_map.at[ticker, "fund_name"])
            has_current_nasdaq_match = similarity >= 0.72
            sources.append("nasdaq_trader_current_snapshot")
            evidence.append(f"Nasdaq name similarity={similarity:.3f}")

        price_identity_match: bool | str = "unknown"
        price_clean = False
        if ticker in audit_map.index:
            audit_row = audit_map.loc[ticker]
            first_price = pd.to_datetime(audit_row["first_price_date"], errors="coerce")
            ipo = pd.to_datetime(record.ipo_date, errors="coerce")
            if pd.notna(first_price) and pd.notna(ipo):
                price_identity_match = bool(first_price >= ipo - pd.Timedelta(days=31))
            nonpositive_count = int(audit_row.get("nonpositive_price_count", 0))
            logic_error_count = int(audit_row.get("ohlc_logic_error_count", 0))
            price_clean = bool(
                int(audit_row.get("duplicate_dates", 0)) == 0
                and int(audit_row.get("missing_adj_close", 0)) == 0
                and int(audit_row.get("missing_volume", 0)) == 0
                and nonpositive_count == 0
                and logic_error_count == 0
                and _audit_bool(audit_row.get("adjusted_close_positive", True), True)
                and _audit_bool(audit_row.get("volume_nonnegative", True), True)
                and not _audit_bool(audit_row.get("large_missing_gap", True), True)
            )

        verified = ticker in official_map.index
        if verified:
            verified_row = official_map.loc[ticker]
            preliminary.update(
                {
                    "instrument_type": verified_row["instrument_type"],
                    "asset_class": verified_row["asset_class"],
                    "geographic_scope": verified_row["geographic_scope"],
                    "leveraged_flag": _bool_text(verified_row["leveraged_flag"]),
                    "inverse_flag": _bool_text(verified_row["inverse_flag"]),
                    "single_stock_flag": _bool_text(verified_row["single_stock_flag"]),
                    "etn_flag": _bool_text(verified_row["etn_flag"]),
                    "option_overlay_flag": _bool_text(
                        verified_row["option_overlay_flag"]
                    ),
                    "thematic_flag": _bool_text(verified_row["thematic_flag"]),
                    "exclusion_reasons": [],
                }
            )
            sources.append(verified_row["source_url"])
            evidence.append(verified_row["evidence"])
            if ticker in nasdaq_map.index and str(record.status).lower() == "active":
                # The issuer page tied to the exact ticker resolves a pure legal-
                # name/display-name variation between the two vendor tables.
                has_current_nasdaq_match = True

        known_flags = all(
            preliminary[key] in {True, False}
            for key in (
                "leveraged_flag",
                "inverse_flag",
                "single_stock_flag",
                "etn_flag",
                "option_overlay_flag",
            )
        )
        eligible = bool(
            verified
            and str(record.status).lower() == "active"
            and record.ipo_date != "unknown"
            and preliminary["instrument_type"] == "ETF"
            and preliminary["asset_class"] == "equity"
            and preliminary["geographic_scope"] == "United States"
            and preliminary["leveraged_flag"] is False
            and preliminary["inverse_flag"] is False
            and preliminary["single_stock_flag"] is False
            and preliminary["etn_flag"] is False
            and preliminary["option_overlay_flag"] is False
            and preliminary["thematic_flag"] is False
            and known_flags
            and has_current_nasdaq_match
            and not lifecycle_conflict
            and price_identity_match is True
            and price_clean
        )

        exclusion_reasons = list(preliminary["exclusion_reasons"])
        unresolved_identity = bool(
            lifecycle_conflict
            or record.source == "nasdaq_trader_current_snapshot_only"
            or str(record.status).lower() == "active_needs_review"
        )
        if unresolved_identity:
            review_status = "needs_review"
            confidence = "low"
            exclusion_reasons.append("unresolved_lifecycle_or_vendor_conflict")
        elif exclusion_reasons:
            review_status = "excluded"
            confidence = "medium"
        elif eligible:
            review_status = "clean"
            confidence = "high"
        else:
            review_status = "needs_review"
            confidence = "high" if verified else "low"
            if lifecycle_conflict:
                exclusion_reasons.append("unresolved_lifecycle_or_vendor_conflict")
            if preliminary["thematic_flag"] is True:
                exclusion_reasons.append("thematic_manual_review")
            if not verified:
                exclusion_reasons.append("insufficient_structured_classification_evidence")
            if not price_clean:
                exclusion_reasons.append("price_data_not_clean")
            if price_identity_match is not True:
                exclusion_reasons.append("price_identity_not_confirmed")

        rows.append(
            {
                "lifecycle_id": record.lifecycle_id,
                "ticker": ticker,
                "fund_name": record.fund_name,
                "exchange": record.exchange,
                "listing_date": record.ipo_date,
                "delisting_date": record.delisting_date,
                "status": record.status,
                "instrument_type": preliminary["instrument_type"],
                "asset_class": preliminary["asset_class"],
                "geographic_scope": preliminary["geographic_scope"],
                "leveraged_flag": preliminary["leveraged_flag"],
                "inverse_flag": preliminary["inverse_flag"],
                "single_stock_flag": preliminary["single_stock_flag"],
                "etn_flag": preliminary["etn_flag"],
                "option_overlay_flag": preliminary["option_overlay_flag"],
                "thematic_flag": preliminary["thematic_flag"],
                "classification_confidence": confidence,
                "classification_sources": "|".join(map(str, sources)),
                "classification_evidence": "|".join(map(str, evidence)),
                "review_status": review_status,
                "exclusion_reason": "|".join(sorted(set(exclusion_reasons))),
                "eligible_us_equity_etf": eligible,
                "lifecycle_conflict": lifecycle_conflict,
                "price_identity_match": price_identity_match,
            }
        )
    return pd.DataFrame(rows, columns=METADATA_V2_COLUMNS)


def rebuild_metadata() -> MetadataRebuildResult:
    """Build lifecycle, conflict, classification, and conservative output tables."""
    lifecycle = load_alpha_lifecycles()
    current_nasdaq = pd.read_csv(
        NASDAQ_CURRENT_SNAPSHOT_PATH, dtype=str
    ).fillna("unknown")
    lifecycle = add_nasdaq_only_current_lifecycles(lifecycle, current_nasdaq)
    audit = pd.read_csv(DATA_AUDIT_PATH)
    official = pd.read_csv(OFFICIAL_VERIFICATION_PATH, dtype=str).fillna("unknown")
    reuse = _ticker_reuse_table(lifecycle)
    reuse_tickers = set(reuse["ticker"])
    vendor = _vendor_conflict_table(
        lifecycle, current_nasdaq, audit, reuse_tickers
    )
    vendor = pd.concat(
        [vendor, _alpha_non_etf_conflicts(current_nasdaq)], ignore_index=True
    ).drop_duplicates(["ticker", "conflict_types"])
    # An issuer-verified identity can resolve a pure vendor naming variation,
    # but never a reuse, date, delisting, or multi-cause conflict.
    verified_tickers = set(official["ticker"].str.upper())
    pure_name_mismatch = vendor["conflict_types"].eq("active_name_mismatch")
    issuer_verified = vendor["ticker"].isin(verified_tickers)
    vendor.loc[
        pure_name_mismatch & issuer_verified, "resolution_status"
    ] = "resolved_by_official_issuer_evidence"
    vendor_tickers = set(
        vendor.loc[vendor["resolution_status"].eq("needs_review"), "ticker"]
    )
    metadata = _build_metadata(
        lifecycle,
        current_nasdaq,
        official,
        audit,
        reuse_tickers,
        vendor_tickers,
    )
    clean = metadata[metadata["review_status"].eq("clean")].copy()
    excluded = metadata[metadata["review_status"].eq("excluded")].copy()
    needs_review = metadata[metadata["review_status"].eq("needs_review")].copy()

    _atomic_csv(lifecycle, ETF_LIFECYCLE_PATH)
    _atomic_csv(metadata, ETF_METADATA_PATH)
    _atomic_csv(clean, CLEAN_US_EQUITY_ETFS_PATH)
    _atomic_csv(excluded, EXCLUDED_ETPS_PATH)
    _atomic_csv(needs_review, METADATA_NEEDS_REVIEW_PATH)
    _atomic_csv(reuse, TICKER_REUSE_CONFLICTS_PATH)
    _atomic_csv(vendor, VENDOR_CONFLICTS_PATH)
    return MetadataRebuildResult(
        lifecycle=lifecycle,
        metadata=metadata,
        clean=clean,
        excluded=excluded,
        needs_review=needs_review,
        ticker_reuse_conflicts=reuse,
        vendor_conflicts=vendor,
    )


def write_rebuild_summary(result: MetadataRebuildResult, path: Path) -> None:
    """Write a machine-readable summary without embedding any credentials."""
    summary = {
        "generated_on": date.today().isoformat(),
        "lifecycle_records": len(result.lifecycle),
        "clean_records": len(result.clean),
        "excluded_records": len(result.excluded),
        "needs_review_records": len(result.needs_review),
        "ticker_reuse_conflicts": len(result.ticker_reuse_conflicts),
        "vendor_conflicts": len(result.vendor_conflicts),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
