"""Lifecycle-aware price audit and clean-universe acceptance assertions."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.data_loader import PRICE_COLUMNS, read_cached_prices
from src.paths import (
    CLEAN_US_EQUITY_ETFS_PATH,
    DATA_AUDIT_PATH,
    DATA_AUDIT_V2_PATH,
    ETF_METADATA_PATH,
    PRICE_ANOMALY_RESOLUTION_PATH,
    RAW_DATA_DIR,
)

AUDIT_V2_COLUMNS: Final = [
    "lifecycle_id",
    "ticker",
    "fund_name",
    "listing_date",
    "delisting_date",
    "first_price_date",
    "last_price_date",
    "total_observations",
    "missing_adj_close",
    "missing_volume",
    "duplicate_dates",
    "nonpositive_price_count",
    "ohlc_logic_error_count",
    "large_missing_gap",
    "sufficient_60_month_history_at_latest_date",
    "instrument_type",
    "asset_class",
    "geographic_scope",
    "leveraged_flag",
    "inverse_flag",
    "single_stock_flag",
    "etn_flag",
    "thematic_flag",
    "eligible_us_equity_etf",
    "review_status",
    "exclusion_reason",
    "price_data_status",
    "lifecycle_conflict",
    "download_status",
]


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _known_date(value: object) -> pd.Timestamp | None:
    if str(value).strip().lower() in {"", "unknown", "nan", "nat", "none"}:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def _slice_lifecycle(prices: pd.DataFrame, listing: object, delisting: object) -> pd.DataFrame:
    frame = prices.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    start = _known_date(listing)
    end = _known_date(delisting)
    if start is not None:
        frame = frame[frame["Date"] >= start]
    if end is not None:
        frame = frame[frame["Date"] <= end]
    return frame.reset_index(drop=True)


def _max_gap_and_missing_run(frame: pd.DataFrame) -> tuple[int, int]:
    dates = pd.DatetimeIndex(frame["Date"].dropna().unique()).sort_values()
    maximum_gap = 0
    if len(dates) >= 2:
        previous = dates[:-1].values.astype("datetime64[D]")
        current = dates[1:].values.astype("datetime64[D]")
        gaps = np.busday_count(previous, current) - 1
        maximum_gap = int(max(gaps.max(initial=0), 0))
    numeric = frame[PRICE_COLUMNS[1:]].apply(pd.to_numeric, errors="coerce")
    maximum_run = current_run = 0
    for missing in numeric.isna().any(axis=1):
        current_run = current_run + 1 if missing else 0
        maximum_run = max(maximum_run, current_run)
    return maximum_gap, maximum_run


def _price_metrics(frame: pd.DataFrame) -> dict[str, object]:
    dates = pd.to_datetime(frame["Date"], errors="coerce")
    valid_dates = dates.dropna()
    numeric = frame[PRICE_COLUMNS[1:]].apply(pd.to_numeric, errors="coerce")
    duplicate_dates = int(valid_dates.duplicated(keep=False).sum())
    price_columns = ["Open", "High", "Low", "Close", "Adjusted Close"]
    nonpositive = (numeric[price_columns] <= 0).any(axis=1)
    logic = (
        (numeric["High"] < numeric["Low"])
        | (numeric["Open"] < numeric["Low"])
        | (numeric["Open"] > numeric["High"])
        | (numeric["Close"] < numeric["Low"])
        | (numeric["Close"] > numeric["High"])
    )
    max_gap, max_missing_run = _max_gap_and_missing_run(frame)
    first = valid_dates.min() if not valid_dates.empty else pd.NaT
    last = valid_dates.max() if not valid_dates.empty else pd.NaT
    sufficient = bool(
        pd.notna(first)
        and pd.notna(last)
        and first + pd.DateOffset(months=60) <= last
    )
    large_gap = bool(max_gap > 5 or max_missing_run > 5)
    missing_adj = int(numeric["Adjusted Close"].isna().sum())
    missing_volume = int(numeric["Volume"].isna().sum())
    negative_volume = bool((numeric["Volume"].dropna() < 0).any())
    clean = bool(
        len(frame)
        and missing_adj == 0
        and missing_volume == 0
        and duplicate_dates == 0
        and int(nonpositive.sum()) == 0
        and int(logic.sum()) == 0
        and not negative_volume
        and not large_gap
    )
    return {
        "first_price_date": first.date().isoformat() if pd.notna(first) else "unknown",
        "last_price_date": last.date().isoformat() if pd.notna(last) else "unknown",
        "total_observations": int(len(frame)),
        "missing_adj_close": missing_adj,
        "missing_volume": missing_volume,
        "duplicate_dates": duplicate_dates,
        "nonpositive_price_count": int(nonpositive.sum()),
        "ohlc_logic_error_count": int(logic.sum()),
        "large_missing_gap": large_gap,
        "sufficient_60_month_history_at_latest_date": sufficient,
        "clean": clean,
    }


def build_data_audit_v2(
    *,
    metadata_path: Path = ETF_METADATA_PATH,
    raw_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path, dtype=str).fillna("unknown")
    prior_audit = (
        pd.read_csv(DATA_AUDIT_PATH, dtype=str).fillna("")
        if DATA_AUDIT_PATH.exists()
        else pd.DataFrame()
    )
    status_map = (
        prior_audit.drop_duplicates("ticker").set_index("ticker")
        if not prior_audit.empty
        else pd.DataFrame()
    )
    resolutions = (
        pd.read_csv(PRICE_ANOMALY_RESOLUTION_PATH, dtype=str).fillna("")
        if PRICE_ANOMALY_RESOLUTION_PATH.exists()
        else pd.DataFrame()
    )
    resolution_map = (
        resolutions.drop_duplicates("ticker").set_index("ticker")
        if not resolutions.empty
        else pd.DataFrame()
    )
    cached: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    for record in metadata.itertuples(index=False):
        ticker = str(record.ticker).upper()
        if ticker not in cached:
            try:
                cached[ticker] = read_cached_prices(ticker, raw_dir)
            except Exception:
                cached[ticker] = pd.DataFrame(columns=PRICE_COLUMNS)
        prices = _slice_lifecycle(
            cached[ticker], record.listing_date, record.delisting_date
        )
        metrics = _price_metrics(prices)
        if ticker in resolution_map.index:
            price_status = resolution_map.at[ticker, "final_status"]
        elif metrics["clean"]:
            price_status = "clean"
        elif len(prices) == 0:
            price_status = "missing_or_unmatched_price_data"
        else:
            price_status = "price_anomalies_detected"
        download_status = (
            status_map.at[ticker, "download_status"]
            if not status_map.empty and ticker in status_map.index
            else ("cached" if len(prices) else "not_downloaded")
        )
        row = {
            "lifecycle_id": record.lifecycle_id,
            "ticker": ticker,
            "fund_name": record.fund_name,
            "listing_date": record.listing_date,
            "delisting_date": record.delisting_date,
            **{key: value for key, value in metrics.items() if key != "clean"},
            "instrument_type": record.instrument_type,
            "asset_class": record.asset_class,
            "geographic_scope": record.geographic_scope,
            "leveraged_flag": record.leveraged_flag,
            "inverse_flag": record.inverse_flag,
            "single_stock_flag": record.single_stock_flag,
            "etn_flag": record.etn_flag,
            "thematic_flag": record.thematic_flag,
            "eligible_us_equity_etf": record.eligible_us_equity_etf,
            "review_status": record.review_status,
            "exclusion_reason": record.exclusion_reason,
            "price_data_status": price_status,
            "lifecycle_conflict": record.lifecycle_conflict,
            "download_status": download_status,
        }
        rows.append(row)
    audit = pd.DataFrame(rows, columns=AUDIT_V2_COLUMNS)
    _atomic_csv(audit, DATA_AUDIT_V2_PATH)
    return audit


def refresh_audit_metadata(
    audit_path: Path = DATA_AUDIT_V2_PATH,
    metadata_path: Path = ETF_METADATA_PATH,
) -> pd.DataFrame:
    """Refresh classification columns when price files have not changed."""
    audit = pd.read_csv(audit_path, dtype=str).fillna("unknown")
    metadata = pd.read_csv(metadata_path, dtype=str).fillna("unknown")
    if audit["lifecycle_id"].duplicated().any():
        raise AssertionError("data_audit_v2 contains duplicate lifecycle_id values")
    if metadata["lifecycle_id"].duplicated().any():
        raise AssertionError("etf_metadata contains duplicate lifecycle_id values")
    metadata_map = metadata.set_index("lifecycle_id")
    if set(audit["lifecycle_id"]) != set(metadata_map.index):
        raise AssertionError("Audit and metadata lifecycle_id sets do not match")
    refresh_columns = [
        "fund_name",
        "listing_date",
        "delisting_date",
        "instrument_type",
        "asset_class",
        "geographic_scope",
        "leveraged_flag",
        "inverse_flag",
        "single_stock_flag",
        "etn_flag",
        "thematic_flag",
        "eligible_us_equity_etf",
        "review_status",
        "exclusion_reason",
        "lifecycle_conflict",
    ]
    ordered_ids = audit["lifecycle_id"]
    for column in refresh_columns:
        audit[column] = metadata_map.loc[ordered_ids, column].to_numpy()
    audit = audit[AUDIT_V2_COLUMNS]
    _atomic_csv(audit, audit_path)
    return audit


def assert_clean_universe(
    clean_path: Path = CLEAN_US_EQUITY_ETFS_PATH,
    audit_path: Path = DATA_AUDIT_V2_PATH,
) -> None:
    """Fail loudly if any prohibited or unknown record enters the clean universe."""
    clean = pd.read_csv(clean_path, dtype=str).fillna("unknown")
    audit = pd.read_csv(audit_path, dtype=str).fillna("unknown")
    required_known = [
        "listing_date",
        "instrument_type",
        "asset_class",
        "geographic_scope",
        "leveraged_flag",
        "inverse_flag",
        "single_stock_flag",
        "etn_flag",
        "option_overlay_flag",
    ]
    for column in required_known:
        assert not clean[column].str.lower().isin({"", "unknown", "nan", "none"}).any(), column
    assert clean["instrument_type"].eq("ETF").all()
    assert clean["asset_class"].eq("equity").all()
    assert clean["geographic_scope"].eq("United States").all()
    for column in (
        "leveraged_flag",
        "inverse_flag",
        "single_stock_flag",
        "etn_flag",
        "option_overlay_flag",
        "thematic_flag",
        "lifecycle_conflict",
    ):
        assert clean[column].str.lower().eq("false").all(), column
    assert clean["eligible_us_equity_etf"].str.lower().eq("true").all()
    assert clean["review_status"].eq("clean").all()
    clean_audit = audit[audit["lifecycle_id"].isin(clean["lifecycle_id"])]
    assert len(clean_audit) == len(clean)
    assert clean_audit["price_data_status"].eq("clean").all()
    assert clean_audit["nonpositive_price_count"].astype(int).eq(0).all()
    assert clean_audit["ohlc_logic_error_count"].astype(int).eq(0).all()
    assert clean_audit["large_missing_gap"].str.lower().eq("false").all()
