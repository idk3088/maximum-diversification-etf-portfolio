"""Re-download, cross-check, and explicitly isolate known OHLCV anomalies."""

from __future__ import annotations

import io
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import pandas as pd
import requests

from src.data_loader import (
    PRICE_COLUMNS,
    _download_yahoo_chart,
    _normalize_download,
    _write_cache,
    read_cached_prices,
)
from src.paths import PRICE_ANOMALY_RESOLUTION_PATH, RAW_DATA_DIR

ANOMALY_TICKERS: Final = (
    "BSMV",
    "FTDS",
    "IBCA",
    "INDS",
    "LKOR",
    "MSBT",
    "ROCQ",
    "SMHD",
    "XAGG",
    "XT",
    "AUGC",
    "KEO",
)

RESOLUTION_COLUMNS: Final = [
    "ticker",
    "anomaly_type",
    "anomaly_dates",
    "primary_source_result",
    "secondary_source_result",
    "resolution",
    "rows_removed_or_replaced",
    "evidence",
    "final_status",
]


@dataclass(frozen=True)
class AnomalyDetails:
    types: tuple[str, ...]
    dates: tuple[str, ...]
    invalid_mask: pd.Series
    max_business_gap: int


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _max_business_gap(dates: pd.Series) -> tuple[int, list[str]]:
    ordered = pd.DatetimeIndex(pd.to_datetime(dates, errors="coerce").dropna().unique())
    ordered = ordered.sort_values()
    maximum = 0
    intervals: list[str] = []
    for previous, current in zip(ordered[:-1], ordered[1:]):
        missing = max(int(pd.bdate_range(previous, current).size) - 2, 0)
        if missing > maximum:
            maximum = missing
        if missing > 5:
            intervals.append(
                f"gap:{previous.date().isoformat()}..{current.date().isoformat()}"
            )
    return maximum, intervals


def detect_anomalies(frame: pd.DataFrame) -> AnomalyDetails:
    """Return row-level invalidity without filling or altering source values."""
    if frame.empty:
        empty = pd.Series(dtype=bool)
        return AnomalyDetails(("empty_price_file",), (), empty, 0)
    working = frame.copy()
    dates = pd.to_datetime(working["Date"], errors="coerce")
    numeric = working[PRICE_COLUMNS[1:]].apply(pd.to_numeric, errors="coerce")
    ohlc = numeric[["Open", "High", "Low", "Close"]]
    missing = numeric.isna().any(axis=1) | dates.isna()
    nonpositive = (numeric[["Open", "High", "Low", "Close", "Adjusted Close"]] <= 0).any(axis=1)
    negative_volume = numeric["Volume"] < 0
    logic = (
        (numeric["High"] < numeric["Low"])
        | (numeric["Open"] < numeric["Low"])
        | (numeric["Open"] > numeric["High"])
        | (numeric["Close"] < numeric["Low"])
        | (numeric["Close"] > numeric["High"])
    )
    duplicate = dates.duplicated(keep=False) & dates.notna()
    invalid = missing | nonpositive | negative_volume | logic | duplicate
    types: list[str] = []
    if missing.any():
        types.append("missing_ohlcv")
    if nonpositive.any():
        types.append("nonpositive_price")
    if negative_volume.any():
        types.append("negative_volume")
    if logic.any():
        types.append("ohlc_logic_error")
    if duplicate.any():
        types.append("duplicate_date")
    max_gap, gap_intervals = _max_business_gap(dates)
    if max_gap > 5:
        types.append("large_missing_gap")
    invalid_dates = [
        "unknown" if pd.isna(value) else value.date().isoformat()
        for value in dates[invalid]
    ]
    return AnomalyDetails(
        tuple(types) or ("none",),
        tuple(dict.fromkeys(invalid_dates + gap_intervals)),
        invalid,
        max_gap,
    )


def _download_stooq(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download secondary raw OHLCV evidence; Stooq has no adjusted close."""
    url = "https://stooq.com/q/d/l/"
    response = requests.get(
        url,
        params={
            "s": f"{ticker.lower()}.us",
            "i": "d",
            "d1": start_date.replace("-", ""),
            "d2": end_date.replace("-", ""),
        },
        timeout=45,
        headers={"User-Agent": "Mozilla/5.0 ETF-Portfolio-2.0/1.0"},
    )
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(response.text))
    if frame.empty or "Date" not in frame.columns:
        return pd.DataFrame()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    return frame.dropna(subset=["Date"])


def _secondary_summary(frame: pd.DataFrame, anomaly_dates: tuple[str, ...]) -> str:
    if frame.empty:
        return "stooq_no_data"
    exact_dates = {
        pd.Timestamp(value)
        for value in anomaly_dates
        if value != "unknown" and not value.startswith("gap:")
    }
    matched = frame[pd.to_datetime(frame["Date"]).isin(exact_dates)]
    numeric_columns = [column for column in ("Open", "High", "Low", "Close", "Volume") if column in matched]
    valid = int(matched[numeric_columns].notna().all(axis=1).sum()) if numeric_columns else 0
    return (
        f"stooq_rows={len(frame)};anomaly_dates_found={len(matched)};"
        f"raw_ohlcv_rows_valid={valid};adjusted_close_unavailable"
    )


def resolve_price_anomalies(
    *,
    start_date: str = "2011-08-01",
    end_date: str | None = None,
    raw_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Resolve the named anomalies and preserve the original files in quarantine."""
    end = end_date or pd.Timestamp.today().date().isoformat()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = raw_dir / "quarantine" / f"archive_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for ticker in ANOMALY_TICKERS:
        cache_path = raw_dir / f"{ticker}.csv"
        if cache_path.exists():
            shutil.copy2(cache_path, backup_dir / cache_path.name)
        original = read_cached_prices(ticker, raw_dir)
        original_details = detect_anomalies(original)
        primary_summary = "not_attempted"
        secondary_summary = "not_attempted"
        resolution = "unresolved_needs_review"
        final_status = "unresolved_needs_review"
        change_summary = "0"
        evidence_parts = [
            f"original_rows={len(original)}",
            f"original_invalid_rows={int(original_details.invalid_mask.sum())}",
            f"original_backup={backup_dir / cache_path.name}",
        ]

        try:
            downloaded = _normalize_download(
                _download_yahoo_chart(ticker, start_date, end)
            )
            downloaded_details = detect_anomalies(downloaded)
            primary_summary = (
                f"yahoo_chart_rows={len(downloaded)};"
                f"invalid_rows={int(downloaded_details.invalid_mask.sum())};"
                f"types={'|'.join(downloaded_details.types)}"
            )
            try:
                secondary = _download_stooq(ticker, start_date, end)
                secondary_summary = _secondary_summary(secondary, original_details.dates)
            except Exception as exc:  # secondary failure must not erase primary evidence
                secondary_summary = f"stooq_error={type(exc).__name__}: {exc}"

            if downloaded.empty:
                resolution = "ticker_quarantined"
                final_status = "quarantined_primary_returned_no_data"
            else:
                clean_download = downloaded.loc[~downloaded_details.invalid_mask].copy()
                clean_download = (
                    clean_download.drop_duplicates("Date", keep="last")
                    .sort_values("Date")
                    .reset_index(drop=True)
                )
                removed = len(downloaded) - len(clean_download)
                if clean_download.empty:
                    # Never convert a non-empty cache into a misleading empty
                    # "clean" file when every downloaded row is invalid.
                    resolution = "ticker_quarantined"
                    final_status = "quarantined_no_verified_valid_rows"
                    change_summary = "original_cache_retained;verified_valid_rows=0"
                    evidence_parts.append(
                        "Primary returned no valid rows after strict OHLCV checks"
                    )
                else:
                    _write_cache(ticker, clean_download, raw_dir)
                    change_summary = (
                        f"cache_replaced_from_primary={len(clean_download)};"
                        f"invalid_primary_rows_removed={removed}"
                    )
                    final_details = detect_anomalies(clean_download)
                    if removed:
                        resolution = "invalid_rows_removed"
                    else:
                        resolution = "corrected_from_verified_source"
                    if final_details.max_business_gap > 5:
                        final_status = "isolated_invalid_dates_remaining_gap_needs_review"
                    else:
                        final_status = "clean_after_primary_redownload"
                    evidence_parts.extend(
                        [
                            f"final_rows={len(clean_download)}",
                            f"final_max_missing_business_days={final_details.max_business_gap}",
                            "No forward-fill, zero substitution, or synthetic OHLCV used",
                        ]
                    )
        except Exception as exc:
            primary_summary = f"yahoo_chart_error={type(exc).__name__}: {exc}"
            try:
                secondary = _download_stooq(ticker, start_date, end)
                secondary_summary = _secondary_summary(secondary, original_details.dates)
            except Exception as secondary_exc:
                secondary_summary = (
                    f"stooq_error={type(secondary_exc).__name__}: {secondary_exc}"
                )
            resolution = "ticker_quarantined"
            final_status = "quarantined_primary_download_failed"

        rows.append(
            {
                "ticker": ticker,
                "anomaly_type": "|".join(original_details.types),
                "anomaly_dates": "|".join(original_details.dates),
                "primary_source_result": primary_summary,
                "secondary_source_result": secondary_summary,
                "resolution": resolution,
                "rows_removed_or_replaced": change_summary,
                "evidence": ";".join(evidence_parts),
                "final_status": final_status,
            }
        )

    result = pd.DataFrame(rows, columns=RESOLUTION_COLUMNS)
    _atomic_csv(result, PRICE_ANOMALY_RESOLUTION_PATH)
    return result
