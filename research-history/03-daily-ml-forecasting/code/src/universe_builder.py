"""Acquire ETF lifecycle metadata without inventing historical eligibility.

Alpha Vantage LISTING_STATUS is the preferred source because it exposes listing
and delisting dates. When no key is available, the fallback is a current Nasdaq
Trader symbol-directory snapshot. That fallback is explicitly marked as
current-only and must not be used to reconstruct a historical point-in-time
universe.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Final

import pandas as pd
import requests

from src.paths import ETF_METADATA_PATH, METADATA_SNAPSHOT_DIR

ALPHA_VANTAGE_URL: Final = "https://www.alphavantage.co/query"
NASDAQ_LISTED_URL: Final = (
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
)
OTHER_LISTED_URL: Final = (
    "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
)

METADATA_COLUMNS: Final = [
    "ticker",
    "fund_name",
    "asset_type",
    "exchange",
    "listing_date",
    "delisting_date",
    "status",
    "metadata_status",
    "point_in_time_coverage",
    "source",
    "source_as_of",
    "notes",
]

UNKNOWN_VALUES: Final = {"", "none", "nan", "nat", "null", "unknown"}


@dataclass(frozen=True)
class MetadataBuildResult:
    """Summary of one candidate-list acquisition run."""

    metadata: pd.DataFrame
    source: str
    warning: str


class MetadataSourceError(RuntimeError):
    """Raised when a metadata provider returns an unusable response."""


def _clean_text(value: object, default: str = "unknown") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    return default if text.lower() in UNKNOWN_VALUES else text


def _clean_date(value: object) -> str:
    text = _clean_text(value)
    if text == "unknown":
        return text
    parsed = pd.to_datetime(text, errors="coerce")
    return "unknown" if pd.isna(parsed) else parsed.date().isoformat()


def _column_or_unknown(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series("unknown", index=frame.index, dtype="object")


def _request_text(url: str, *, params: dict[str, str] | None = None) -> str:
    response = requests.get(
        url,
        params=params,
        timeout=60,
        headers={"User-Agent": "ETF-Portfolio-2.0/1.0"},
    )
    response.raise_for_status()
    if not response.text.strip():
        raise MetadataSourceError(f"Empty response from {response.url}")
    return response.text


def _save_source_snapshot(text: str, source_name: str) -> None:
    METADATA_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = METADATA_SNAPSHOT_DIR / f"{source_name}_{date.today().isoformat()}.txt"
    temporary = snapshot.with_suffix(snapshot.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(snapshot)


def fetch_alpha_vantage_etfs(api_key: str) -> pd.DataFrame:
    """Fetch active and delisted US ETFs from Alpha Vantage LISTING_STATUS."""
    frames: list[pd.DataFrame] = []
    for state in ("active", "delisted"):
        text = _request_text(
            ALPHA_VANTAGE_URL,
            params={
                "function": "LISTING_STATUS",
                "state": state,
                "apikey": api_key,
            },
        )
        _save_source_snapshot(text, f"alpha_vantage_{state}")
        frame = pd.read_csv(StringIO(text), dtype=str)
        required = {"symbol", "name", "exchange", "assetType"}
        if not required.issubset(frame.columns):
            preview = text[:200].replace("\n", " ")
            raise MetadataSourceError(
                f"Alpha Vantage returned no listing table for state={state}: {preview}"
            )
        frame = frame[frame["assetType"].str.upper().eq("ETF")].copy()
        frame["requested_state"] = state
        frames.append(frame)

    source = pd.concat(frames, ignore_index=True)
    output = pd.DataFrame(
        {
            "ticker": source["symbol"].map(_clean_text).str.upper(),
            "fund_name": source["name"].map(_clean_text),
            "asset_type": "ETF",
            "exchange": source["exchange"].map(_clean_text),
            "listing_date": _column_or_unknown(source, "ipoDate").map(_clean_date),
            "delisting_date": _column_or_unknown(source, "delistingDate").map(
                _clean_date
            ),
            "status": (
                source["status"] if "status" in source else source["requested_state"]
            ).map(_clean_text),
            "metadata_status": "confirmed_from_provider",
            "point_in_time_coverage": "listing_and_delisting_dates",
            "source": "alpha_vantage_listing_status",
            "source_as_of": date.today().isoformat(),
            "notes": "",
        }
    )
    return output[METADATA_COLUMNS].sort_values(
        ["ticker", "listing_date", "delisting_date"], ignore_index=True
    )


def _parse_pipe_table(text: str) -> pd.DataFrame:
    rows = list(csv.DictReader(StringIO(text), delimiter="|"))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise MetadataSourceError("Nasdaq Trader symbol directory was empty")
    first_column = frame.columns[0]
    return frame[
        ~frame[first_column].fillna("").str.startswith("File Creation Time")
    ].copy()


def fetch_nasdaq_current_etfs() -> pd.DataFrame:
    """Fetch a current-only ETF snapshot from Nasdaq Trader symbol directories."""
    nasdaq_text = _request_text(NASDAQ_LISTED_URL)
    other_text = _request_text(OTHER_LISTED_URL)
    _save_source_snapshot(nasdaq_text, "nasdaq_listed")
    _save_source_snapshot(other_text, "other_listed")

    nasdaq = _parse_pipe_table(nasdaq_text)
    other = _parse_pipe_table(other_text)

    nasdaq = nasdaq[
        nasdaq["ETF"].eq("Y") & nasdaq["Test Issue"].eq("N")
    ].copy()
    other = other[other["ETF"].eq("Y") & other["Test Issue"].eq("N")].copy()

    first = pd.DataFrame(
        {
            "ticker": nasdaq["Symbol"],
            "fund_name": nasdaq["Security Name"],
            "exchange": "NASDAQ",
        }
    )
    exchange_names = {
        "A": "NYSE American",
        "N": "NYSE",
        "P": "NYSE Arca",
        "Z": "Cboe BZX",
        "V": "IEX",
    }
    second = pd.DataFrame(
        {
            "ticker": other["ACT Symbol"],
            "fund_name": other["Security Name"],
            "exchange": other["Exchange"].map(exchange_names).fillna("unknown"),
        }
    )
    source = pd.concat([first, second], ignore_index=True)
    source["ticker"] = source["ticker"].map(_clean_text).str.upper()
    source["fund_name"] = source["fund_name"].map(_clean_text)
    source = source[source["ticker"].ne("unknown")].drop_duplicates("ticker")

    source["asset_type"] = "ETF"
    source["listing_date"] = "unknown"
    source["delisting_date"] = "unknown"
    source["status"] = "active"
    source["metadata_status"] = "needs_review"
    source["point_in_time_coverage"] = "current_snapshot_only"
    source["source"] = "nasdaq_trader_symbol_directory"
    source["source_as_of"] = date.today().isoformat()
    source["notes"] = (
        "Current listing snapshot only; listing/delisting dates require Alpha Vantage."
    )
    return source[METADATA_COLUMNS].sort_values("ticker", ignore_index=True)


def _merge_reviewed_values(fresh: pd.DataFrame, existing_path: Path) -> pd.DataFrame:
    """Preserve prior non-unknown values when a fresh source cannot replace them."""
    if not existing_path.exists():
        return fresh
    try:
        existing = pd.read_csv(existing_path, dtype=str).fillna("unknown")
    except (OSError, pd.errors.ParserError):
        return fresh
    if "ticker" not in existing.columns:
        return fresh

    prior = existing.drop_duplicates("ticker", keep="last").set_index("ticker")
    merged = fresh.copy()
    for index, ticker in merged["ticker"].items():
        if ticker not in prior.index:
            continue
        for column in METADATA_COLUMNS:
            if column not in prior.columns or column == "ticker":
                continue
            current = _clean_text(merged.at[index, column])
            previous = _clean_text(prior.at[ticker, column])
            if current == "unknown" and previous != "unknown":
                merged.at[index, column] = previous
    return merged


def write_metadata(metadata: pd.DataFrame, output_path: Path = ETF_METADATA_PATH) -> None:
    """Atomically write the normalized ETF metadata table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = metadata.reindex(columns=METADATA_COLUMNS).fillna("unknown")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    normalized.to_csv(temporary, index=False)
    temporary.replace(output_path)


def acquire_etf_candidates(
    *,
    api_key: str | None = None,
    output_path: Path = ETF_METADATA_PATH,
    allow_nasdaq_fallback: bool = True,
) -> MetadataBuildResult:
    """Acquire and persist the best available ETF candidate metadata."""
    key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY") or os.getenv(
        "MARKET_DATA_API_KEY"
    )
    warning = ""
    if key:
        try:
            metadata = fetch_alpha_vantage_etfs(key)
            source = "alpha_vantage_listing_status"
        except (requests.RequestException, MetadataSourceError, ValueError) as exc:
            if not allow_nasdaq_fallback:
                raise
            warning = f"Alpha Vantage unavailable ({exc}); used current Nasdaq snapshot."
            metadata = fetch_nasdaq_current_etfs()
            source = "nasdaq_trader_symbol_directory"
    else:
        if not allow_nasdaq_fallback:
            raise MetadataSourceError("ALPHA_VANTAGE_API_KEY is not configured")
        warning = (
            "ALPHA_VANTAGE_API_KEY is not configured; listing/delisting dates "
            "remain unknown and historical candidate coverage is incomplete."
        )
        metadata = fetch_nasdaq_current_etfs()
        source = "nasdaq_trader_symbol_directory"

    metadata = _merge_reviewed_values(metadata, output_path)
    write_metadata(metadata, output_path)
    return MetadataBuildResult(metadata=metadata, source=source, warning=warning)


def eligible_lifecycles_as_of(metadata: pd.DataFrame, as_of: str | pd.Timestamp) -> pd.DataFrame:
    """Return only lifecycle rows whose known dates make them eligible as of `as_of`.

    Rows with unknown listing dates are excluded instead of guessed. This helper
    does not apply the later 60-month, leverage, inverse, GICS, or ADV rules.
    """
    point = pd.Timestamp(as_of).normalize()
    listing = pd.to_datetime(metadata["listing_date"], errors="coerce", format="mixed")
    delisting = pd.to_datetime(
        metadata["delisting_date"], errors="coerce", format="mixed"
    )
    known_listing = listing.notna()
    active_on_date = listing.le(point) & (delisting.isna() | delisting.gt(point))
    return metadata.loc[known_listing & active_on_date].copy()
