"""Incremental per-ticker OHLCV cache and data-quality audit utilities."""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Final, Iterable

import numpy as np
import pandas as pd

from src.paths import DATA_AUDIT_PATH, RAW_DATA_DIR

PRICE_COLUMNS: Final = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adjusted Close",
    "Volume",
]

AUDIT_COLUMNS: Final = [
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
    "sufficient_60_month_history",
    "download_status",
    "error_message",
    "dates_strictly_increasing",
    "adjusted_close_positive",
    "volume_nonnegative",
    "large_missing_gap",
    "max_missing_business_days",
    "max_consecutive_missing_rows",
    "metadata_status",
    "point_in_time_coverage",
    "price_source",
    "audit_date",
]


@dataclass(frozen=True)
class DownloadResult:
    ticker: str
    status: str
    error_message: str = ""
    new_observations: int = 0
    price_source: str = "unknown"


def _cache_name(ticker: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", ticker.strip().upper())
    if not safe:
        raise ValueError("Ticker cannot be empty")
    return f"{safe}.csv"


def cache_path_for_ticker(ticker: str, raw_dir: Path = RAW_DATA_DIR) -> Path:
    return raw_dir / _cache_name(ticker)


def _yahoo_symbol(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "-")


def read_cached_prices(ticker: str, raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    path = cache_path_for_ticker(ticker, raw_dir)
    if not path.exists():
        return pd.DataFrame(columns=PRICE_COLUMNS)
    frame = pd.read_csv(path)
    missing = set(PRICE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    frame = frame[PRICE_COLUMNS].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    return frame


def validate_step3_price_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Validate one ticker's cached OHLCV without sorting or filling it.

    Step 3 intentionally fails on unordered or duplicate dates so an upstream
    cache problem cannot be hidden by feature construction.
    """
    missing = set(PRICE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{ticker}: missing OHLCV columns {sorted(missing)}")
    validated = frame.loc[:, PRICE_COLUMNS].copy()
    validated["Date"] = pd.to_datetime(validated["Date"], errors="coerce")
    if validated.empty:
        raise ValueError(f"{ticker}: cached price file is empty")
    if validated["Date"].isna().any():
        raise ValueError(f"{ticker}: one or more dates are invalid")
    if validated["Date"].duplicated().any():
        duplicates = validated.loc[
            validated["Date"].duplicated(keep=False), "Date"
        ].dt.strftime("%Y-%m-%d")
        raise ValueError(f"{ticker}: duplicate dates found: {sorted(set(duplicates))}")
    if not validated["Date"].is_monotonic_increasing:
        raise ValueError(f"{ticker}: dates are not strictly increasing")
    for column in PRICE_COLUMNS[1:]:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    validated.insert(0, "ticker", ticker.strip().upper())
    return validated


def load_step3_prices(
    tickers: Iterable[str],
    *,
    raw_dir: Path = RAW_DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """Load isolated local caches for Step 3 without making network requests."""
    normalized = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
    if not normalized:
        raise ValueError("At least one ticker is required")
    if len(normalized) != len(set(normalized)):
        raise ValueError("Ticker list contains duplicates")

    paths = [cache_path_for_ticker(ticker, raw_dir).resolve() for ticker in normalized]
    if len(paths) != len(set(paths)):
        raise ValueError("Ticker cache paths are not isolated")

    loaded: dict[str, pd.DataFrame] = {}
    for ticker, path in zip(normalized, paths, strict=True):
        if not path.exists():
            raise FileNotFoundError(f"{ticker}: local cache not found at {path}")
        loaded[ticker] = validate_step3_price_frame(
            read_cached_prices(ticker, raw_dir), ticker
        )
    return loaded


def _normalize_download(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    output = frame.copy()
    if "Date" not in output.columns and "Datetime" not in output.columns:
        output = output.reset_index()
    output = output.rename(
        columns={
            "Adj Close": "Adjusted Close",
            "Datetime": "Date",
            "index": "Date",
        }
    )
    missing = set(PRICE_COLUMNS) - set(output.columns)
    if missing:
        raise ValueError(f"Downloaded data is missing columns: {sorted(missing)}")
    output = output[PRICE_COLUMNS]
    output["Date"] = pd.to_datetime(output["Date"], errors="coerce", utc=True)
    output["Date"] = output["Date"].dt.tz_convert(None).dt.normalize()
    for column in PRICE_COLUMNS[1:]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output.dropna(subset=["Date"])


def _extract_ticker_frame(download: pd.DataFrame, yahoo_ticker: str) -> pd.DataFrame:
    if download.empty:
        return pd.DataFrame()
    if not isinstance(download.columns, pd.MultiIndex):
        return download
    for level in range(download.columns.nlevels):
        values = download.columns.get_level_values(level).astype(str)
        matches = [value for value in values.unique() if value.upper() == yahoo_ticker]
        if matches:
            return download.xs(matches[0], axis=1, level=level, drop_level=True)
    return pd.DataFrame()


def _download_yahoo_chart(
    yahoo_ticker: str,
    start_date: str,
    end_date: str | None,
) -> pd.DataFrame:
    """Download one ticker from Yahoo's chart response, including adjclose."""
    import requests

    start = pd.Timestamp(start_date, tz="UTC")
    end = (
        pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
        if end_date
        else pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_ticker}"
    params = {
        "period1": str(int(start.timestamp())),
        "period2": str(int(end.timestamp())),
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    last_error = ""
    for attempt in range(3):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=45,
                headers={"User-Agent": "Mozilla/5.0 ETF-Portfolio-2.0/1.0"},
            )
            if response.status_code in {429, 500, 502, 503, 504}:
                last_error = f"HTTP {response.status_code}"
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            chart = response.json().get("chart", {})
            if chart.get("error"):
                error = chart["error"]
                raise ValueError(f"{error.get('code')}: {error.get('description')}")
            results = chart.get("result") or []
            if not results:
                return pd.DataFrame()
            result = results[0]
            timestamps = result.get("timestamp") or []
            quotes = (result.get("indicators", {}).get("quote") or [{}])[0]
            adjusted = (result.get("indicators", {}).get("adjclose") or [{}])[0]
            if not timestamps:
                return pd.DataFrame()
            length = len(timestamps)

            def values(name: str, source: dict[str, list[object]]) -> list[object]:
                found = source.get(name) or []
                return (list(found) + [None] * max(length - len(found), 0))[:length]

            return pd.DataFrame(
                {
                    "Date": pd.to_datetime(timestamps, unit="s", utc=True),
                    "Open": values("open", quotes),
                    "High": values("high", quotes),
                    "Low": values("low", quotes),
                    "Close": values("close", quotes),
                    "Adjusted Close": values("adjclose", adjusted),
                    "Volume": values("volume", quotes),
                }
            )
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"Yahoo Chart failed after retries: {last_error}")


def _write_cache(ticker: str, frame: pd.DataFrame, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path_for_ticker(ticker, raw_dir)
    output = frame.copy()
    output["Date"] = pd.to_datetime(output["Date"]).dt.strftime("%Y-%m-%d")
    temporary = path.with_suffix(path.suffix + ".tmp")
    output.to_csv(temporary, index=False)
    temporary.replace(path)


def _merge_prices(cached: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([cached, incoming], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined = combined.dropna(subset=["Date"])
    return (
        combined.drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date")
        .reset_index(drop=True)[PRICE_COLUMNS]
    )


def _incremental_start(cached: pd.DataFrame, configured_start: str) -> str:
    if cached.empty or cached["Date"].dropna().empty:
        return pd.Timestamp(configured_start).date().isoformat()
    next_date = pd.Timestamp(cached["Date"].max()).date() + timedelta(days=1)
    return max(next_date, pd.Timestamp(configured_start).date()).isoformat()


def download_prices(
    tickers: Iterable[str],
    *,
    start_date: str,
    end_date: str | None = None,
    raw_dir: Path = RAW_DATA_DIR,
    batch_size: int = 50,
    provider: str = "auto",
    chart_workers: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> list[DownloadResult]:
    """Incrementally update per-ticker CSV caches from Yahoo Finance.

    Existing caches determine each ticker's next start date. Downloads are
    grouped by that date and split into bounded batches. Each ticker is written
    atomically, so rerunning after interruption continues from its last cache.
    """
    if provider not in {"auto", "yfinance", "yahoo_chart"}:
        raise ValueError("provider must be auto, yfinance, or yahoo_chart")
    yf = None
    if provider in {"auto", "yfinance"}:
        try:
            import yfinance as yf_module

            yf = yf_module
        except ModuleNotFoundError:
            if provider == "yfinance":
                raise RuntimeError(
                    "yfinance is required only for price downloading; install it "
                    "with 'python -m pip install yfinance'"
                ) from None
    unique = sorted({ticker.strip().upper() for ticker in tickers if ticker.strip()})
    cached_by_ticker: dict[str, pd.DataFrame] = {}
    groups: dict[str, list[str]] = {}
    initial_errors: dict[str, str] = {}

    for ticker in unique:
        try:
            cached = read_cached_prices(ticker, raw_dir)
            cached_by_ticker[ticker] = cached
            incremental_start = _incremental_start(cached, start_date)
            groups.setdefault(incremental_start, []).append(ticker)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            initial_errors[ticker] = f"Invalid existing cache: {exc}"

    results: dict[str, DownloadResult] = {
        ticker: DownloadResult(ticker, "failed", message)
        for ticker, message in initial_errors.items()
    }
    processed = len(results)
    total = len(unique)
    yahoo_end = (
        (pd.Timestamp(end_date).date() + timedelta(days=1)).isoformat()
        if end_date
        else None
    )

    for group_start, group_tickers in sorted(groups.items()):
        if end_date and pd.Timestamp(group_start) > pd.Timestamp(end_date):
            for ticker in group_tickers:
                results[ticker] = DownloadResult(ticker, "up_to_date")
                processed += 1
                if progress:
                    progress(processed, total)
            continue

        for offset in range(0, len(group_tickers), batch_size):
            batch = group_tickers[offset : offset + batch_size]
            yahoo_symbols = {_yahoo_symbol(ticker): ticker for ticker in batch}
            downloaded = pd.DataFrame()
            batch_error = ""
            if provider in {"auto", "yfinance"} and yf is not None:
                try:
                    downloaded = yf.download(
                        list(yahoo_symbols),
                        start=group_start,
                        end=yahoo_end,
                        interval="1d",
                        auto_adjust=False,
                        actions=False,
                        threads=True,
                        group_by="ticker",
                        progress=False,
                        repair=True,
                        keepna=True,
                        multi_level_index=True,
                        timeout=30,
                    )
                except Exception as exc:  # provider/client exceptions vary by version
                    batch_error = (
                        f"Yahoo Finance batch error: {type(exc).__name__}: {exc}"
                    )

            incoming_by_ticker: dict[str, pd.DataFrame] = {}
            source_by_ticker: dict[str, str] = {}
            fallback: dict[str, str] = {}
            extraction_errors: dict[str, str] = {}
            for yahoo_ticker, ticker in yahoo_symbols.items():
                try:
                    incoming = _normalize_download(
                        _extract_ticker_frame(downloaded, yahoo_ticker)
                    )
                    if incoming.empty and provider in {"auto", "yahoo_chart"}:
                        fallback[yahoo_ticker] = ticker
                    else:
                        incoming_by_ticker[ticker] = incoming
                        source_by_ticker[ticker] = "yfinance_yahoo"
                except Exception as exc:
                    extraction_errors[ticker] = f"{type(exc).__name__}: {exc}"
                    if provider in {"auto", "yahoo_chart"}:
                        fallback[yahoo_ticker] = ticker

            if fallback:
                with ThreadPoolExecutor(max_workers=max(1, chart_workers)) as executor:
                    futures = {
                        executor.submit(
                            _download_yahoo_chart,
                            yahoo_ticker,
                            group_start,
                            end_date,
                        ): (yahoo_ticker, ticker)
                        for yahoo_ticker, ticker in fallback.items()
                    }
                    for future in as_completed(futures):
                        _, ticker = futures[future]
                        try:
                            incoming_by_ticker[ticker] = _normalize_download(future.result())
                            source_by_ticker[ticker] = "yahoo_chart_api"
                        except Exception as exc:
                            extraction_errors[ticker] = f"{type(exc).__name__}: {exc}"

            for yahoo_ticker, ticker in yahoo_symbols.items():
                cached = cached_by_ticker[ticker]
                try:
                    incoming = incoming_by_ticker.get(
                        ticker, pd.DataFrame(columns=PRICE_COLUMNS)
                    )
                    if incoming.empty:
                        if cached.empty:
                            message = (
                                extraction_errors.get(ticker)
                                or batch_error
                                or "Yahoo Finance returned no daily data"
                            )
                            result = DownloadResult(
                                ticker, "failed", message, price_source="unknown"
                            )
                        else:
                            result = DownloadResult(
                                ticker,
                                "up_to_date",
                                extraction_errors.get(ticker) or batch_error,
                                price_source="cache",
                            )
                    else:
                        merged = _merge_prices(cached, incoming)
                        _write_cache(ticker, merged, raw_dir)
                        status = "downloaded" if cached.empty else "updated"
                        result = DownloadResult(
                            ticker,
                            status,
                            "",
                            len(incoming),
                            source_by_ticker.get(ticker, "unknown"),
                        )
                except Exception as exc:
                    result = DownloadResult(
                        ticker,
                        "failed",
                        f"{type(exc).__name__}: {exc}",
                    )
                results[ticker] = result
                processed += 1
                if progress:
                    progress(processed, total)

    return [results[ticker] for ticker in unique]


def _max_missing_business_days(dates: pd.Series) -> int:
    unique = pd.DatetimeIndex(pd.to_datetime(dates, errors="coerce").dropna().unique())
    unique = unique.sort_values()
    if len(unique) < 2:
        return 0
    previous = unique[:-1].values.astype("datetime64[D]")
    current = unique[1:].values.astype("datetime64[D]")
    missing = np.busday_count(previous, current) - 1
    return int(max(missing.max(initial=0), 0))


def _max_consecutive_true(values: pd.Series) -> int:
    maximum = current = 0
    for value in values.fillna(False).astype(bool):
        if value:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def audit_cached_prices(
    metadata: pd.DataFrame,
    download_results: Iterable[DownloadResult],
    *,
    raw_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Audit each unique ticker's cache and return the required audit table."""
    result_map = {result.ticker: result for result in download_results}
    rows: list[dict[str, object]] = []
    for ticker, lifecycle_rows in metadata.groupby("ticker", sort=True):
        latest = lifecycle_rows.iloc[-1]
        result = result_map.get(
            ticker,
            DownloadResult(ticker, "not_attempted", "Ticker was not submitted"),
        )
        try:
            prices = read_cached_prices(ticker, raw_dir)
            dates = pd.to_datetime(prices["Date"], errors="coerce")
            valid_dates = dates.dropna()
            duplicate_dates = int(valid_dates.duplicated(keep=False).sum())
            first_date = valid_dates.min() if not valid_dates.empty else pd.NaT
            last_date = valid_dates.max() if not valid_dates.empty else pd.NaT
            sufficient = bool(
                pd.notna(first_date)
                and pd.notna(last_date)
                and first_date + pd.DateOffset(months=60) <= last_date
            )
            max_gap = _max_missing_business_days(valid_dates)
            adj = pd.to_numeric(prices["Adjusted Close"], errors="coerce")
            volume = pd.to_numeric(prices["Volume"], errors="coerce")
            numeric = prices[PRICE_COLUMNS[1:]].apply(pd.to_numeric, errors="coerce")
            max_missing_run = _max_consecutive_true(numeric.isna().any(axis=1))
            row = {
                "ticker": ticker,
                "fund_name": latest.get("fund_name", "unknown"),
                "listing_date": latest.get("listing_date", "unknown"),
                "delisting_date": latest.get("delisting_date", "unknown"),
                "first_price_date": (
                    first_date.date().isoformat() if pd.notna(first_date) else "unknown"
                ),
                "last_price_date": (
                    last_date.date().isoformat() if pd.notna(last_date) else "unknown"
                ),
                "total_observations": int(len(prices)),
                "missing_adj_close": int(adj.isna().sum()),
                "missing_volume": int(volume.isna().sum()),
                "duplicate_dates": duplicate_dates,
                "sufficient_60_month_history": sufficient,
                "download_status": result.status,
                "error_message": result.error_message,
                "dates_strictly_increasing": bool(
                    valid_dates.is_monotonic_increasing and duplicate_dates == 0
                ),
                "adjusted_close_positive": bool(
                    len(prices) > 0 and (adj.dropna() > 0).all()
                ),
                "volume_nonnegative": bool(
                    len(prices) > 0 and (volume.dropna() >= 0).all()
                ),
                "large_missing_gap": bool(max_gap > 5 or max_missing_run > 5),
                "max_missing_business_days": max_gap,
                "max_consecutive_missing_rows": max_missing_run,
                "metadata_status": latest.get("metadata_status", "needs_review"),
                "point_in_time_coverage": latest.get(
                    "point_in_time_coverage", "unknown"
                ),
                "price_source": result.price_source,
                "audit_date": date.today().isoformat(),
            }
        except Exception as exc:
            row = {
                "ticker": ticker,
                "fund_name": latest.get("fund_name", "unknown"),
                "listing_date": latest.get("listing_date", "unknown"),
                "delisting_date": latest.get("delisting_date", "unknown"),
                "first_price_date": "unknown",
                "last_price_date": "unknown",
                "total_observations": 0,
                "missing_adj_close": 0,
                "missing_volume": 0,
                "duplicate_dates": 0,
                "sufficient_60_month_history": False,
                "download_status": "failed",
                "error_message": f"Audit error: {type(exc).__name__}: {exc}",
                "dates_strictly_increasing": False,
                "adjusted_close_positive": False,
                "volume_nonnegative": False,
                "large_missing_gap": False,
                "max_missing_business_days": 0,
                "max_consecutive_missing_rows": 0,
                "metadata_status": latest.get("metadata_status", "needs_review"),
                "point_in_time_coverage": latest.get(
                    "point_in_time_coverage", "unknown"
                ),
                "price_source": result.price_source,
                "audit_date": date.today().isoformat(),
            }
        rows.append(row)
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def write_data_audit(
    audit: pd.DataFrame, output_path: Path = DATA_AUDIT_PATH
) -> None:
    """Atomically write the data audit CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    audit.reindex(columns=AUDIT_COLUMNS).to_csv(temporary, index=False)
    temporary.replace(output_path)
