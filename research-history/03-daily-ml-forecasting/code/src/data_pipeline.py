"""Command-line entry point for ETF metadata, price caching, and data auditing."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from src.data_loader import audit_cached_prices, download_prices, write_data_audit
from src.logging_config import configure_logging
from src.paths import CONFIG_PATH, DATA_AUDIT_PATH, ENV_PATH, ETF_METADATA_PATH
from src.universe_builder import acquire_etf_candidates


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire ETF candidates, incrementally cache OHLCV, and audit data."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--tickers", nargs="+", help="Optional explicit ticker subset")
    parser.add_argument("--limit", type=int, help="Optional deterministic smoke-test limit")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--price-source",
        choices=("auto", "yfinance", "yahoo_chart"),
        default="auto",
    )
    parser.add_argument("--chart-workers", type=int, default=8)
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for required in ("data_start_date", "end_date", "minimum_history_months"):
        if required not in config:
            raise KeyError(f"Missing config field: {required}")
    return config


def run_data_stage(
    *,
    config_path: Path = CONFIG_PATH,
    tickers: list[str] | None = None,
    limit: int | None = None,
    batch_size: int = 50,
    price_source: str = "auto",
    chart_workers: int = 8,
    metadata_only: bool = False,
) -> pd.DataFrame:
    """Run only the project data-acquisition and audit stage."""
    load_dotenv(ENV_PATH, override=False)
    logger = logging.getLogger("etf_portfolio.data")
    config = _load_config(config_path)
    metadata_result = acquire_etf_candidates(output_path=ETF_METADATA_PATH)
    metadata = metadata_result.metadata
    logger.info("Metadata source: %s; rows: %s", metadata_result.source, len(metadata))
    if metadata_result.warning:
        logger.warning(metadata_result.warning)

    if tickers:
        wanted = {ticker.upper() for ticker in tickers}
        selected = metadata[metadata["ticker"].isin(wanted)].copy()
        missing = sorted(wanted - set(selected["ticker"]))
        if missing:
            additions = pd.DataFrame(
                {
                    "ticker": missing,
                    "fund_name": "unknown",
                    "asset_type": "ETF",
                    "exchange": "unknown",
                    "listing_date": "unknown",
                    "delisting_date": "unknown",
                    "status": "needs_review",
                    "metadata_status": "needs_review",
                    "point_in_time_coverage": "unknown",
                    "source": "user_supplied_ticker",
                    "source_as_of": pd.Timestamp.today().date().isoformat(),
                    "notes": "Ticker supplied explicitly; ETF identity requires review.",
                }
            )
            metadata = pd.concat([metadata, additions], ignore_index=True)
            selected = pd.concat([selected, additions], ignore_index=True)
        metadata = metadata.sort_values("ticker", ignore_index=True)
        from src.universe_builder import write_metadata

        write_metadata(metadata, ETF_METADATA_PATH)
    else:
        selected = metadata.drop_duplicates("ticker", keep="last")

    if limit is not None:
        selected = selected.sort_values("ticker").head(limit)

    if metadata_only:
        return pd.DataFrame()

    def show_progress(done: int, total: int) -> None:
        if done == total or done % max(batch_size, 1) == 0:
            logger.info("Price download progress: %s/%s", done, total)

    results = download_prices(
        selected["ticker"],
        start_date=str(config["data_start_date"]),
        end_date=(str(config["end_date"]) if config.get("end_date") else None),
        batch_size=batch_size,
        provider=price_source,
        chart_workers=chart_workers,
        progress=show_progress,
    )
    audit = audit_cached_prices(selected, results)
    write_data_audit(audit, DATA_AUDIT_PATH)
    logger.info("Data audit written to %s", DATA_AUDIT_PATH)
    return audit


def main() -> None:
    args = _arguments()
    configure_logging()
    run_data_stage(
        config_path=args.config,
        tickers=args.tickers,
        limit=args.limit,
        batch_size=args.batch_size,
        price_source=args.price_source,
        chart_workers=args.chart_workers,
        metadata_only=args.metadata_only,
    )


if __name__ == "__main__":
    main()
