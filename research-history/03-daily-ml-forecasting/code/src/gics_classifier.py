"""Point-in-time GICS sector naming and effective-date rules for Step 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd


CURRENT_GICS_SECTORS: Final = (
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

REAL_ESTATE_CLASSIFICATION_EFFECTIVE_DATE: Final = pd.Timestamp("2016-08-31")
REAL_ESTATE_EFFECTIVE_TRADE_DATE: Final = pd.Timestamp("2016-09-01")
COMMUNICATION_CLASSIFICATION_EFFECTIVE_DATE: Final = pd.Timestamp("2018-09-28")
COMMUNICATION_EFFECTIVE_TRADE_DATE: Final = pd.Timestamp("2018-10-01")

# These two funds existed as broad Telecommunication Services sector funds
# before the 2018 GICS restructuring and continued as Communication Services
# sector funds afterwards. Later-launched/current-only products are not
# backfilled into the earlier taxonomy.
LEGACY_TELECOMMUNICATION_FUNDS: Final = frozenset({"VOX", "FCOM"})


@dataclass(frozen=True)
class PointInTimeClassification:
    """The classification that may be used for the next trade date."""

    historical_sector_name: str
    classification_valid: bool
    classification_level_at_date: str
    classification_effective_date: pd.Timestamp | None


def active_sector_names(effective_trade_date: pd.Timestamp) -> tuple[str, ...]:
    """Return the GICS Sector names effective for the next holding period."""
    names = list(CURRENT_GICS_SECTORS)
    if effective_trade_date < COMMUNICATION_EFFECTIVE_TRADE_DATE:
        names[names.index("Communication Services")] = "Telecommunication Services"
    if effective_trade_date < REAL_ESTATE_EFFECTIVE_TRADE_DATE:
        names.remove("Real Estate")
    return tuple(names)


def classify_sector_for_period(
    *,
    ticker: str,
    current_sector: str,
    listing_date: pd.Timestamp,
    selection_date: pd.Timestamp,
    effective_trade_date: pd.Timestamp,
) -> PointInTimeClassification:
    """Apply documented GICS transitions without backcasting current labels."""
    if selection_date < listing_date:
        return PointInTimeClassification(
            current_sector, False, "sector", None
        )

    if current_sector == "Real Estate":
        valid = effective_trade_date >= REAL_ESTATE_EFFECTIVE_TRADE_DATE
        return PointInTimeClassification(
            "Real Estate",
            valid,
            "sector" if valid else "industry_group",
            REAL_ESTATE_CLASSIFICATION_EFFECTIVE_DATE if valid else None,
        )

    if current_sector == "Communication Services":
        if effective_trade_date >= COMMUNICATION_EFFECTIVE_TRADE_DATE:
            return PointInTimeClassification(
                "Communication Services",
                True,
                "sector",
                COMMUNICATION_CLASSIFICATION_EFFECTIVE_DATE,
            )
        valid = ticker in LEGACY_TELECOMMUNICATION_FUNDS
        return PointInTimeClassification(
            "Telecommunication Services",
            valid,
            "sector" if valid else "unclassified",
            listing_date if valid else None,
        )

    return PointInTimeClassification(
        current_sector, True, "sector", listing_date
    )
