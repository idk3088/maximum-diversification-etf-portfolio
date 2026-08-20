"""Acceptance tests for the conservative clean U.S. equity ETF universe."""

from src.audit_v2 import assert_clean_universe


def test_clean_universe_has_no_unknown_conflict_or_price_anomaly() -> None:
    assert_clean_universe()
