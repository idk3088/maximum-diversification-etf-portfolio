"""Offline unit tests for Step 3 backward-looking features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import FeatureConfig, build_features


def synthetic_prices(rows: int = 330, ticker: str = "TEST") -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=rows)
    adjusted = pd.Series(100.0 + np.arange(rows) * 0.25)
    close = adjusted + 0.10
    return pd.DataFrame(
        {
            "ticker": ticker,
            "Date": dates,
            "Open": close - 0.20,
            "High": close + 0.50,
            "Low": close - 0.50,
            "Close": close,
            "Adjusted Close": adjusted,
            "Volume": 1_000_000 + np.arange(rows) * 1_000,
        }
    )


def test_return_21d_formula_and_rolling_boundary() -> None:
    prices = synthetic_prices()
    result = build_features(prices, "TEST", FeatureConfig()).features
    assert result["return_21d"].iloc[:21].isna().all()
    expected = prices["Adjusted Close"].iloc[21] / prices["Adjusted Close"].iloc[0] - 1
    assert result["return_21d"].iloc[21] == pytest.approx(expected)
    assert result["sma_21"].iloc[:20].isna().all()
    assert result["sma_21"].iloc[20] == pytest.approx(
        prices["Adjusted Close"].iloc[:21].mean()
    )


def test_volatility_21d_uses_only_trailing_21_daily_returns() -> None:
    prices = synthetic_prices()
    features = build_features(prices, "TEST", FeatureConfig()).features
    position = 40
    daily = prices["Adjusted Close"].pct_change(fill_method=None)
    expected = daily.iloc[position - 20 : position + 1].std(ddof=1)
    assert features["volatility_21d"].iloc[position] == pytest.approx(expected)


def test_rsi_and_drawdown_do_not_change_when_future_prices_change() -> None:
    original = synthetic_prices()
    modified = original.copy()
    cutoff = 200
    modified.loc[cutoff + 1 :, "Adjusted Close"] *= 10
    first = build_features(original, "TEST", FeatureConfig()).features
    second = build_features(modified, "TEST", FeatureConfig()).features
    for column in ["RSI_14", "rolling_max_drawdown_21d", "rolling_max_drawdown_63d"]:
        pd.testing.assert_series_equal(
            first.loc[:cutoff, column], second.loc[:cutoff, column], check_names=False
        )


def test_overnight_dollar_volume_and_adv_formulas() -> None:
    prices = synthetic_prices()
    features = build_features(prices, "TEST", FeatureConfig()).features
    position = 25
    expected_gap = prices["Open"].iloc[position] / prices["Close"].iloc[position - 1] - 1
    expected_dollar_volume = prices["Close"] * prices["Volume"]
    assert features["overnight_gap"].iloc[position] == pytest.approx(expected_gap)
    assert features["dollar_volume"].iloc[position] == pytest.approx(
        expected_dollar_volume.iloc[position]
    )
    assert features["ADV20"].iloc[position] == pytest.approx(
        expected_dollar_volume.iloc[position - 19 : position + 1].mean()
    )


def test_invalid_ohlc_is_nan_and_classified_without_imputation() -> None:
    prices = synthetic_prices()
    prices.loc[30, "Open"] = 0
    result = build_features(prices, "TEST", FeatureConfig())
    assert pd.isna(result.features.loc[30, "open_close_return"])
    assert result.missing_reasons["open_close_return"].iloc[30] == "calculation_invalid"
    assert result.features["return_252d"].iloc[:252].isna().all()


def test_mixed_tickers_are_rejected_before_rolling() -> None:
    prices = synthetic_prices()
    prices.loc[100, "ticker"] = "OTHER"
    with pytest.raises(ValueError, match="exactly one ticker"):
        build_features(prices, "TEST", FeatureConfig())
