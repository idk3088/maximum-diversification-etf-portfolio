"""Strictly backward-looking daily OHLCV features for Step 3.

This module deliberately contains no target construction, fitting, scaling,
imputation, feature selection, or cross-ticker rolling operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data_loader import PRICE_COLUMNS


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration needed by the deterministic feature calculations."""

    return_windows: tuple[int, ...] = (1, 5, 10, 21, 63, 126, 252)
    sma_windows: tuple[int, ...] = (5, 10, 21, 63, 126, 252)
    volatility_windows: tuple[int, ...] = (5, 10, 21, 63, 126, 252)
    drawdown_windows: tuple[int, ...] = (21, 63, 252)
    rsi_window: int = 14
    adv_window_days: int = 20
    annualize_volatility: bool = False

    @classmethod
    def from_mapping(cls, config: Mapping[str, object]) -> "FeatureConfig":
        return cls(
            return_windows=_positive_windows(config["feature_return_windows"]),
            sma_windows=_positive_windows(config["feature_sma_windows"]),
            volatility_windows=_positive_windows(
                config["feature_volatility_windows"]
            ),
            drawdown_windows=_positive_windows(config["feature_drawdown_windows"]),
            rsi_window=_positive_integer(config["rsi_window"], "rsi_window"),
            adv_window_days=_positive_integer(
                config["adv_window_days"], "adv_window_days"
            ),
            annualize_volatility=bool(
                config.get("annualize_feature_volatility", False)
            ),
        )


@dataclass(frozen=True)
class FeatureBuildResult:
    """Feature values plus per-feature missing-value classifications."""

    features: pd.DataFrame
    missing_reasons: Mapping[str, pd.Series]


FEATURE_DEFINITIONS: Final[dict[str, tuple[str, int, str]]] = {}


def _positive_integer(value: object, name: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _positive_windows(value: object) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("Feature windows must be a sequence of positive integers")
    windows = tuple(_positive_integer(item, "feature window") for item in value)
    if len(windows) != len(set(windows)):
        raise ValueError("Feature windows cannot contain duplicates")
    return windows


def _missing_reason(
    values: pd.Series,
    *,
    expected_initial_rows: int,
    invalid_mask: pd.Series | None = None,
) -> pd.Series:
    """Classify every NaN without filling it."""
    missing = values.isna()
    positions = pd.Series(np.arange(len(values)), index=values.index)
    reason = pd.Series("", index=values.index, dtype="string")
    reason.loc[missing & positions.lt(expected_initial_rows)] = "expected_missing"
    if invalid_mask is not None:
        reason.loc[missing & invalid_mask.fillna(False)] = "calculation_invalid"
    reason.loc[missing & reason.eq("")] = "source_missing"
    return reason


def _rolling_any(mask: pd.Series, window: int) -> pd.Series:
    return mask.astype(float).rolling(window=window, min_periods=1).max().eq(1.0)


def _maximum_drawdown(values: np.ndarray) -> float:
    if len(values) == 0 or not np.isfinite(values).all() or (values <= 0).any():
        return np.nan
    peaks = np.maximum.accumulate(values)
    return float(np.min(values / peaks - 1.0))


def _register(name: str, category: str, lookback: int, formula: str) -> None:
    FEATURE_DEFINITIONS[name] = (category, lookback, formula)


def build_features(
    prices: pd.DataFrame,
    ticker: str,
    config: FeatureConfig,
) -> FeatureBuildResult:
    """Build one ticker's features using only the current and prior rows.

    The input must already be validated and ordered.  Processing one ticker per
    call is an explicit guard against sharing rolling windows across products.
    """
    missing_columns = set(PRICE_COLUMNS) - set(prices.columns)
    if missing_columns:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing_columns)}")
    if "ticker" in prices.columns:
        input_tickers = set(prices["ticker"].dropna().astype(str).str.upper())
        if input_tickers and input_tickers != {ticker.upper()}:
            raise ValueError("build_features accepts exactly one ticker per call")

    frame = prices.loc[:, PRICE_COLUMNS].copy().reset_index(drop=True)
    dates = pd.to_datetime(frame["Date"], errors="coerce")
    if dates.isna().any() or not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError("Prices must have unique, strictly increasing dates")
    for column in PRICE_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    adjusted = frame["Adjusted Close"]
    open_price = frame["Open"]
    high = frame["High"]
    low = frame["Low"]
    close = frame["Close"]
    volume = frame["Volume"]
    positive_adjusted = adjusted.where(adjusted.gt(0))
    adjusted_invalid = adjusted.notna() & adjusted.le(0)

    output = pd.DataFrame(
        {"ticker": ticker.upper(), "feature_date": dates.dt.normalize()}
    )
    reasons: dict[str, pd.Series] = {}

    def add(
        name: str,
        values: pd.Series,
        *,
        category: str,
        lookback: int,
        expected_initial_rows: int,
        formula: str,
        invalid_mask: pd.Series | None = None,
    ) -> None:
        output[name] = values.astype(float)
        reasons[name] = _missing_reason(
            output[name],
            expected_initial_rows=expected_initial_rows,
            invalid_mask=invalid_mask,
        )
        _register(name, category, lookback, formula)

    daily_return = positive_adjusted / positive_adjusted.shift(1) - 1.0
    for window in config.return_windows:
        lagged = positive_adjusted.shift(window)
        invalid = adjusted_invalid | adjusted_invalid.shift(window).fillna(False)
        add(
            f"return_{window}d",
            positive_adjusted / lagged - 1.0,
            category="return",
            lookback=window,
            expected_initial_rows=window,
            formula=f"AdjustedClose_t / AdjustedClose_(t-{window}) - 1",
            invalid_mask=invalid,
        )

    for window in config.sma_windows:
        sma = positive_adjusted.rolling(window=window, min_periods=window).mean()
        invalid = _rolling_any(adjusted_invalid, window)
        add(
            f"sma_{window}",
            sma,
            category="moving_average",
            lookback=window,
            expected_initial_rows=window - 1,
            formula=f"mean of AdjustedClose over trailing {window} rows including t",
            invalid_mask=invalid,
        )
        add(
            f"price_to_sma_{window}",
            positive_adjusted / sma.where(sma.gt(0)) - 1.0,
            category="price_distance",
            lookback=window,
            expected_initial_rows=window - 1,
            formula=f"AdjustedClose_t / SMA_{window}_t - 1",
            invalid_mask=invalid | adjusted_invalid | sma.notna() & sma.le(0),
        )

    for window in config.volatility_windows:
        volatility = daily_return.rolling(window=window, min_periods=window).std(ddof=1)
        if config.annualize_volatility:
            volatility = volatility * np.sqrt(252.0)
        formula = f"sample std of trailing {window} simple daily returns"
        if config.annualize_volatility:
            formula += " multiplied by sqrt(252)"
        add(
            f"volatility_{window}d",
            volatility,
            category="volatility",
            lookback=window,
            expected_initial_rows=window,
            formula=formula,
            invalid_mask=_rolling_any(adjusted_invalid, window + 1),
        )

    add(
        "high_low_range",
        (high - low) / close.where(close.gt(0)),
        category="ohlc",
        lookback=1,
        expected_initial_rows=0,
        formula="(High_t - Low_t) / Close_t",
        invalid_mask=close.notna() & close.le(0),
    )
    add(
        "open_close_return",
        close.where(close.gt(0)) / open_price.where(open_price.gt(0)) - 1.0,
        category="ohlc",
        lookback=1,
        expected_initial_rows=0,
        formula="Close_t / Open_t - 1",
        invalid_mask=(open_price.notna() & open_price.le(0))
        | (close.notna() & close.le(0)),
    )
    previous_close = close.shift(1)
    add(
        "overnight_gap",
        open_price.where(open_price.gt(0)) / previous_close.where(previous_close.gt(0))
        - 1.0,
        category="ohlc",
        lookback=1,
        expected_initial_rows=1,
        formula="Open_t / Close_(t-1) - 1",
        invalid_mask=(open_price.notna() & open_price.le(0))
        | (previous_close.notna() & previous_close.le(0)),
    )

    previous_volume = volume.shift(1)
    add(
        "volume_change_1d",
        volume.where(volume.ge(0)) / previous_volume.where(previous_volume.gt(0)) - 1.0,
        category="volume_liquidity",
        lookback=1,
        expected_initial_rows=1,
        formula="Volume_t / Volume_(t-1) - 1",
        invalid_mask=(volume.notna() & volume.lt(0))
        | (previous_volume.notna() & previous_volume.le(0)),
    )
    for window in (5, 21):
        average_volume = volume.where(volume.ge(0)).rolling(
            window=window, min_periods=window
        ).mean()
        add(
            f"volume_to_average_{window}d",
            volume.where(volume.ge(0)) / average_volume.where(average_volume.gt(0)),
            category="volume_liquidity",
            lookback=window,
            expected_initial_rows=window - 1,
            formula=f"Volume_t / trailing {window}-day mean Volume including t",
            invalid_mask=_rolling_any(volume.notna() & volume.lt(0), window)
            | average_volume.notna() & average_volume.le(0),
        )

    valid_close = close.where(close.gt(0))
    valid_volume = volume.where(volume.ge(0))
    dollar_volume = valid_close * valid_volume
    liquidity_invalid = (close.notna() & close.le(0)) | (volume.notna() & volume.lt(0))
    add(
        "dollar_volume",
        dollar_volume,
        category="volume_liquidity",
        lookback=1,
        expected_initial_rows=0,
        formula="Close_t * Volume_t",
        invalid_mask=liquidity_invalid,
    )
    adv_window = config.adv_window_days
    adv = dollar_volume.rolling(window=adv_window, min_periods=adv_window).mean()
    add(
        "ADV20",
        adv,
        category="volume_liquidity",
        lookback=adv_window,
        expected_initial_rows=adv_window - 1,
        formula=f"mean of trailing {adv_window} daily Close * Volume values including t",
        invalid_mask=_rolling_any(liquidity_invalid, adv_window),
    )
    add(
        "relative_ADV20",
        dollar_volume / adv.where(adv.gt(0)),
        category="volume_liquidity",
        lookback=adv_window,
        expected_initial_rows=adv_window - 1,
        formula="DollarVolume_t / ADV20_t",
        invalid_mask=_rolling_any(liquidity_invalid, adv_window)
        | adv.notna() & adv.le(0),
    )

    delta = positive_adjusted.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(
        alpha=1.0 / config.rsi_window,
        adjust=False,
        min_periods=config.rsi_window,
    ).mean()
    average_loss = losses.ewm(
        alpha=1.0 / config.rsi_window,
        adjust=False,
        min_periods=config.rsi_window,
    ).mean()
    relative_strength = average_gain / average_loss.where(average_loss.gt(0))
    rsi = 100.0 - 100.0 / (1.0 + relative_strength)
    rsi = rsi.mask(average_loss.eq(0) & average_gain.gt(0), 100.0)
    rsi = rsi.mask(average_loss.eq(0) & average_gain.eq(0), 50.0)
    add(
        "RSI_14",
        rsi,
        category="rsi",
        lookback=config.rsi_window,
        expected_initial_rows=config.rsi_window,
        formula=f"Wilder RSI from historical gains/losses with window {config.rsi_window}",
        invalid_mask=_rolling_any(adjusted_invalid, config.rsi_window + 1),
    )

    for window in config.drawdown_windows:
        drawdown = positive_adjusted.rolling(
            window=window, min_periods=window
        ).apply(_maximum_drawdown, raw=True)
        add(
            f"rolling_max_drawdown_{window}d",
            drawdown,
            category="drawdown",
            lookback=window,
            expected_initial_rows=window - 1,
            formula=(
                f"minimum price/previous-window-peak - 1 within trailing {window} rows"
            ),
            invalid_mask=_rolling_any(adjusted_invalid, window),
        )

    return FeatureBuildResult(features=output, missing_reasons=reasons)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return model feature names while excluding identifiers and any targets."""
    forbidden_fragments = ("target", "future", "label_")
    columns = [
        column
        for column in frame.columns
        if column not in {"ticker", "feature_date"}
        and not any(fragment in column.casefold() for fragment in forbidden_fragments)
    ]
    if len(columns) != len(frame.columns) - 2:
        raise ValueError("Target-like columns were found in the feature frame")
    return columns
