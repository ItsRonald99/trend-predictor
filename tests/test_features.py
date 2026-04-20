import numpy as np
import pandas as pd
import pytest

from trend_predictor.features import build_features_from_prices, rsi


def _make_prices(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "date":      pd.date_range("2020-01-01", periods=n, freq="B"),
        "adj_close": prices,
        "volume":    rng.integers(1_000_000, 10_000_000, n).astype(float),
    })


def test_build_features_output_columns():
    df = _make_prices()
    out = build_features_from_prices(df)
    expected = {"r1", "r5", "r10", "vol10", "vol20",
                "sma10_rel", "sma20_rel", "ema12_rel", "ema26_rel",
                "macd", "macd_hist", "rsi14", "vol_z20",
                "y_reg", "y_cls", "date"}
    assert expected.issubset(set(out.columns))


def test_build_features_no_nans_after_warmup():
    # macd_hist = macd - ema(macd, 9) needs EMA26 (26 periods) + EMA9 signal (9 more) = 35 min warmup
    df = _make_prices(n=100)
    out = build_features_from_prices(df, warmup=35)
    feature_cols = [c for c in out.columns if c not in ("date", "y_reg", "y_cls")]
    assert out[feature_cols].isna().sum().sum() == 0


def test_rsi_bounds():
    prices = pd.Series(np.linspace(100, 150, 60))
    result = rsi(prices, period=14).dropna()
    assert (result >= 0).all() and (result <= 100).all()


def test_build_features_row_count():
    df = _make_prices(n=80)
    out = build_features_from_prices(df, warmup=30)
    assert len(out) == 80 - 30
