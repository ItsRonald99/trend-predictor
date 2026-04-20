import numpy as np
import pandas as pd
import pytest

from trend_predictor.modeling import perf_metrics, run_strategy, make_X_y


def _make_dataset(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    y_reg = rng.normal(0, 0.01, n)
    df = pd.DataFrame({
        "date":  dates,
        "feat1": rng.normal(0, 1, n),
        "feat2": rng.normal(0, 1, n),
        "y_reg": y_reg,
        "y_cls": (y_reg > 0).astype(int),
    })
    return df


def test_perf_metrics_flat():
    r = pd.Series(np.zeros(100))
    m = perf_metrics(r)
    assert m["CAGR"] == pytest.approx(0.0)
    assert not np.isfinite(m["Sharpe"])


def test_perf_metrics_positive_returns():
    r = pd.Series(np.full(252, 0.001))
    m = perf_metrics(r)
    assert m["CAGR"] > 0
    assert m["Sharpe"] > 0
    assert m["MaxDD"] == pytest.approx(0.0, abs=1e-6)


def test_run_strategy_all_cash():
    df = _make_dataset()
    signal = np.zeros(len(df))
    bt = run_strategy(df, signal, "reg", thr=0.0, cost_bps=0.0)
    assert (bt["r_strategy"] == 0).all()


def test_run_strategy_all_long_no_cost():
    df = _make_dataset()
    signal = np.ones(len(df)) * 1.0
    bt = run_strategy(df, signal, "reg", thr=0.0, cost_bps=0.0)
    assert np.allclose(bt["r_strategy"].values, df["y_reg"].values)


def test_make_X_y_drops_nan_rows():
    df = _make_dataset(n=60)
    df.loc[5, "feat1"] = np.nan
    X, y, feat_names, good = make_X_y(df, "reg")
    assert not np.isnan(X).any()
    assert len(X) == good.sum()
    assert good.sum() < len(df)


def test_make_X_y_cls_binary():
    df = _make_dataset(n=60)
    _, y, _, _ = make_X_y(df, "cls")
    assert set(y).issubset({0, 1})
