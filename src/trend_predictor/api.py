import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse

from .io_paths import MODELS, DATA_PROCESSED
from .modeling import load_dataset, make_X_y, wf_predict, run_strategy, perf_metrics

# ---------------------------------------------------------------------------
# Startup: load all available model bundles into memory
# ---------------------------------------------------------------------------
_models: dict[str, dict] = {}

_TAG_RE = re.compile(r"^(.+)_(ridge_reg|hgb_reg|logit_cls)(_tuned)?\.pkl$")


def _scan_models() -> dict[str, dict]:
    bundles: dict[str, dict] = {}
    for pkl in MODELS.glob("*.pkl"):
        m = _TAG_RE.match(pkl.name)
        if not m:
            continue
        symbol, kind, tuned_suffix = m.group(1), m.group(2), m.group(3)
        tuned = tuned_suffix == "_tuned"
        key = f"{symbol}|{kind}|{'tuned' if tuned else 'base'}"
        bundles[key] = joblib.load(pkl)
    return bundles


@asynccontextmanager
async def lifespan(app: FastAPI):
    _models.update(_scan_models())
    yield


app = FastAPI(
    title="Trend Predictor API",
    description=(
        "ML-based ETF trend prediction pipeline. "
        "Supports QQQ, VFV.TO, and XEQT.TO. "
        "Models trained with walk-forward time-series cross-validation."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_bundle(symbol: str, kind: str, tuned: bool) -> dict:
    key = f"{symbol}|{kind}|{'tuned' if tuned else 'base'}"
    bundle = _models.get(key)
    if bundle is None:
        raise HTTPException(
            status_code=404,
            detail=f"No model found for symbol={symbol}, kind={kind}, tuned={tuned}. "
                   f"Run `tp train` (and `tp tune` for tuned models) first.",
        )
    return bundle


def _available_symbols() -> list[str]:
    seen: set[str] = set()
    for key in _models:
        seen.add(key.split("|")[0])
    return sorted(seen)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok", "models_loaded": len(_models)}


@app.get("/symbols", tags=["Meta"])
def symbols():
    """List symbols that have at least one trained model available."""
    return {"symbols": _available_symbols()}


@app.get("/predict/{symbol}", tags=["Prediction"])
def predict(
    symbol: str,
    kind: Literal["ridge_reg", "hgb_reg", "logit_cls"] = Query(
        "logit_cls", description="Model type"
    ),
    tuned: bool = Query(True, description="Use tuned (GridSearchCV) model"),
):
    """
    Return the latest trading signal for a symbol using the last available row
    of feature data. Classification models return a probability and LONG/CASH
    signal; regression models return a predicted next-day log return.
    """
    if symbol not in _available_symbols():
        raise HTTPException(status_code=404, detail=f"Unknown symbol: {symbol}")

    dataset_path = DATA_PROCESSED / f"{symbol}_dataset.parquet"
    if not dataset_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Feature dataset not found for {symbol}. Run `tp features --symbols {symbol}` first.",
        )

    bundle = _get_bundle(symbol, kind, tuned)
    model = bundle["model"]

    df = load_dataset(symbol)
    last_row = df.iloc[[-1]]
    task = "cls" if "logit" in kind else "reg"

    try:
        X, _, _, good = make_X_y(last_row, task)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}")

    if len(X) == 0 or not good.any():
        raise HTTPException(status_code=422, detail="Latest row has missing features — cannot predict.")

    date_val = str(last_row["date"].iloc[0])[:10]

    if task == "cls":
        prob = float(model.predict_proba(X)[0, 1])
        signal = "LONG" if prob > 0.55 else "CASH"
        return {
            "symbol":      symbol,
            "date":        date_val,
            "kind":        kind,
            "tuned":       tuned,
            "signal":      signal,
            "probability": round(prob, 4),
            "threshold":   0.55,
        }
    else:
        pred = float(model.predict(X)[0])
        signal = "LONG" if pred > 0.0 else "CASH"
        return {
            "symbol":              symbol,
            "date":                date_val,
            "kind":                kind,
            "tuned":               tuned,
            "signal":              signal,
            "predicted_log_return": round(pred, 6),
            "threshold":           0.0,
        }


@app.get("/metrics/{symbol}", tags=["Backtest"])
def metrics(
    symbol: str,
    kind: Literal["ridge_reg", "hgb_reg", "logit_cls"] = Query(
        "logit_cls", description="Model type"
    ),
    tuned: bool = Query(True, description="Use tuned (GridSearchCV) model"),
    splits: int = Query(5, ge=2, le=10, description="Walk-forward CV splits"),
    cls_thr: float = Query(0.55, description="Classification threshold"),
    reg_thr: float = Query(0.0, description="Regression threshold"),
    cost: float = Query(0.0005, description="Trading cost per trade (bps as decimal)"),
):
    """
    Run a walk-forward backtest and return strategy performance metrics:
    CAGR, Volatility, Sharpe ratio, and Max Drawdown — alongside buy-and-hold.
    """
    if symbol not in _available_symbols():
        raise HTTPException(status_code=404, detail=f"Unknown symbol: {symbol}")

    bundle = _get_bundle(symbol, kind, tuned)
    model = bundle["model"]

    df = load_dataset(symbol)
    task = "cls" if "logit" in kind else "reg"

    preds = wf_predict(df, model, task, n_splits=splits)
    thr = cls_thr if task == "cls" else reg_thr
    bt = run_strategy(df, preds, task, thr, cost_bps=cost)

    strat = perf_metrics(bt["r_strategy"])
    bh = perf_metrics(bt["r_bh"])

    def _fmt(d: dict) -> dict:
        return {k: (round(v, 4) if np.isfinite(v) else None) for k, v in d.items()}

    return {
        "symbol":    symbol,
        "kind":      kind,
        "tuned":     tuned,
        "n_splits":  splits,
        "threshold": thr,
        "strategy":  _fmt(strat),
        "buy_and_hold": _fmt(bh),
    }
