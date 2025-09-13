import numpy as np, pandas as pd, joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score
from sklearn.base import clone
from .io_paths import DATA_PROCESSED, MODELS, REPORTS

TRADING_DAYS = 252

def load_dataset(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_PROCESSED / f"{symbol}_dataset.parquet")
    return df.sort_values("date").reset_index(drop=True)

def make_X_y(df: pd.DataFrame, task: str):
    feats = df.drop(columns=["date","y_reg","y_cls"])
    X = feats.values.astype(float)
    y = df["y_reg"].values if task=="reg" else df["y_cls"].values
    return X, y, feats.columns.tolist()

def ts_splits(n_samples: int, n_splits=5):
    return list(TimeSeriesSplit(n_splits=n_splits).split(np.arange(n_samples)))

def train_baselines(symbol: str, n_splits=5):
    df = load_dataset(symbol)
    # Regression: Ridge + HGB
    Xr, yr, feat_names = make_X_y(df, "reg")
    folds = ts_splits(len(Xr), n_splits)

    ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    hgb   = HistGradientBoostingRegressor()
    # CV (ridge)
    rows = []
    for name, model in [("ridge_reg", ridge), ("hgb_reg", hgb)]:
        yh_all = np.full(len(yr), np.nan)
        for tr, te in folds:
            model.fit(Xr[tr], yr[tr]); yh = model.predict(Xr[te]); yh_all[te] = yh
        mae = mean_absolute_error(yr[~np.isnan(yh_all)], yh_all[~np.isnan(yh_all)])
        rmse = mean_squared_error(yr[~np.isnan(yh_all)], yh_all[~np.isnan(yh_all)], squared=False) if hasattr(mean_squared_error, "__call__") else np.sqrt(mean_squared_error(yr, yh_all))
        r2 = r2_score(yr[~np.isnan(yh_all)], yh_all[~np.isnan(yh_all)])
        rows.append(dict(model=name, MAE=mae, RMSE=rmse, R2=r2))
        # fit on all & save
        model.fit(Xr, yr)
        joblib.dump({"model": model, "features": feat_names}, MODELS / f"{symbol}_{name}.pkl")

    # Classification: Logistic
    Xc, yc, feat_names = make_X_y(df, "cls")
    logit = Pipeline([("scaler", StandardScaler()), ("logit", LogisticRegression(max_iter=200, class_weight="balanced"))])
    proba_all = np.full(len(yc), np.nan)
    for tr, te in folds:
        logit.fit(Xc[tr], yc[tr]); proba_all[te] = logit.predict_proba(Xc[te])[:,1]
    acc = accuracy_score(yc[~np.isnan(proba_all)], (proba_all[~np.isnan(proba_all)]>0.5).astype(int))
    f1  = f1_score(yc[~np.isnan(proba_all)], (proba_all[~np.isnan(proba_all)]>0.5).astype(int))
    try:
        auc = roc_auc_score(yc[~np.isnan(proba_all)], proba_all[~np.isnan(proba_all)])
    except ValueError:
        auc = np.nan
    rows.append(dict(model="logit_cls", ACC=acc, F1=f1, AUC=auc))
    logit.fit(Xc, yc)
    joblib.dump({"model": logit, "features": feat_names}, MODELS / f"{symbol}_logit_cls.pkl")

    pd.DataFrame(rows).to_csv(REPORTS / f"{symbol}_day4_baseline_cv.csv", index=False)

def equity_from_logrets(r: np.ndarray, start: float = 1.0) -> np.ndarray:
    return start * np.exp(np.nancumsum(r))

def perf_metrics(logrets: pd.Series) -> dict:
    r = pd.Series(logrets).dropna()
    if len(r)==0: return dict(CAGR=np.nan, Vol=np.nan, Sharpe=np.nan, MaxDD=np.nan)
    cagr = r.mean()*TRADING_DAYS
    vol  = r.std(ddof=0)*np.sqrt(TRADING_DAYS)
    sharpe = cagr/vol if vol>0 else np.nan
    eq = equity_from_logrets(r.values)
    peak = np.maximum.accumulate(eq); maxdd = ((eq - peak)/peak).min()
    return dict(CAGR=cagr, Vol=vol, Sharpe=sharpe, MaxDD=maxdd)

def wf_predict(df: pd.DataFrame, pipe: Pipeline, task: str, n_splits=5) -> np.ndarray:
    X, y, _ = make_X_y(df, task)
    preds = np.full(len(X), np.nan)
    for tr, te in ts_splits(len(X), n_splits):
        m = clone(pipe); m.fit(X[tr], y[tr]); preds[te] = (m.predict_proba(X[te])[:,1] if task=="cls" else m.predict(X[te]))
    return preds

def run_strategy(df: pd.DataFrame, signal: np.ndarray, kind: str, thr: float, cost_bps=0.0005) -> pd.DataFrame:
    out = pd.DataFrame({"date": df["date"], "y_reg": df["y_reg"]})
    pos = (signal > thr).astype(int)
    trades = pd.Series(pos).diff().abs().fillna(0.0).values
    costs = cost_bps * trades
    out["r_strategy"] = pos*out["y_reg"] - costs
    out["r_bh"] = out["y_reg"]
    out["eq_strat"] = equity_from_logrets(out["r_strategy"].values)
    out["eq_bh"] = equity_from_logrets(out["r_bh"].values)
    return out