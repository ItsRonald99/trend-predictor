import numpy as np, pandas as pd, joblib
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score, make_scorer
from sklearn.base import clone
from .io_paths import DATA_PROCESSED, MODELS, REPORTS
from sklearn.impute import SimpleImputer

TRADING_DAYS = 252

def load_dataset(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_PROCESSED / f"{symbol}_dataset.parquet")
    return df.sort_values("date").reset_index(drop=True)

def make_X_y(df: pd.DataFrame, task: str):
    feats = df.drop(columns=["date","y_reg","y_cls"]).copy()
    feats = feats.replace([np.inf, -np.inf], np.nan)
    target = df["y_reg"].copy() if task == "reg" else df["y_cls"].copy()

    good = feats.notna().all(axis=1) & np.isfinite(target)
    X = feats.loc[good].values.astype(float)
    y = target.loc[good].values
    feat_names = feats.columns.tolist()
    # return a boolean mask aligned to df.index
    return X, y, feat_names, good.values

def ts_splits(n_samples: int, n_splits=5):
    return list(TimeSeriesSplit(n_splits=n_splits).split(np.arange(n_samples)))

def train_baselines(symbol: str, n_splits=5):
    df = load_dataset(symbol)
    # Regression: Ridge + HGB
    Xr, yr, feat_names, good_r = make_X_y(df, "reg")
    folds = ts_splits(len(Xr), n_splits)

    ridge = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
    ])

    hgb = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # HGB can handle NaNs, but keep consistent
    ("hgb", HistGradientBoostingRegressor())
    ])
    
    # CV (ridge)
    rows = []
    for name, model in [("ridge_reg", ridge), ("hgb_reg", hgb)]:
        yh_all = np.full(len(yr), np.nan)
        for tr, te in folds:
            model.fit(Xr[tr], yr[tr])
            yh = model.predict(Xr[te])
            yh_all[te] = yh
    
        # ---- NaN-safe slice of CV preds/targets
        mask = ~np.isnan(yh_all)
        y_true = yr[mask]
        y_pred = yh_all[mask]
    
        if len(y_true) == 0:
            mae = rmse = r2 = np.nan
        else:
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)  # returns MSE on all versions
            rmse = float(np.sqrt(mse))                # compute RMSE manually
            r2 = r2_score(y_true, y_pred)
    
        rows.append(dict(model=name, MAE=mae, RMSE=rmse, R2=r2))
    
        # fit on ALL data and save the final model bundle
        model.fit(Xr, yr)
        joblib.dump({"model": model, "features": feat_names}, MODELS / f"{symbol}_{name}.pkl")
        
    # Classification: Logistic
    Xc, yc, feat_names_c, good_c = make_X_y(df, "cls")
    logit = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("logit", LogisticRegression(max_iter=200, class_weight="balanced"))
    ])
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
    # get filtered matrix + mask
    X, y, _, good = make_X_y(df, task)
    n = len(X)
    preds_compact = np.full(n, np.nan, dtype=float)

    folds = ts_splits(n, n_splits)
    for tr, te in folds:
        m = clone(pipe)
        m.fit(X[tr], y[tr])
        if task == "cls":
            preds_compact[te] = m.predict_proba(X[te])[:, 1]
        else:
            preds_compact[te] = m.predict(X[te])

    # expand back to full length, align to df.index
    preds_full = np.full(len(df), np.nan, dtype=float)
    preds_full[good] = preds_compact
    return preds_full

_RIDGE_GRID = {"ridge__alpha": [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]}
_HGB_GRID   = {
    "hgb__learning_rate":    [0.03, 0.05, 0.08, 0.1],
    "hgb__max_depth":        [3, 5, None],
    "hgb__max_leaf_nodes":   [15, 31, 63],
    "hgb__min_samples_leaf": [10, 20, 50],
}
_LOGIT_GRID = {
    "logit__C":        [0.1, 0.5, 1.0, 2.0, 5.0],
    "logit__penalty":  ["l2"],
    "logit__solver":   ["lbfgs"],
    "logit__max_iter": [200],
}

def _rmse_score(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def tune_baselines(symbol: str, n_splits: int = 5):
    df = load_dataset(symbol)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_scorer = make_scorer(_rmse_score, greater_is_better=False)

    # --- Ridge ---
    Xr, yr, feat_names_r, _ = make_X_y(df, "reg")
    ridge_pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    gs_ridge = GridSearchCV(
        ridge_pipe, _RIDGE_GRID,
        scoring={"rmse": rmse_scorer, "mae": make_scorer(mean_absolute_error, greater_is_better=False), "r2": make_scorer(r2_score)},
        refit="rmse", cv=tscv, n_jobs=2, return_train_score=False,
    )
    gs_ridge.fit(Xr, yr)
    joblib.dump({"model": gs_ridge.best_estimator_, "features": feat_names_r, "best_params": gs_ridge.best_params_},
                MODELS / f"{symbol}_ridge_reg_tuned.pkl")
    pd.DataFrame(gs_ridge.cv_results_).to_csv(REPORTS / f"day7_{symbol}_ridge_grid.csv", index=False)

    # --- HGB ---
    hgb_pipe = Pipeline([("hgb", HistGradientBoostingRegressor(random_state=42))])
    gs_hgb = GridSearchCV(
        hgb_pipe, _HGB_GRID,
        scoring={"rmse": rmse_scorer, "mae": make_scorer(mean_absolute_error, greater_is_better=False), "r2": make_scorer(r2_score)},
        refit="rmse", cv=tscv, n_jobs=2, return_train_score=False,
    )
    gs_hgb.fit(Xr, yr)
    joblib.dump({"model": gs_hgb.best_estimator_, "features": feat_names_r, "best_params": gs_hgb.best_params_},
                MODELS / f"{symbol}_hgb_reg_tuned.pkl")
    pd.DataFrame(gs_hgb.cv_results_).to_csv(REPORTS / f"day7_{symbol}_hgb_grid.csv", index=False)

    # --- Logit ---
    Xc, yc, feat_names_c, _ = make_X_y(df, "cls")
    logit_pipe = Pipeline([("scaler", StandardScaler()), ("logit", LogisticRegression())])
    gs_logit = GridSearchCV(
        logit_pipe, _LOGIT_GRID,
        scoring={"auc": "roc_auc", "f1": "f1", "acc": "accuracy"},
        refit="auc", cv=TimeSeriesSplit(n_splits=n_splits), n_jobs=2, return_train_score=False,
    )
    gs_logit.fit(Xc, yc)
    joblib.dump({"model": gs_logit.best_estimator_, "features": feat_names_c, "best_params": gs_logit.best_params_},
                MODELS / f"{symbol}_logit_cls_tuned.pkl")
    pd.DataFrame(gs_logit.cv_results_).to_csv(REPORTS / f"day7_{symbol}_logit_grid.csv", index=False)

    return {
        "ridge_best_params": gs_ridge.best_params_,
        "hgb_best_params":   gs_hgb.best_params_,
        "logit_best_params": gs_logit.best_params_,
    }

def run_strategy(df: pd.DataFrame, signal: np.ndarray, kind: str, thr: float, cost_bps=0.0005) -> pd.DataFrame:
    out = pd.DataFrame({"date": df["date"], "y_reg": df["y_reg"]})

    sig = pd.Series(signal, index=df.index, dtype=float)
    # stay in cash (0) where signal is NaN
    pos = (sig > thr).astype(float)
    pos = pos.fillna(0.0)

    trades = pos.diff().abs().fillna(0.0)
    costs = cost_bps * trades

    out["pos"] = pos.values
    out["costs"] = costs.values
    out["r_strategy"] = out["pos"] * out["y_reg"] - out["costs"]
    out["r_bh"] = out["y_reg"]
    out["eq_strat"] = equity_from_logrets(out["r_strategy"].values)
    out["eq_bh"] = equity_from_logrets(out["r_bh"].values)
    return out