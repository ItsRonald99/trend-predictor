import argparse, sys, numpy as np, pandas as pd, joblib
from pathlib import Path
from .io_paths import DATA_PROCESSED, MODELS, REPORTS
from .features import save_symbol_dataset
from .modeling import load_dataset, train_baselines, wf_predict, run_strategy, perf_metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor

def cmd_features(args):
    for sym in args.symbols:
        inp = DATA_PROCESSED / f"{sym}.parquet"
        out = DATA_PROCESSED / f"{sym}_dataset.parquet"
        if not inp.exists():
            print(f"[features] Missing {inp}. Generate Day 2 parquet first.", file=sys.stderr); continue
        save_symbol_dataset(str(inp), str(out))
        print(f"[features] Wrote {out}")

def _baseline_pipe(kind: str) -> Pipeline:
    if kind == "ridge_reg": return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    if kind == "hgb_reg":   return Pipeline([("hgb", HistGradientBoostingRegressor())])
    if kind == "logit_cls": return Pipeline([("scaler", StandardScaler()), ("logit", LogisticRegression(max_iter=200, class_weight="balanced"))])
    raise ValueError("unknown kind")

def cmd_train(args):
    for sym in args.symbols:
        train_baselines(sym)
        print(f"[train] Saved baselines for {sym} under models/ and CV metrics under reports/")

def cmd_backtest(args):
    for sym in args.symbols:
        df = load_dataset(sym)
        # choose pipe
        if args.kind in {"ridge_reg","hgb_reg","logit_cls"} and not args.tuned:
            pipe = _baseline_pipe(args.kind)
        else:
            tag = args.kind.replace("_base","") + ("_tuned" if args.tuned else "")
            bundle = joblib.load(MODELS / f"{sym}_{tag}.pkl")
            pipe = bundle["model"]

        task = "cls" if "logit" in args.kind else "reg"
        preds = wf_predict(df, pipe, task, n_splits=args.splits)
        thr = args.cls_thr if task=="cls" else args.reg_thr
        bt = run_strategy(df, preds, "cls" if task=="cls" else "reg", thr, cost_bps=args.cost)
        outcsv = REPORTS / f"cli_bt_{sym}_{args.kind}{'_tuned' if args.tuned else ''}.csv"
        bt.to_csv(outcsv, index=False)
        m = perf_metrics(bt["r_strategy"])
        print(f"[backtest] {sym} {args.kind}{'_tuned' if args.tuned else ''} | Sharpe={m['Sharpe']:.3f}, CAGR={m['CAGR']:.3%} | saved {outcsv}")

def cmd_thresholds(args):
    # simple calibration→test split (80/20) on OOS preds
    cal_frac = args.cal_frac
    for sym in args.symbols:
        df = load_dataset(sym)
        # tuned by default
        bundle = joblib.load(MODELS / f"{sym}_{args.kind}_tuned.pkl")
        pipe = bundle["model"]
        task = "cls" if "logit" in args.kind else "reg"
        preds = wf_predict(df, pipe, task, n_splits=args.splits)
        n = len(preds); split = int(n*cal_frac)
        cal_idx = np.arange(split); tes_idx = np.arange(split, n)
        grid = (np.round(np.linspace(0.45,0.65,21),3) if task=="cls" else np.round(np.linspace(-0.0005,0.0005,21),6))
        best_thr, best_val = None, -1e9
        for thr in grid:
            cal_bt = run_strategy(df.iloc[cal_idx], preds[cal_idx], task, thr, cost_bps=args.cost)
            m = perf_metrics(cal_bt["r_strategy"]); val = m["Sharpe"]
            if np.isfinite(val) and val > best_val:
                best_val, best_thr = val, thr
        tes_bt = run_strategy(df.iloc[tes_idx], preds[tes_idx], task, best_thr, cost_bps=args.cost)
        mtest = perf_metrics(tes_bt["r_strategy"])
        outcsv = REPORTS / f"cli_thr_{sym}_{args.kind}_cal{int(cal_frac*100)}.csv"
        tes_bt.to_csv(outcsv, index=False)
        print(f"[thresholds] {sym} {args.kind} best_thr={best_thr} | Test Sharpe={mtest['Sharpe']:.3f}, Test CAGR={mtest['CAGR']:.3%} | saved {outcsv}")

def main():
    p = argparse.ArgumentParser(prog="tp", description="Trend Predictor CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("features", help="Build feature datasets from Day 2 Parquet prices")
    s1.add_argument("--symbols", nargs="+", required=True)
    s1.set_defaults(func=cmd_features)

    s2 = sub.add_parser("train", help="Train baseline models and save them")
    s2.add_argument("--symbols", nargs="+", required=True)
    s2.set_defaults(func=cmd_train)

    s3 = sub.add_parser("backtest", help="Walk-forward backtest for a model")
    s3.add_argument("--symbols", nargs="+", required=True)
    s3.add_argument("--kind", choices=["ridge_reg","hgb_reg","logit_cls"], required=True)
    s3.add_argument("--tuned", action="store_true")
    s3.add_argument("--splits", type=int, default=5)
    s3.add_argument("--reg-thr", type=float, default=0.0)
    s3.add_argument("--cls-thr", type=float, default=0.55)
    s3.add_argument("--cost", type=float, default=0.0005)
    s3.set_defaults(func=cmd_backtest)

    s4 = sub.add_parser("thresholds", help="Calibrate thresholds on tuned model, evaluate on holdout")
    s4.add_argument("--symbols", nargs="+", required=True)
    s4.add_argument("--kind", choices=["ridge_reg","hgb_reg","logit_cls"], required=True)
    s4.add_argument("--splits", type=int, default=5)
    s4.add_argument("--cal-frac", type=float, default=0.8)
    s4.add_argument("--cost", type=float, default=0.0005)
    s4.set_defaults(func=cmd_thresholds)

    args = p.parse_args()
    args.func(args)