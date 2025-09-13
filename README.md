# 📈 Trend Predictor: Stock & ETF ML Backtesting Pipeline

This project is a **full-stack data engineering + ML pipeline** for exploring whether simple machine learning models can predict **stock and ETF returns**.  

---

## 🚀 Features

- **ETL Pipeline**
  - Fetches historical data (Alpha Vantage / Stooq APIs).
  - Stores in Parquet with efficient schemas (`pyarrow`, `fastparquet`).
  - Cleans and feature-engineers technical indicators (MACD, RSI, SMAs, volatility).

- **Machine Learning**
  - Baseline models:
    - Regression: Ridge, HistGradientBoostingRegressor
    - Classification: Logistic Regression
  - Cross-validation (time-series aware).
  - Hyperparameter tuning via GridSearchCV.
  - Walk-forward backtesting with trading costs.

- **Evaluation**
  - Metrics: MAE, RMSE, R², Accuracy, F1, AUC.
  - Strategy performance: CAGR, Volatility, Sharpe, Max Drawdown.
  - Feature importances (permutation-based).
  - Threshold calibration (Day 9) on a holdout split.

- **Packaging**
  - Jupyter notebooks for transparency (Day 1 → Day 10).
  - CLI (`tp`) to reproduce full experiments without notebooks.
  - Makefile for automation.
  - Git-managed virtualenv + pinned dependencies (`requirements.lock.txt`).

---

## 📂 Repo Structure
trend-predictor/
├── data/ # raw, interim, processed datasets (Parquet)
├── models/ # saved model artifacts (.pkl)
├── notebooks/ # day-by-day development (Day1 → Day10)
├── reports/ # backtest logs, plots, metrics
├── src/
│ └── trend_predictor/ # CLI + reusable modules
│ ├── cli.py
│ ├── features.py
│ ├── modeling.py
│ └── io_paths.py
├── pyproject.toml # package metadata (tp CLI entrypoint)
├── requirements.lock.txt
├── Makefile
└── README.md

---

## 🛠 Setup

### 1. Clone + create environment
git clone git@github.com:yourusername/trend-predictor.git
cd trend-predictor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt

### 2. Install package (editable mode)
pip install -e .

This installs the CLI command tp.

---

## ⚡ Usage
Build features
tp features --symbols QQQ VFV.TO XEQT.TO

Train models
tp train --symbols QQQ VFV.TO XEQT.TO

Backtest
tp backtest --symbols QQQ --kind ridge_reg
tp backtest --symbols QQQ --kind logit_cls --tuned

Threshold calibration
tp thresholds --symbols QQQ --kind logit_cls --cal-frac 0.8
Reports are saved in reports/.

---

## 📊 Example Output
Cross-validation (Day 4):
Model	MAE	RMSE	R²
Ridge	0.009	0.012	-0.02
HGB	0.008	0.011	0.01

Backtest performance (Day 6/8):
Strategy	CAGR	Vol	Sharpe	MaxDD
Buy & Hold	8.2%	15.0%	0.55	-0.35
Logistic ML	6.1%	13.0%	0.47	-0.28

---

👤 Author: Ronald Ma

📧 r3ma@uwaterloo.ca
