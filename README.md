# 📈 Trend Predictor: Stock & ETF ML Backtesting Pipeline

A **full-stack data engineering + ML pipeline** that explores whether simple machine learning models can predict ETF returns — with a live REST API deployed for real-time signals.

**Live API:** https://trend-predictor.onrender.com/docs

---

## 🚀 Features

- **Data Pipeline**
  - Fetches historical OHLCV data from Alpha Vantage (Stooq fallback).
  - Cleans, validates, and stores in Parquet (`pyarrow`, `fastparquet`).
  - Feature-engineers technical indicators: MACD, RSI, SMAs, rolling volatility.

- **Machine Learning**
  - Models: Ridge Regression, HistGradientBoostingRegressor, Logistic Regression.
  - Time-series aware cross-validation (`TimeSeriesSplit`).
  - Hyperparameter tuning via `GridSearchCV`.
  - Walk-forward backtesting with configurable trading costs.

- **Evaluation**
  - ML metrics: MAE, RMSE, R², Accuracy, F1, AUC.
  - Strategy metrics: CAGR, Volatility, Sharpe, Max Drawdown.
  - Threshold calibration on holdout split.

- **Deployment**
  - FastAPI REST API with interactive Swagger UI (`/docs`).
  - Dockerized and deployed to Render.com.
  - GitHub Actions CI: runs tests and builds Docker image on every push.

---

## 🛠 Setup

```bash
git clone git@github.com:ItsRonald99/trend-predictor.git
cd trend-predictor
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt
pip install -e .
```

---

## ⚡ Usage

### Data ingestion
```bash
tp ingest --symbols QQQ VFV.TO XEQT.TO
```

### ML pipeline
```bash
tp features  --symbols QQQ VFV.TO XEQT.TO
tp train     --symbols QQQ VFV.TO XEQT.TO
tp tune      --symbols QQQ VFV.TO XEQT.TO
tp backtest  --symbols QQQ --kind logit_cls --tuned
tp thresholds --symbols QQQ --kind logit_cls --cal-frac 0.8
```

### Run API locally
```bash
uvicorn trend_predictor.api:app --reload
# Open http://localhost:8000/docs
```

---

## 🌐 API Endpoints

| Endpoint | Description |
| --- | --- |
| `GET /predict/{symbol}` | Latest LONG/CASH signal for a symbol |
| `GET /metrics/{symbol}` | Walk-forward backtest performance (CAGR, Sharpe, MaxDD) |
| `GET /symbols` | List available symbols |
| `GET /health` | Health check |

Supported symbols: `QQQ`, `VFV.TO`, `XEQT.TO`

Query parameters for `/predict` and `/metrics`: `kind` (`ridge_reg`, `hgb_reg`, `logit_cls`), `tuned` (bool).

---

## 📊 Example Output

Cross-validation (Day 4):
| Model | MAE   | RMSE  | R²    |
| ----- | ----- | ----- | ----- |
| Ridge | 0.009 | 0.012 | -0.02 |
| HGB   | 0.008 | 0.011 | 0.01  |

Backtest performance (Day 6/8):
| Strategy    | CAGR | Vol   | Sharpe | MaxDD |
| ----------- | ---- | ----- | ------ | ----- |
| Buy & Hold  | 8.2% | 15.0% | 0.55   | -0.35 |
| Logistic ML | 6.1% | 13.0% | 0.47   | -0.28 |

---

👤 Author: Ronald Ma — r3ma@uwaterloo.ca
