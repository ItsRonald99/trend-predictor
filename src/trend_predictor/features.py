import numpy as np, pandas as pd
from .io_paths import DATA_PROCESSED

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = gain / (loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out

def build_features_from_prices(df: pd.DataFrame, warmup: int = 30) -> pd.DataFrame:
    df = df.sort_values("date").copy().set_index("date")
    px  = df["adj_close"].astype(float)
    vol = df["volume"].astype(float)
    out = pd.DataFrame(index=df.index)

    r1 = np.log(px / px.shift(1))
    out["r1"]    = r1
    out["r5"]    = r1.rolling(5).sum()
    out["r10"]   = r1.rolling(10).sum()
    out["vol10"] = r1.rolling(10).std()
    out["vol20"] = r1.rolling(20).std()

    sma10 = px.rolling(10).mean()
    sma20 = px.rolling(20).mean()
    out["sma10_rel"] = sma10 / px - 1.0
    out["sma20_rel"] = sma20 / px - 1.0

    ema12 = ema(px, 12); ema26 = ema(px, 26)
    out["ema12_rel"] = ema12 / px - 1.0
    out["ema26_rel"] = ema26 / px - 1.0
    macd = ema12 - ema26; signal = ema(macd, 9)
    out["macd"] = macd; out["macd_hist"] = macd - signal

    out["rsi14"]   = rsi(px, 14)
    out["vol_z20"] = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + 1e-9)

    out["y_reg"] = r1.shift(-1)
    out["y_cls"] = (out["y_reg"] > 0).astype(int)
    out = out.iloc[warmup:].reset_index().rename(columns={"index":"date"})
    return out

def save_symbol_dataset(prices_parquet: str, out_dataset: str):
    df = pd.read_parquet(prices_parquet).sort_values("date")
    feats = build_features_from_prices(df)
    feats.to_parquet(out_dataset, index=False)