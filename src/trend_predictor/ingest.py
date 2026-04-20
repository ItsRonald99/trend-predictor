import json, os, time
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from .io_paths import DATA_RAW, DATA_INTERIM, DATA_PROCESSED

STOOQ_MAP = {
    "QQQ":     "qqq.us",
    "VFV.TO":  "vfv.to",
    "XEQT.TO": "xeqt.to",
}

_AV_URL = "https://www.alphavantage.co/query"


def fetch_av(symbol: str, api_key: str, outputsize: str = "full",
             min_sleep: float = 65.0, max_retries: int = 5) -> pd.DataFrame:
    params = {
        "function":   "TIME_SERIES_DAILY",
        "symbol":     symbol,
        "apikey":     api_key,
        "outputsize": outputsize,
    }
    for attempt in range(1, max_retries + 1):
        r = requests.get(_AV_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        key = next((k for k in data if "Time Series" in k), None)
        if key:
            (DATA_RAW / f"{symbol}.json").write_text(
                json.dumps(data, indent=2)[:1_000_000]
            )
            rows = [
                {
                    "date":      dt,
                    "open":      v.get("1. open"),
                    "high":      v.get("2. high"),
                    "low":       v.get("3. low"),
                    "close":     v.get("4. close"),
                    "adj_close": v.get("4. close"),
                    "volume":    v.get("5. volume"),
                }
                for dt, v in sorted(data[key].items())
            ]
            df = pd.DataFrame(rows)
            df.to_csv(DATA_INTERIM / f"{symbol}.csv", index=False)
            return df

        msg = data.get("Note") or data.get("Information") or data.get("Error Message") or str(data)
        print(f"[ingest] AV attempt {attempt}/{max_retries} for {symbol}: {msg}")
        if attempt < max_retries:
            time.sleep(min_sleep + attempt * 10)

    raise RuntimeError(f"Alpha Vantage failed for {symbol} after {max_retries} retries")


def fetch_stooq(symbol: str) -> pd.DataFrame:
    stooq_sym = STOOQ_MAP.get(symbol, symbol.lower().replace(".to", ".to"))
    url = f"https://stooq.com/q/d/l/?s={stooq_sym}&i=d"
    df = pd.read_csv(url)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"vol": "volume"})
    df["adj_close"] = df["close"]
    df.to_csv(DATA_INTERIM / f"{symbol}.csv", index=False)
    return df


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    num_cols = ["open", "high", "low", "close", "adj_close"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df.loc[df["volume"] < 0, "volume"] = np.nan
    df["volume"] = df["volume"].fillna(0).astype("int64")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["high"] >= df["low"]]
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    df["adj_close"] = df["close"]

    assert df["date"].is_monotonic_increasing, "dates not monotonic after clean"
    assert (df["high"] >= df["low"]).all(), "high < low after clean"
    assert df[["open", "high", "low", "close", "adj_close"]].isna().sum().sum() == 0

    return df


def ingest_symbol(symbol: str, api_key: str | None = None) -> Path:
    api_key = api_key or os.getenv("ALPHAVANTAGE_KEY")
    raw_df = None

    if api_key:
        try:
            raw_df = fetch_av(symbol, api_key)
        except Exception as e:
            print(f"[ingest] AV failed for {symbol} ({e}), trying Stooq...")

    if raw_df is None:
        raw_df = fetch_stooq(symbol)

    clean = clean_prices(raw_df)
    out = DATA_PROCESSED / f"{symbol}.parquet"
    clean.to_parquet(out, index=False)
    print(f"[ingest] {symbol}: {len(clean)} rows → {out}")
    return out


def ingest_all(symbols: list[str], api_key: str | None = None) -> None:
    api_key = api_key or os.getenv("ALPHAVANTAGE_KEY")
    for sym in symbols:
        ingest_symbol(sym, api_key)
