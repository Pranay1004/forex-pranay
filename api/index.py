import os
from datetime import timezone
from typing import Any

import pandas as pd
import requests
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

app = FastAPI(title="Forex Pranay API", version="1.0.0")
UTC = timezone.utc

PAIRS = {"EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF"}
TF_MAP: dict[str, dict[str, Any]] = {
    "5m": {"td": "5min", "yf": "5m", "period_days": 30, "resample": None},
    "15m": {"td": "15min", "yf": "15m", "period_days": 60, "resample": None},
    "30m": {"td": "30min", "yf": "30m", "period_days": 90, "resample": None},
    "4h": {"td": "4h", "yf": "60m", "period_days": 365, "resample": "4h"},
    "1d": {"td": "1day", "yf": "1d", "period_days": 3650, "resample": None},
}


def _get_env(name: str) -> str:
    return os.getenv(name, "").strip()


def _fetch_twelvedata(pair: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    api_key = _get_env("TWELVEDATA_API_KEY")
    if not api_key:
        return pd.DataFrame()

    symbol = f"{pair[:3]}/{pair[3:]}"
    interval = TF_MAP[timeframe]["td"]
    params = {
        "symbol": symbol,
        "interval": interval,
        "start_date": start,
        "end_date": end,
        "outputsize": 5000,
        "apikey": api_key,
        "format": "JSON",
        "timezone": "UTC",
    }

    try:
        response = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=30)
        response.raise_for_status()
        values = response.json().get("values", [])
        if not values:
            return pd.DataFrame()

        rows = []
        for item in values:
            rows.append(
                {
                    "datetime": item["datetime"],
                    "open": float(item["open"]),
                    "high": float(item["high"]),
                    "low": float(item["low"]),
                    "close": float(item["close"]),
                    "volume": float(item.get("volume", 0) or 0),
                }
            )

        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.sort_values("datetime").drop_duplicates("datetime")
        return df
    except Exception:
        return pd.DataFrame()


def _fetch_yfinance(pair: str, timeframe: str) -> pd.DataFrame:
    ticker = f"{pair}=X"
    interval = TF_MAP[timeframe]["yf"]
    period_days = TF_MAP[timeframe]["period_days"]

    try:
        if timeframe == "1d":
            hist = yf.download(ticker, period="max", interval=interval, auto_adjust=False, progress=False)
        else:
            hist = yf.download(ticker, period=f"{period_days}d", interval=interval, auto_adjust=False, progress=False)

        if hist.empty:
            return pd.DataFrame()

        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = [column[0] for column in hist.columns]

        df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
        if df.index.tz is None:
            df.index = df.index.tz_localize(UTC)
        else:
            df.index = df.index.tz_convert(UTC)

        if TF_MAP[timeframe]["resample"] == "4h":
            df = (
                df.resample("4h")
                .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                .dropna()
            )

        out = df.reset_index().rename(
            columns={
                "Datetime": "datetime",
                "Date": "datetime",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        out["datetime"] = pd.to_datetime(out["datetime"], utc=True)
        out = out.sort_values("datetime").drop_duplicates("datetime")
        return out
    except Exception:
        return pd.DataFrame()


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "twelvedata_key_present": bool(_get_env("TWELVEDATA_API_KEY")),
        "fmp_key_present": bool(_get_env("FMP_API_KEY")),
    }


@app.get("/api/ohlcv")
def ohlcv(
    pair: str = Query("USDJPY"),
    timeframe: str = Query("1d"),
    start: str = Query("2015-01-01"),
    end: str = Query("2026-02-01"),
) -> JSONResponse:
    pair = pair.upper().strip()
    timeframe = timeframe.strip()

    if pair not in PAIRS:
        raise HTTPException(status_code=400, detail=f"Unsupported pair: {pair}")
    if timeframe not in TF_MAP:
        raise HTTPException(status_code=400, detail=f"Unsupported timeframe: {timeframe}")

    td_df = _fetch_twelvedata(pair, timeframe, start, end)
    source = "TwelveData"
    if td_df.empty:
        td_df = _fetch_yfinance(pair, timeframe)
        source = "yfinance"

    if td_df.empty:
        raise HTTPException(status_code=502, detail="No data available from providers")

    start_ts = pd.Timestamp(start).tz_localize(UTC)
    end_ts = (pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize(UTC)
    td_df = td_df[(td_df["datetime"] >= start_ts) & (td_df["datetime"] < end_ts)].copy()

    if td_df.empty:
        raise HTTPException(status_code=404, detail="No candles found in requested date range")

    td_df["datetime"] = td_df["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return JSONResponse(
        {
            "pair": pair,
            "timeframe": timeframe,
            "source": source,
            "start": start,
            "end": end,
            "count": int(len(td_df)),
            "data": td_df.to_dict(orient="records"),
        }
    )
