# data_loader.py
import os
import time
from datetime import datetime, timezone
import requests
import pandas as pd
from tqdm import tqdm

BINANCE_URL = "https://api.binance.com/api/v3/klines"

def _fetch_chunk(start_ms: int, symbol="BTCUSDT", interval="1h", limit=1000):
    params = dict(symbol=symbol, interval=interval, limit=limit, startTime=start_ms)
    r = requests.get(BINANCE_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def load_btcusdt_hourly(start="2018-01-01", end=None, cache_path="data/BTCUSDT_1h.csv") -> pd.DataFrame:
    """
    Descarga (y cachea) velas 1h de Binance Spot.
    Devuelve DataFrame con columnas: Open, High, Low, Close, Volume y datetime index (UTC).
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Si ya existe el cache, úsalo
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["Open time"])
        df = df.set_index("Open time").sort_index()
        return df

    start_dt = pd.Timestamp(start, tz="UTC")
    if end is None:
        end_dt = datetime.now(timezone.utc)
    else:
        end_dt = pd.Timestamp(end, tz="UTC")

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    rows = []
    cur = start_ms
    pbar = tqdm(total=None, desc="Descargando BTCUSDT 1h")
    while cur < end_ms:
        data = _fetch_chunk(cur)
        if not data:
            break
        rows.extend(data)
        last_open = data[-1][0]
        # siguiente vela (1h)
        cur = last_open + 60 * 60 * 1000
        pbar.update(1)
        time.sleep(0.1)  # cortesía para no saturar

    pbar.close()

    if not rows:
        raise RuntimeError("No se pudo descargar datos de Binance.")

    cols = [
        "Open time","Open","High","Low","Close","Volume",
        "Close time","Quote asset volume","Number of trades",
        "Taker buy base asset volume","Taker buy quote asset volume","Ignore"
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms", utc=True)
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = df[c].astype(float)

    df = df[["Open time","Open","High","Low","Close","Volume"]] \
           .set_index("Open time").sort_index()

    df.to_csv(cache_path, index=True)
    return df
