# signals.py
import pandas as pd
import ta

def add_indicators_and_signals(
    df: pd.DataFrame,
    rsi_window: int,
    rsi_low: int,
    rsi_high: int,
    sma_fast: int,
    sma_slow: int,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
) -> pd.DataFrame:
    data = df.copy()

    # ==== Indicadores ====
    # RSI
    data["RSI"] = ta.momentum.RSIIndicator(close=data["Close"], window=rsi_window).rsi()

    # SMAs
    data["SMA_fast"] = data["Close"].rolling(window=sma_fast, min_periods=sma_fast).mean()
    data["SMA_slow"] = data["Close"].rolling(window=sma_slow, min_periods=sma_slow).mean()

    # MACD
    macd = ta.trend.MACD(
        close=data["Close"],
        window_slow=macd_slow,
        window_fast=macd_fast,
        window_sign=macd_signal,
    )
    data["MACD"] = macd.macd()
    data["MACD_sig"] = macd.macd_signal()

    # ==== Señales individuales ====
    # RSI por CRUCE de umbrales (más actividad que niveles fijos)
    rsi_prev = data["RSI"].shift(1)
    sig_rsi_buy  = (rsi_prev >= rsi_low)  & (data["RSI"] <  rsi_low)   # cruce hacia abajo -> sobreventa
    sig_rsi_sell = (rsi_prev <= rsi_high) & (data["RSI"] >  rsi_high)  # cruce hacia arriba -> sobrecompra

    sig_sma_buy  = data["SMA_fast"] > data["SMA_slow"]
    sig_sma_sell = data["SMA_fast"] < data["SMA_slow"]

    sig_macd_buy  = data["MACD"] > data["MACD_sig"]
    sig_macd_sell = data["MACD"] < data["MACD_sig"]

    # ==== Confirmación 2 de 3 ====
    buys  = (pd.concat([sig_rsi_buy,  sig_sma_buy,  sig_macd_buy],  axis=1).sum(axis=1) >= 2)
    sells = (pd.concat([sig_rsi_sell, sig_sma_sell, sig_macd_sell], axis=1).sum(axis=1) >= 2)

    data["BUY_SIG"]  = buys
    data["SELL_SIG"] = sells

    # Limpieza por NaNs de ventanas
    data = data.dropna(subset=["RSI", "SMA_fast", "SMA_slow", "MACD", "MACD_sig"])
    return data
