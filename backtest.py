# backtest.py
import numpy as np
import pandas as pd
from utils import calmar_ratio, cagr_from_equity, max_drawdown, sharpe_ratio, sortino_ratio

# --- Parámetros globales del backtest ---
TX_FEE = 0.00125             # 0.125% por lado
INITIAL_CASH = 100_000.0
MAX_NOTIONAL = 1_000_000.0
MIN_QTY = 0.1                 # mínimo de BTC por trade

def backtest(df: pd.DataFrame, sl: float, tp: float, n_shares: float,
             tx_fee: float = TX_FEE, initial_cash: float = INITIAL_CASH):
    """
    Backtest sin apalancamiento. 1 posición a la vez (long/short).
    - sl,tp proporciones (0.01=1%).
    - n_shares es objetivo; se recorta dinámicamente por caja y tope de nocional.
    - Cierre por SL/TP y por señal contraria.
    """
    data = df.copy()
    cash = initial_cash
    shares = 0.0
    entry_price = None
    trade_pnls = []
    equity_curve = []

    for i in range(len(data)):
        price_open  = data["Open"].iloc[i]
        price_high  = data["High"].iloc[i]
        price_low   = data["Low"].iloc[i]
        price_close = data["Close"].iloc[i]

        closed_by_sl_tp = False

        # ===== CIERRE por SL/TP =====
        if shares != 0 and entry_price is not None:
            if shares > 0:  # LONG
                sl_price = entry_price * (1 - sl)
                tp_price = entry_price * (1 + tp)
                exit_price = None
                if price_low <= sl_price:
                    exit_price = sl_price
                elif price_high >= tp_price:
                    exit_price = tp_price
                if exit_price is not None:
                    cash += exit_price * shares * (1 - tx_fee)
                    pnl = (exit_price - entry_price) * shares \
                          - exit_price * shares * tx_fee - entry_price * shares * tx_fee
                    trade_pnls.append(pnl)
                    shares = 0.0
                    entry_price = None
                    closed_by_sl_tp = True
            else:  # SHORT
                sl_price = entry_price * (1 + sl)
                tp_price = entry_price * (1 - tp)
                exit_price = None
                if price_high >= sl_price:
                    exit_price = sl_price
                elif price_low <= tp_price:
                    exit_price = tp_price
                if exit_price is not None:
                    qty = abs(shares)
                    cash -= exit_price * qty * (1 + tx_fee)
                    pnl = (entry_price - exit_price) * qty \
                          - exit_price * qty * tx_fee - entry_price * qty * tx_fee
                    trade_pnls.append(pnl)
                    shares = 0.0
                    entry_price = None
                    closed_by_sl_tp = True

        # ===== CIERRE por SEÑAL CONTRARIA =====
        if shares > 0 and entry_price is not None and not closed_by_sl_tp and data["SELL_SIG"].iloc[i]:
            qty = shares
            cash += price_close * qty * (1 - tx_fee)
            pnl = (price_close - entry_price) * qty \
                  - price_close * qty * tx_fee - entry_price * qty * tx_fee
            trade_pnls.append(pnl)
            shares = 0.0
            entry_price = None

        elif shares < 0 and entry_price is not None and not closed_by_sl_tp and data["BUY_SIG"].iloc[i]:
            qty = abs(shares)
            cash -= price_close * qty * (1 + tx_fee)
            pnl = (entry_price - price_close) * qty \
                  - price_close * qty * tx_fee - entry_price * qty * tx_fee
            trade_pnls.append(pnl)
            shares = 0.0
            entry_price = None

        # ===== APERTURAS (dinámicas, sin apalancamiento) =====
        if shares == 0:
            max_qty_by_notional = MAX_NOTIONAL / price_close

            if data["BUY_SIG"].iloc[i]:
                max_qty_by_cash = cash / (price_close * (1 + tx_fee))
                qty = min(n_shares, max_qty_by_cash, max_qty_by_notional)
                qty = float(np.floor(qty * 1e6) / 1e6)
                if qty >= MIN_QTY:
                    cash -= price_close * qty * (1 + tx_fee)
                    shares = qty
                    entry_price = price_close

            elif data["SELL_SIG"].iloc[i]:
                equity_now = cash  # sin posición abierta
                max_qty_by_margin = equity_now / (price_close * (1 + tx_fee))
                qty = min(n_shares, max_qty_by_margin, max_qty_by_notional)
                qty = float(np.floor(qty * 1e6) / 1e6)
                if qty >= MIN_QTY:
                    cash += price_close * qty * (1 - tx_fee)
                    shares = -qty
                    entry_price = price_close

        # ===== Equity al cierre =====
        equity_curve.append(cash + shares * price_close)

    # ===== Cierre forzado al final =====
    if shares != 0 and entry_price is not None:
        last_price = data["Close"].iloc[-1]
        if shares > 0:
            cash += last_price * shares * (1 - tx_fee)
            pnl = (last_price - entry_price) * shares \
                  - last_price * shares * tx_fee - entry_price * shares * tx_fee
        else:
            qty = abs(shares)
            cash -= last_price * qty * (1 + tx_fee)
            pnl = (entry_price - last_price) * qty \
                  - last_price * qty * tx_fee - entry_price * qty * tx_fee
        trade_pnls.append(pnl)
        shares = 0.0
        entry_price = None
        equity_curve[-1] = cash

    equity = pd.Series(equity_curve, index=data.index, name="Equity")

    metrics = {
        "final_equity": float(equity.iloc[-1]),
        "CAGR": cagr_from_equity(equity),
        "MaxDD": max_drawdown(equity),
        "Sharpe": sharpe_ratio(equity),
        "Sortino": sortino_ratio(equity),
        "Calmar": calmar_ratio(equity),
        "WinRate": float((np.array(trade_pnls) > 0).mean()) if len(trade_pnls) else float("nan"),
        "Trades": int(len(trade_pnls)),
    }
    return equity, metrics
