# optimize.py
import numpy as np
import optuna
from signals import add_indicators_and_signals
from backtest import backtest
from utils import calmar_ratio

def objective_factory(train_df, n_splits: int = 5, min_trades_per_chunk: int = 15):
    def objective(trial: optuna.trial.Trial) -> float:
        # ===== RANGOS (más "sueltos" para actividad) =====
        rsi_window  = trial.suggest_int("rsi_window", 8, 20)
        rsi_low     = trial.suggest_int("rsi_low", 20, 40)
        rsi_high    = trial.suggest_int("rsi_high", 60, 80)

        sma_fast    = trial.suggest_int("sma_fast", 8, 40)
        sma_slow    = trial.suggest_int("sma_slow", 50, 120)
        if sma_fast >= sma_slow:
            raise optuna.TrialPruned()

        macd_fast   = trial.suggest_int("macd_fast", 8, 14)
        macd_slow   = trial.suggest_int("macd_slow", 18, 30)
        macd_signal = trial.suggest_int("macd_signal", 5, 12)
        if macd_fast >= macd_slow:
            raise optuna.TrialPruned()

        sl = trial.suggest_float("sl", 0.006, 0.03)     # 0.6%–3%
        tp = trial.suggest_float("tp", 0.02, 0.06)      # 2%–6%
        n_shares = trial.suggest_float("n_shares", 0.1, 5.0, step=0.1)

        data_sig = add_indicators_and_signals(
            train_df, rsi_window, rsi_low, rsi_high,
            sma_fast, sma_slow, macd_fast, macd_slow, macd_signal
        )

        L = len(data_sig)
        if L < n_splits * 200:
            raise optuna.TrialPruned()

        # walk-forward: trozos contiguos
        indices = np.array_split(np.arange(L), n_splits)

        calmars = []
        for idx in indices:
            chunk = data_sig.iloc[idx]
            if len(chunk) < 100:
                raise optuna.TrialPruned()

            equity, metrics = backtest(chunk, sl, tp, n_shares)

            # exigir actividad mínima
            if metrics.get("Trades", 0) < min_trades_per_chunk:
                raise optuna.TrialPruned()

            c = calmar_ratio(equity)
            if np.isnan(c):
                raise optuna.TrialPruned()
            calmars.append(c)

        return float(np.mean(calmars))
    return objective
