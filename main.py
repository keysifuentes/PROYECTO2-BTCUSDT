# main.py
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_loader import load_btcusdt_hourly
from utils import split_by_ratio, returns_table
from signals import add_indicators_and_signals
from backtest import backtest
from optimize import objective_factory
from plots import plot_equity, plot_drawdown
import optuna

def main():
    # 1) Datos
    print("1) Descargando datos...")
    df = load_btcusdt_hourly(start="2018-01-01")
    print(f"Datos: {df.index[0]} \u2192 {df.index[-1]} | {len(df)} velas")

    # 2) Split 60/20/20
    train_df, test_df, val_df = split_by_ratio(df, train=0.6, test=0.2)

    # 2.1) Optuna (Calmar, walk-forward)
    print("2) Optimizando en TRAIN (walk-forward)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    objective = objective_factory(train_df, n_splits=5, min_trades_per_chunk=15)
    study.optimize(objective, n_trials=60, show_progress_bar=True)

    p = study.best_params
    print("\nMejores hiperparámetros:")
    for k, v in p.items():
        print(f"  {k}: {v}")
    print(f"Mejor Calmar (train): {study.best_value:.4f}")

    # Guardar resultados Optuna
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/best_params.json", "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2)
    with open("outputs/best_value.json", "w", encoding="utf-8") as f:
        json.dump({"best_calmar_train": study.best_value}, f, indent=2)
    try:
        df_trials = study.trials_dataframe(attrs=("number","value","state","params","user_attrs","system_attrs","duration"))
        df_trials.to_csv("outputs/optuna_trials.csv", index=False)
    except Exception as e:
        print("Aviso: no se pudo exportar optuna_trials.csv:", e)

    # 3) TEST
    print("\n3) TEST...")
    test_sig = add_indicators_and_signals(
        test_df,
        p["rsi_window"], p["rsi_low"], p["rsi_high"],
        p["sma_fast"], p["sma_slow"], p["macd_fast"], p["macd_slow"], p["macd_signal"],
    )
    eq_test, m_test = backtest(test_sig, p["sl"], p["tp"], p["n_shares"])
    print(m_test)

    # 4) VALIDATION
    print("\n4) VALIDATION...")
    val_sig = add_indicators_and_signals(
        val_df,
        p["rsi_window"], p["rsi_low"], p["rsi_high"],
        p["sma_fast"], p["sma_slow"], p["macd_fast"], p["macd_slow"], p["macd_signal"],
    )
    eq_val, m_val = backtest(val_sig, p["sl"], p["tp"], p["n_shares"])
    print(m_val)

    # Conteos de señales (diagnóstico)
    print("\nConteos de señales (TEST):")
    print("BUY_SIG:", int(test_sig["BUY_SIG"].sum()), "SELL_SIG:", int(test_sig["SELL_SIG"].sum()))
    print("\nConteos de señales (VALIDATION):")
    print("BUY_SIG:", int(val_sig["BUY_SIG"].sum()), "SELL_SIG:", int(val_sig["SELL_SIG"].sum()))

    # 5) Tablas de rendimientos (VALIDATION)
    print("\n5) Tablas de rendimientos (VALIDATION)")
    print(returns_table(eq_val, "ME"))
    print(returns_table(eq_val, "QE"))
    print(returns_table(eq_val, "YE"))

    # Exportar métricas y curvas a outputs/
    with open("outputs/metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(m_test, f, indent=2)
    pd.DataFrame({"Equity": eq_test.values}, index=eq_test.index).to_csv("outputs/equity_test.csv")

    with open("outputs/metrics_validation.json", "w", encoding="utf-8") as f:
        json.dump(m_val, f, indent=2)
    pd.DataFrame({"Equity": eq_val.values}, index=eq_val.index).to_csv("outputs/equity_validation.csv")

    # Exportar tablas returns
    rt_test_m = returns_table(eq_test, "ME"); rt_test_q = returns_table(eq_test, "QE"); rt_test_a = returns_table(eq_test, "YE")
    rt_val_m  = returns_table(eq_val, "ME");  rt_val_q  = returns_table(eq_val, "QE");  rt_val_a  = returns_table(eq_val, "YE")
    rt_test_m.to_csv("outputs/returns_test_monthly.csv"); rt_test_q.to_csv("outputs/returns_test_quarterly.csv"); rt_test_a.to_csv("outputs/returns_test_annual.csv")
    rt_val_m.to_csv("outputs/returns_validation_monthly.csv"); rt_val_q.to_csv("outputs/returns_validation_quarterly.csv"); rt_val_a.to_csv("outputs/returns_validation_annual.csv")

    # 6) Gráficas
    print("\n6) Graficando (guardando en carpeta outputs/)...")
    plot_equity(eq_test, "Equity TEST", show=False, save="equity_test.png")
    plot_drawdown(eq_test, "TEST", show=False, save="drawdown_test.png")
    plot_equity(eq_val, "Equity VALIDATION", show=False, save="equity_validation.png")
    plot_drawdown(eq_val, "VALIDATION", show=False, save="drawdown_validation.png")
    print("Listo. Revisa la carpeta outputs/.")

if __name__ == "__main__":
    main()
