# utils.py
import numpy as np
import pandas as pd

def split_by_ratio(df: pd.DataFrame, train=0.6, test=0.2):
    n = len(df)
    i_train = int(n * train)
    i_test  = int(n * (train + test))
    return df.iloc[:i_train], df.iloc[i_train:i_test], df.iloc[i_test:]

def cagr_from_equity(equity: pd.Series) -> float:
    if len(equity) < 2:
        return np.nan
    start, end = equity.iloc[0], equity.iloc[-1]
    years = (equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 24 * 3600)
    if years <= 0 or start <= 0:
        return np.nan
    return (end / start) ** (1 / years) - 1

def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())

def _ann_factor_from_index(idx: pd.DatetimeIndex) -> float:
    # datos horarios -> 24*365.25 periodos/año
    return np.sqrt(24 * 365.25)

def sharpe_ratio(equity: pd.Series, rf: float = 0.0) -> float:
    rets = equity.pct_change().dropna()
    if rets.std() == 0 or len(rets) == 0:
        return np.nan
    ann = _ann_factor_from_index(equity.index)
    return float(((rets.mean() - rf) / rets.std()) * ann)

def sortino_ratio(equity: pd.Series, rf: float = 0.0) -> float:
    rets = equity.pct_change().dropna()
    downside = rets[rets < 0]
    if downside.std() == 0 or len(rets) == 0:
        return np.nan
    ann = _ann_factor_from_index(equity.index)
    return float(((rets.mean() - rf) / downside.std()) * ann)

def calmar_ratio(equity: pd.Series) -> float:
    cagr = cagr_from_equity(equity)
    mdd = max_drawdown(equity)
    if mdd == 0 or np.isnan(cagr) or np.isnan(mdd):
        return np.nan
    return float(cagr / abs(mdd))

def returns_table(equity: pd.Series, freq: str = "ME") -> pd.DataFrame:
    """
    Construye tabla de retornos para equity.
    Frecuencias válidas:
      'ME' = Month End
      'QE' = Quarter End
      'YE' = Year End
    """
    if freq not in {"ME", "QE", "YE"}:
        raise ValueError("freq debe ser 'ME', 'QE' o 'YE'")

    # Fin de periodo sin FutureWarning (antes usabas 'M','Q','A')
    eq = equity.resample(freq).last()
    ret = eq.pct_change().fillna(0.0)

    out = pd.DataFrame(
        {"Return": ret, "Return_%": (ret * 100)},
        index=eq.index,
    )
    out.index.name = equity.index.name or "Open time"
    return out
