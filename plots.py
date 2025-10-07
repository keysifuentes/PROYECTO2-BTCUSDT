# plots.py
import os
import matplotlib.pyplot as plt
from typing import Optional

def plot_equity(equity, title: str, show: bool = False, save: Optional[str] = None):
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(equity.index, equity.values)
    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel("Equity")
    plt.tight_layout()
    if save:
        plt.savefig(f"outputs/{save}", dpi=150)
    if show:
        plt.show()
    plt.close()

def plot_drawdown(equity, title: str, show: bool = False, save: Optional[str] = None):
    os.makedirs("outputs", exist_ok=True)
    dd = equity / equity.cummax() - 1
    plt.figure(figsize=(10, 3))
    plt.plot(dd.index, dd.values)
    plt.title(title + " - Drawdown")
    plt.xlabel("Fecha")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    if save:
        plt.savefig(f"outputs/{save}", dpi=150)
    if show:
        plt.show()
    plt.close()
