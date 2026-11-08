"""Performance and risk statistics from a daily NAV series.

Cash earns nothing in the simulation, so the Sharpe ratio uses a zero
risk-free rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(nav: pd.Series) -> float:
    return float((1 - nav / nav.cummax()).max())


def stats(nav: pd.Series, benchmark: pd.Series) -> pd.Series:
    r = nav.pct_change().dropna()
    b = benchmark.pct_change().dropna()
    active = r - b
    vol = r.std() * np.sqrt(TRADING_DAYS)
    te = active.std() * np.sqrt(TRADING_DAYS)
    total = nav.iloc[-1] / nav.iloc[0] - 1
    bench_total = benchmark.iloc[-1] / benchmark.iloc[0] - 1
    return pd.Series({
        "Return": total,
        "Excess return vs benchmark": total - bench_total,
        "Volatility": vol,
        "Sharpe ratio": r.mean() * TRADING_DAYS / vol,
        "Beta vs benchmark": r.cov(b) / b.var(),
        "Tracking error": te,
        "Information ratio": active.mean() * TRADING_DAYS / te if te > 1e-12 else np.nan,
        "Max drawdown": max_drawdown(nav),
        "VaR 95% (1-day)": -np.percentile(r, 5),
    })
