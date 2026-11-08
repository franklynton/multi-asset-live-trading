"""Constrained mean-variance optimisation under the mandate.

Maximises  w'mu - (gamma/2) w'Sigma w - c * sum|w - w_current|
subject to:
  - long only, each position <= $4.75m / NAV (soft limit under the $5m cap)
  - each asset class within its benchmark weight +/- 8% (soft limit inside +/- 10%)
  - 90% <= total invested <= 99%
The turnover term stops the optimiser trading back and forth on small changes
in its inputs; |x| is smoothed as sqrt(x^2 + eps) so SLSQP can use gradients.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import config


def ewma_update(cov_daily: np.ndarray, r: np.ndarray, lam: float = 0.97) -> np.ndarray:
    """RiskMetrics-style update of the daily covariance with one day's returns."""
    return lam * cov_daily + (1 - lam) * np.outer(r, r)


def constraint_bounds(nav: float, tickers: list[str], blocked: set[str] = frozenset()) -> list[tuple[float, float]]:
    cap = min(1.0, config.SOFT_POSITION / nav)
    return [(0.0, 0.0) if t in blocked else (0.0, cap) for t in tickers]


def optimise(mu: pd.Series, cov: pd.DataFrame, current: pd.Series, nav: float,
             blocked: set[str] = frozenset(), risk_aversion: float = 4.0,
             turnover_cost: float = 0.002) -> pd.Series:
    tickers = list(mu.index)
    m, sigma, w0 = mu.to_numpy(), cov.to_numpy(), current.reindex(tickers).fillna(0).to_numpy()
    eps = 1e-8

    def objective(w):
        d = w - w0
        return -(w @ m - 0.5 * risk_aversion * w @ sigma @ w - turnover_cost * np.sqrt(d * d + eps).sum())

    def gradient(w):
        d = w - w0
        return -(m - risk_aversion * sigma @ w - turnover_cost * d / np.sqrt(d * d + eps))

    bounds = constraint_bounds(nav, tickers, blocked)
    invested = [{"type": "ineq", "fun": lambda w: w.sum() - config.MIN_INVESTED},
                {"type": "ineq", "fun": lambda w: config.MAX_INVESTED - w.sum()}]
    bands = []
    for cls in config.CLASSES:
        mask = np.array([config.ASSET_CLASS[t] == cls for t in tickers], dtype=float)
        lo = config.CLASS_BENCHMARK[cls] - config.SOFT_BAND
        hi = config.CLASS_BENCHMARK[cls] + config.SOFT_BAND
        bands += [{"type": "ineq", "fun": lambda w, k=mask, lo=lo: k @ w - lo},
                  {"type": "ineq", "fun": lambda w, k=mask, hi=hi: hi - k @ w}]

    start = np.clip(w0, [b[0] for b in bounds], [b[1] for b in bounds])
    if start.sum() < config.MIN_INVESTED:
        start = np.array([b[1] for b in bounds]) * 0.95 / max(sum(b[1] for b in bounds), 1e-9)
    for constraints in (invested + bands, invested):   # drop class bands only if infeasible
        result = minimize(objective, start, jac=gradient, method="SLSQP", bounds=bounds,
                          constraints=constraints, options={"ftol": 1e-10, "maxiter": 300})
        if result.success:
            return pd.Series(np.clip(result.x, 0, None), index=tickers)
    raise RuntimeError(f"optimisation failed: {result.message}")


def check_mandate(weights: pd.Series, nav: float, tol: float = 1e-4) -> list[str]:
    """List of mandate breaches for a set of NAV weights (empty if compliant)."""
    breaches = []
    for t, w in weights.items():
        if w < -tol:
            breaches.append(f"{t} short")
        if w * nav > config.MAX_POSITION * (1 + tol):
            breaches.append(f"{t} above ${config.MAX_POSITION / 1e6:.0f}m cap")
    for cls in config.CLASSES:
        cw = sum(w for t, w in weights.items() if config.ASSET_CLASS[t] == cls)
        if abs(cw - config.CLASS_BENCHMARK[cls]) > config.CLASS_BAND + tol:
            breaches.append(f"{cls} at {cw:.1%}, outside band")
    if weights.sum() < config.MIN_INVESTED - tol:
        breaches.append(f"only {weights.sum():.1%} invested")
    return breaches
