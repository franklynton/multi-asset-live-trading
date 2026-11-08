"""Historical prices for calibration: daily adjusted closes from Yahoo Finance,
with a synthetic fallback so the project still runs offline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import ASSET_CLASS


def load_prices(tickers: list[str], start: str, end: str, source: str = "yahoo") -> pd.DataFrame:
    if source == "yahoo":
        try:
            return _yahoo_prices(tickers, start, end)
        except Exception as exc:
            print(f"Yahoo Finance download failed ({exc}); falling back to synthetic prices")
    return synthetic_prices(tickers, start, end)


def _yahoo_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    prices = data[tickers].dropna()
    if prices.empty:
        raise ValueError("no prices returned")
    return prices


def synthetic_prices(tickers: list[str], start: str, end: str, seed: int = 42) -> pd.DataFrame:
    """Correlated daily prices: instruments in the same asset class are more
    correlated with each other (0.6) than with the rest (0.1)."""
    dates = pd.bdate_range(start, end)
    n = len(tickers)
    rng = np.random.default_rng(seed)
    classes = [ASSET_CLASS.get(t, t) for t in tickers]
    vol_by_class = {"Equity": (0.15, 0.24), "Bonds": (0.05, 0.15), "Commodities": (0.15, 0.35)}
    drift = rng.uniform(0.02, 0.10, n)
    vol = np.array([rng.uniform(*vol_by_class.get(c, (0.15, 0.25))) for c in classes])

    corr = np.array([[0.6 if a == b else 0.1 for b in classes] for a in classes])
    np.fill_diagonal(corr, 1.0)
    shocks = rng.standard_normal((len(dates), n)) @ np.linalg.cholesky(corr).T

    log_returns = (drift - 0.5 * vol**2) / 252 + vol / np.sqrt(252) * shocks
    return pd.DataFrame(100 * np.exp(np.cumsum(log_returns, axis=0)), index=dates, columns=tickers)
