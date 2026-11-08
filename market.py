"""Simulated live market, calibrated on historical returns.

Each session is one trading year. Daily returns are multivariate Student-t
(fat tails) with the historical covariance matrix. Each session draws its own
true drift around the shrunk historical mean, so the trader never knows it. Macro headlines arrive at random during the year. Each one moves prices
on the release day and then keeps moving them over the next few days, with
volatility raised while the market digests the news.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import DRIFT_DAYS, EVENTS, RELEASE_SHARE

TRADING_DAYS = 252


@dataclass
class Calibration:
    mu: pd.Series     # annualised expected return
    cov: pd.DataFrame  # annualised covariance matrix
    last_prices: pd.Series
    years: float       # length of the history the estimates come from


def calibrate(prices: pd.DataFrame, shrink: float = 0.5) -> Calibration:
    """Annualised mean and covariance of daily returns. Mean returns are shrunk
    halfway towards their cross-sectional average, since a few years of history
    estimate them very noisily."""
    returns = prices.pct_change().dropna()
    raw_mu = returns.mean() * TRADING_DAYS
    mu = (1 - shrink) * raw_mu + shrink * raw_mu.mean()
    return Calibration(mu=mu, cov=returns.cov() * TRADING_DAYS, last_prices=prices.iloc[-1],
                       years=len(returns) / TRADING_DAYS)


@dataclass
class Session:
    prices: pd.DataFrame   # simulated closes, row 0 is the opening price
    returns: pd.DataFrame  # simulated daily returns (days 1..T)
    events: pd.DataFrame   # day, event, headline


def simulate_session(cal: Calibration, seed: int, days: int = TRADING_DAYS, events_per_year: float = 12,
                     df: int = 5, start: str = "2026-01-02", follow_through: float = 1.0) -> Session:
    rng = np.random.default_rng(seed)
    tickers = list(cal.mu.index)
    n = len(tickers)

    # Fat-tailed noise: Student-t scaled to the daily covariance
    cov_d = cal.cov.to_numpy() / TRADING_DAYS
    normal = rng.multivariate_normal(np.zeros(n), cov_d * (df - 2) / df, size=days)
    noise = normal / np.sqrt(rng.chisquare(df, size=(days, 1)) / df)

    # The session's true drift differs from the calibrated estimate by the standard
    # error of a mean estimated from cal.years of data, so the trader's expected
    # returns are a noisy estimate rather than the real answer
    true_mu = cal.mu.to_numpy() + rng.multivariate_normal(np.zeros(n), cal.cov.to_numpy() / cal.years)
    drift = np.tile(true_mu / TRADING_DAYS, (days, 1))
    vol_scale = np.ones((days, 1))

    # Macro headlines: random days (not the first or last few), random types
    n_events = rng.poisson(events_per_year * days / TRADING_DAYS)
    event_days = np.sort(rng.choice(np.arange(5, days - DRIFT_DAYS), size=n_events, replace=False))
    names = list(EVENTS)
    rows = []
    for day in event_days:
        name = names[rng.integers(len(names))]
        size = rng.lognormal(0, 0.3)                       # some headlines are bigger than others
        move = np.array([EVENTS[name]["move"][t] for t in tickers]) / 100 * size
        drift[day] += RELEASE_SHARE * move
        # Follow-through: positive on average (follow_through=1 means the full second
        # half of the move arrives on average), but noisy and sometimes reverses
        follow = (1 - RELEASE_SHARE) * move * rng.normal(follow_through, 0.8)
        weights = np.arange(DRIFT_DAYS, 0, -1) / np.arange(DRIFT_DAYS, 0, -1).sum()  # front-loaded
        end = min(day + 1 + DRIFT_DAYS, days)
        drift[day + 1:end] += np.outer(weights[: end - day - 1], follow)
        vol_scale[day:day + 5] = np.maximum(vol_scale[day:day + 5], 1.5)
        rows.append({"day": int(day) + 1, "event": name, "headline": EVENTS[name]["headline"]})

    returns = drift + noise * vol_scale
    dates = pd.bdate_range(start, periods=days + 1)
    rets = pd.DataFrame(returns, index=dates[1:], columns=tickers)
    path = np.vstack([np.ones(n), np.cumprod(1 + returns, axis=0)])
    prices = pd.DataFrame(path * cal.last_prices.to_numpy(), index=dates, columns=tickers)
    events = pd.DataFrame(rows, columns=["day", "event", "headline"])
    events["date"] = [dates[d] for d in events["day"]]
    return Session(prices=prices, returns=rets, events=events)
