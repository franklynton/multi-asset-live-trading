"""Runs a strategy through a live session, one trading day at a time.

At each day's close the trader sees that day's prices and any headline
released during the day, and only then decides what to do. A headline's
release-day move has already happened by then, so the trader can only
capture the follow-through.

Strategies:
  - benchmark: 50/30/20 weights held constant (frictionless index)
  - monthly:   mean-variance optimised, re-optimised every 21 trading days, with
               mandate checks and stop-losses every day, but no reaction to news
  - live:      the same, plus re-optimising on each headline with a view taken
               from the macro playbook, and again when that view runs out
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from execution import Book
from market import TRADING_DAYS, Calibration, Session
from optimiser import check_mandate, ewma_update, optimise

REBALANCE_DAYS = 21


@dataclass
class Result:
    name: str
    nav: pd.Series
    weights: pd.DataFrame
    blotter: pd.DataFrame = field(default_factory=pd.DataFrame)
    decisions: pd.DataFrame = field(default_factory=pd.DataFrame)
    costs: float = 0.0
    stop_losses: int = 0
    end_of_day_breaches: int = 0


def run_benchmark(session: Session) -> Result:
    w = pd.Series(config.benchmark_weights())[session.returns.columns]
    daily = session.returns @ w
    nav = config.STARTING_CAPITAL * pd.concat([pd.Series([1.0], index=session.prices.index[:1]), (1 + daily).cumprod()])
    weights = pd.DataFrame([w] * len(nav), index=nav.index)
    return Result("Benchmark", nav, weights)


def playbook_view(session: Session, day: int, confidence: float) -> pd.Series:
    """Expected annualised drift still to come from headlines in the last
    DRIFT_DAYS days. Uses the playbook's average move; the trader doesn't know
    how big this particular headline will turn out to be."""
    tickers = session.returns.columns
    view = pd.Series(0.0, index=tickers)
    weights = np.arange(config.DRIFT_DAYS, 0, -1) / np.arange(config.DRIFT_DAYS, 0, -1).sum()
    for _, ev in session.events[(session.events.day <= day) & (session.events.day > day - config.DRIFT_DAYS)].iterrows():
        elapsed = day - ev.day
        left = weights[elapsed:].sum()                     # share of the drift still to come
        days_left = config.DRIFT_DAYS - elapsed
        move = pd.Series(config.EVENTS[ev.event]["move"])[tickers] / 100
        view += (1 - config.RELEASE_SHARE) * move * left / days_left * TRADING_DAYS
    return confidence * view


def run_strategy(session: Session, cal: Calibration, react_to_news: bool, confidence: float = 0.5,
                 risk_aversion: float = 4.0) -> Result:
    name = "Live (news-driven)" if react_to_news else "Monthly rebalance"
    tickers = list(session.returns.columns)
    book = Book(tickers)
    cov_d = cal.cov.to_numpy() / TRADING_DAYS
    blocked: dict[str, int] = {}
    event_days = set(session.events.day)
    expiry_days = {d + config.DRIFT_DAYS for d in event_days}
    navs, weights, decisions = [], [], []
    stops = breaches_eod = 0

    for i, date in enumerate(session.prices.index):
        prices = session.prices.iloc[i]
        if i > 0:
            cov_d = ewma_update(cov_d, session.returns.iloc[i - 1].to_numpy())
        vol_d = pd.Series(np.sqrt(np.diag(cov_d)), index=tickers)
        reasons = []

        # Stop-losses
        for t, pnl in book.unrealised(prices).items():
            if pnl <= -config.STOP_LOSS:
                book.close(t, prices[t], vol_d[t], date, f"stop-loss ({pnl / 1e3:,.0f}k)")
                blocked[t] = i + config.STOP_COOLDOWN
                stops += 1
                reasons.append(f"stop-loss {t}")
        blocked = {t: until for t, until in blocked.items() if until > i}

        if i == 0:
            reasons.append("initial allocation")
        if i > 0 and i % REBALANCE_DAYS == 0:
            reasons.append("monthly rebalance")
        if react_to_news and i in event_days:
            headline = session.events.loc[session.events.day == i, "headline"].iloc[0]
            reasons.append(f"news: {headline}")
        if react_to_news and i in expiry_days and i not in event_days:
            reasons.append("news view expired")
        breaches = check_mandate(book.weights(prices), book.nav(prices)) if i > 0 else []
        if breaches:
            reasons.append("mandate: " + "; ".join(breaches))

        if reasons:
            mu = cal.mu + (playbook_view(session, i, confidence) if react_to_news else 0)
            cov = pd.DataFrame(cov_d * TRADING_DAYS, index=tickers, columns=tickers)
            target = optimise(mu, cov, book.weights(prices), book.nav(prices), set(blocked), risk_aversion)
            book.rebalance(target, prices, vol_d, date, reasons[-1] if len(reasons) == 1 else " / ".join(reasons))
            decisions.append({"date": date, "day": i, "reason": " / ".join(reasons),
                              **{t: round(w, 4) for t, w in target.items()}})

        nav = book.nav(prices)
        w = book.weights(prices)
        breaches_eod += bool(check_mandate(w, nav, tol=1e-3))
        navs.append(nav)
        weights.append(w)

    return Result(name, pd.Series(navs, index=session.prices.index), pd.DataFrame(weights, index=session.prices.index),
                  pd.DataFrame(book.blotter), pd.DataFrame(decisions), book.costs, stops, breaches_eod)
