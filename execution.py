"""Order execution and position keeping.

Orders fill at the day's closing price plus a cost: half the bid-offer spread
plus square-root market impact, coeff x daily vol x sqrt(order size / ADV).
The book tracks quantities, average entry prices, cash, a trade blotter and
per-position unrealised P&L for the stop-loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


class Book:
    def __init__(self, tickers: list[str], capital: float = config.STARTING_CAPITAL):
        self.tickers = tickers
        self.cash = float(capital)
        self.qty = pd.Series(0.0, index=tickers)
        self.avg_price = pd.Series(0.0, index=tickers)
        self.blotter: list[dict] = []
        self.costs = 0.0
        self.realised = 0.0

    def nav(self, prices: pd.Series) -> float:
        return self.cash + float(self.qty @ prices)

    def weights(self, prices: pd.Series) -> pd.Series:
        return self.qty * prices / self.nav(prices)

    def unrealised(self, prices: pd.Series) -> pd.Series:
        return self.qty * (prices - self.avg_price)

    @staticmethod
    def cost(ticker: str, notional: float, daily_vol: float) -> float:
        spread = config.HALF_SPREAD_BPS[ticker] / 1e4
        impact = config.IMPACT_COEFF * daily_vol * np.sqrt(abs(notional) / (config.ADV_MILLIONS[ticker] * 1e6))
        return abs(notional) * (spread + impact)

    def trade(self, ticker: str, quantity: float, price: float, daily_vol: float, date, reason: str) -> None:
        notional = quantity * price
        cost = self.cost(ticker, notional, daily_vol)
        old = self.qty[ticker]
        new = old + quantity
        if quantity > 0:  # buying: blend the entry price
            self.avg_price[ticker] = (old * self.avg_price[ticker] + quantity * price) / new
        else:             # selling: book realised P&L at the old entry price
            self.realised += -quantity * (price - self.avg_price[ticker])
        if abs(new) < 1e-9:
            new, self.avg_price[ticker] = 0.0, 0.0
        self.qty[ticker] = new
        self.cash -= notional + cost
        self.costs += cost
        self.blotter.append({"date": date, "ticker": ticker, "side": "BUY" if quantity > 0 else "SELL",
                             "quantity": round(abs(quantity), 2), "price": round(price, 4),
                             "notional": round(abs(notional), 2), "cost": round(cost, 2), "reason": reason})

    def rebalance(self, target: pd.Series, prices: pd.Series, daily_vol: pd.Series, date, reason: str) -> None:
        """Trade to target NAV weights. Sells go first so buys are funded."""
        nav = self.nav(prices)
        delta = (target.reindex(self.tickers).fillna(0) * nav / prices) - self.qty
        orders = [(t, q) for t, q in delta.items() if abs(q * prices[t]) >= config.MIN_TRADE
                  or (q < 0 and abs(q + self.qty[t]) < 1e-9 and self.qty[t] > 0)]
        for t, q in sorted(orders, key=lambda o: o[1]):
            self.trade(t, q, prices[t], daily_vol[t], date, reason)

    def close(self, ticker: str, price: float, daily_vol: float, date, reason: str) -> None:
        if self.qty[ticker] > 0:
            self.trade(ticker, -self.qty[ticker], price, daily_vol, date, reason)
