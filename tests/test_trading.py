import numpy as np
import pandas as pd
import pytest

import config
from data import synthetic_prices
from execution import Book
from market import calibrate, simulate_session
from metrics import max_drawdown
from optimiser import check_mandate, ewma_update, optimise
from trader import playbook_view, run_benchmark, run_strategy


@pytest.fixture(scope="module")
def cal():
    return calibrate(synthetic_prices(config.TICKERS, "2018-01-01", "2025-12-31"))


@pytest.fixture(scope="module")
def session(cal):
    return simulate_session(cal, seed=3)


def test_benchmark_weights_match_class_split():
    w = pd.Series(config.benchmark_weights())
    assert w.sum() == pytest.approx(1.0)
    for cls, target in config.CLASS_BENCHMARK.items():
        assert w[[t for t in w.index if config.ASSET_CLASS[t] == cls]].sum() == pytest.approx(target)


def test_optimiser_respects_mandate(cal):
    nav = config.STARTING_CAPITAL
    w = optimise(cal.mu, cal.cov, pd.Series(0.0, index=cal.mu.index), nav)
    assert check_mandate(w, nav) == []
    assert (w * nav <= config.SOFT_POSITION + 1).all()
    assert config.MIN_INVESTED - 1e-6 <= w.sum() <= config.MAX_INVESTED + 1e-6


def test_blocked_instrument_gets_zero_weight(cal):
    w = optimise(cal.mu, cal.cov, pd.Series(0.0, index=cal.mu.index), config.STARTING_CAPITAL, blocked={"SPY"})
    assert w["SPY"] == pytest.approx(0.0, abs=1e-8)


def test_check_mandate_flags_each_breach():
    w = pd.Series(0.0, index=config.TICKERS)
    w["SPY"] = 0.30  # $6m on $20m: over the cap, and equity at 30% is under its 40% floor
    w["IEF"] = 0.20
    w["GLD"] = 0.20
    breaches = check_mandate(w, 20e6)
    assert any("SPY above" in b for b in breaches)
    assert any("Equity" in b for b in breaches)
    assert any("invested" in b for b in breaches)


def test_trade_accounting():
    book = Book(["SPY"], capital=1_000_000)
    book.trade("SPY", 1000, 100.0, 0.01, None, "buy")
    book.trade("SPY", 1000, 110.0, 0.01, None, "buy")
    assert book.avg_price["SPY"] == pytest.approx(105.0)
    book.trade("SPY", -500, 120.0, 0.01, None, "sell")
    assert book.realised == pytest.approx(500 * 15)
    prices = pd.Series({"SPY": 120.0})
    # NAV = starting capital + realised + unrealised - costs
    assert book.nav(prices) == pytest.approx(1_000_000 + 500 * 15 + 1500 * 15 - book.costs)


def test_cost_has_spread_and_impact():
    small = Book.cost("SPY", 1e5, 0.01) / 1e5
    large = Book.cost("SPY", 1e9, 0.01) / 1e9
    assert small >= config.HALF_SPREAD_BPS["SPY"] / 1e4
    assert large > small  # impact grows with order size


def test_ewma_update():
    cov = np.eye(2) * 1e-4
    r = np.array([0.02, -0.01])
    new = ewma_update(cov, r, lam=0.9)
    assert new[0, 1] == pytest.approx(0.1 * 0.02 * -0.01)
    assert new[0, 0] == pytest.approx(0.9e-4 + 0.1 * 0.0004)


def test_max_drawdown():
    nav = pd.Series([100, 120, 90, 110, 130])
    assert max_drawdown(nav) == pytest.approx(0.25)


def test_session_shapes_and_events(session):
    assert len(session.prices) == len(session.returns) + 1
    assert (session.events.day >= 1).all() and (session.events.day < len(session.prices)).all()


def test_playbook_view_only_after_headline(session):
    ev = session.events.iloc[0]
    assert (playbook_view(session, ev.day - 1, 1.0) == 0).all()  # nothing known before the first headline
    view = playbook_view(session, ev.day, 1.0)
    move = pd.Series(config.EVENTS[ev.event]["move"])[view.index]
    if not (session.events.day.iloc[1:] <= ev.day).any():
        assert np.sign(view[move != 0]).eq(np.sign(move[move != 0])).all()
    last = session.events.day.max()
    assert (playbook_view(session, last + config.DRIFT_DAYS, 1.0) == 0).all()  # view has run out


def test_strategies_run_inside_mandate(session, cal):
    bench = run_benchmark(session)
    assert bench.nav.iloc[0] == config.STARTING_CAPITAL
    for news in (False, True):
        r = run_strategy(session, cal, react_to_news=news)
        assert r.end_of_day_breaches == 0
        assert (r.weights.min() >= -1e-9).all()
        assert r.nav.iloc[0] == pytest.approx(config.STARTING_CAPITAL, rel=1e-3)
