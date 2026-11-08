"""Instruments, mandate rules and the macro news playbook."""

from __future__ import annotations

# Instruments and asset classes

ASSET_CLASS = {
    "SPY": "Equity",       # US equity
    "EFA": "Equity",       # developed ex-US equity
    "EWJ": "Equity",       # Japan equity
    "EEM": "Equity",       # emerging-market equity
    "IEF": "Bonds",        # 7-10y US Treasuries
    "TLT": "Bonds",        # 20y+ US Treasuries
    "BNDX": "Bonds",       # international bonds (USD-hedged)
    "GLD": "Commodities",  # gold
    "USO": "Commodities",  # crude oil
    "DBA": "Commodities",  # agriculture
}
TICKERS = list(ASSET_CLASS)
CLASSES = ["Equity", "Bonds", "Commodities"]

# Benchmark: 50% equity / 30% bonds / 20% commodities, equal-weighted within each class
CLASS_BENCHMARK = {"Equity": 0.50, "Bonds": 0.30, "Commodities": 0.20}


def benchmark_weights() -> dict[str, float]:
    counts = {c: sum(1 for t in TICKERS if ASSET_CLASS[t] == c) for c in CLASSES}
    return {t: CLASS_BENCHMARK[ASSET_CLASS[t]] / counts[ASSET_CLASS[t]] for t in TICKERS}


# Mandate

STARTING_CAPITAL = 20_000_000
CLASS_BAND = 0.10            # each asset class within benchmark +/- 10%
MAX_POSITION = 5_000_000     # single-instrument notional cap ($)
MIN_INVESTED = 0.90          # at least 90% of NAV invested
MAX_INVESTED = 0.99          # leaves a small cash buffer for trading costs
STOP_LOSS = 500_000          # exit a position once its unrealised loss reaches $500k
STOP_COOLDOWN = 10           # trading days before a stopped-out instrument can be rebought
MIN_TRADE = 25_000           # ignore rebalancing trades smaller than this ($)
# The optimiser targets soft limits inside the hard ones, so normal price moves
# don't push positions straight back over a limit
SOFT_POSITION = 4_750_000
SOFT_BAND = 0.08

# Execution costs: half bid-offer spread (bps) and average daily traded value ($m)

HALF_SPREAD_BPS = {"SPY": 0.5, "EFA": 1.0, "EWJ": 1.5, "EEM": 1.5, "IEF": 1.0, "TLT": 1.0,
                   "BNDX": 2.0, "GLD": 1.0, "USO": 2.5, "DBA": 5.0}
ADV_MILLIONS = {"SPY": 30_000, "EFA": 1_500, "EWJ": 300, "EEM": 1_200, "IEF": 1_000, "TLT": 3_000,
                "BNDX": 200, "GLD": 1_500, "USO": 300, "DBA": 15}
IMPACT_COEFF = 0.1           # square-root impact: coeff x daily vol x sqrt(order / ADV)

# Macro news playbook: total expected move (%) for each instrument after the headline.
# Half of the move happens on the day of release, before anyone can trade on it;
# the rest drifts in over the following days, with a lot of noise around it.

EVENTS = {
    "hawkish_fed": {
        "headline": "Fed hikes 50bp, signals more to come",
        "move": {"SPY": -2.5, "EFA": -1.8, "EWJ": -1.2, "EEM": -3.0, "IEF": -1.5, "TLT": -3.0,
                 "BNDX": -0.8, "GLD": -1.5, "USO": -1.0, "DBA": -0.5},
    },
    "dovish_fed": {
        "headline": "Fed signals cuts as inflation cools",
        "move": {"SPY": 2.5, "EFA": 1.8, "EWJ": 1.0, "EEM": 3.0, "IEF": 1.5, "TLT": 3.0,
                 "BNDX": 0.8, "GLD": 1.5, "USO": 0.5, "DBA": 0.3},
    },
    "inflation_surprise": {
        "headline": "CPI comes in well above consensus",
        "move": {"SPY": -1.8, "EFA": -1.2, "EWJ": -0.8, "EEM": -1.5, "IEF": -1.2, "TLT": -2.5,
                 "BNDX": -0.6, "GLD": 2.0, "USO": 2.5, "DBA": 2.0},
    },
    "growth_scare": {
        "headline": "Payrolls miss badly, recession fears grow",
        "move": {"SPY": -3.0, "EFA": -2.5, "EWJ": -2.0, "EEM": -3.5, "IEF": 1.5, "TLT": 3.0,
                 "BNDX": 0.8, "GLD": 1.5, "USO": -4.0, "DBA": -1.0},
    },
    "soft_landing": {
        "headline": "Strong earnings and steady growth: soft landing priced in",
        "move": {"SPY": 2.5, "EFA": 2.0, "EWJ": 1.5, "EEM": 2.5, "IEF": -0.3, "TLT": -0.8,
                 "BNDX": -0.2, "GLD": -0.8, "USO": 1.5, "DBA": 0.5},
    },
    "oil_supply_shock": {
        "headline": "Suez Canal disruption: oil supply shock",
        "move": {"SPY": -1.2, "EFA": -1.8, "EWJ": -1.5, "EEM": -1.5, "IEF": 0.5, "TLT": 0.8,
                 "BNDX": 0.3, "GLD": 2.0, "USO": 8.0, "DBA": 1.5},
    },
    "em_crisis": {
        "headline": "Emerging-market currency crisis spreads",
        "move": {"SPY": -1.0, "EFA": -1.5, "EWJ": -1.0, "EEM": -6.0, "IEF": 0.8, "TLT": 1.2,
                 "BNDX": 0.4, "GLD": 1.5, "USO": -2.0, "DBA": -1.0},
    },
    "boj_shift": {
        "headline": "BoJ abandons yield curve control",
        "move": {"SPY": -0.5, "EFA": -0.5, "EWJ": -4.0, "EEM": -0.8, "IEF": -0.6, "TLT": -1.2,
                 "BNDX": -1.0, "GLD": 0.5, "USO": 0.0, "DBA": 0.0},
    },
}
RELEASE_SHARE = 0.5          # share of the move that happens on the release day
DRIFT_DAYS = 10              # days over which the rest of the move drifts in
