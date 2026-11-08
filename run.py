"""Command-line runner.

1. Calibrates the market on historical prices.
2. Runs one live session in full: trade blotter, decision log, daily NAV and
   weights, summary table and charts.
3. Runs the same strategies over many simulated sessions (Monte Carlo) to see
   whether reacting to news helps in general or just in one lucky year.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config
import plots
from data import load_prices
from market import calibrate, simulate_session
from metrics import stats
from trader import run_benchmark, run_strategy

OUT = Path(__file__).parent / "output"


def run_all(session, cal, args):
    bench = run_benchmark(session)
    monthly = run_strategy(session, cal, react_to_news=False, risk_aversion=args.risk_aversion)
    live = run_strategy(session, cal, react_to_news=True, confidence=args.confidence, risk_aversion=args.risk_aversion)
    return bench, monthly, live


def summary_table(results, bench) -> pd.DataFrame:
    table = pd.DataFrame({r.name: stats(r.nav, bench.nav) for r in results})
    table.loc["Trades"] = [len(r.blotter) for r in results]
    table.loc["Trading costs ($)"] = [r.costs for r in results]
    table.loc["Stop-losses hit"] = [r.stop_losses for r in results]
    table.loc["Days ending in breach"] = [r.end_of_day_breaches for r in results]
    return table


def fmt(table: pd.DataFrame) -> pd.DataFrame:
    pct = ["Return", "Excess return vs benchmark", "Volatility", "Tracking error", "Max drawdown", "VaR 95% (1-day)"]
    out = table.astype(object).copy()
    for row in table.index:
        if row in pct:
            out.loc[row] = [f"{v:.2%}" for v in table.loc[row]]
        elif row in ("Trades", "Stop-losses hit", "Days ending in breach"):
            out.loc[row] = [f"{v:.0f}" for v in table.loc[row]]
        elif row == "Trading costs ($)":
            out.loc[row] = [f"{v:,.0f}" for v in table.loc[row]]
        else:
            out.loc[row] = [f"{v:.2f}" for v in table.loc[row]]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2016-01-01", help="calibration start date")
    p.add_argument("--end", default="2026-01-01", help="calibration end date")
    p.add_argument("--source", choices=["yahoo", "synthetic"], default="yahoo")
    p.add_argument("--seed", type=int, default=7, help="seed for the featured session")
    p.add_argument("--sessions", type=int, default=200, help="Monte Carlo sessions (0 to skip)")
    p.add_argument("--confidence", type=float, default=0.5, help="weight on the news playbook view")
    p.add_argument("--risk-aversion", type=float, default=4.0)
    p.add_argument("--follow-through", type=float, default=1.0,
                   help="average post-headline drift, as a multiple of the playbook (0 = none)")
    p.add_argument("--out", default=str(OUT), help="output folder")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    prices = load_prices(config.TICKERS, args.start, args.end, args.source)
    cal = calibrate(prices)
    print(f"Calibrated on {len(prices)} days of prices ({prices.index[0]:%b %Y} - {prices.index[-1]:%b %Y})\n")
    print(pd.DataFrame({"Class": pd.Series(config.ASSET_CLASS), "Expected return": cal.mu.map("{:.1%}".format),
                        "Volatility": (cal.cov.to_numpy().diagonal() ** 0.5).round(3)}).to_string(), "\n")

    # Featured session
    session = simulate_session(cal, seed=args.seed, follow_through=args.follow_through)
    bench, monthly, live = run_all(session, cal, args)
    table = summary_table([live, monthly, bench], bench)
    print(f"Session {args.seed}: {len(session.events)} headlines\n")
    print(session.events[["date", "headline"]].to_string(index=False), "\n")
    print(fmt(table).to_string(), "\n")

    table.to_csv(out / "session_summary.csv")
    session.events.to_csv(out / "session_events.csv", index=False)
    live.blotter.to_csv(out / "trade_blotter.csv", index=False)
    live.decisions.to_csv(out / "decision_log.csv", index=False)
    pd.DataFrame({r.name: r.nav for r in (live, monthly, bench)}).to_csv(out / "nav.csv")
    live.weights.to_csv(out / "live_weights.csv")
    plots.session_chart(session, live, monthly, bench, out / "live_session.png")
    plots.exposure_chart(live, out / "exposure.png")

    # Monte Carlo
    if args.sessions:
        rows = []
        for seed in range(1000, 1000 + args.sessions):
            s = simulate_session(cal, seed=seed, follow_through=args.follow_through)
            b, m, l = run_all(s, cal, args)
            for r in (l, m, b):
                st = stats(r.nav, b.nav)
                rows.append({"seed": seed, "strategy": r.name, **st, "Trades": len(r.blotter),
                             "Trading costs ($)": r.costs, "Stop-losses hit": r.stop_losses,
                             "Days ending in breach": r.end_of_day_breaches, "Headlines": len(s.events)})
        mc = pd.DataFrame(rows)
        mc.to_csv(out / "monte_carlo.csv", index=False)
        agg = mc.groupby("strategy", sort=False).mean(numeric_only=True).drop(columns=["seed", "Headlines"]).T
        agg = agg[[live.name, monthly.name, bench.name]]
        live_x = mc[mc.strategy == live.name].set_index("seed")
        month_x = mc[mc.strategy == monthly.name].set_index("seed")
        print(f"Monte Carlo: averages over {args.sessions} sessions\n")
        print(fmt(agg).to_string(), "\n")
        print(f"Live beats benchmark in {(live_x['Excess return vs benchmark'] > 0).mean():.0%} of sessions; "
              f"beats monthly rebalancing in {(live_x['Return'] > month_x['Return']).mean():.0%}")
        print(f"Live minus monthly return: mean {(live_x['Return'] - month_x['Return']).mean():.2%}, "
              f"std {(live_x['Return'] - month_x['Return']).std():.2%}")
        plots.monte_carlo_chart(mc, live.name, monthly.name, out / "monte_carlo.png")

    print(f"\nwrote outputs to {out}")


if __name__ == "__main__":
    main()
