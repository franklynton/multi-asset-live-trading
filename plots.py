"""Matplotlib charts."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

import config

COLOURS = {"Live (news-driven)": "#1f4e79", "Monthly rebalance": "#d08c2c", "Benchmark": "#8c8c8c"}
CLASS_COLOURS = {"Equity": "#1f4e79", "Bonds": "#3a9a6e", "Commodities": "#c9732c"}
SHORT = {"hawkish_fed": "Fed hike", "dovish_fed": "Fed cuts", "inflation_surprise": "CPI beat",
         "growth_scare": "Payrolls miss", "soft_landing": "Soft landing", "oil_supply_shock": "Oil shock",
         "em_crisis": "EM crisis", "boj_shift": "BoJ shift"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def session_chart(session, live, monthly, bench, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    for r in (bench, monthly, live):
        ax1.plot(r.nav.index, r.nav / 1e6, color=COLOURS[r.name], lw=1.8 if r is live else 1.2, label=r.name)
    top = max(r.nav.max() for r in (bench, monthly, live)) / 1e6
    for k, ev in enumerate(session.events.itertuples()):
        for ax in (ax1, ax2):
            ax.axvline(ev.date, color="#b23b3b", lw=0.6, alpha=0.5, ls="--")
        ax1.text(ev.date, top * (1.004 - 0.012 * (k % 3)), SHORT[ev.event], rotation=0, fontsize=6.5,
                 color="#b23b3b", ha="center", va="bottom")
    ax1.set_ylabel("Portfolio value ($m)")
    ax1.set_title("Simulated live session: $20m multi-asset portfolio, macro headlines in red")
    ax1.legend(loc="lower left", frameon=False)

    for r in (live, monthly):
        active = r.nav / bench.nav - 1
        ax2.plot(active.index, active, color=COLOURS[r.name], lw=1.4)
    ax2.axhline(0, color="black", lw=0.6)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=1))
    ax2.set_ylabel("Return vs benchmark")
    _save(fig, path)


def exposure_chart(live, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    w = live.weights
    for cls in config.CLASSES:
        cw = w[[t for t in w.columns if config.ASSET_CLASS[t] == cls]].sum(axis=1)
        b = config.CLASS_BENCHMARK[cls]
        ax1.fill_between(w.index, b - config.CLASS_BAND, b + config.CLASS_BAND, color=CLASS_COLOURS[cls], alpha=0.08)
        ax1.axhline(b, color=CLASS_COLOURS[cls], lw=0.6, ls=":")
        ax1.plot(w.index, cw, color=CLASS_COLOURS[cls], lw=1.6, label=f"{cls} (benchmark {b:.0%} ± 10%)")
    ax1.plot(w.index, w.sum(axis=1), color="black", lw=1, ls="--", label="Total invested (min 90%)")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax1.set_ylabel("Share of NAV")
    ax1.set_title("Live strategy: asset-class exposure inside the mandate bands")
    ax1.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)

    nav = live.nav
    positions = w.mul(nav, axis=0) / 1e6
    shades = {"Equity": ["#1f4e79", "#3c6e9f", "#6d97c2", "#a3c1df"], "Bonds": ["#2b7a55", "#56a77f", "#93cfb0"],
              "Commodities": ["#a85a1c", "#d08c4f", "#ebbf91"]}
    used = {c: 0 for c in shades}
    colours = []
    for t in w.columns:
        c = config.ASSET_CLASS[t]
        colours.append(shades[c][used[c]])
        used[c] += 1
    ax2.stackplot(positions.index, positions.T.to_numpy(), labels=w.columns, colors=colours, alpha=0.9)
    ax2.set_ylabel("Position ($m)")
    ax2.set_title("Positions by instrument (each capped at $5m)")
    ax2.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, ncol=1)
    _save(fig, path)


def monte_carlo_chart(mc: pd.DataFrame, live_name: str, monthly_name: str, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in (monthly_name, live_name):
        x = mc.loc[mc.strategy == name, "Excess return vs benchmark"]
        ax1.hist(x, bins=30, alpha=0.6, color=COLOURS[name], label=f"{name} (mean {x.mean():.1%})")
    ax1.axvline(0, color="black", lw=0.8)
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax1.set_xlabel("Return minus benchmark return, one simulated year")
    ax1.set_ylabel("Sessions")
    ax1.set_title(f"Excess return across {mc.seed.nunique()} simulated sessions")
    ax1.legend(frameon=False)

    live = mc[mc.strategy == live_name].set_index("seed")["Return"]
    monthly = mc[mc.strategy == monthly_name].set_index("seed")["Return"]
    diff = live - monthly
    ax2.hist(diff, bins=30, color=COLOURS[live_name], alpha=0.8)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.axvline(diff.mean(), color="#b23b3b", lw=1.2, ls="--", label=f"mean {diff.mean():+.1%}")
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax2.set_xlabel("Live return minus monthly-rebalance return")
    ax2.set_title(f"Value of reacting to news: live wins {(diff > 0).mean():.0%} of sessions")
    ax2.legend(frameon=False)
    _save(fig, path)
