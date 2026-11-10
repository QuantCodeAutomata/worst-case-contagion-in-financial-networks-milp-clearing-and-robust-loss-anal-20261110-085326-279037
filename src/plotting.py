"""Matplotlib visualizations for the worst-case contagion experiments.

These plots render the quantities defined in the paper (the robust
worst-case loss curve eta_wc(epsilon) of Sec. 5-6, and the scenario
lower bound / fixed-pattern upper bound pair of Sec. 6) from the result
dictionaries produced by exp/exp_1.py .. exp/exp_5.py. No experiment
logic lives here; this module only visualizes already-computed numbers.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(
    eps_grid: list[float],
    losses: list[float],
    eps_ub: float,
    out_path: Path,
    title: str,
    eps_star: float | None = None,
    breakpoints: list[dict] | None = None,
    xlabel: str = "epsilon",
) -> None:
    """Plot a traced worst-case loss curve eta_wc(epsilon) against epsilon,
    with vertical reference lines at epsilon* and epsilon_ub (Theorem 14 /
    Theorem 22) and markers at reported salient points / kinks.
    """
    order = np.argsort(eps_grid)
    x = np.asarray(eps_grid)[order]
    y = np.asarray(losses)[order]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, color="tab:blue", linewidth=1.8, label=r"$\eta_{wc}(\epsilon)$")
    if eps_star is not None:
        ax.axvline(eps_star, color="tab:green", linestyle=":", label=r"$\epsilon^*$")
    ax.axvline(eps_ub, color="tab:red", linestyle="--", label=r"$\epsilon_{ub}$")
    if breakpoints:
        bx = [bp["epsilon"] for bp in breakpoints]
        by = np.interp(bx, x, y)
        ax.scatter(bx, by, color="black", zorder=5, marker="x", label="breakpoint / kink")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"normalized loss $\eta_{wc}(\epsilon)\,/\,\bar{p}^\top\mathbf{1}$")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bounds_vs_radius(
    radius_ratios: list[float],
    lower_bounds: list[float],
    upper_bounds: list[float],
    out_path: Path,
    title: str,
    localized_upper_bounds: list[float] | None = None,
) -> None:
    """Plot the scenario lower bound LB_Delta(epsilon) against the fixed-pattern
    upper bound UB_Z(epsilon) (and, where available, the localized certificate)
    over the stress radius ratio epsilon/epsilon_ub, as in Tables 6-11.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(radius_ratios, lower_bounds, "o-", color="tab:blue", label="scenario LB")
    ax.plot(radius_ratios, upper_bounds, "s--", color="tab:red", label="fixed-pattern UB (global)")
    if localized_upper_bounds is not None:
        valid = [(r, u) for r, u in zip(radius_ratios, localized_upper_bounds) if u is not None]
        if valid:
            rx, ry = zip(*valid)
            ax.plot(rx, ry, "^-.", color="tab:green", label="localized UB")
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1, label=r"$\epsilon_{ub}$")
    ax.set_xlabel(r"stress radius ratio $\epsilon/\epsilon_{ub}$")
    ax.set_ylabel(r"normalized worst-case loss")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feasibility_bars(
    radius_ratios: list[float],
    lp_feasible_fraction: list[float],
    avg_insolvent_banks: list[float],
    out_path: Path,
    title: str,
) -> None:
    """Bar chart of the LP-feasible fraction and average number of externally
    insolvent banks per stress radius ratio (Table 2).
    """
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(radius_ratios))
    width = 0.35
    ax1.bar(x - width / 2, lp_feasible_fraction, width, color="tab:blue", label="LP-feasible fraction")
    ax1.set_ylabel("LP-feasible fraction", color="tab:blue")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{r:g}" for r in radius_ratios])
    ax1.set_xlabel(r"stress radius ratio $\epsilon/\epsilon_{ub}$")

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, avg_insolvent_banks, width, color="tab:orange", label="avg insolvent banks")
    ax2.set_ylabel("avg. externally insolvent banks", color="tab:orange")

    fig.suptitle(title)
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
