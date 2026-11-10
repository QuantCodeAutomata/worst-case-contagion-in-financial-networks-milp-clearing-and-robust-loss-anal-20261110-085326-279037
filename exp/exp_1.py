"""exp_1: Random Pointwise Clearing Validation and the Role of the
Insolvency-Resilience Threshold (paper Sec. 7.2, Table 2).

Validates the clearing LP (Eq. 8) against an independent exact clearing
oracle (exhaustive regime enumeration, 3^n states) on the D-val instance
(n=8, m=3, ER(0.35), seed 11), at stress radii epsilon/epsilon_ub in
{0.5, 1.0, 1.2, 2.0}, and reports the fraction of LP-feasible shocks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.generate_synthetic import build_network
from src import config
from src.clearing import (
    build_regime_table,
    clearing_lp,
    default_resilience_margin,
    exact_clearing_regime_enumeration,
    insolvency_resilience_threshold,
    insolvency_resilience_threshold_lower_bound,
)
from src.plotting import plot_feasibility_bars

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def sample_l1_boundary_shock(m: int, eps: float, rng: np.random.Generator) -> np.ndarray:
    """delta = eps * g / ||g||_1 with g ~ N(0, I_m) (paper Sec. 7.2 step 5):
    a uniform-direction sample on the boundary of the l1 ball of radius eps.
    """
    g = rng.normal(size=m)
    return eps * g / np.sum(np.abs(g))


def run(seed: int = config.EXP1_SEED, n_shocks: int = config.EXP1_N_SHOCKS,
        radius_ratios=config.EXP1_RADIUS_RATIOS) -> dict:
    inst = build_network(config.EXP1_N, config.EXP1_M, config.EXP1_GRAPH_TYPE,
                          config.EXP1_DENSITY, seed)
    p_bar, a, c_bar, s_matrix = inst["p_bar"], inst["A"], inst["c_bar"], inst["S"]
    total_p_bar = float(np.sum(p_bar))

    eps_star = default_resilience_margin(p_bar, a, c_bar, s_matrix)
    eps_ub, eps_ub_info = insolvency_resilience_threshold(p_bar, a, c_bar, s_matrix)
    eps_ub_lower = insolvency_resilience_threshold_lower_bound(p_bar, a, c_bar, s_matrix)

    rng = np.random.default_rng(seed)
    regime_table = build_regime_table(p_bar, a)  # shared 3^n affine maps (paper Sec. 7.2)
    table = []
    for ratio in radius_ratios:
        eps = ratio * eps_ub
        lp_losses, exact_losses, insolvent_counts, lp_feasible_flags = [], [], [], []
        mismatches = []
        for _ in range(n_shocks):
            delta = sample_l1_boundary_shock(config.EXP1_M, eps, rng)
            c = c_bar + s_matrix @ delta

            p_lp, lp_feasible = clearing_lp(p_bar, a, c)
            lp_feasible_flags.append(lp_feasible)
            if lp_feasible:
                lp_loss = float(np.sum(p_bar - p_lp))
                lp_losses.append(lp_loss)

            exact = exact_clearing_regime_enumeration(p_bar, a, c, regime_table=regime_table)
            exact_losses.append(exact.loss)
            resources = c + a.T @ exact.p
            insolvent_counts.append(int(np.sum(resources < -1e-9)))

            if lp_feasible and abs(lp_loss - exact.loss) > 1e-5 * max(1.0, total_p_bar):
                mismatches.append({"delta": delta.tolist(), "lp_loss": lp_loss,
                                    "exact_loss": exact.loss})

        table.append({
            "radius_ratio": ratio,
            "epsilon": eps,
            "avg_lp_loss_normalized": float(np.mean(lp_losses) / total_p_bar) if lp_losses else None,
            "avg_exact_loss_normalized": float(np.mean(exact_losses) / total_p_bar),
            "avg_insolvent_banks": float(np.mean(insolvent_counts)),
            "lp_feasible_fraction": float(np.mean(lp_feasible_flags)),
            "n_mismatches": len(mismatches),
        })

    results = {
        "instance": "D-val",
        "n": config.EXP1_N, "m": config.EXP1_M,
        "graph_type": config.EXP1_GRAPH_TYPE, "density": config.EXP1_DENSITY, "seed": seed,
        "epsilon_star": eps_star,
        "epsilon_ub": eps_ub,
        "epsilon_ub_lower": eps_ub_lower,
        "epsilon_ub_lower_vs_exact_pct": float(100 * (eps_ub_lower - eps_ub) / eps_ub),
        "table_2": table,
    }
    return results


def main():
    results = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp1_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    print(json.dumps(results, indent=2))

    plot_feasibility_bars(
        radius_ratios=[row["radius_ratio"] for row in results["table_2"]],
        lp_feasible_fraction=[row["lp_feasible_fraction"] for row in results["table_2"]],
        avg_insolvent_banks=[row["avg_insolvent_banks"] for row in results["table_2"]],
        out_path=RESULTS_DIR / "exp1_feasibility.png",
        title="D-val: LP feasibility and insolvent banks vs. stress radius (Table 2)",
    )
    print(f"Saved {RESULTS_DIR / 'exp1_feasibility.png'}")


if __name__ == "__main__":
    main()
