"""exp_2: Pre-Insolvency Worst-Case Loss Curve: Piecewise Linearity and
Breakpoint Diagnostics (paper Sec. 7.3, Tables 3-4).

Traces the exact robust worst-case loss curve eta_wc(epsilon) on [0,
epsilon_ub] for the D-pre instance (n=12, m=4, CP(0.28), seed 22), verifies
convexity/monotonicity/piecewise linearity (Theorem 14), diagnoses each
breakpoint's mechanism (Theorem 18: outer shock-direction switch vs. inner
default-set switch, Sec. 5.3), and runs a growth experiment measuring how the
number of affine pieces and salient points scales with network size n.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from data.generate_synthetic import build_network
from src import config
from src.clearing import (
    default_resilience_margin,
    insolvency_resilience_threshold,
    insolvency_resilience_threshold_lower_bound,
)
from src.robust_loss import breakpoint_mechanism, eta_wc_point, trace_loss_curve

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _curve_properties(segments: list) -> dict:
    """Checks the properties of Theorem 14: nondecreasing, convex (slopes mu_k
    nondecreasing with epsilon), piecewise linear on the traced segments."""
    slopes = [piece.mu for _, _, piece in segments]
    nondecreasing_slopes = all(slopes[k] <= slopes[k + 1] + 1e-8 for k in range(len(slopes) - 1))
    values = []
    for lo, hi, piece in segments:
        values.append(piece.gamma + piece.mu * lo)
        values.append(piece.gamma + piece.mu * hi)
    is_nondecreasing = all(values[k] <= values[k + 1] + 1e-7 for k in range(len(values) - 1))
    return {"convex_slopes_nondecreasing": bool(nondecreasing_slopes),
            "curve_nondecreasing": bool(is_nondecreasing)}


def run_main_instance() -> dict:
    inst = build_network(config.EXP2_N, config.EXP2_M, config.EXP2_GRAPH_TYPE,
                          config.EXP2_DENSITY, config.EXP2_SEED)
    p_bar, a, c_bar, s_matrix = inst["p_bar"], inst["A"], inst["c_bar"], inst["S"]
    total_p_bar = float(np.sum(p_bar))

    eps_star = default_resilience_margin(p_bar, a, c_bar, s_matrix)
    eps_ub, _ = insolvency_resilience_threshold(p_bar, a, c_bar, s_matrix)
    eps_ub_lower = insolvency_resilience_threshold_lower_bound(p_bar, a, c_bar, s_matrix)

    trace = trace_loss_curve(p_bar, a, c_bar, s_matrix, eps_ub, tol=config.BREAKPOINT_TRACE_TOL)
    properties = _curve_properties(trace["segments"])

    breakpoints = []
    for bp in trace["breakpoints"]:
        if not (eps_star < bp["epsilon"] < eps_ub):
            continue  # paper restricts breakpoints to the open interval (epsilon*, epsilon_ub)
        mech = breakpoint_mechanism(p_bar, a, c_bar, s_matrix, bp)
        breakpoints.append({
            "epsilon": bp["epsilon"],
            "radius_ratio": bp["epsilon"] / eps_ub,
            "left_slope": bp["left_slope"] / total_p_bar,
            "right_slope": bp["right_slope"] / total_p_bar,
            "mechanism": mech["mechanism"],
            "banks_changed": mech["banks_changed"],
        })

    # Validation gap: resolve the robust LP exactly at each segment endpoint
    # and compare with the traced envelope value (paper Sec. 7.3).
    max_gap = 0.0
    for lo, hi, piece in trace["segments"]:
        for eps in (lo, hi):
            val, _ = eta_wc_point(p_bar, a, c_bar, s_matrix, eps)
            predicted = piece.gamma + piece.mu * eps
            max_gap = max(max_gap, abs(val - predicted))

    no_breakpoint_beyond_lower_ub = all(bp["epsilon"] <= eps_ub_lower for bp in breakpoints) \
        if eps_ub_lower < eps_ub else True

    return {
        "instance": "D-pre",
        "n": config.EXP2_N, "m": config.EXP2_M,
        "graph_type": config.EXP2_GRAPH_TYPE, "density": config.EXP2_DENSITY,
        "seed": config.EXP2_SEED,
        "epsilon_star": eps_star,
        "epsilon_ub": eps_ub,
        "epsilon_ub_lower": eps_ub_lower,
        "epsilon_ub_lower_vs_exact_pct": float(100 * (eps_ub_lower - eps_ub) / eps_ub),
        "curve_properties": properties,
        "n_salient_points": len(breakpoints),
        "breakpoints": breakpoints,
        "no_breakpoint_beyond_eps_ub_lower": bool(no_breakpoint_beyond_lower_ub),
        "max_validation_gap": max_gap,
        "n_affine_pieces": len(trace["segments"]),
    }


def run_growth_experiment(sizes=config.EXP2_GROWTH_SIZES_RUN,
                           seeds=config.EXP2_GROWTH_SEEDS) -> list:
    """Growth experiment (paper Table 4): for each network size n, build CP
    graphs with m=4 over two random seeds, trace the pre-insolvency loss
    curve, and count affine pieces / salient points on (epsilon*, epsilon_ub).
    """
    rows = []
    for n in sizes:
        # Hold the expected mean in-degree constant across n (see src/config.py).
        density = min(1.0, config.EXP2_GROWTH_DENSITY_ANCHOR_D
                      * config.EXP2_GROWTH_DENSITY_ANCHOR_N / n)
        pieces_counts, salient_counts, times, gaps = [], [], [], []
        for seed in seeds:
            inst = build_network(n, config.EXP2_GROWTH_M, "CP", density, seed)
            p_bar, a, c_bar, s_matrix = inst["p_bar"], inst["A"], inst["c_bar"], inst["S"]
            eps_star = default_resilience_margin(p_bar, a, c_bar, s_matrix)
            eps_ub, _ = insolvency_resilience_threshold(p_bar, a, c_bar, s_matrix)
            t0 = time.perf_counter()
            trace = trace_loss_curve(p_bar, a, c_bar, s_matrix, eps_ub,
                                      tol=config.BREAKPOINT_TRACE_TOL)
            elapsed = time.perf_counter() - t0

            n_salient = sum(1 for bp in trace["breakpoints"] if eps_star < bp["epsilon"] < eps_ub)
            max_gap = 0.0
            for lo, hi, piece in trace["segments"]:
                for eps in (lo, hi):
                    val, _ = eta_wc_point(p_bar, a, c_bar, s_matrix, eps)
                    max_gap = max(max_gap, abs(val - (piece.gamma + piece.mu * eps)))

            pieces_counts.append(len(trace["segments"]))
            salient_counts.append(n_salient)
            times.append(elapsed)
            gaps.append(max_gap)

        rows.append({
            "n": n,
            "mean_pieces": float(np.mean(pieces_counts)),
            "mean_salient_points": float(np.mean(salient_counts)),
            "mean_time_sec": float(np.mean(times)),
            "max_validation_gap": float(np.max(gaps)),
        })
    return rows


def run() -> dict:
    main_instance = run_main_instance()
    growth = run_growth_experiment()
    return {"table_3": main_instance, "table_4": growth}


def main():
    results = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp2_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
