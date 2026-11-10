"""exp_5: Scenario Lower Bounds and Localized Certificates on a 353-Bank
Network (paper Sec. 7.6, Tables 10-11).

On the L-353 instance (n=353, m=5, CP(0.02), seed 353), compute deterministic
scenario lower bounds LB_Delta(eps) (Sec. 6) via the monotone fixed-point
clearing iteration (Sec. 7.6), track the number of distinct realized payment
regimes as the stress radius grows, and certify three post-insolvency radii
(1.5, 2.0, 3.0 times epsilon_ub) with localized fixed-pattern upper bounds
(Proposition 13).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from data.generate_synthetic import build_network
from src import config
from src.certificates import adverse_simplex_vertices, localized_certificate, scenario_lower_bound
from src.clearing import (
    default_resilience_margin,
    insolvency_resilience_threshold,
    insolvency_resilience_threshold_lower_bound,
)
from src.robust_loss import robust_dual_lp

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def heuristic_dual_direction(p_bar, a, c_bar, s_matrix, eps_ub: float) -> int:
    """Heuristic shock direction from the pre-insolvency dual problem (Eq. 43):
    solve the robust dual LP at radius epsilon_ub for every asset and return
    the index of the asset attaining the largest affine piece. For S >= 0 and
    the l1 ball, this is the single-asset direction the pre-insolvency
    envelope would pick as the radius approaches epsilon_ub (paper Sec. 7.6,
    step 3(d); the exact composition of this heuristic is not detailed in the
    paper beyond "a heuristic direction obtained from the pre-insolvency dual
    problem" -- documented assumption).
    """
    m = s_matrix.shape[1]
    best_val, best_j = -np.inf, 0
    for j in range(m):
        piece = robust_dual_lp(p_bar, a, c_bar, s_matrix, eps_ub, j)
        if piece is None:
            continue
        val = piece.gamma + piece.mu * eps_ub
        if val > best_val:
            best_val, best_j = val, j
    return best_j


def build_scenario_set(m: int, eps: float, n_total: int, heuristic_asset: int,
                        rng: np.random.Generator) -> list[np.ndarray]:
    """Scenario set Delta for radius ``eps`` (paper Sec. 7.6, step 3):
    (a) the m single-asset price-decline directions -eps*e_j,
    (c) the zero shock,
    (d) the heuristic direction from the pre-insolvency dual problem
        (already one of the m directions above when it coincides; kept as a
        separate scenario for traceability), and
    (b) random boundary scenarios filling the set up to ``n_total`` (Dirichlet
        draws on the adverse l1 simplex; concentration not specified in the
        paper -- documented assumption, same construction as exp_4).
    """
    vertices = adverse_simplex_vertices(m, eps)  # [0, -eps e_1, ..., -eps e_m]
    scenarios = [vertices[0]]  # (c) zero shock
    scenarios.extend(vertices[1:])  # (a) single-asset directions
    scenarios.append(vertices[1 + heuristic_asset].copy())  # (d) heuristic direction

    vertex_matrix = np.array(vertices)
    n_remaining = max(0, n_total - len(scenarios))
    if n_remaining > 0:
        alpha = np.full(len(vertices), 1.0)
        weights = rng.dirichlet(alpha, size=n_remaining)  # (b) random boundary scenarios
        scenarios.extend(weights @ vertex_matrix)
    return scenarios[:n_total]


def run(radius_ratios=config.EXP5_RADIUS_RATIOS,
        localized_radii=config.EXP5_LOCALIZED_RADII) -> dict:
    inst = build_network(config.EXP5_N, config.EXP5_M, config.EXP5_GRAPH_TYPE,
                          config.EXP5_DENSITY, config.EXP5_SEED)
    p_bar, a, c_bar, s_matrix = inst["p_bar"], inst["A"], inst["c_bar"], inst["S"]
    total_p_bar = float(np.sum(p_bar))

    eps_star = default_resilience_margin(p_bar, a, c_bar, s_matrix)
    eps_ub, _ = insolvency_resilience_threshold(p_bar, a, c_bar, s_matrix)
    eps_ub_lower = insolvency_resilience_threshold_lower_bound(p_bar, a, c_bar, s_matrix)

    heuristic_asset = heuristic_dual_direction(p_bar, a, c_bar, s_matrix, eps_ub)
    rng = np.random.default_rng(config.EXP5_SEED)

    table_10 = []
    for ratio in radius_ratios:
        eps = ratio * eps_ub
        scenarios = build_scenario_set(config.EXP5_M, eps, config.EXP5_N_SCENARIOS_RUN,
                                        heuristic_asset, rng)
        t0 = time.perf_counter()
        lb_result = scenario_lower_bound(p_bar, a, c_bar, s_matrix, scenarios,
                                          clearing_method="fixed_point")
        wall_time = time.perf_counter() - t0

        table_10.append({
            "radius_ratio": ratio, "epsilon": eps,
            "lb": lb_result["lb"] / total_p_bar,
            "n_scenarios": lb_result["n_scenarios"],
            "n_patterns": lb_result["n_patterns"],
            "lp_feasible_at_best": lb_result["lp_feasible_at_best"],
            "median_time_sec": lb_result["median_time"],
            "wall_time_sec": wall_time,
        })

    table_11 = []
    for ratio in localized_radii:
        eps = ratio * eps_ub
        scenarios = build_scenario_set(config.EXP5_M, eps, config.EXP5_N_SCENARIOS_RUN,
                                        heuristic_asset, rng)
        lb_result = scenario_lower_bound(p_bar, a, c_bar, s_matrix, scenarios,
                                          clearing_method="fixed_point")
        lb = lb_result["lb"] / total_p_bar

        loc = localized_certificate(p_bar, a, c_bar, s_matrix, eps,
                                     max_cells=config.EXP5_MAX_CELLS_RUN,
                                     clearing_method="fixed_point")
        localized_ub = loc["localized_ub"] / total_p_bar
        table_11.append({
            "radius_ratio": ratio, "lb": lb, "localized_ub": localized_ub,
            "gap_vs_lb": float(localized_ub - lb),
            "n_cells": loc["n_cells"],
            "n_local_clearings": loc["n_local_clearings"],
            "n_pattern_lps": loc["n_pattern_lps"],
            "wall_time_sec": loc["wall_time"],
        })

    results = {
        "instance": "L-353",
        "n": config.EXP5_N, "m": config.EXP5_M,
        "graph_type": config.EXP5_GRAPH_TYPE, "density": config.EXP5_DENSITY,
        "seed": config.EXP5_SEED,
        "epsilon_star": eps_star, "epsilon_ub": eps_ub, "epsilon_ub_lower": eps_ub_lower,
        "epsilon_ub_lower_vs_exact_pct": float(100 * (eps_ub_lower - eps_ub) / eps_ub),
        "heuristic_dual_asset": int(heuristic_asset),
        "n_scenarios": config.EXP5_N_SCENARIOS_RUN,
        "fixed_point_tol": config.EXP5_FIXED_POINT_TOL,
        "table_10_scenario_lower_bounds": table_10,
        "table_11_localized_certificates": table_11,
    }
    return results


def main():
    results = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp5_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
