"""exp_4: Certified Bounds and Regime Generation for Post-Insolvency
Worst-Case Losses (paper Sec. 7.5, Tables 6-9).

On the M-bound instance (n=25, m=5, CP(0.12), seed 5), compute scenario-based
lower bounds LB_Delta(eps) (Sec. 6), fixed-pattern dual upper bounds UB_z(eps)
(Proposition 10) refined by the regime-generation loop (Proposition 11), and
localized fixed-pattern certificates via adaptive bisection of the adverse l1
simplex (Proposition 13), at radii epsilon/epsilon_ub in
{0.8, 1.0, 1.2, 1.5, 2.0, 3.0}.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from data.generate_synthetic import build_network
from src import config
from src.certificates import (
    adverse_simplex_vertices,
    global_upper_bound,
    greedy_fixed_regime_search,
    localized_certificate,
    regime_generation,
    scenario_lower_bound,
)
from src.clearing import (
    default_resilience_margin,
    insolvency_resilience_threshold,
    insolvency_resilience_threshold_lower_bound,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_scenario_set(m: int, eps: float, n_total: int, concentrations,
                        rng: np.random.Generator) -> list[np.ndarray]:
    """Scenario set Delta for the radius ``eps`` (paper Sec. 7.5, step 3):
    (a) the m single-asset price-decline directions -eps*e_j,
    (b) the pairwise edge midpoints of the adverse l1 simplex
        conv{0, -eps*e_1, ..., -eps*e_m},
    (c) Dirichlet boundary samples of that simplex at several concentration
        parameters (not individually specified by the paper; documented
        assumption), filling the set up to ``n_total`` scenarios, and
    (d) the zero shock (used for regime identification).
    """
    vertices = adverse_simplex_vertices(m, eps)  # [0, -eps e_1, ..., -eps e_m]
    scenarios = [vertices[0]]  # (d) zero shock
    scenarios.extend(vertices[1:])  # (a) single-asset directions
    for i in range(len(vertices)):  # (b) pairwise edge midpoints
        for j in range(i + 1, len(vertices)):
            scenarios.append(0.5 * (vertices[i] + vertices[j]))

    vertex_matrix = np.array(vertices)
    n_remaining = max(0, n_total - len(scenarios))
    n_each = n_remaining // len(concentrations)
    for k, conc in enumerate(concentrations):
        n_draw = n_each if k < len(concentrations) - 1 else n_remaining - n_each * (len(concentrations) - 1)
        if n_draw <= 0:
            continue
        alpha = np.full(len(vertices), conc)
        weights = rng.dirichlet(alpha, size=n_draw)  # (c) Dirichlet boundary samples
        scenarios.extend(weights @ vertex_matrix)
    return scenarios[:n_total]


def run(radius_ratios=config.EXP4_RADIUS_RATIOS) -> dict:
    inst = build_network(config.EXP4_N, config.EXP4_M, config.EXP4_GRAPH_TYPE,
                          config.EXP4_DENSITY, config.EXP4_SEED)
    p_bar, a, c_bar, s_matrix = inst["p_bar"], inst["A"], inst["c_bar"], inst["S"]
    total_p_bar = float(np.sum(p_bar))

    eps_star = default_resilience_margin(p_bar, a, c_bar, s_matrix)
    eps_ub, _ = insolvency_resilience_threshold(p_bar, a, c_bar, s_matrix)
    eps_ub_lower = insolvency_resilience_threshold_lower_bound(p_bar, a, c_bar, s_matrix)

    rng = np.random.default_rng(config.EXP4_SEED)

    table_6, table_7, table_8, table_9 = [], None, [], []
    for ratio in radius_ratios:
        eps = ratio * eps_ub

        scenarios = build_scenario_set(config.EXP4_M, eps, config.EXP4_N_SCENARIOS_RUN,
                                        config.EXP4_DIRICHLET_CONCENTRATIONS, rng)
        lb_result = scenario_lower_bound(p_bar, a, c_bar, s_matrix, scenarios, clearing_method="milp")
        lb = lb_result["lb"] / total_p_bar

        t0 = time.perf_counter()
        trace = regime_generation(p_bar, a, c_bar, s_matrix, eps,
                                   max_iter=config.EXP4_REGIME_GEN_MAX_ITER)
        regime_time = time.perf_counter() - t0
        candidate_patterns = list(trace.patterns)
        if ratio > 1.0:
            # Beyond epsilon_ub (post-insolvency), the regime-generation loop
            # of Proposition 11 can stall at the trivial all-zero pattern: the
            # single worst-case shock it extracts favors one asset direction,
            # and the realized pattern typically fails the robust-domain
            # condition at the other assets' vertices. The paper's step 5
            # therefore adds a greedy fixed-regime search, activating banks
            # one at a time subject to the robust-domain condition.
            _, _, greedy_patterns = greedy_fixed_regime_search(
                p_bar, a, c_bar, s_matrix, eps)
            candidate_patterns.extend(greedy_patterns)
        ub_global = global_upper_bound(p_bar, a, c_bar, s_matrix, eps, candidate_patterns) / total_p_bar

        loc = localized_certificate(p_bar, a, c_bar, s_matrix, eps,
                                     max_cells=config.EXP4_MAX_CELLS_RUN)
        localized_ub = loc["localized_ub"] / total_p_bar
        enhanced_ub = min(ub_global, localized_ub)
        # Table 6 reports the gap of the global bound UB_Z(eps) only; the
        # localized refinement is reported separately in Table 8 (Sec. 6,
        # Proposition 13). Mixing the two would misreport the monotone
        # closing of the gap that Table 7's regime generation documents.
        gap_global = (ub_global - lb) / ub_global if ub_global > 1e-12 else 0.0

        table_6.append({
            "radius_ratio": ratio, "epsilon": eps,
            "lb": lb, "ub": ub_global, "enhanced_ub": enhanced_ub,
            "relative_gap_pct": float(100 * gap_global),
        })
        closed = bool(np.isfinite(localized_ub))
        table_8.append({
            "radius_ratio": ratio,
            "localized_ub": localized_ub if closed else None,
            "lb": lb,
            "gap_vs_lb": float(localized_ub - lb) if closed else None,
            "n_cells": loc["n_cells"],
            "closed_within_cell_budget": closed,
        })
        table_9.append({
            "radius_ratio": ratio, "n_cells": loc["n_cells"],
            "n_local_clearings": loc["n_local_clearings"],
            "n_pattern_lps": loc["n_pattern_lps"],
            "wall_time_sec": loc["wall_time"], "regime_gen_time_sec": regime_time,
        })
        if abs(ratio - 2.0) < 1e-9:
            table_7 = {
                "radius_ratio": ratio,
                "ub_history_normalized": [float(v / total_p_bar) for v in trace.ub_history],
                "n_candidate_regimes": trace.n_candidate,
                "n_finite_regimes": trace.n_finite,
            }

    results = {
        "instance": "M-bound",
        "n": config.EXP4_N, "m": config.EXP4_M,
        "graph_type": config.EXP4_GRAPH_TYPE, "density": config.EXP4_DENSITY,
        "seed": config.EXP4_SEED,
        "epsilon_star": eps_star, "epsilon_ub": eps_ub, "epsilon_ub_lower": eps_ub_lower,
        "epsilon_ub_lower_vs_exact_pct": float(100 * (eps_ub_lower - eps_ub) / eps_ub),
        "n_scenarios": config.EXP4_N_SCENARIOS_RUN,
        "table_6_global_bounds": table_6,
        "table_7_regime_generation_at_2x": table_7,
        "table_8_localized_refinement": table_8,
        "table_9_computational_effort": table_9,
    }
    return results


def main():
    results = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp4_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
