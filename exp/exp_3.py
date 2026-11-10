"""exp_3: Post-Insolvency Scalar Geometry: Exact Regime Enumeration on a
Single-Asset Instance (paper Sec. 7.4, Table 5).

For the fixed D-post instance (n=6, m=1, long-only exposure S >= 0), the
adverse l1 shock is simply a price decline delta = -epsilon, so
c(epsilon) = c_bar - epsilon*S. The exact maximal clearing vector at each
epsilon is obtained by exhaustive enumeration of all 2^6 = 64 zero/active
payment patterns z (the same pattern decomposition used by the fixed-pattern
machinery of Sec. 6, here applied scalar-wise with no uncertainty set): for
each z the clearing reduces to the regime-restricted LP
    max 1^T p  s.t.  0 <= p <= Z p_bar,  Z p <= Z (c + A^T p),
and the maximal clearing vector is the feasible pattern with minimal loss
(Theorem 1 pattern decomposition, Sec. 3). The loss curve is then traced on a
fine grid of epsilon in [0, ~4*epsilon_ub], kinks (slope changes) are located,
and at each kink the left/right zero-payment bank sets are recorded. No-jump
is verified by evaluating the curve at epsilon_k +/- 1e-6 around each kink.
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np

from data.generate_synthetic import build_d_post
from src import config
from src.certificates import fixed_pattern_loss
from src.clearing import default_resilience_margin, insolvency_resilience_threshold
from src.plotting import plot_loss_curve

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def exact_clearing_loss_and_zero_set(p_bar: np.ndarray, a: np.ndarray, c: np.ndarray):
    """Exact maximal clearing loss and zero-payment bank set via exhaustive
    enumeration of all 2^n patterns z in {0,1}^n (paper Sec. 7.4)."""
    n = len(p_bar)
    best_loss, best_z = np.inf, None
    for combo in product((0, 1), repeat=n):
        z = np.array(combo)
        loss = fixed_pattern_loss(p_bar, a, c, z)
        if loss is None:
            continue
        if loss < best_loss:
            best_loss, best_z = loss, z
    assert best_z is not None, "no feasible payment pattern found"
    zero_set = sorted((np.where(best_z == 0)[0] + 1).tolist())  # 1-indexed banks
    return float(best_loss), zero_set


def trace_grid(p_bar, a, c_bar, s_vector, eps_grid: np.ndarray, total_p_bar: float):
    losses, zero_sets = [], []
    for eps in eps_grid:
        c = c_bar - eps * s_vector
        loss, zero_set = exact_clearing_loss_and_zero_set(p_bar, a, c)
        losses.append(loss / total_p_bar)
        zero_sets.append(zero_set)
    return np.array(losses), zero_sets


def locate_exact_kinks(p_bar, a, c_bar, s_vector, eps_grid: np.ndarray,
                        zero_sets: list, total_p_bar: float, eps_min_report: float,
                        probe: float = config.EXP3_KINK_PROBE,
                        merge_tol: float = 1e-6) -> list:
    """Exact regime-switch kink location (paper Sec. 7.4, steps 6-7).

    A kink of the piecewise-linear loss curve coincides exactly with a change
    of the active zero-payment bank pattern z (Theorem 1 pattern
    decomposition), because within one fixed pattern the regime-restricted LP
    of ``exact_clearing_loss_and_zero_set`` reduces to a linear system that is
    affine in epsilon. Consecutive grid points whose exact zero-payment set
    differs therefore bracket a true switch, which is localized exactly by
    bisection rather than read off at the (finite) grid resolution -- this
    removes the duplicate detections a naive finite-difference slope test
    produces when a single switch falls inside one grid cell. Only switches
    at or beyond ``eps_min_report`` are kept, matching research objective 3
    ("post-insolvency kinks"): Table 5 of the paper reports no breakpoint
    below epsilon_ub.
    """
    def pattern_at(eps: float) -> tuple:
        c = c_bar - eps * s_vector
        _, zs = exact_clearing_loss_and_zero_set(p_bar, a, c)
        return tuple(zs)

    located = []
    for k in range(len(eps_grid) - 1):
        if tuple(zero_sets[k]) == tuple(zero_sets[k + 1]):
            continue
        lo, hi, pat_lo = eps_grid[k], eps_grid[k + 1], tuple(zero_sets[k])
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if pattern_at(mid) == pat_lo:
                lo = mid
            else:
                hi = mid
        located.append(0.5 * (lo + hi))

    kinks = []
    for eps_k in located:
        if kinks and abs(eps_k - kinks[-1]) < merge_tol:
            continue
        if eps_k < eps_min_report:
            continue
        kinks.append(eps_k)

    table = []
    for eps_k in kinks:
        left_loss, zs_left = exact_clearing_loss_and_zero_set(p_bar, a, c_bar - (eps_k - probe) * s_vector)
        right_loss, zs_right = exact_clearing_loss_and_zero_set(p_bar, a, c_bar - (eps_k + probe) * s_vector)
        at_loss, _ = exact_clearing_loss_and_zero_set(p_bar, a, c_bar - eps_k * s_vector)
        left_slope = (at_loss - left_loss) / probe / total_p_bar
        right_slope = (right_loss - at_loss) / probe / total_p_bar
        table.append({
            "epsilon": float(eps_k),
            "left_slope": float(left_slope),
            "right_slope": float(right_slope),
            "zero_set_left": zs_left,
            "zero_set_right": zs_right,
        })
    return table


def verify_no_jumps(p_bar, a, c_bar, s_vector, kinks: list, total_p_bar: float,
                     probe: float = config.EXP3_KINK_PROBE) -> list:
    """At each detected kink epsilon_k, evaluate the exact loss at
    epsilon_k - probe and epsilon_k + probe and confirm continuity
    (paper Sec. 7.4, step 9)."""
    checks = []
    for kink in kinks:
        eps0 = kink["epsilon"]
        left_loss, _ = exact_clearing_loss_and_zero_set(p_bar, a, c_bar - (eps0 - probe) * s_vector)
        right_loss, _ = exact_clearing_loss_and_zero_set(p_bar, a, c_bar - (eps0 + probe) * s_vector)
        at_loss, _ = exact_clearing_loss_and_zero_set(p_bar, a, c_bar - eps0 * s_vector)
        jump = max(abs(at_loss - left_loss), abs(at_loss - right_loss)) / total_p_bar
        checks.append({"epsilon": eps0, "jump_size": jump, "is_continuous": bool(jump < 1e-3)})
    return checks


def run() -> dict:
    inst = build_d_post()
    p_bar, a, c_bar, s_matrix = inst["p_bar"], inst["A"], inst["c_bar"], inst["S"]
    s_vector = s_matrix[:, 0]
    total_p_bar = float(np.sum(p_bar))

    eps_star = default_resilience_margin(p_bar, a, c_bar, s_matrix)
    eps_ub, _ = insolvency_resilience_threshold(p_bar, a, c_bar, s_matrix)

    eps_max = config.EXP3_RADIUS_MULTIPLE * eps_ub
    eps_grid = np.linspace(0.0, eps_max, config.EXP3_GRID_POINTS)

    losses, zero_sets = trace_grid(p_bar, a, c_bar, s_vector, eps_grid, total_p_bar)
    # Research objective 3 / Table 5 of the paper report only post-insolvency
    # kinks (epsilon >= epsilon_ub); the pre-insolvency envelope (epsilon* to
    # epsilon_ub) is the subject of experiment 2 (D-pre), not of this one.
    kinks = locate_exact_kinks(p_bar, a, c_bar, s_vector, eps_grid, zero_sets,
                                total_p_bar, eps_min_report=eps_ub)
    continuity = verify_no_jumps(p_bar, a, c_bar, s_vector, kinks, total_p_bar)

    # Theorem 22: beyond epsilon_ub the curve is nondecreasing and piecewise
    # linear but need not be convex. Check both properties on the traced grid.
    post_mask = eps_grid > eps_ub
    post_losses = losses[post_mask]
    post_eps = eps_grid[post_mask]
    nondecreasing = bool(np.all(np.diff(post_losses) >= -1e-9))
    post_slopes = np.diff(post_losses) / np.diff(post_eps)
    convex_post = bool(np.all(np.diff(post_slopes) >= -1e-6))

    pre_mask = eps_grid <= eps_ub
    pre_losses = losses[pre_mask]
    pre_eps = eps_grid[pre_mask]
    pre_slopes = np.diff(pre_losses) / np.diff(pre_eps)
    convex_pre = bool(np.all(np.diff(pre_slopes) >= -1e-6))

    kink_table = [{
        "radius_ratio": k["epsilon"] / eps_ub,
        "epsilon": k["epsilon"],
        "left_slope": k["left_slope"],
        "right_slope": k["right_slope"],
        "zero_set_left": k["zero_set_left"],
        "zero_set_right": k["zero_set_right"],
    } for k in kinks]
    # Theorem 22 only requires each piece's slope to be nonnegative (loss
    # nondecreasing); it does not require slopes to decrease at every kink,
    # which is exactly how the curve can fail to be convex post-insolvency.
    assert all(k["left_slope"] >= -1e-6 and k["right_slope"] >= -1e-6 for k in kink_table), \
        "post-insolvency slopes must stay nonnegative (Theorem 22: nondecreasing loss curve)"

    return {
        "instance": "D-post",
        "n": config.EXP3_N, "m": config.EXP3_M, "seed": config.EXP3_SEED,
        "epsilon_star": eps_star,
        "epsilon_ub": eps_ub,
        "epsilon_ub_lower": eps_ub,  # coincide exactly since m=1 (paper Sec. 7.4)
        "loss_at_zero": float(losses[0]),
        "loss_at_eps_max": float(losses[-1]),
        "pre_insolvency_convex": convex_pre,
        "pre_insolvency_nondecreasing": bool(np.all(np.diff(pre_losses) >= -1e-9)),
        "post_insolvency_nondecreasing": nondecreasing,
        "post_insolvency_convex": convex_post,
        "table_5_kinks": kink_table,
        "no_jump_checks": continuity,
        "n_kinks": len(kinks),
        "curve_eps_grid": eps_grid.tolist(),
        "curve_losses": losses.tolist(),
    }


def main():
    results = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp3_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    print(json.dumps({k: v for k, v in results.items()
                       if k not in ("curve_eps_grid", "curve_losses")}, indent=2))

    plot_loss_curve(
        eps_grid=results["curve_eps_grid"],
        losses=results["curve_losses"],
        eps_ub=results["epsilon_ub"],
        eps_star=results["epsilon_star"],
        breakpoints=[{"epsilon": k["epsilon"]} for k in results["table_5_kinks"]],
        out_path=RESULTS_DIR / "exp3_loss_curve.png",
        title="D-post: post-insolvency worst-case loss curve eta_wc(epsilon) (Theorem 22)",
    )
    print(f"Saved {RESULTS_DIR / 'exp3_loss_curve.png'}")


if __name__ == "__main__":
    main()
