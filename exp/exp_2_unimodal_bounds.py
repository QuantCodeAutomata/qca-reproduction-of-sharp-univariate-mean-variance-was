"""
Experiment 2: Reproduction of unimodality-constrained distortion risk bounds
and projection algorithms.

Validates Propositions 2, 3, 4 and Theorem 3:
- Analytic projection examples (GD and MMD)
- Step-function approximation and Proposition 4 error bounds
- Unimodality + Wasserstein combined thresholds and worst-case values
"""

from __future__ import annotations
import os
import sys
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reference_distributions import make_gaussian_reference, make_student_t_reference
from src.distortion_metrics import make_gd, make_mmd, make_es, make_gluevar, DistortionMetric
from src.unimodal_projection import (
    project_gamma_unimodal,
    compute_unimodal_worst_case_no_wasserstein,
    compute_unimodal_worst_case_quantile,
    step_approx_gamma,
    compute_step_projection_and_error,
    compute_unimodal_wasserstein_thresholds,
    compute_unimodal_wasserstein_bound,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def validate_gd_projection(xi: float = 0.5, n_grid: int = 1000) -> dict:
    """
    Validate GD projection: gamma_GD(u) = 1-2u. For xi=0.5, the projection
    should remain 1-2u since it is already unimodal (non-decreasing on [0,0.5],
    non-increasing on [0.5,1]).

    Returns dict with analytic and numeric values.
    """
    metric = make_gd()
    u_grid, proj_vals, _ = project_gamma_unimodal(metric.gamma, xi, n_grid)
    gamma_vals = metric.gamma(u_grid)

    # Check: proj should match gamma for GD since 1-2u is already unimodal at xi=0.5
    # GD: gamma(u) = 1-2u, which is strictly decreasing. At xi=0.5,
    # left part [0,0.5]: 1-2u goes from 1 to 0 (non-increasing, not non-decreasing)
    # The unimodal cone requires non-decreasing on [0,xi] then non-increasing on [xi,1]
    # So GD gamma is NOT in the cone naturally; its projection differs
    # For MMD: gamma(u) = -1 on (0,0.5), +1 on (0.5,1)
    # This IS unimodal (non-decr, then non-incr)? No: -1,-1,...,1,1 is increasing then flat
    # The projection of MMD should be 3*(u-0.5) according to the paper

    max_abs_diff = float(np.max(np.abs(proj_vals - gamma_vals)))
    proj_l2 = float(np.sqrt(np.mean((proj_vals - gamma_vals)**2)))

    wc, _, proj_v, a_xi, b_xi = compute_unimodal_worst_case_no_wasserstein(
        metric, xi, mu=0.0, sigma=1.0, n_grid=n_grid
    )

    return {
        "metric": "GD",
        "xi": xi,
        "max_diff_to_original_gamma": max_abs_diff,
        "proj_l2_vs_gamma": proj_l2,
        "a_xi": float(a_xi),
        "b_xi": float(b_xi),
        "worst_case_value": float(wc),
        "formula_value": float(0.0 * metric.g_val_at_1 + 1.0 * b_xi),
    }


def validate_mmd_projection(xi: float = 0.5, n_grid: int = 2000) -> dict:
    """
    Validate MMD projection: gamma_MMD(u) = -1 on (0,0.5), +1 on (0.5,1).
    Paper claims projection at xi=0.5 is 3*(u-0.5).

    Returns dict with validation results.
    """
    metric = make_mmd()
    u_grid, proj_vals, _ = project_gamma_unimodal(metric.gamma, xi, n_grid)

    # Analytic target: 3*(u-0.5)
    analytic = 3.0 * (u_grid - 0.5)
    abs_diff = np.abs(proj_vals - analytic)
    max_diff = float(np.max(abs_diff))
    l2_diff = float(np.sqrt(np.mean(abs_diff**2)))

    wc, _, proj_v, a_xi, b_xi = compute_unimodal_worst_case_no_wasserstein(
        metric, xi, mu=0.0, sigma=1.0, n_grid=n_grid
    )

    # For 3*(u-0.5): E = 0, E[sq] = 9/12 = 0.75, so std = sqrt(0.75) = sqrt(3)/2
    analytic_b_xi = np.sqrt(3) / 2  # = sqrt(integral_0^1 9*(u-0.5)^2 du) = sqrt(9/12)
    analytic_wc = 0.0 * metric.g_val_at_1 + 1.0 * analytic_b_xi

    return {
        "metric": "MMD",
        "xi": xi,
        "max_diff_to_analytic": max_diff,
        "l2_diff_to_analytic": l2_diff,
        "analytic_b_xi": analytic_b_xi,
        "numeric_b_xi": float(b_xi),
        "analytic_wc": analytic_wc,
        "numeric_wc": float(wc),
        "projection_matches_analytic": max_diff < 0.15,  # tolerance for step approx
    }


def run_step_approximation_study() -> pd.DataFrame:
    """Run step-function approximation study validating Proposition 4."""
    xi = 0.5
    metrics = [make_gd(), make_mmd(), make_es(alpha=0.05)]
    n_pieces_list = [10, 20, 50, 100, 200]

    results = []
    for metric in metrics:
        print(f"  Step approx: {metric.name}")
        errors = compute_step_projection_and_error(metric.gamma, xi, n_pieces_list)
        for err in errors:
            err["metric"] = metric.name
            results.append(err)

    return pd.DataFrame(results)


def run_unimodal_wasserstein_study() -> pd.DataFrame:
    """Run unimodality + Wasserstein bound study."""
    refs = [
        make_gaussian_reference(mu=0.0, sigma=1.0),
        make_student_t_reference(mu=0.0, sigma=1.0, nu=5.0),
    ]
    metrics = [make_gd(), make_mmd(), make_es(alpha=0.05)]
    xi = 0.5
    mu, sigma = 0.0, 1.0

    results = []
    for ref in refs:
        for metric in metrics:
            print(f"  Unimodal+Wass: {ref.name[:20]}, {metric.name}")
            try:
                eps_min_U, eps_switch_U, c0, c1 = compute_unimodal_wasserstein_thresholds(
                    ref, mu, sigma, metric, xi
                )

                # Unrestricted worst-case (no unimodality)
                from src.wasserstein_bounds import compute_worst_case_value
                unrestr_result = compute_worst_case_value(
                    metric, ref, mu, sigma, epsilon=eps_switch_U * 1.5,
                    n_grid=400, include_benchmark=False
                )
                unrestr_wc = unrestr_result.worst_case_value or float('nan')

                # No-Wasserstein unimodal
                wc_no_wass, _, _, _, b_xi = compute_unimodal_worst_case_no_wasserstein(
                    metric, xi, mu, sigma
                )

                eps_grid = np.array([
                    max(eps_min_U * 0.5, 1e-6),
                    eps_min_U * 1.01,
                    (eps_min_U + eps_switch_U) / 2,
                    eps_switch_U * 0.99,
                    eps_switch_U * 1.5,
                ])

                for eps in eps_grid:
                    try:
                        uni_res = compute_unimodal_wasserstein_bound(
                            metric, ref, mu, sigma, eps, xi, n_grid=500
                        )
                        results.append({
                            "ref": ref.name[:25],
                            "metric": metric.name,
                            "xi": xi,
                            "epsilon": eps,
                            "epsilon_min_U": eps_min_U,
                            "epsilon_switch_U": eps_switch_U,
                            "regime": uni_res.regime,
                            "lambda_epsilon": uni_res.lambda_epsilon,
                            "unimodal_wc": uni_res.worst_case_value,
                            "unrestricted_wc": unrestr_wc,
                            "unimodal_le_unrestricted": (
                                uni_res.worst_case_value <= unrestr_wc + 0.01
                                if np.isfinite(uni_res.worst_case_value) else None
                            ),
                        })
                    except Exception as e:
                        warnings.warn(f"eps={eps:.4f} failed: {e}")

            except Exception as e:
                warnings.warn(f"Study failed: {e}")

    return pd.DataFrame(results)


def plot_projections(gd_result: dict, mmd_result: dict) -> None:
    """Plot analytic vs numeric projections for GD and MMD."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    xi = 0.5
    u_plot = np.linspace(0.001, 0.999, 500)

    # GD panel
    ax = axes[0]
    metric_gd = make_gd()
    _, proj_gd, proj_fn_gd = project_gamma_unimodal(metric_gd.gamma, xi, 1000)
    u_grid_gd = np.linspace(1e-9, 1-1e-9, 1000)
    ax.plot(u_plot, metric_gd.gamma(u_plot), "b-", linewidth=2, label="γ_GD(u) = 1-2u")
    ax.plot(u_grid_gd, proj_gd, "r--", linewidth=2, label="Projection γ_GD^↑")
    ax.axvline(xi, color="g", linestyle=":", label=f"ξ={xi}")
    ax.set_xlabel("u")
    ax.set_ylabel("γ(u)")
    ax.set_title("GD: original and unimodal projection")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MMD panel
    ax = axes[1]
    metric_mmd = make_mmd()
    _, proj_mmd, _ = project_gamma_unimodal(metric_mmd.gamma, xi, 1000)
    u_grid_mmd = np.linspace(1e-9, 1-1e-9, 1000)
    ax.plot(u_plot, metric_mmd.gamma(u_plot), "b-", linewidth=2, label="γ_MMD(u)")
    ax.plot(u_grid_mmd, proj_mmd, "r--", linewidth=2, label="Projection γ_MMD^↑")
    ax.plot(u_plot, 3*(u_plot - 0.5), "g:", linewidth=2, label="Analytic: 3(u-0.5)")
    ax.axvline(xi, color="k", linestyle=":", label=f"ξ={xi}")
    ax.set_xlabel("u")
    ax.set_ylabel("γ(u)")
    ax.set_title("MMD: original, projection, and analytic")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Exp 2: Unimodal projections (xi=0.5)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp2_projections.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: exp2_projections.png")


def plot_step_errors(step_df: pd.DataFrame) -> None:
    """Plot Proposition 4 error bounds vs n_pieces."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for metric_name in step_df["metric"].unique():
        sub = step_df[step_df["metric"] == metric_name].sort_values("n_pieces")
        ax.semilogy(sub["n_pieces"], sub["gamma_approx_l2"], "--",
                    label=f"{metric_name[:12]} γ_n err", linewidth=1.5)
        ax.semilogy(sub["n_pieces"], sub["proj_error_l2"], "-o",
                    label=f"{metric_name[:12]} proj err", linewidth=1.5, ms=4)
    ax.set_xlabel("n_pieces")
    ax.set_ylabel("L2 error")
    ax.set_title("Exp 2: Step approx errors vs refinement (Prop 4 validation)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp2_step_approx_errors.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: exp2_step_approx_errors.png")


def save_results(gd_result: dict, mmd_result: dict, step_df: pd.DataFrame, uni_df: pd.DataFrame) -> None:
    """Save all Exp 2 results."""
    step_df.to_csv(RESULTS_DIR / "exp2_step_approx.csv", index=False)
    if len(uni_df) > 0:
        uni_df.to_csv(RESULTS_DIR / "exp2_unimodal_wasserstein.csv", index=False)

    summary = {
        "gd_validation": gd_result,
        "mmd_validation": mmd_result,
        "prop4_satisfied_fraction": (
            float(step_df["prop4_satisfied"].mean()) if "prop4_satisfied" in step_df.columns else None
        ),
        "pieces_bound_ok_fraction": (
            float(step_df["pieces_bound_ok"].mean()) if "pieces_bound_ok" in step_df.columns else None
        ),
    }
    with open(RESULTS_DIR / "exp2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved exp2 results. MMD analytic match: {mmd_result['projection_matches_analytic']}")


def main() -> None:
    print("=" * 60)
    print("Experiment 2: Unimodal Bounds and Projection Algorithms")
    print("=" * 60)

    print("\n1. Analytic example validation (GD and MMD)")
    gd_result = validate_gd_projection(xi=0.5)
    mmd_result = validate_mmd_projection(xi=0.5)
    print(f"   GD b_xi={gd_result['b_xi']:.4f}, wc={gd_result['worst_case_value']:.4f}")
    print(f"   MMD analytic_b_xi={mmd_result['analytic_b_xi']:.4f}, numeric={mmd_result['numeric_b_xi']:.4f}")
    print(f"   MMD max diff to analytic: {mmd_result['max_diff_to_analytic']:.4f}")

    print("\n2. Step-function approximation study (Prop 4)")
    step_df = run_step_approximation_study()
    print(f"   Prop 4 satisfied: {step_df['prop4_satisfied'].mean():.2%} of cases")
    print(f"   Pieces bound OK: {step_df['pieces_bound_ok'].mean():.2%} of cases")

    print("\n3. Unimodality + Wasserstein combined study")
    uni_df = run_unimodal_wasserstein_study()
    print(f"   Total cases: {len(uni_df)}")

    print("\n4. Saving results and plots")
    plot_projections(gd_result, mmd_result)
    plot_step_errors(step_df)
    save_results(gd_result, mmd_result, step_df, uni_df)

    print("\nExp 2 complete.")


if __name__ == "__main__":
    main()
