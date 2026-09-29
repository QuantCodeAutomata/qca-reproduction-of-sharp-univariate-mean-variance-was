"""
Experiment 1: Reproduction of sharp univariate mean-variance-Wasserstein bounds
for distortion risk metrics.

Validates Theorems 1, 2 and Corollaries 1, 2. Covers:
- Feasibility thresholds and active/inactive Wasserstein regimes
- Worst-case and best-case values for GD, MMD, ES, VaR+, IQD+, GlueVaR
- Gaussian and Student-t reference distributions
- Benchmark comparison via direct constrained optimization
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

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reference_distributions import (
    make_gaussian_reference,
    make_student_t_reference,
    STANDARD_GAUSSIAN,
    STANDARD_STUDENT_T5,
)
from src.distortion_metrics import (
    make_gd, make_mmd, make_es, make_var_plus, make_iqd_plus,
    make_gluevar, make_distortion_metrics_for_exp1, eval_rho_direct,
)
from src.wasserstein_bounds import (
    compute_worst_case_value,
    compute_best_case_value,
    compute_thresholds,
    BoundResult,
)
from src.concave_envelope import (
    least_concave_majorant,
    compute_b_lambda,
    compute_a_lambda,
    wasserstein_sq_distance,
    construct_h_lambda,
    build_g_lambda_vals,
    _make_g_grid_local,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _make_epsilon_grid(eps_min: float, eps_switch: float, n: int = 3) -> np.ndarray:
    """
    Create epsilon grid covering infeasible, boundary, active, switch, and inactive regimes.

    Parameters
    ----------
    eps_min : float
        Feasibility threshold.
    eps_switch : float
        Active/inactive switching threshold.
    n : int
        Number of interior active-regime samples (default 3 for speed).
    """
    grid = []
    # Infeasible
    if eps_min > 1e-6:
        grid.append(eps_min * 0.5)
    # Boundary
    grid.append(eps_min * 1.001)
    # Active regime (between min and switch)
    if eps_switch > eps_min + 1e-6:
        for f in np.linspace(0.2, 0.8, n):
            grid.append(eps_min + f * (eps_switch - eps_min))
    # Near switch
    grid.append(eps_switch * 0.999)
    grid.append(eps_switch * 1.001)
    # Inactive
    grid.append(eps_switch * 2.0)
    return np.array(sorted(set([float(x) for x in grid if x > 0])))


def run_exp1() -> pd.DataFrame:
    """
    Main experiment 1: sweep over references, metrics, moment pairs, and epsilon values.

    Returns
    -------
    pd.DataFrame with all results.
    """
    # Reference distributions
    refs = [
        make_gaussian_reference(mu=0.0, sigma=1.0),
        make_student_t_reference(mu=0.0, sigma=1.0, nu=5.0),
    ]

    # Moment pairs: (mu, sigma)
    moment_pairs = [
        (0.0, 1.0),   # Same as reference
        (0.1, 1.0),   # Shifted mean
    ]

    # Distortion metrics
    metrics = [
        make_gd(),
        make_mmd(),
        make_es(alpha=0.05),
        make_es(alpha=0.25),
        make_var_plus(alpha=0.05),
        make_iqd_plus(alpha=0.1),
        make_gluevar(alpha=0.05, beta=0.1, h1=0.6, h2=0.9),
    ]

    all_results = []

    for ref in refs:
        for (mu, sigma) in moment_pairs:
            print(f"\nRef: {ref.name}, mu={mu}, sigma={sigma}")

            for metric in metrics:
                try:
                    # Compute thresholds
                    eps_min, eps_switch, c0 = compute_thresholds(
                        ref, mu, sigma, metric, n_grid=200, n_gauss=100
                    )
                    print(f"  {metric.name}: eps_min={eps_min:.4f}, eps_switch={eps_switch:.4f}, c0={c0:.4f}")

                    # Build epsilon grid
                    eps_grid = _make_epsilon_grid(eps_min, eps_switch, n=3)

                    # Benchmark only for a subset of metrics to keep runtime manageable
                    benchmark_metrics = {"GD", "MMD", "ES(alpha=0.05)"}
                    for eps in eps_grid:
                        try:
                            run_bench = (
                                eps > eps_min
                                and metric.name in benchmark_metrics
                            )
                            res = compute_worst_case_value(
                                metric=metric,
                                ref=ref,
                                mu=mu,
                                sigma=sigma,
                                epsilon=eps,
                                n_grid=200,
                                n_gauss=100,
                                tol=1e-8,
                                include_benchmark=run_bench,
                            )
                            # Best-case
                            if res.regime != "infeasible":
                                bc_val = compute_best_case_value(
                                    metric, ref, mu, sigma, eps,
                                    n_grid=200, n_gauss=100
                                )
                            else:
                                bc_val = None

                            all_results.append({
                                "ref": ref.name[:30],
                                "metric": metric.name,
                                "mu": mu,
                                "sigma": sigma,
                                "epsilon": eps,
                                "epsilon_min": eps_min,
                                "epsilon_switch": eps_switch,
                                "c0": c0,
                                "regime": res.regime,
                                "lambda_epsilon": res.lambda_epsilon,
                                "worst_case": res.worst_case_value,
                                "best_case": bc_val,
                                "benchmark": res.benchmark_value,
                                "abs_gap": res.abs_gap,
                                "constraints_ok": res.constraints_satisfied,
                            })
                        except Exception as e:
                            warnings.warn(f"epsilon={eps:.4f} failed: {e}")
                            all_results.append({
                                "ref": ref.name[:30],
                                "metric": metric.name,
                                "mu": mu,
                                "sigma": sigma,
                                "epsilon": eps,
                                "regime": "error",
                                "error": str(e),
                            })

                except Exception as e:
                    warnings.warn(f"Metric {metric.name} failed: {e}")

    return pd.DataFrame(all_results)


def plot_worst_case_curves(df: pd.DataFrame) -> None:
    """
    Plot worst-case values vs epsilon for each metric/reference combo.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    refs = df["ref"].dropna().unique()
    metrics_to_plot = ["GD", "MMD", "ES(alpha=0.05)", "ES(alpha=0.25)"]

    for ax_idx, ref_name in enumerate(refs[:4]):
        ax = axes[ax_idx]
        sub = df[(df["ref"] == ref_name) & (df["regime"] != "error") &
                 (df["mu"] == 0.0) & (df["sigma"] == 1.0)].copy()

        for metric_name in sub["metric"].unique():
            m_sub = sub[sub["metric"] == metric_name].sort_values("epsilon")
            if len(m_sub) == 0:
                continue
            wc = m_sub["worst_case"].values
            eps_vals = m_sub["epsilon"].values
            valid = np.isfinite(wc)
            if valid.sum() > 1:
                ax.plot(eps_vals[valid], wc[valid], marker="o", ms=3,
                        label=metric_name[:15], linewidth=1.5)

        ax.set_xlabel(r"$\varepsilon$", fontsize=11)
        ax.set_ylabel("Worst-case value", fontsize=11)
        ax.set_title(f"Ref: {ref_name[:25]}", fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Exp 1: Worst-case distortion risk bounds vs Wasserstein radius",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp1_worst_case_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: exp1_worst_case_curves.png")


def plot_quantile_deformation(df: pd.DataFrame) -> None:
    """
    Plot the worst-case quantile h_lambda for different epsilon values (GD, Gaussian ref).
    """
    from src.wasserstein_bounds import _make_g_grid
    from src.concave_envelope import build_g_lambda_vals, least_concave_majorant, compute_a_lambda, compute_b_lambda, construct_h_lambda

    ref = make_gaussian_reference(mu=0.0, sigma=1.0)
    metric = make_gd()
    mu, sigma = 0.0, 1.0

    eps_min, eps_switch, _ = compute_thresholds(ref, mu, sigma, metric, n_grid=200)
    eps_vals = [eps_min * 1.05, (eps_min + eps_switch) / 2, eps_switch * 1.5]
    u_eval = np.linspace(0.001, 0.999, 300)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(u_eval, ref.quantile(u_eval), "k--", linewidth=2, label="F^{-1}(u) [reference]")

    from src.wasserstein_bounds import compute_worst_case_value, solve_lambda_epsilon
    u_grid, g_vals = _make_g_grid(metric, n=200)

    for eps in eps_vals:
        try:
            res = compute_worst_case_value(
                metric, ref, mu, sigma, eps, n_grid=200, include_benchmark=False
            )
            lam = res.lambda_epsilon if res.lambda_epsilon is not None and np.isfinite(res.lambda_epsilon) else 0.0
            g_lam = build_g_lambda_vals(u_grid, g_vals, lam, ref.quantile, ref.mu_F)
            hull_u, hull_f, _ = least_concave_majorant(u_grid, g_lam)
            a_lam = compute_a_lambda(0.0, metric.g_val_at_1, lam, ref.mu_F)
            b_lam = compute_b_lambda(hull_u, hull_f, a_lam)
            if b_lam > 1e-12:
                h_vals = construct_h_lambda(hull_u, hull_f, a_lam, b_lam, mu, sigma, u_eval)
                label = f"ε={eps:.3f} ({res.regime})"
                ax.plot(u_eval, h_vals, linewidth=1.5, label=label)
        except Exception as e:
            warnings.warn(f"Quantile deformation failed for eps={eps}: {e}")

    ax.set_xlabel("u", fontsize=11)
    ax.set_ylabel("Quantile", fontsize=11)
    ax.set_title("Exp 1: Worst-case quantile h_λ(u) for varying ε (GD, Gaussian ref)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp1_quantile_deformation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: exp1_quantile_deformation.png")


def plot_benchmark_comparison(df: pd.DataFrame) -> None:
    """Plot absolute gap between theorem and benchmark values."""
    bench_df = df[df["benchmark"].notna() & df["abs_gap"].notna() & (df["regime"] != "error")]
    if len(bench_df) == 0:
        print("No benchmark data to plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for metric_name in bench_df["metric"].unique():
        m_sub = bench_df[bench_df["metric"] == metric_name].sort_values("epsilon")
        gaps = m_sub["abs_gap"].values
        eps_vals = m_sub["epsilon"].values
        valid = np.isfinite(gaps)
        if valid.sum() > 0:
            ax.semilogy(eps_vals[valid], np.maximum(gaps[valid], 1e-12),
                        marker="s", ms=4, linewidth=1.5, label=metric_name[:15])

    ax.set_xlabel(r"$\varepsilon$", fontsize=11)
    ax.set_ylabel("|Theorem - Benchmark|", fontsize=11)
    ax.set_title("Exp 1: Absolute gap between theorem and benchmark optimization", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp1_benchmark_gap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: exp1_benchmark_gap.png")


def save_results(df: pd.DataFrame) -> None:
    """Save results to CSV and JSON."""
    df.to_csv(RESULTS_DIR / "exp1_results.csv", index=False)
    # Save summary to JSON
    summary = {
        "n_total_cases": len(df),
        "n_infeasible": int((df["regime"] == "infeasible").sum()),
        "n_active": int((df["regime"] == "active").sum()),
        "n_inactive": int((df["regime"] == "inactive").sum()),
        "n_errors": int((df["regime"] == "error").sum()),
    }
    if "abs_gap" in df.columns:
        bench_df = df[df["abs_gap"].notna()]
        if len(bench_df) > 0:
            summary["max_abs_gap"] = float(bench_df["abs_gap"].max())
            summary["mean_abs_gap"] = float(bench_df["abs_gap"].mean())
            summary["n_benchmark_cases"] = int(len(bench_df))
    with open(RESULTS_DIR / "exp1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved exp1_results.csv, exp1_summary.json. Summary: {summary}")


def main() -> None:
    print("=" * 60)
    print("Experiment 1: Univariate Wasserstein Distortion Risk Bounds")
    print("=" * 60)

    df = run_exp1()
    save_results(df)
    plot_worst_case_curves(df)
    plot_quantile_deformation(df)
    plot_benchmark_comparison(df)

    print("\nExp 1 complete.")
    return df


if __name__ == "__main__":
    main()
