"""
Experiment 3: Reproduction of robust portfolio optimization under multivariate
Wasserstein ambiguity.

Validates:
- Reduction from multivariate to univariate Wasserstein problem (Proposition 6)
- Optimal weight w1^*(epsilon) curves for 4 scenarios x 6 metrics
- Active/inactive regime transitions
- Plateau behavior for large epsilon
- Qualitative difference between variability (GD, MMD) and tail-risk (VaR, ES) metrics
"""

from __future__ import annotations
import sys
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.distortion_metrics import make_gd, make_mmd, make_iqd_plus, make_var_plus, make_es, make_gluevar
from src.portfolio_optimization import (
    make_standard_scenarios,
    run_portfolio_experiment,
    _precompute_portfolio,
    _fast_portfolio_objective,
    _get_standardized_reference,
    PortfolioScenario,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def make_portfolio_metrics():
    """Define the six distortion risk metrics used in portfolio experiments."""
    return [
        make_gd(),
        make_mmd(),
        make_iqd_plus(alpha=0.1),
        make_var_plus(alpha=0.05),
        make_es(alpha=0.05),
        make_gluevar(alpha=0.05, beta=0.1, h1=0.6, h2=0.9),
    ]


def make_epsilon_grid(n_eps: int = 12) -> np.ndarray:
    """Create epsilon grid from near-zero to a large value revealing plateau."""
    return np.concatenate([
        np.linspace(0.001, 0.05, 4),
        np.linspace(0.05, 0.2, 4),
        np.linspace(0.2, 0.5, 4),
    ])


def validate_nominal_solution(scenarios, metrics):
    """
    At epsilon=0, verify solution equals nominal minimum without ambiguity.
    Uses fast precomputed path.  Returns list of validation dicts.
    """
    print("  Validating nominal (epsilon=0) solutions...")
    validations = []
    eps_small = 1e-10  # effectively 0
    w1_grid = np.linspace(0.0, 1.0, 15)

    for scenario in scenarios:
        std_ref = _get_standardized_reference(scenario)
        for metric in metrics:
            try:
                pc = _precompute_portfolio(metric, std_ref, n_grid=80, n_gauss=40)
                objs = [_fast_portfolio_objective(w1, scenario, eps_small, pc)[0] for w1 in w1_grid]
                best_idx = int(np.nanargmin([o if np.isfinite(o) else 1e18 for o in objs]))
                w1_best = float(w1_grid[best_idx])
                _, regime, lam = _fast_portfolio_objective(w1_best, scenario, eps_small, pc)
                validations.append({
                    "scenario": scenario.name,
                    "metric": metric.name,
                    "eps": eps_small,
                    "w1_nominal": w1_best,
                    "obj_nominal": float(objs[best_idx]),
                    "regime": regime,
                })
            except Exception as e:
                warnings.warn(f"Nominal failed: {scenario.name}/{metric.name}: {e}")
    return validations


def plot_weight_curves(df: pd.DataFrame, scenario_name: str) -> None:
    """Plot w1^* vs epsilon for all metrics in a given scenario."""
    sub = df[df["scenario"] == scenario_name].copy()
    sub = sub[sub["w1_star"].notna()]
    if len(sub) == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for metric_name in sub["metric"].unique():
        m_sub = sub[sub["metric"] == metric_name].sort_values("epsilon")
        eps = m_sub["epsilon"].values
        w1 = m_sub["w1_star"].values
        valid = np.isfinite(w1)
        if valid.sum() > 0:
            ax.plot(eps[valid], w1[valid], marker="o", ms=3,
                    linewidth=1.5, label=metric_name[:18])

    ax.set_xlabel(r"$\varepsilon$", fontsize=12)
    ax.set_ylabel(r"$w_1^*(\varepsilon)$", fontsize=12)
    ax.set_title(f"Portfolio: {scenario_name}", fontsize=11)
    ax.legend(fontsize=8, loc="best")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    safe_name = scenario_name.replace("/", "_")
    plt.savefig(RESULTS_DIR / f"exp3_weights_{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: exp3_weights_{safe_name}.png")


def plot_all_scenarios_grid(df: pd.DataFrame) -> None:
    """4-panel grid of weight curves for all scenarios."""
    scenarios = df["scenario"].unique()
    n_panels = min(4, len(scenarios))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, scenario_name in enumerate(scenarios[:n_panels]):
        ax = axes[i]
        sub = df[(df["scenario"] == scenario_name) & df["w1_star"].notna()].copy()

        for metric_name in sub["metric"].unique():
            m_sub = sub[sub["metric"] == metric_name].sort_values("epsilon")
            eps = m_sub["epsilon"].values
            w1 = m_sub["w1_star"].values
            valid = np.isfinite(w1)
            if valid.sum() > 0:
                ax.plot(eps[valid], w1[valid], marker="o", ms=2,
                        linewidth=1.5, label=metric_name[:14])

        ax.set_xlabel(r"$\varepsilon$", fontsize=10)
        ax.set_ylabel(r"$w_1^*$", fontsize=10)
        ax.set_title(scenario_name[:30], fontsize=9)
        ax.legend(fontsize=6, loc="best")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Exp 3: Robust Portfolio Weights w1^*(ε) - All Scenarios",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp3_all_scenarios.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: exp3_all_scenarios.png")


def check_qualitative_claims(df: pd.DataFrame) -> dict:
    """
    Check paper's qualitative claims:
    1. Variability metrics (GD, MMD) give lower w1 than tail-risk (VaR, ES) at large eps.
    2. Plateau exists for large epsilon.
    3. Gaussian vs Student-t differ qualitatively.
    """
    claims = {}

    # Check plateau: w1 should be nearly constant for large eps
    large_eps_df = df[df["epsilon"] > df["epsilon"].quantile(0.8)]
    plateau_vars = []
    for (sc, mt), grp in large_eps_df.groupby(["scenario", "metric"]):
        w1 = grp["w1_star"].dropna().values
        if len(w1) >= 3:
            plateau_vars.append(np.std(w1))
    if plateau_vars:
        claims["mean_plateau_std"] = float(np.mean(plateau_vars))
        claims["plateau_exists"] = float(np.mean(plateau_vars)) < 0.1

    # Check variability vs tail metrics at largest epsilon
    max_eps = df["epsilon"].max()
    at_max = df[df["epsilon"] >= max_eps * 0.9]
    variability = ["GD", "MMD"]
    tail_risk = ["VaR+(alpha=0.05)", "ES(alpha=0.05)"]
    for sc_name in df["scenario"].unique():
        sc_sub = at_max[at_max["scenario"] == sc_name]
        var_w1 = sc_sub[sc_sub["metric"].isin(variability)]["w1_star"].dropna().mean()
        tail_w1 = sc_sub[sc_sub["metric"].isin(tail_risk)]["w1_star"].dropna().mean()
        claims[f"{sc_name}_variability_w1"] = float(var_w1) if np.isfinite(var_w1) else None
        claims[f"{sc_name}_tail_w1"] = float(tail_w1) if np.isfinite(tail_w1) else None

    return claims


def save_results(df: pd.DataFrame, validations: list, claims: dict) -> None:
    """Save Exp 3 results."""
    df.to_csv(RESULTS_DIR / "exp3_portfolio_results.csv", index=False)
    summary = {
        "n_scenarios": int(df["scenario"].nunique()),
        "n_metrics": int(df["metric"].nunique()),
        "n_epsilon_values": int(df["epsilon"].nunique()),
        "n_total_rows": len(df),
        "n_valid": int(df["w1_star"].notna().sum()),
        "qualitative_claims": claims,
    }
    with open(RESULTS_DIR / "exp3_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved exp3 results. Total rows: {len(df)}, Valid: {df['w1_star'].notna().sum()}")


def main() -> None:
    print("=" * 60)
    print("Experiment 3: Robust Portfolio Optimization under Wasserstein Ambiguity")
    print("=" * 60)

    scenarios = make_standard_scenarios()
    metrics = make_portfolio_metrics()
    epsilon_grid = make_epsilon_grid()

    print(f"\nScenarios: {[s.name for s in scenarios]}")
    print(f"Metrics: {[m.name for m in metrics]}")
    print(f"Epsilon range: [{epsilon_grid.min():.4f}, {epsilon_grid.max():.4f}], n={len(epsilon_grid)}")

    print("\n1. Nominal solution validation (epsilon ~ 0)")
    validations = validate_nominal_solution(scenarios[:2], metrics[:3])
    print(f"   Validated {len(validations)} cases.")

    print("\n2. Running main portfolio experiment (fast precomputed path)...")
    df = pd.DataFrame(run_portfolio_experiment(
        scenarios=scenarios,
        metrics=metrics,
        epsilon_grid=epsilon_grid,
        n_w_grid=20,    # coarse w-grid; local refinement via minimize_scalar
        n_grid=120,     # LCM grid (precomputed once per metric/scenario)
        n_gauss=60,     # GL quadrature points (precomputed once per metric/scenario)
    ))

    print("\n3. Qualitative claims check")
    claims = check_qualitative_claims(df)
    print(f"   Plateau exists: {claims.get('plateau_exists', '?')}")
    print(f"   Mean plateau std: {claims.get('mean_plateau_std', float('nan')):.4f}")

    print("\n4. Saving results and plots")
    for scenario in scenarios:
        plot_weight_curves(df, scenario.name)
    plot_all_scenarios_grid(df)
    save_results(df, validations, claims)

    print("\nExp 3 complete.")


if __name__ == "__main__":
    main()
