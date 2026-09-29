"""
Robust portfolio optimization under multivariate 2-Wasserstein ambiguity (Exp 3).

Implements the reduction from the multivariate Wasserstein portfolio problem to
a univariate mean-variance-Wasserstein problem parameterized by portfolio weight w.

Reference: Proposition 6 and Corollary 3 of the paper.

Custom implementation — Context7 found no library equivalent for paper-specific
distortion-risk robust portfolio objective.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize_scalar
from scipy import stats
import warnings

from .reference_distributions import ReferenceDistribution, make_gaussian_reference, make_student_t_reference
from .distortion_metrics import DistortionMetric, eval_rho_direct
from .concave_envelope import (
    _leggauss,
    least_concave_majorant,
    lcm_derivative,
    compute_a_lambda,
    compute_b_lambda,
    construct_h_lambda,
    wasserstein_sq_distance,
    build_g_lambda_vals,
)
from .wasserstein_bounds import (
    compute_worst_case_value,
    _make_g_grid,
    solve_lambda_epsilon,
    N_GRID,
)


@dataclass
class PortfolioScenario:
    """
    A multivariate portfolio scenario with known mean and covariance.

    For a 2-asset loss portfolio:
    L_w = w^T X, X is the loss vector.

    Attributes
    ----------
    name : str
    mu_vec : np.ndarray, shape (2,)
        Mean loss vector.
    cov_mat : np.ndarray, shape (2,2)
        Covariance matrix.
    ref_family : str
        'gaussian' or 'student_t'.
    nu : float
        Degrees of freedom (for Student-t), > 2.
    """

    name: str
    mu_vec: np.ndarray
    cov_mat: np.ndarray
    ref_family: str
    nu: float = 5.0


def make_standard_scenarios() -> List[PortfolioScenario]:
    """
    Create the four standard scenarios from the paper:
    Gaussian/positive correlation, Gaussian/negative correlation,
    Student-t/positive correlation, Student-t/negative correlation.

    Parameters from the paper (Section portfolio):
    mu = (-0.06, -0.03)^T  (loss means)
    sigma1 = 0.15, sigma2 = 0.10, correlation = ±0.5
    nu = 5 for Student-t

    Returns
    -------
    List[PortfolioScenario]
    """
    mu_vec = np.array([-0.06, -0.03])
    s1, s2, rho = 0.15, 0.10, 0.5

    cov_pos = np.array([
        [s1**2, rho * s1 * s2],
        [rho * s1 * s2, s2**2],
    ])
    cov_neg = np.array([
        [s1**2, -rho * s1 * s2],
        [-rho * s1 * s2, s2**2],
    ])

    scenarios = [
        PortfolioScenario(
            name="Gaussian_positive",
            mu_vec=mu_vec.copy(),
            cov_mat=cov_pos.copy(),
            ref_family="gaussian",
            nu=5.0,
        ),
        PortfolioScenario(
            name="Gaussian_negative",
            mu_vec=mu_vec.copy(),
            cov_mat=cov_neg.copy(),
            ref_family="gaussian",
            nu=5.0,
        ),
        PortfolioScenario(
            name="StudentT5_positive",
            mu_vec=mu_vec.copy(),
            cov_mat=cov_pos.copy(),
            ref_family="student_t",
            nu=5.0,
        ),
        PortfolioScenario(
            name="StudentT5_negative",
            mu_vec=mu_vec.copy(),
            cov_mat=cov_neg.copy(),
            ref_family="student_t",
            nu=5.0,
        ),
    ]
    return scenarios


def _make_scalar_reference(
    scenario: PortfolioScenario,
    w: np.ndarray,
) -> ReferenceDistribution:
    """
    For a given weight vector w, construct the scalar reference distribution
    for w^T X_0 under the elliptical model.

    Under elliptical family E_n(mu, Sigma0, psi):
    w^T X_0 ~ E_1(w^T mu, w^T Sigma0 w, psi)

    Standardized scalar reference F_0: E_1(0, 1, psi).

    For Gaussian: F_0 = N(0,1).
    For Student-t with nu DOF: F_0 = t_nu scaled to have unit variance.
    The actual quantile of w^T X_0 is:
    F_{w^T X_0}^{-1}(u) = m_w + sqrt(s_w^2) * F_0^{-1}(u)

    Parameters
    ----------
    scenario : PortfolioScenario
    w : np.ndarray, shape (2,)

    Returns
    -------
    ReferenceDistribution for the portfolio
    """
    m_w = float(w @ scenario.mu_vec)
    s_w2 = float(w @ scenario.cov_mat @ w)
    s_w = float(np.sqrt(max(s_w2, 1e-15)))

    if scenario.ref_family == "gaussian":
        return make_gaussian_reference(mu=m_w, sigma=s_w)
    elif scenario.ref_family == "student_t":
        nu = scenario.nu
        # Scale student-t to have variance s_w^2:
        # X = m_w + s_w * sqrt((nu-2)/nu) * T_nu
        return make_student_t_reference(mu=m_w, sigma=s_w, nu=nu)
    else:
        raise ValueError(f"Unknown ref_family: {scenario.ref_family}")


def _get_standardized_reference(scenario: PortfolioScenario) -> ReferenceDistribution:
    """Return the standardized scalar reference F_0 ~ E_1(0,1,psi)."""
    if scenario.ref_family == "gaussian":
        return make_gaussian_reference(mu=0.0, sigma=1.0)
    elif scenario.ref_family == "student_t":
        nu = scenario.nu
        return make_student_t_reference(mu=0.0, sigma=1.0, nu=nu)
    else:
        raise ValueError(f"Unknown ref_family: {scenario.ref_family}")


def portfolio_robust_objective(
    w1: float,
    scenario: PortfolioScenario,
    metric: DistortionMetric,
    epsilon: float,
    n_grid: int = 500,
    n_gauss: int = 300,
) -> Tuple[float, str, float]:
    """
    Compute the robust portfolio objective J(w; epsilon) for w = (w1, 1-w1).

    Using the reduction (Proposition 6):
    J(w; epsilon) = w^T mu * g(1) + sqrt(w^T Sigma w) *
                    sup_{G in M_{epsilon_w}(0,1)} rho_g(G)

    where:
    epsilon_w = epsilon * ||w||_2^2 / (w^T Sigma w)  (standardized radius)

    For the standardized problem: mean=0, sigma=1, Wasserstein radius = epsilon_w.

    Parameters
    ----------
    w1 : float
        Weight on asset 1; w2 = 1 - w1.
    scenario : PortfolioScenario
    metric : DistortionMetric
    epsilon : float
        Global Wasserstein radius squared.
    n_grid, n_gauss : int

    Returns
    -------
    (objective_value, regime, lambda_w)
    """
    w = np.array([w1, 1.0 - w1])
    m_w = float(w @ scenario.mu_vec)
    s_w2 = float(w @ scenario.cov_mat @ w)
    s_w = float(np.sqrt(max(s_w2, 1e-15)))
    norm_w2 = float(w @ w)

    # Standardized Wasserstein radius
    epsilon_w = epsilon * norm_w2 / max(s_w2, 1e-15)

    # Standardized reference F_0 ~ E_1(0, 1, psi)
    std_ref = _get_standardized_reference(scenario)

    # Compute sup rho_g over M_{epsilon_w}(0, 1) with reference F_0
    result = compute_worst_case_value(
        metric=metric,
        ref=std_ref,
        mu=0.0,
        sigma=1.0,
        epsilon=epsilon_w,
        n_grid=n_grid,
        n_gauss=n_gauss,
        tol=1e-8,
        include_benchmark=False,
    )

    if result.regime == "infeasible":
        # The standardized problem is infeasible => objective is infeasible
        # Return a large value to indicate infeasibility
        return float("inf"), "infeasible", np.nan

    sup_rho = result.worst_case_value if result.worst_case_value is not None else 0.0

    # J(w; epsilon) = m_w * g(1) + s_w * sup_rho
    objective = m_w * metric.g_val_at_1 + s_w * sup_rho

    regime = result.regime
    lambda_w = result.lambda_epsilon

    return float(objective), regime, float(lambda_w)


def optimize_portfolio(
    scenario: PortfolioScenario,
    metric: DistortionMetric,
    epsilon: float,
    n_w_grid: int = 50,
    n_grid: int = 400,
    n_gauss: int = 300,
) -> Tuple[float, float, str, float]:
    """
    Find the optimal portfolio weight w1^* minimizing J(w; epsilon).

    Since n=2, sweep over w1 in [0, 1] and use bounded scalar minimization.

    Parameters
    ----------
    scenario : PortfolioScenario
    metric : DistortionMetric
    epsilon : float
    n_w_grid : int
        Initial coarse grid size.
    n_grid, n_gauss : int

    Returns
    -------
    (w1_star, objective_star, regime_star, lambda_star)
    """
    # Coarse sweep
    w1_candidates = np.linspace(0.0, 1.0, n_w_grid)
    objectives = []
    for w1 in w1_candidates:
        try:
            obj, _, _ = portfolio_robust_objective(
                w1, scenario, metric, epsilon, n_grid, n_gauss
            )
        except Exception:
            obj = float("inf")
        objectives.append(obj)

    best_coarse_idx = int(np.nanargmin([o if np.isfinite(o) else 1e18 for o in objectives]))
    best_w1_coarse = w1_candidates[best_coarse_idx]

    # Refine with bounded minimization
    lo = max(0.0, best_w1_coarse - 2.0 / n_w_grid)
    hi = min(1.0, best_w1_coarse + 2.0 / n_w_grid)

    try:
        res = minimize_scalar(
            lambda w1: portfolio_robust_objective(w1, scenario, metric, epsilon, n_grid, n_gauss)[0],
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-4},
        )
        w1_star = float(res.x)
        obj_star = float(res.fun)
    except Exception:
        w1_star = best_w1_coarse
        obj_star = objectives[best_coarse_idx]

    # Get regime and lambda at the optimum
    try:
        _, regime_star, lambda_star = portfolio_robust_objective(
            w1_star, scenario, metric, epsilon, n_grid, n_gauss
        )
    except Exception:
        regime_star = "unknown"
        lambda_star = np.nan

    return w1_star, obj_star, regime_star, lambda_star


@dataclass
class _PortfolioPrecomputed:
    """Cached computations shared across all (w1, epsilon) evaluations for a fixed (metric, scenario)."""
    u_grid: np.ndarray
    g_vals: np.ndarray
    # LCM at lambda=0 (for inactive-regime and threshold computation)
    hull_u0: np.ndarray
    hull_f0: np.ndarray
    a_lam0: float
    b_lam0: float
    wc_val_at_lam0: float    # rho_g(H_0)
    epsilon_min: float        # threshold: (mu_F-0)^2 + (sigma_F-1)^2
    epsilon_switch: float     # threshold above which lambda=0 is optimal
    std_ref: ReferenceDistribution
    metric: DistortionMetric
    n_gauss: int


def _precompute_portfolio(
    metric: DistortionMetric,
    std_ref: ReferenceDistribution,
    n_grid: int = 150,
    n_gauss: int = 80,
) -> _PortfolioPrecomputed:
    """
    Precompute all lambda=0 (inactive-regime) shared state for one (metric, scenario).

    This is called once per (metric, scenario) pair; subsequent calls for different
    w1/epsilon values reuse the cached results.

    Parameters
    ----------
    metric : DistortionMetric
    std_ref : ReferenceDistribution  Standardised reference (mu=0, sigma=1).
    n_grid, n_gauss : int

    Returns
    -------
    _PortfolioPrecomputed
    """
    mu_F, sigma_F = std_ref.mu_F, std_ref.sigma_F
    # Feasibility threshold for standardised (mu=0, sigma=1) target
    epsilon_min = (mu_F - 0.0) ** 2 + (sigma_F - 1.0) ** 2

    u_grid, g_vals = _make_g_grid(metric, n=n_grid)

    # LCM at lambda=0 (concave envelope g*)
    hull_u0, hull_f0, _ = least_concave_majorant(u_grid, g_vals)
    a_lam0 = compute_a_lambda(0.0, metric.g_val_at_1, 0.0, mu_F)
    b_lam0 = compute_b_lambda(hull_u0, hull_f0, a_lam0)

    # c_0 = Corr(F^{-1}(V), (g*)'(1-V)) via GL quadrature
    nodes, weights = _leggauss(n_gauss)
    eps_clip = 1e-9
    half = 0.5 * (1.0 - 2 * eps_clip)
    u_pts = 0.5 * (1.0 - 2 * eps_clip) * nodes + 0.5 * (1.0 + 2 * eps_clip)
    f_inv_pts = std_ref.quantile(u_pts)
    g_star_deriv_pts = lcm_derivative(hull_u0, hull_f0, 1.0 - u_pts)

    if np.std(g_star_deriv_pts) < 1e-12:
        c_0 = 0.0
    else:
        e_f = float(half * np.sum(weights * f_inv_pts))
        e_g = float(half * np.sum(weights * g_star_deriv_pts))
        e_fg = float(half * np.sum(weights * f_inv_pts * g_star_deriv_pts))
        e_f2 = float(half * np.sum(weights * f_inv_pts ** 2))
        e_g2 = float(half * np.sum(weights * g_star_deriv_pts ** 2))
        var_f = e_f2 - e_f ** 2
        var_g = e_g2 - e_g ** 2
        c_0 = (e_fg - e_f * e_g) / np.sqrt(var_f * var_g) if (var_f > 1e-15 and var_g > 1e-15) else 0.0

    c_0 = float(np.clip(c_0, -1.0, 1.0))
    epsilon_switch = epsilon_min + 2.0 * sigma_F * 1.0 * (1.0 - c_0)

    # WC value at lambda=0
    if b_lam0 < 1e-12:
        wc_val_at_lam0 = 0.0 * metric.g_val_at_1  # = 0 for variability metrics
    else:
        def h0_fn(u: np.ndarray) -> np.ndarray:
            return construct_h_lambda(hull_u0, hull_f0, a_lam0, b_lam0, 0.0, 1.0, u)
        wc_val_at_lam0 = float(eval_rho_direct(metric, h0_fn))

    return _PortfolioPrecomputed(
        u_grid=u_grid,
        g_vals=g_vals,
        hull_u0=hull_u0,
        hull_f0=hull_f0,
        a_lam0=a_lam0,
        b_lam0=b_lam0,
        wc_val_at_lam0=wc_val_at_lam0,
        epsilon_min=epsilon_min,
        epsilon_switch=epsilon_switch,
        std_ref=std_ref,
        metric=metric,
        n_gauss=n_gauss,
    )


def _fast_wc_value(
    epsilon_w: float,
    pc: _PortfolioPrecomputed,
    tol: float = 1e-7,
) -> Tuple[float, str, float]:
    """
    Quickly evaluate sup rho_g over M_{epsilon_w}(0,1) using precomputed state.

    Parameters
    ----------
    epsilon_w : float
        Standardised Wasserstein radius squared.
    pc : _PortfolioPrecomputed
    tol : float

    Returns
    -------
    (wc_value, regime, lambda_epsilon)
    """
    if epsilon_w < pc.epsilon_min - tol:
        return np.nan, "infeasible", np.nan

    if epsilon_w >= pc.epsilon_switch - tol:
        return pc.wc_val_at_lam0, "inactive", 0.0

    # Active regime: solve for lambda_epsilon
    lam = solve_lambda_epsilon(
        epsilon_w,
        pc.u_grid, pc.g_vals, pc.metric.g_val_at_1,
        pc.std_ref, 0.0, 1.0,
        tol=tol,
        n_gauss=pc.n_gauss,
    )

    g_lam_vals = build_g_lambda_vals(
        pc.u_grid, pc.g_vals, lam, pc.std_ref.quantile, pc.std_ref.mu_F
    )
    hull_u, hull_f, _ = least_concave_majorant(pc.u_grid, g_lam_vals)
    a_lam = compute_a_lambda(0.0, pc.metric.g_val_at_1, lam, pc.std_ref.mu_F)
    b_lam = compute_b_lambda(hull_u, hull_f, a_lam)

    if b_lam < 1e-12:
        return 0.0, "active", float(lam)

    def h_fn(u: np.ndarray) -> np.ndarray:
        return construct_h_lambda(hull_u, hull_f, a_lam, b_lam, 0.0, 1.0, u)

    wc_val = float(eval_rho_direct(pc.metric, h_fn))
    return wc_val, "active", float(lam)


def _fast_portfolio_objective(
    w1: float,
    scenario: PortfolioScenario,
    epsilon: float,
    pc: _PortfolioPrecomputed,
    tol: float = 1e-7,
) -> Tuple[float, str, float]:
    """
    Fast portfolio objective using precomputed shared state.

    J(w; epsilon) = m_w * g(1) + s_w * sup rho_g over M_{epsilon_w}(0,1)

    Parameters
    ----------
    w1 : float
    scenario : PortfolioScenario
    epsilon : float  Global radius squared.
    pc : _PortfolioPrecomputed
    tol : float

    Returns
    -------
    (objective, regime, lambda_w)
    """
    w = np.array([w1, 1.0 - w1])
    m_w = float(w @ scenario.mu_vec)
    s_w2 = float(w @ scenario.cov_mat @ w)
    s_w = float(np.sqrt(max(s_w2, 1e-15)))
    norm_w2 = float(w @ w)

    epsilon_w = epsilon * norm_w2 / max(s_w2, 1e-15)

    sup_rho, regime, lam = _fast_wc_value(epsilon_w, pc, tol=tol)

    if regime == "infeasible" or not np.isfinite(sup_rho):
        return float("inf"), "infeasible", np.nan

    objective = m_w * pc.metric.g_val_at_1 + s_w * sup_rho
    return float(objective), regime, float(lam)


def run_portfolio_experiment(
    scenarios: List[PortfolioScenario],
    metrics: List[DistortionMetric],
    epsilon_grid: np.ndarray,
    n_w_grid: int = 20,
    n_grid: int = 120,
    n_gauss: int = 60,
) -> List[dict]:
    """
    Run the full portfolio optimization experiment for all scenarios, metrics, and epsilons.

    Uses precomputed shared state (_PortfolioPrecomputed) to avoid redundant
    LCM and threshold computations across (w1, epsilon) sweeps.

    Parameters
    ----------
    scenarios : List[PortfolioScenario]
    metrics : List[DistortionMetric]
    epsilon_grid : np.ndarray
    n_w_grid : int  Coarse w1 grid size for minimization.
    n_grid, n_gauss : int  Grid sizes for LCM and quadrature.

    Returns
    -------
    List of result dicts with columns:
    scenario, metric, epsilon, w1_star, w2_star, objective, regime, lambda_w
    """
    results = []
    w1_candidates = np.linspace(0.0, 1.0, n_w_grid)

    for scenario in scenarios:
        std_ref = _get_standardized_reference(scenario)
        for metric in metrics:
            print(f"  Precomputing: Scenario={scenario.name}, Metric={metric.name}")
            try:
                pc = _precompute_portfolio(metric, std_ref, n_grid=n_grid, n_gauss=n_gauss)
            except Exception as e:
                warnings.warn(f"Precompute failed for {scenario.name}/{metric.name}: {e}")
                for eps in epsilon_grid:
                    results.append({
                        "scenario": scenario.name, "metric": metric.name,
                        "epsilon": float(eps), "w1_star": None, "w2_star": None,
                        "objective": None, "regime": "error", "lambda_w": None,
                    })
                continue

            print(f"    eps_min={pc.epsilon_min:.4f}, eps_switch={pc.epsilon_switch:.4f}")

            for eps in epsilon_grid:
                try:
                    # Coarse sweep using fast objective
                    objectives = []
                    for w1 in w1_candidates:
                        try:
                            obj, _, _ = _fast_portfolio_objective(w1, scenario, eps, pc)
                        except Exception:
                            obj = float("inf")
                        objectives.append(obj)

                    finite_objs = [o if np.isfinite(o) else 1e18 for o in objectives]
                    best_idx = int(np.nanargmin(finite_objs))
                    best_w1 = w1_candidates[best_idx]

                    # Refine with bounded scalar minimization
                    lo = max(0.0, best_w1 - 2.0 / n_w_grid)
                    hi = min(1.0, best_w1 + 2.0 / n_w_grid)
                    try:
                        res = minimize_scalar(
                            lambda w1: _fast_portfolio_objective(w1, scenario, eps, pc)[0],
                            bounds=(lo, hi),
                            method="bounded",
                            options={"xatol": 1e-3},
                        )
                        w1_star = float(res.x)
                        obj_star = float(res.fun)
                    except Exception:
                        w1_star = best_w1
                        obj_star = objectives[best_idx]

                    # Regime at optimum
                    _, regime_star, lam_star = _fast_portfolio_objective(
                        w1_star, scenario, eps, pc
                    )

                    results.append({
                        "scenario": scenario.name,
                        "metric": metric.name,
                        "epsilon": float(eps),
                        "w1_star": float(w1_star),
                        "w2_star": float(1.0 - w1_star),
                        "objective": float(obj_star),
                        "regime": regime_star,
                        "lambda_w": float(lam_star) if np.isfinite(lam_star) else None,
                    })

                except Exception as e:
                    warnings.warn(
                        f"Failed: scenario={scenario.name}, metric={metric.name}, eps={eps}: {e}"
                    )
                    results.append({
                        "scenario": scenario.name, "metric": metric.name,
                        "epsilon": float(eps), "w1_star": None, "w2_star": None,
                        "objective": None, "regime": "error", "lambda_w": None,
                    })

    return results
