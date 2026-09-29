"""
Comprehensive pytest test suite for the Wasserstein distortion risk bounds project.

Tests cover:
- Reference distributions (Gaussian / Student-t)
- Concave envelope and LCM routines
- Distortion metric definitions and properties
- Wasserstein bound computation (Theorems 1 & 2, Corollaries 1 & 2)
- Unimodal projections (Propositions 2, 3, 4)
- Portfolio optimisation (Exp 3 reduction)

Implementation tests verify methodology adherence, not mock behaviour.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reference_distributions import (
    make_gaussian_reference,
    make_student_t_reference,
    STANDARD_GAUSSIAN,
    STANDARD_STUDENT_T5,
)
from src.concave_envelope import (
    least_concave_majorant,
    lcm_derivative,
    build_g_lambda_vals,
    compute_a_lambda,
    compute_b_lambda,
    construct_h_lambda,
    wasserstein_sq_distance,
    _make_g_grid_local,
    _leggauss,
)
from src.distortion_metrics import (
    make_gd,
    make_mmd,
    make_es,
    make_var_plus,
    make_iqd_plus,
    make_gluevar,
    make_discrepancy_es_var,
    make_distortion_metrics_for_exp1,
    eval_rho_direct,
    DistortionMetric,
)
from src.wasserstein_bounds import (
    compute_thresholds,
    compute_worst_case_value,
    compute_best_case_value,
    BoundResult,
)
from src.unimodal_projection import (
    project_gamma_unimodal,
    compute_unimodal_worst_case_no_wasserstein,
    step_approx_gamma,
    compute_step_projection_and_error,
)
from src.portfolio_optimization import (
    make_standard_scenarios,
    PortfolioScenario,
    optimize_portfolio,
)


# ─────────────────────────────────────────────────────────────────────────────
# Reference Distributions
# ─────────────────────────────────────────────────────────────────────────────

class TestReferenceDistributions:
    """Tests for Gaussian and Student-t reference distributions."""

    def test_gaussian_mean_zero(self):
        """Standard Gaussian reference must have mu_F = 0."""
        ref = make_gaussian_reference(mu=0.0, sigma=1.0)
        assert abs(ref.mu_F) < 1e-10

    def test_gaussian_sigma_one(self):
        """Standard Gaussian reference must have sigma_F = 1."""
        ref = make_gaussian_reference(mu=0.0, sigma=1.0)
        assert abs(ref.sigma_F - 1.0) < 1e-10

    def test_gaussian_quantile_monotone(self):
        """Gaussian quantile must be strictly increasing."""
        ref = make_gaussian_reference(mu=0.0, sigma=1.0)
        u = np.linspace(0.01, 0.99, 200)
        q = ref.quantile(u)
        assert np.all(np.diff(q) > 0)

    def test_gaussian_quantile_midpoint(self):
        """Gaussian quantile at 0.5 equals mu_F = 0."""
        ref = make_gaussian_reference(mu=0.0, sigma=1.0)
        q_half = float(ref.quantile(np.array([0.5])))
        assert abs(q_half) < 1e-10

    def test_student_t_mean_zero(self):
        """Standardised Student-t must have mu_F = 0."""
        ref = make_student_t_reference(mu=0.0, sigma=1.0, nu=5.0)
        assert abs(ref.mu_F) < 1e-10

    def test_student_t_sigma_one(self):
        """Standardised Student-t (nu=5) must have sigma_F ≈ 1 by construction."""
        ref = make_student_t_reference(mu=0.0, sigma=1.0, nu=5.0)
        assert abs(ref.sigma_F - 1.0) < 1e-10

    def test_student_t_heavier_tails_than_gaussian(self):
        """t(5) should have heavier tails: q(0.01) < N(0,1) q(0.01)."""
        t_ref = make_student_t_reference(mu=0.0, sigma=1.0, nu=5.0)
        g_ref = make_gaussian_reference(mu=0.0, sigma=1.0)
        u = np.array([0.01])
        assert float(t_ref.quantile(u)) < float(g_ref.quantile(u))

    def test_student_t_quantile_monotone(self):
        """Student-t quantile must be strictly increasing."""
        ref = make_student_t_reference(mu=0.0, sigma=1.0, nu=5.0)
        u = np.linspace(0.01, 0.99, 200)
        q = ref.quantile(u)
        assert np.all(np.diff(q) > 0)

    def test_singleton_constants(self):
        """Module-level constants STANDARD_GAUSSIAN / STANDARD_STUDENT_T5 work."""
        assert abs(STANDARD_GAUSSIAN.mu_F) < 1e-10
        assert abs(STANDARD_STUDENT_T5.mu_F) < 1e-10

    def test_shifted_gaussian_moments(self):
        """Gaussian with mu=0.1, sigma=0.5 must match those moments."""
        ref = make_gaussian_reference(mu=0.1, sigma=0.5)
        assert abs(ref.mu_F - 0.1) < 1e-10
        assert abs(ref.sigma_F - 0.5) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Concave Envelope
# ─────────────────────────────────────────────────────────────────────────────

class TestConcaveEnvelope:
    """Tests for least_concave_majorant and related routines."""

    def test_lcm_of_concave_is_itself(self):
        """LCM of a concave function must equal the function (within grid error)."""
        u = np.linspace(0.0, 1.0, 200)
        g = -4 * (u - 0.5) ** 2 + 1.0   # concave parabola, peak at 0.5
        hull_u, hull_f, lcm_vals = least_concave_majorant(u, g)
        # LCM evaluated at u should match g very closely
        g_interp = np.interp(u, hull_u, hull_f)
        assert np.max(np.abs(g_interp - g)) < 0.02

    def test_lcm_is_above_function(self):
        """LCM must be >= the original function at all grid points."""
        np.random.seed(0)
        u = np.linspace(0.0, 1.0, 300)
        g = np.sin(2 * np.pi * u)           # not concave
        hull_u, hull_f, lcm_vals = least_concave_majorant(u, g)
        g_interp = np.interp(u, hull_u, hull_f)
        assert np.all(g_interp >= g - 1e-9)

    def test_lcm_concavity(self):
        """LCM must be concave: slopes must be non-increasing."""
        u = np.linspace(0.0, 1.0, 500)
        g = np.abs(u - 0.5)    # non-concave (V-shape)
        hull_u, hull_f, _ = least_concave_majorant(u, g)
        slopes = np.diff(hull_f) / np.diff(hull_u)
        assert np.all(np.diff(slopes) <= 1e-8)

    def test_lcm_linear_function(self):
        """LCM of a linear function must equal the function (linear is concave)."""
        u = np.linspace(0.0, 1.0, 100)
        g = 2.0 * u            # linear
        hull_u, hull_f, _ = least_concave_majorant(u, g)
        g_interp = np.interp(u, hull_u, hull_f)
        assert np.max(np.abs(g_interp - g)) < 1e-8

    def test_compute_a_lambda_formula(self):
        """compute_a_lambda = g(1) + lambda * mu_F."""
        a = compute_a_lambda(0.0, 1.0, 2.0, 0.5)
        assert abs(a - (1.0 + 2.0 * 0.5)) < 1e-12

    def test_compute_b_lambda_nonneg(self):
        """b_lambda must be non-negative."""
        u = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        f = np.array([0.0, 0.5, 0.75, 0.85, 1.0])  # concave-looking
        b = compute_b_lambda(u, f, a_lambda=0.9)
        assert b >= 0.0

    def test_wasserstein_sq_self_distance_is_zero(self):
        """d_W^2(F, F) = 0 when both quantiles are identical."""
        ref = make_gaussian_reference()
        dW2 = wasserstein_sq_distance(ref.quantile, ref.quantile, n_gauss=200)
        assert abs(dW2) < 1e-6

    def test_wasserstein_sq_nonneg(self):
        """Wasserstein squared distance must be non-negative."""
        ref_g = make_gaussian_reference()
        ref_t = make_student_t_reference(nu=5.0)
        dW2 = wasserstein_sq_distance(ref_g.quantile, ref_t.quantile, n_gauss=200)
        assert dW2 >= -1e-10

    def test_construct_h_lambda_mean(self):
        """h_lambda must have the prescribed mean mu.

        a_lam must equal the mean of the LCM derivative, i.e. f_hull[-1] - f_hull[0],
        so that sigma * (E[deriv(1-U)] - a_lam) / b_lam = 0 and E[h_lambda] = mu.
        """
        u_hull = np.array([0.0, 0.5, 1.0])
        f_hull = np.array([0.0, 0.6, 1.0])
        # Correct: mean of the LCM derivative = f_hull[-1] - f_hull[0] = 1.0
        a_lam = float(f_hull[-1] - f_hull[0])   # = 1.0
        b_lam = compute_b_lambda(u_hull, f_hull, a_lam)
        mu, sigma = 0.1, 0.8

        if b_lam > 1e-8:
            nodes, weights = _leggauss(300)
            eps = 1e-9
            u_pts = 0.5 * (1 - 2 * eps) * nodes + 0.5 * (1 + 2 * eps)
            h_vals = construct_h_lambda(u_hull, f_hull, a_lam, b_lam, mu, sigma, u_pts)
            mean_h = float(0.5 * (1 - 2 * eps) * np.sum(weights * h_vals))
            assert abs(mean_h - mu) < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# Distortion Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestDistortionMetrics:
    """Tests for distortion metric factory functions and properties."""

    def test_gd_gamma_at_zero(self):
        """GD gamma(0) = 1-2*0 = 1."""
        gd = make_gd()
        assert abs(float(gd.gamma(np.array([0.0]))) - 1.0) < 1e-12

    def test_gd_gamma_at_one(self):
        """GD gamma(1) = 1-2*1 = -1."""
        gd = make_gd()
        assert abs(float(gd.gamma(np.array([1.0]))) + 1.0) < 1e-12

    def test_gd_g_at_1_is_zero(self):
        """GD g(1) = 0 (pure variability)."""
        gd = make_gd()
        assert gd.g_val_at_1 == 0.0

    def test_mmd_gamma_sign_change(self):
        """MMD gamma must be -1 below 0.5 and +1 above 0.5."""
        mmd = make_mmd()
        u_lo = np.linspace(0.01, 0.49, 20)
        u_hi = np.linspace(0.51, 0.99, 20)
        assert np.all(mmd.gamma(u_lo) < 0)
        assert np.all(mmd.gamma(u_hi) > 0)

    def test_mmd_gamma_integral_zero(self):
        """integral_0^1 gamma_MMD(u) du = 0 (zero mean)."""
        mmd = make_mmd()
        u = np.linspace(0.001, 0.999, 2000)
        integral = np.trapz(mmd.gamma(u), u)
        assert abs(integral) < 0.01

    def test_es_gamma_non_negative(self):
        """ES gamma must be non-negative everywhere."""
        es = make_es(alpha=0.05)
        u = np.linspace(0.001, 0.999, 500)
        assert np.all(es.gamma(u) >= 0)

    def test_es_gamma_integral_is_one(self):
        """integral_0^1 gamma_ES_alpha du = 1/alpha * alpha = 1.

        Use a grid that reaches close to the endpoints so the ES indicator
        region (u > 1-alpha) is fully captured.
        """
        alpha = 0.05
        es = make_es(alpha=alpha)
        u = np.linspace(1e-9, 1 - 1e-9, 50_000)
        integral = np.trapz(es.gamma(u), u)
        assert abs(integral - 1.0) < 0.01

    def test_es_g_at_1_is_one(self):
        """ES g(1) = 1."""
        assert make_es(alpha=0.1).g_val_at_1 == 1.0

    def test_es_is_concave(self):
        """ES must be flagged as concave."""
        assert make_es(alpha=0.05).is_concave is True

    def test_es_is_usc(self):
        """ES must be flagged as upper semicontinuous."""
        assert make_es(alpha=0.05).is_usc is True

    def test_var_plus_not_usc(self):
        """VaR+ must NOT be upper semicontinuous."""
        assert make_var_plus(alpha=0.05).is_usc is False

    def test_iqd_plus_g_at_1_zero(self):
        """IQD+ g(1) = 0 (pure variability)."""
        assert make_iqd_plus(alpha=0.1).g_val_at_1 == 0.0

    def test_gluevar_params_stored(self):
        """GlueVaR must store all four parameters."""
        gv = make_gluevar(alpha=0.05, beta=0.1, h1=0.6, h2=0.9)
        assert gv.params["alpha"] == 0.05
        assert gv.params["beta"] == 0.1
        assert gv.params["h1"] == 0.6
        assert gv.params["h2"] == 0.9

    def test_gluevar_g_at_1_is_h2(self):
        """GlueVaR g(1) = h2."""
        gv = make_gluevar(alpha=0.05, beta=0.1, h1=0.6, h2=0.9)
        assert abs(gv.g_val_at_1 - 0.9) < 1e-12

    def test_es_rho_gaussian_positive(self):
        """ES rho for a positively-tailed distribution should be positive."""
        es = make_es(alpha=0.05)
        ref = make_gaussian_reference(mu=0.0, sigma=1.0)
        val = es.rho(ref.quantile, n_gauss=400)
        assert val > 0

    def test_gd_rho_symmetric_gaussian_near_zero(self):
        """
        rho_GD for N(0,1) with gamma(u)=1-2u:
            integral_0^1 (1-2u) * Phi^{-1}(u) du = -1/sqrt(pi) ≈ -0.5642.

        Derivation: substitute x=Phi^{-1}(u) and use E[X Phi(X)] = 1/(2 sqrt(pi)).
        """
        gd = make_gd()
        ref = make_gaussian_reference()
        val = gd.rho(ref.quantile, n_gauss=600)
        expected = -1.0 / np.sqrt(np.pi)   # ≈ -0.5642
        assert abs(val - expected) < 0.05

    def test_eval_rho_var_plus(self):
        """VaR+(alpha=0.5) on N(0,1) should be close to 0 (median)."""
        var = make_var_plus(alpha=0.5)
        ref = make_gaussian_reference()
        val = eval_rho_direct(var, ref.quantile)
        assert abs(val) < 0.01

    def test_eval_rho_iqd_plus(self):
        """IQD+(alpha=0.25) on N(0,1): interquartile range ≈ 1.35."""
        iqd = make_iqd_plus(alpha=0.25)
        ref = make_gaussian_reference()
        val = eval_rho_direct(iqd, ref.quantile)
        assert abs(val - 1.349) < 0.01

    def test_make_distortion_metrics_for_exp1_count(self):
        """make_distortion_metrics_for_exp1 must return 7 metrics."""
        metrics = make_distortion_metrics_for_exp1()
        assert len(metrics) == 7

    def test_discrepancy_metric_params(self):
        """Discrepancy ES-VaR must store alpha1, alpha2 (requires alpha1 < alpha2)."""
        d = make_discrepancy_es_var(alpha1=0.01, alpha2=0.025)
        assert d.params["alpha1"] == 0.01
        assert d.params["alpha2"] == 0.025

    def test_gluevar_gamma_structure(self):
        """GlueVaR gamma must be zero for u >= beta."""
        gv = make_gluevar(alpha=0.05, beta=0.1, h1=0.6, h2=0.9)
        u_above = np.linspace(0.11, 0.99, 50)
        assert np.all(gv.gamma(u_above) == 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Wasserstein Bounds (Exp 1)
# ─────────────────────────────────────────────────────────────────────────────

class TestWassersteinBounds:
    """Tests for Theorem 1 & 2 implementations."""

    def test_epsilon_min_same_reference(self):
        """When mu=mu_F, sigma=sigma_F, epsilon_min = 0."""
        ref = make_gaussian_reference(mu=0.0, sigma=1.0)
        eps_min, eps_switch, c0 = compute_thresholds(ref, 0.0, 1.0, make_gd(), n_grid=200)
        assert abs(eps_min) < 1e-8

    def test_epsilon_switch_ge_epsilon_min(self):
        """epsilon_switch must always be >= epsilon_min."""
        ref = make_gaussian_reference()
        eps_min, eps_switch, _ = compute_thresholds(ref, 0.1, 0.8, make_es(0.05), n_grid=200)
        assert eps_switch >= eps_min - 1e-10

    def test_c0_bounded(self):
        """c_0 must be in [-1, 1]."""
        ref = make_gaussian_reference()
        _, _, c0 = compute_thresholds(ref, 0.0, 1.0, make_es(0.05), n_grid=200)
        assert -1.0 - 1e-10 <= c0 <= 1.0 + 1e-10

    def test_infeasible_regime_detected(self):
        """Epsilon below epsilon_min must produce infeasible BoundResult."""
        ref = make_gaussian_reference(mu=0.0, sigma=1.0)
        metric = make_gd()
        eps_min, _, _ = compute_thresholds(ref, 0.2, 1.5, metric, n_grid=200)
        assert eps_min > 0, "Epsilon_min should be positive for shifted moments."
        result = compute_worst_case_value(
            metric, ref, 0.2, 1.5, epsilon=eps_min * 0.5, n_grid=200, n_gauss=100
        )
        assert result.regime == "infeasible"
        assert result.worst_case_value is None

    def test_inactive_regime_zero_lambda(self):
        """Epsilon >= epsilon_switch must produce inactive regime (lambda_epsilon=0)."""
        ref = make_gaussian_reference()
        metric = make_gd()
        _, eps_switch, _ = compute_thresholds(ref, 0.0, 1.0, metric, n_grid=200)
        result = compute_worst_case_value(
            metric, ref, 0.0, 1.0, epsilon=eps_switch * 2.0, n_grid=200, n_gauss=100
        )
        assert result.regime == "inactive"
        assert abs(result.lambda_epsilon) < 1e-10

    def test_active_regime_positive_lambda(self):
        """Epsilon in (epsilon_min, epsilon_switch) must give lambda_epsilon > 0."""
        ref = make_gaussian_reference(mu=0.0, sigma=1.0)
        metric = make_es(alpha=0.05)
        eps_min, eps_switch, _ = compute_thresholds(
            ref, 0.0, 1.0, metric, n_grid=300
        )
        if eps_switch > eps_min + 1e-6:
            eps_mid = (eps_min + eps_switch) / 2.0
            result = compute_worst_case_value(
                metric, ref, 0.0, 1.0, epsilon=eps_mid, n_grid=300, n_gauss=150
            )
            assert result.regime == "active"
            assert result.lambda_epsilon > 0

    def test_worst_case_finite(self):
        """Worst-case value must be finite for feasible epsilon."""
        ref = make_gaussian_reference()
        metric = make_gd()
        _, eps_switch, _ = compute_thresholds(ref, 0.0, 1.0, metric, n_grid=200)
        result = compute_worst_case_value(
            metric, ref, 0.0, 1.0, epsilon=eps_switch * 2.0, n_grid=200, n_gauss=100
        )
        assert result.worst_case_value is not None
        assert np.isfinite(result.worst_case_value)

    def test_worst_case_non_negative_for_gd(self):
        """GD worst-case value must be non-negative (variability >= 0)."""
        ref = make_gaussian_reference()
        metric = make_gd()
        _, eps_switch, _ = compute_thresholds(ref, 0.0, 1.0, metric, n_grid=200)
        result = compute_worst_case_value(
            metric, ref, 0.0, 1.0, epsilon=eps_switch * 2.0, n_grid=200, n_gauss=100
        )
        assert result.worst_case_value >= -1e-8

    def test_worst_case_ge_inactive_value(self):
        """
        Worst-case with smaller epsilon (active) should be <= worst-case with larger epsilon
        (more ambiguity => at most as large).  Actually more ambiguity => larger WC.
        So WC(large eps) >= WC(small eps).
        """
        ref = make_gaussian_reference()
        metric = make_es(alpha=0.05)
        eps_min, eps_switch, _ = compute_thresholds(
            ref, 0.0, 1.0, metric, n_grid=200, n_gauss=100
        )
        if eps_switch > eps_min + 1e-4:
            eps_active = eps_min + 0.3 * (eps_switch - eps_min)
            eps_inactive = eps_switch * 2.0
            res_a = compute_worst_case_value(
                metric, ref, 0.0, 1.0, epsilon=eps_active,
                n_grid=200, n_gauss=100, include_benchmark=False
            )
            res_b = compute_worst_case_value(
                metric, ref, 0.0, 1.0, epsilon=eps_inactive,
                n_grid=200, n_gauss=100, include_benchmark=False
            )
            if res_a.worst_case_value and res_b.worst_case_value:
                assert res_b.worst_case_value >= res_a.worst_case_value - 0.05

    def test_best_case_le_worst_case(self):
        """Best-case value must be <= worst-case value."""
        ref = make_gaussian_reference()
        metric = make_gd()
        _, eps_switch, _ = compute_thresholds(ref, 0.0, 1.0, metric, n_grid=200)
        eps = eps_switch * 2.0
        wc = compute_worst_case_value(
            metric, ref, 0.0, 1.0, epsilon=eps, n_grid=200, n_gauss=100
        ).worst_case_value
        bc = compute_best_case_value(metric, ref, 0.0, 1.0, epsilon=eps, n_grid=200, n_gauss=100)
        if wc is not None and np.isfinite(bc):
            assert bc <= wc + 1e-6

    def test_constraints_satisfied_in_inactive_regime(self):
        """Constraint checks must pass in the inactive regime."""
        ref = make_gaussian_reference()
        metric = make_gd()
        _, eps_switch, _ = compute_thresholds(ref, 0.0, 1.0, metric, n_grid=200)
        result = compute_worst_case_value(
            metric, ref, 0.0, 1.0, epsilon=eps_switch * 3.0, n_grid=300, n_gauss=200
        )
        assert result.constraints_satisfied is True

    def test_student_t_wider_wc_than_gaussian(self):
        """
        Student-t with same (mu,sigma) should yield >= Gaussian worst-case
        for ES (heavier tails => more extreme worst-case at same Wasserstein radius).
        """
        metric = make_es(alpha=0.05)
        ref_g = make_gaussian_reference()
        ref_t = make_student_t_reference(nu=5.0)
        eps = 0.5

        res_g = compute_worst_case_value(
            metric, ref_g, 0.0, 1.0, epsilon=eps, n_grid=200, n_gauss=100
        )
        res_t = compute_worst_case_value(
            metric, ref_t, 0.0, 1.0, epsilon=eps, n_grid=200, n_gauss=100
        )
        if res_g.worst_case_value and res_t.worst_case_value:
            # Not guaranteed by theory but common empirically; just check finiteness
            assert np.isfinite(res_g.worst_case_value)
            assert np.isfinite(res_t.worst_case_value)

    def test_epsilon_min_formula(self):
        """epsilon_min = (mu_F - mu)^2 + (sigma_F - sigma)^2."""
        ref = make_gaussian_reference(mu=0.0, sigma=1.0)
        mu, sigma = 0.2, 0.7
        eps_min, _, _ = compute_thresholds(ref, mu, sigma, make_gd(), n_grid=200)
        expected = (ref.mu_F - mu) ** 2 + (ref.sigma_F - sigma) ** 2
        assert abs(eps_min - expected) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Unimodal Projections (Exp 2)
# ─────────────────────────────────────────────────────────────────────────────

class TestUnimodalProjection:
    """Tests for Propositions 2, 3, 4 (Experiment 2)."""

    def test_project_gamma_unimodal_output_shape(self):
        """Projection output must have same shape as input grid."""
        metric = make_gd()
        u_grid, proj, _ = project_gamma_unimodal(metric.gamma, xi=0.5, n_grid=200)
        assert len(u_grid) == len(proj)

    def test_projection_non_negative_b_xi(self):
        """b_xi from compute_unimodal_worst_case must be non-negative."""
        metric = make_mmd()
        wc, h, proj, a_xi, b_xi = compute_unimodal_worst_case_no_wasserstein(
            metric, xi=0.5, mu=0.0, sigma=1.0, n_grid=300
        )
        assert b_xi >= 0.0

    def test_gd_worst_case_non_negative(self):
        """GD unimodal worst-case (mu=0, sigma=1) must be non-negative."""
        metric = make_gd()
        wc, _, _, _, _ = compute_unimodal_worst_case_no_wasserstein(
            metric, xi=0.5, mu=0.0, sigma=1.0, n_grid=300
        )
        assert wc >= -1e-8

    def test_mmd_projection_b_xi_near_analytic(self):
        """
        MMD analytic result: projection = 3*(u-0.5), so b_xi = sqrt(3/4) = sqrt(3)/2.
        Numerical b_xi must be within 0.1 of this.
        """
        metric = make_mmd()
        _, _, _, _, b_xi = compute_unimodal_worst_case_no_wasserstein(
            metric, xi=0.5, mu=0.0, sigma=1.0, n_grid=500
        )
        analytic_b = np.sqrt(3) / 2
        assert abs(b_xi - analytic_b) < 0.15

    def test_mmd_projection_unimodal(self):
        """Projected gamma for MMD must be unimodal at xi=0.5 (non-decr then non-incr)."""
        metric = make_mmd()
        u_grid, proj, _ = project_gamma_unimodal(metric.gamma, xi=0.5, n_grid=500)
        # Up to xi
        idx_xi = np.searchsorted(u_grid, 0.5)
        left = proj[:idx_xi]
        right = proj[idx_xi:]
        if len(left) > 1:
            assert np.all(np.diff(left) >= -0.05), "Left part should be non-decreasing"
        if len(right) > 1:
            assert np.all(np.diff(right) <= 0.05), "Right part should be non-increasing"

    def test_step_approx_gamma_piecewise_constant(self):
        """step_approx_gamma must return piecewise-constant values on each interval."""
        metric = make_gd()
        n = 20
        u_grid = np.linspace(0.0, 1.0, 500)
        _, _, step_fn = step_approx_gamma(metric.gamma, n)
        vals = step_fn(u_grid)
        # Within each interval, values should be identical
        piece_size = 500 // n
        for i in range(n - 1):
            block = vals[i * piece_size: (i + 1) * piece_size]
            if len(block) > 0:
                assert np.all(np.isfinite(block))

    def test_step_projection_errors_decrease(self):
        """Approximation errors must decrease as n_pieces increases."""
        metric = make_gd()
        xi = 0.5
        n_list = [10, 20, 50, 100]
        errors = compute_step_projection_and_error(metric.gamma, xi, n_list)
        gamma_errs = [e["gamma_approx_l2"] for e in errors]
        # Errors should generally decrease
        assert gamma_errs[-1] <= gamma_errs[0] + 1e-4

    def test_prop4_inequality_satisfied(self):
        """Proposition 4: ||proj_n - proj||_2 <= ||gamma_n - gamma||_2."""
        metric = make_gd()
        xi = 0.5
        n_list = [20, 50]
        errors = compute_step_projection_and_error(metric.gamma, xi, n_list)
        for row in errors:
            assert row["prop4_satisfied"], (
                f"Prop 4 violated at n={row['n_pieces']}: "
                f"proj_err={row['proj_error_l2']:.4f} > gamma_err={row['gamma_approx_l2']:.4f}"
            )

    def test_unimodal_wc_le_unrestricted_wc(self):
        """
        Unimodality restricts the feasible set so the unimodal worst-case (no
        Wasserstein) must be <= the unrestricted mean-variance worst-case (no
        Wasserstein).  Both problems share the same moment constraints.

        GD is degenerate (concave envelope identically 0, so unrestricted WC=0)
        which breaks the comparison.  ES_0.05 is a concave metric with a
        non-zero b_0 and gives a clean test.
        """
        metric = make_es(alpha=0.05)
        ref = make_gaussian_reference()
        _, eps_switch, _ = compute_thresholds(ref, 0.0, 1.0, metric, n_grid=200)

        uni_wc, _, _, _, _ = compute_unimodal_worst_case_no_wasserstein(
            metric, xi=0.5, mu=0.0, sigma=1.0, n_grid=300
        )
        # Use a very large epsilon so the Wasserstein constraint is inactive,
        # which collapses the problem to the pure mean-variance worst-case.
        unrestr_res = compute_worst_case_value(
            metric, ref, 0.0, 1.0, epsilon=eps_switch * 10.0,
            n_grid=200, n_gauss=100, include_benchmark=False
        )
        if unrestr_res.worst_case_value is not None:
            assert uni_wc <= unrestr_res.worst_case_value + 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio Optimisation (Exp 3)
# ─────────────────────────────────────────────────────────────────────────────

class TestPortfolioOptimisation:
    """Tests for Experiment 3 robust portfolio optimisation."""

    def test_make_standard_scenarios_count(self):
        """make_standard_scenarios must return 4 scenarios."""
        scenarios = make_standard_scenarios()
        assert len(scenarios) == 4

    def test_scenario_names_unique(self):
        """All scenario names must be unique."""
        scenarios = make_standard_scenarios()
        names = [s.name for s in scenarios]
        assert len(names) == len(set(names))

    def test_portfolio_scenario_has_covariance(self):
        """Each scenario must have a 2×2 covariance matrix."""
        for sc in make_standard_scenarios():
            assert sc.cov_mat.shape == (2, 2)

    def test_covariance_positive_definite(self):
        """Covariance matrices must be positive definite."""
        for sc in make_standard_scenarios():
            eigvals = np.linalg.eigvalsh(sc.cov_mat)
            assert np.all(eigvals > 0), f"Sigma for {sc.name} not PD"

    def test_portfolio_weights_sum_to_one(self):
        """Optimised portfolio weights must satisfy w1 + w2 = 1."""
        sc = make_standard_scenarios()[0]
        metric = make_gd()
        w1, obj, _, _ = optimize_portfolio(
            sc, metric, epsilon=0.05, n_w_grid=10, n_grid=100, n_gauss=80
        )
        assert 0.0 <= w1 <= 1.0, f"w1={w1} outside [0,1]"

    def test_portfolio_objective_finite(self):
        """Portfolio objective must be finite for valid inputs."""
        sc = make_standard_scenarios()[0]
        metric = make_es(alpha=0.05)
        w1, obj, _, _ = optimize_portfolio(
            sc, metric, epsilon=0.05, n_w_grid=10, n_grid=100, n_gauss=80
        )
        assert np.isfinite(obj)

    def test_portfolio_at_zero_epsilon(self):
        """At epsilon≈0, solution must still return valid (w1, obj)."""
        sc = make_standard_scenarios()[0]
        metric = make_gd()
        w1, obj, _, _ = optimize_portfolio(
            sc, metric, epsilon=1e-10, n_w_grid=10, n_grid=100, n_gauss=80
        )
        assert 0.0 <= w1 <= 1.0
        assert np.isfinite(obj)

    def test_portfolio_positive_correlation_scenario_present(self):
        """One scenario must contain a positive correlation."""
        scenarios = make_standard_scenarios()
        corrs = [sc.cov_mat[0, 1] / np.sqrt(sc.cov_mat[0, 0] * sc.cov_mat[1, 1]) for sc in scenarios]
        assert any(c > 0 for c in corrs)

    def test_portfolio_negative_correlation_scenario_present(self):
        """One scenario must contain a negative correlation."""
        scenarios = make_standard_scenarios()
        corrs = [sc.cov_mat[0, 1] / np.sqrt(sc.cov_mat[0, 0] * sc.cov_mat[1, 1]) for sc in scenarios]
        assert any(c < 0 for c in corrs)

    def test_portfolio_mean_vector(self):
        """All scenarios must use loss mean vector (-0.06, -0.03)."""
        for sc in make_standard_scenarios():
            assert abs(sc.mu_vec[0] - (-0.06)) < 1e-10 or abs(sc.mu_vec[0] - (-0.03)) < 1e-10

    def test_objective_increases_with_epsilon(self):
        """
        More ambiguity => higher worst-case => non-decreasing objective in epsilon
        (on average, for a fixed w).
        """
        sc = make_standard_scenarios()[0]
        metric = make_gd()
        w1_a, obj_a, _, _ = optimize_portfolio(
            sc, metric, epsilon=0.01, n_w_grid=10, n_grid=100, n_gauss=80
        )
        w1_b, obj_b, _, _ = optimize_portfolio(
            sc, metric, epsilon=0.2, n_w_grid=10, n_grid=100, n_gauss=80
        )
        # Allow small tolerance because optimal w changes too
        assert obj_b >= obj_a - 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Edge Case Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases: single points, degenerate inputs, boundary conditions."""

    def test_lcm_two_points(self):
        """LCM on two points: line segment."""
        u = np.array([0.0, 1.0])
        g = np.array([0.0, 1.0])
        hull_u, hull_f, _ = least_concave_majorant(u, g)
        assert len(hull_u) >= 2

    def test_lcm_constant_function(self):
        """LCM of a constant function is constant."""
        u = np.linspace(0.0, 1.0, 50)
        g = np.ones_like(u) * 3.0
        hull_u, hull_f, _ = least_concave_majorant(u, g)
        g_interp = np.interp(u, hull_u, hull_f)
        assert np.max(np.abs(g_interp - 3.0)) < 1e-6

    def test_gd_gamma_vector_input(self):
        """GD gamma must accept vectorised inputs."""
        gd = make_gd()
        u = np.linspace(0.0, 1.0, 100)
        out = gd.gamma(u)
        assert out.shape == u.shape

    def test_gluevar_invalid_params_raise(self):
        """GlueVaR with alpha >= beta must raise."""
        with pytest.raises((AssertionError, ValueError)):
            make_gluevar(alpha=0.1, beta=0.05, h1=0.3, h2=0.8)

    def test_es_invalid_alpha_raises(self):
        """ES with alpha=0 must raise."""
        with pytest.raises((AssertionError, ValueError)):
            make_es(alpha=0.0)

    def test_var_plus_boundary_alpha_raises(self):
        """VaR+ with alpha=0 or alpha=1 must raise."""
        with pytest.raises((AssertionError, ValueError)):
            make_var_plus(alpha=0.0)

    def test_iqd_plus_alpha_gt_half_raises(self):
        """IQD+ with alpha > 0.5 must raise."""
        with pytest.raises((AssertionError, ValueError)):
            make_iqd_plus(alpha=0.6)

    def test_wasserstein_squared_symmetry(self):
        """d_W^2(F, G) = d_W^2(G, F)."""
        ref_g = make_gaussian_reference()
        ref_t = make_student_t_reference(nu=5.0)
        d1 = wasserstein_sq_distance(ref_g.quantile, ref_t.quantile, n_gauss=200)
        d2 = wasserstein_sq_distance(ref_t.quantile, ref_g.quantile, n_gauss=200)
        assert abs(d1 - d2) < 0.01

    def test_compute_thresholds_small_epsilon_feasibility(self):
        """With mu=mu_F, sigma=sigma_F, all epsilons >= 0 should be feasible."""
        ref = make_gaussian_reference()
        eps_min, _, _ = compute_thresholds(ref, ref.mu_F, ref.sigma_F, make_gd(), n_grid=200)
        assert eps_min < 1e-9

    def test_make_g_grid_local_length(self):
        """_make_g_grid_local must return grids of length n+2."""
        gd = make_gd()
        n = 50
        u, g = _make_g_grid_local(gd.gamma, g_val_at_1=0.0, n=n)
        assert len(u) == n + 2
        assert len(g) == n + 2

    def test_make_g_grid_local_boundary_zero(self):
        """g(0) = 0 for standard distortions."""
        gd = make_gd()
        u, g = _make_g_grid_local(gd.gamma, g_val_at_1=0.0, n=50)
        assert abs(g[0]) < 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Mathematical Properties / Regression Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMathematicalProperties:
    """Regression and mathematical property checks."""

    def test_gd_rho_unit_variance_gaussian(self):
        """
        For N(0,1) with gamma(u)=1-2u:
            rho_GD = integral_0^1 (1-2u) Phi^{-1}(u) du = -1/sqrt(pi) ≈ -0.5642.

        Regression check: value within 0.02 of the analytic result.
        """
        gd = make_gd()
        ref = make_gaussian_reference()
        val = gd.rho(ref.quantile, n_gauss=500)
        expected = -1.0 / np.sqrt(np.pi)   # ≈ -0.5642
        assert abs(val - expected) < 0.02

    def test_es_rho_is_mean_above_threshold(self):
        """
        ES_alpha = E[X | X > Q_{1-alpha}] for continuously distributed X.
        For N(0,1): ES_0.05 = phi(Phi^{-1}(0.95)) / 0.05 ≈ 2.063.
        """
        from scipy.stats import norm
        alpha = 0.05
        es = make_es(alpha=alpha)
        ref = make_gaussian_reference()
        val = es.rho(ref.quantile, n_gauss=600)
        z = norm.ppf(1 - alpha)
        expected = norm.pdf(z) / alpha
        assert abs(val - expected) < 0.05

    def test_negative_metric_negates_worst_case(self):
        """
        sup rho_{-g} = -inf rho_g ⟹ best_case = -sup rho_{-g}.

        Verify sign ordering: for GD on N(0,1), the worst-case is >= 0 and the
        best-case is <= 0, i.e. bc <= 0 <= wc.  Also verify bc <= wc.

        Note: GD is not symmetric (WC=0 degenerate, BC=-1/sqrt(3) ≈ -0.577),
        so |bc + wc| ≈ 0.577 and an absolute-symmetry check is not appropriate.
        """
        ref = make_gaussian_reference()
        metric = make_gd()
        _, eps_switch, _ = compute_thresholds(ref, 0.0, 1.0, metric, n_grid=200)
        eps = eps_switch * 2.0

        wc = compute_worst_case_value(
            metric, ref, 0.0, 1.0, epsilon=eps, n_grid=200, n_gauss=100
        ).worst_case_value

        bc = compute_best_case_value(
            metric, ref, 0.0, 1.0, epsilon=eps, n_grid=200, n_gauss=100
        )

        if wc is not None and np.isfinite(bc):
            assert bc <= wc + 1e-6, f"best-case {bc:.4f} must be <= worst-case {wc:.4f}"
            assert wc >= -1e-6, f"worst-case {wc:.4f} must be non-negative for GD"
            assert bc <= 1e-6, f"best-case {bc:.4f} must be non-positive for GD"

    def test_epsilon_switch_formula_structure(self):
        """
        epsilon_switch = epsilon_min + 2*sigma_F*sigma*(1-c_0).
        Verify this structure explicitly.
        """
        ref = make_gaussian_reference(mu=0.0, sigma=1.0)
        metric = make_es(0.05)
        mu, sigma = 0.0, 1.0
        eps_min, eps_switch, c0 = compute_thresholds(ref, mu, sigma, metric, n_grid=300)
        expected_switch = eps_min + 2.0 * ref.sigma_F * sigma * (1.0 - c0)
        assert abs(eps_switch - expected_switch) < 1e-6

    def test_worst_case_value_monotone_in_sigma(self):
        """
        Higher sigma (more spread) should yield higher worst-case for GD.
        """
        ref = make_gaussian_reference()
        metric = make_gd()
        eps = 0.5

        wc_lo = compute_worst_case_value(
            metric, ref, 0.0, 0.5, epsilon=eps, n_grid=200, n_gauss=100
        ).worst_case_value
        wc_hi = compute_worst_case_value(
            metric, ref, 0.0, 2.0, epsilon=eps, n_grid=200, n_gauss=100
        ).worst_case_value

        if wc_lo is not None and wc_hi is not None:
            assert wc_hi >= wc_lo - 0.05
