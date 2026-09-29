# Reproduction of Sharp Univariate Mean-Variance-Wasserstein Bounds for Distortion Risk Metrics

## Overview

This repository reproduces and validates the theoretical results of the paper on sharp bounds for distortion risk metrics under mean, variance, and 2-Wasserstein distance constraints.  Three experiments are implemented:

| Experiment | Coverage |
|---|---|
| **Exp 1** | Theorems 1 & 2, Corollaries 1 & 2 — univariate worst-case and best-case bounds |
| **Exp 2** | Propositions 2–4, Theorem 3 — unimodality-constrained bounds and step-function projection |
| **Exp 3** | Propositions 6–7, Corollaries 3–6 — robust portfolio optimization under multivariate Wasserstein ambiguity |

## Repository Structure

```
repo/
├── data/               # Placeholder; no external market data used
├── src/                # Core implementations
│   ├── distortion_metrics.py    # GD, MMD, ES, VaR+, IQD+, GlueVaR registry
│   ├── reference_distributions.py  # Gaussian / Student-t quantile wrappers
│   ├── concave_envelope.py      # Least-concave-majorant, b_λ, h_λ construction
│   ├── wasserstein_bounds.py    # Theorem 1 & 2 solver, benchmarks, validation
│   ├── unimodal_projection.py   # Cone projection (Propositions 2–4, Theorem 3)
│   └── portfolio_optimization.py # Multivariate → univariate reduction (Exp 3)
├── exp/                # Experiment scripts
│   ├── exp_1_univariate_bounds.py
│   ├── exp_2_unimodal_bounds.py
│   └── exp_3_portfolio_optimization.py
├── tests/
│   └── test_all.py     # 89-test pytest suite
├── results/            # Saved metrics, plots, and RESULTS.md
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

No API key is needed — all computations are purely synthetic/mathematical.

## Running Tests

```bash
pytest tests/ -q
```

All 89 tests should pass.

## Running Experiments

```bash
# Experiment 1 — univariate bounds (~1–2 min)
python exp/exp_1_univariate_bounds.py

# Experiment 2 — unimodal projection (~1–2 min)
python exp/exp_2_unimodal_bounds.py

# Experiment 3 — portfolio optimization (~2–3 min)
python exp/exp_3_portfolio_optimization.py
```

Results (CSV, JSON, PNG) are written to `results/`.  A summary is in `results/RESULTS.md`.

## Key Implementation Notes

- **No off-the-shelf package** covers the paper-specific distortion-Wasserstein formulas; all core routines are custom.
- `scipy.stats.norm.ppf` / `scipy.stats.t.ppf` are used for reference quantiles.  Student-t is rescaled by `sqrt((ν-2)/ν)` to achieve unit variance.
- Concave envelopes are computed via an upper-convex-hull / least-concave-majorant algorithm on a dense grid.
- Scalar root-finding (`scipy.optimize.brentq`) solves for λ_ε in the active Wasserstein regime.
- The direct benchmark uses `scipy.optimize.minimize` with SLSQP over discretised monotone quantiles.
- Unimodal projections are formulated as quadratic programs solved with `cvxpy` / OSQP.
