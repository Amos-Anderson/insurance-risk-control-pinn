# Insurance Risk Control with Reinsurance using Physics-Informed Neural Networks

This repository implements a stochastic control framework for an insurance company that jointly optimizes investment and reinsurance decisions under jump–diffusion surplus dynamics.

The problem is formulated as a Hamilton–Jacobi–Bellman (HJB) equation with nonlocal jump terms and solved numerically using Physics-Informed Neural Networks (PINNs).

## Key Features
- Jump–diffusion surplus model with compound Poisson claims
- Proportional and excess-of-loss reinsurance structures
- Exponential (CARA) utility maximization
- Nonlinear HJB equations with expectation operators
- PINN-based solution using PyTorch and automatic differentiation
- Gaussian quadrature for nonlocal jump terms
- Comparative analysis across claim-size distributions

## Model Overview
The insurer controls:
- Investment in a risky asset
- Reinsurance strategy (proportional or deductible-based)

We solve:
$$
\max_{\pi_t, \rho_t} \mathbb{E}[U(X_T)]
$$
subject to stochastic surplus dynamics with jumps.

## Numerical Experiments
### Experiment 1 — Claim Distribution Effects
- Exponential
- Pareto (heavy-tailed)
- Lognormal

### Experiment 2 — Reinsurance Structure
- Proportional vs Excess-of-Loss (XoL)
- Fixed claim distribution

## Key Findings
- Heavy-tailed claims induce more conservative investment and higher reinsurance
- Excess-of-loss reinsurance leads to lower optimal investment and lower certainty-equivalent utility
- Optimal XoL deductibles converge to stable interior values
- PINNs successfully handle nonlocal HJB equations that are difficult for grid-based solvers

## Repository Structure
- `src/`: core model and PINN implementation
- `experiments/`: numerical experiments
- `paper/`: final report and slides
- `results/`: generated figures and logs

## Tools
- Python, PyTorch
- Automatic differentiation
- Gaussian quadrature
- Physics-Informed Neural Networks

## Author
**Amos Anderson**  
MS, Quantitative Finance  
Stony Brook University
