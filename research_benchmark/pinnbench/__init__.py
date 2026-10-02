"""pinnbench -- a small, documented PyTorch implementation of the low-dimensional PDE problems of the
article, built for a seeded, multi-configuration benchmark.

Modules
-------
problems   : PDE definitions (domain, residual, boundary/initial data, exact solution,
             held-out evaluation grid) for heat1d, wave1d, wave2d, laplace2d, laplace3d.
samplers   : interior collocation strategies at matched budgets: grid, random (fixed i.i.d.),
             resample (fresh i.i.d. every step), sobol (scrambled Sobol), rad (residual-based
             adaptive distribution, Wu et al. 2023).
core       : MLP, derivative helpers, loss assembly, training (Adam and Adam->L-BFGS branch),
             metrics (relative L2 / L_inf on held-out grids).
classical  : finite-difference baselines with convergence studies.
"""
