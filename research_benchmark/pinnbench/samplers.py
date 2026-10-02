"""Interior collocation strategies at a matched point budget N.

grid      tensor-product, cell-centred grid with N^(1/d) points per axis (N must be a perfect power)
random    N i.i.d. uniform points drawn once (fixed for the whole run)
resample  N fresh i.i.d. uniform points at every Adam step; a fresh fixed draw is used for the L-BFGS phase
sobol     first N points of a scrambled Sobol sequence (scipy.stats.qmc), fixed
rad       residual-based adaptive distribution (Wu, Zhu, Tan, Kartha & Lu, CMAME 2023):
          start from a Sobol set; every `period` Adam steps draw a uniform candidate pool of
          `pool_factor * N` points and resample N of them without replacement with
          p(x) ∝ |r(x)|^k / mean|r|^k + c   (k = 1, c = 1)
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import qmc


class Sampler:
    resample_each_step = False
    adaptive = False

    def __init__(self, problem, n, seed):
        self.p, self.n, self.seed = problem, n, seed
        self.rng = np.random.default_rng([seed, 7919])
        self.lo, self.hi = problem.lo, problem.hi

    def _to_tensor(self, U):
        """Map unit-cube points U (n,d) to the problem box and return a leaf tensor with grad."""
        X = self.lo + (self.hi - self.lo) * U
        return torch.as_tensor(X.astype(np.float32), device=self.p.device).requires_grad_(True)

    def initial(self):
        raise NotImplementedError

    def draw(self):  # used by `resample`
        return self.initial()

    def for_lbfgs(self, net, current):
        return current


class GridSampler(Sampler):
    def initial(self):
        d = self.p.d
        m = round(self.n ** (1.0 / d))
        assert m ** d == self.n, f"grid needs a perfect power, got N={self.n}, d={d}"
        a = (np.arange(m) + 0.5) / m
        mesh = np.meshgrid(*([a] * d), indexing="ij")
        return self._to_tensor(np.stack([g.ravel() for g in mesh], 1))


class RandomSampler(Sampler):
    def initial(self):
        return self._to_tensor(self.rng.random((self.n, self.p.d)))


class ResampleSampler(RandomSampler):
    resample_each_step = True

    def for_lbfgs(self, net, current):
        return self.initial()


class SobolSampler(Sampler):
    def initial(self):
        eng = qmc.Sobol(d=self.p.d, scramble=True, seed=self.rng)
        m = int(np.ceil(np.log2(self.n)))
        return self._to_tensor(eng.random_base2(m)[: self.n])


class RADSampler(SobolSampler):
    adaptive = True

    def __init__(self, problem, n, seed, period=500, pool_factor=8, k=1.0, c=1.0):
        super().__init__(problem, n, seed)
        self.period, self.pool_factor, self.k, self.c = period, pool_factor, k, c

    def propose(self):
        """Uniform candidate pool in the problem box, numpy (pool_factor*N, d)."""
        return self.lo + (self.hi - self.lo) * self.rng.random((self.pool_factor * self.n, self.p.d))

    def select(self, pool, abs_residual):
        """Draw N pool points without replacement with p ∝ |r|^k / mean(|r|^k) + c."""
        w = np.asarray(abs_residual, dtype=np.float64) ** self.k
        w = w / max(w.mean(), 1e-300) + self.c
        idx = self.rng.choice(len(w), size=self.n, replace=False, p=w / w.sum())
        return pool[idx]

    def update(self, net):
        """Single-model convenience wrapper: propose -> residual -> select."""
        pool = self.propose()
        r = []
        for s in range(0, pool.shape[0], 8192):  # chunked to bound memory
            Xc = torch.as_tensor(pool[s:s + 8192].astype(np.float32), device=self.p.device).requires_grad_(True)
            r.append(self.p.residual(net, Xc).detach().abs())
        r = torch.cat(r).cpu().double().numpy()
        X = self.select(pool, r)
        return torch.as_tensor(X.astype(np.float32), device=self.p.device).requires_grad_(True)

    def for_lbfgs(self, net, current):
        return self.update(net)


SAMPLERS = {"grid": GridSampler, "random": RandomSampler, "resample": ResampleSampler,
            "sobol": SobolSampler, "rad": RADSampler}


def make_sampler(name, problem, n, seed, **kw):
    return SAMPLERS[name](problem, n, seed, **kw)
