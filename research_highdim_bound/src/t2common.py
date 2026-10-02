# -- added for the public repository: absolute paths of the development machine are replaced by
# -- paths relative to the repository root, through the helper _repo_path defined here.
import os as _os


def _repo_path(rel=""):
    d = _os.path.dirname(_os.path.abspath(__file__))
    while not (_os.path.isdir(_os.path.join(d, "research_benchmark")) and _os.path.isdir(_os.path.join(d, "research_highdim"))):
        p = _os.path.dirname(d)
        if p == d:
            raise RuntimeError("repository root not found")
        d = p
    return _os.path.join(d, rel) if rel else d
# -- end of the added lines

"""Shared definitions for research_highdim_bound.

Omega = (0,1)^d.  T: L2(Omega) -> L2(boundary), T zeta = dn psi_zeta, where psi_zeta in H^2 cap H^1_0
solves -Lap psi = zeta.  C_d = ||T||.  Theorem 2 (the working document of the proofs):
    L_d <= C_d^2 <= U_d,
    U_d = tanh(pi sqrt(d-1)/2) / (pi sqrt(d-1))   (U_1 = 1/2),
    L_d = (4/pi^2) sum_{j odd} j^2 / (j^2 + d - 1)^2.
"""
import math
import os
import json
import hashlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
HD_SCRIPTS = _repo_path("research_highdim/scripts")
CKPT_DIR = _repo_path("research_highdim/results/ckpt")
PILOT_E_NETS = _repo_path("research_highdim_bound/pilot_networks")


def U(d):
    """Upper bound U_d of C_d^2."""
    if d == 1:
        return 0.5
    a = math.sqrt(d - 1)
    return math.tanh(math.pi * a / 2) / (math.pi * a)


def L(d, jmax=4_000_001):
    """Lower bound L_d of C_d^2 (series summed to j < jmax, tail < (4/pi^2)/(2 jmax))."""
    j = np.arange(1, jmax, 2, dtype=float)
    return float(4 / math.pi ** 2 * (j ** 2 / (j ** 2 + d - 1) ** 2).sum())


def L_tail_bound(jmax=4_000_001):
    # sum_{j odd >= jmax} j^2/(j^2+a)^2 <= sum_{j odd >= jmax} 1/j^2 <= 1/(2 (jmax - 2))
    return 4 / math.pi ** 2 / (2 * (jmax - 2))


def odd_sum_closed(a):
    """sum_{j odd >= 1} 1/(j^2 + a^2) = pi tanh(pi a / 2) / (4 a)  (a > 0); pi^2/8 at a = 0."""
    if a == 0:
        return math.pi ** 2 / 8
    return math.pi * math.tanh(math.pi * a / 2) / (4 * a)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _np_default(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(type(o))


def dump(name, obj):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1, default=_np_default)


def append_jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]
