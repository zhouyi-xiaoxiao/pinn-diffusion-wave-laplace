"""Closed forms for P5 used in the working document of the proofs (Proposition 3(e) and the source of P5) against src/floors.py and
src/presolve.py. Writes results/check_p5_formulas.json."""
import sys, os, json, math
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import floors as fl
import presolve as ps
out = {}
for d in (2, 3, 5, 10, 20, 50):
    a = math.pi / math.sqrt(d)
    phi = lambda b: math.sin(b) / b
    psi = (math.sin(a) - a * math.cos(a)) / a ** 2          # E[t sin(a t)], t uniform on [-1, 1]
    Eu = math.cos(1.0) * phi(a) ** d
    Eu2 = 0.5 * (1.0 + math.cos(2.0) * phi(2 * a) ** d)
    Etu = math.sin(1.0) * psi * phi(a) ** (d - 1)           # |E[t_i u*]|, sign -e_i
    aff = math.sqrt(1.0 - (Eu ** 2 + 12.0 * d * (0.5 * Etu) ** 2) / Eu2)
    ex = fl.exact_floors("altridge", d)
    # interior moments r_u = int u* phi via presolve: r_f = kappa r_u
    r_f, _ = ps.moments("altridge", d)
    r_u = r_f / (4 * math.pi ** 2)
    e = np.where(np.arange(d) % 2 == 0, 1.0, -1.0)
    ru_closed = np.concatenate([[Eu], Eu / 2 + 0.5 * (-e * Etu)])   # E[x_i u] = E[u]/2 + E[t_i u]/2
    out[f"d{d}"] = dict(E_u=abs(Eu - ex["E_u"]), E_u2=abs(Eu2 - ex["E_u2"]), affine=abs(aff - ex["affine"]),
                        affine_value=aff, r_u=float(np.abs(ru_closed - r_u).max()))
    print(d, out[f"d{d}"])
out["max_diff"] = max(max(v[k] for k in ("E_u", "E_u2", "affine", "r_u")) for v in out.values() if isinstance(v, dict))
json.dump(out, open(os.path.join(ROOT, "results", "check_p5_formulas.json"), "w"), indent=1)
print("max", out["max_diff"])
