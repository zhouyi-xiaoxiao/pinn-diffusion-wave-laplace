"""Section 7.1 and Supplementary Section S6: a PDE residual made identically zero by differentiating
with respect to a slice of the input and replacing the missing derivative by zeros, demonstrated on a
minimal example.

Self-contained; needs only numpy, torch and matplotlib.  Run from anywhere:

    python code/s7_pitfalls.py                 # train (about 1-3 min, 1 thread) + figure
    python code/s7_pitfalls.py --plot          # figure only, from the stored JSON
    python code/s7_pitfalls.py --passes 1500   # the same run with ten times as many passes
                                               # -> data/s7_pitfalls_autograd_1500passes.json (no figure)

What it does
  (i)   shows that torch.autograd.grad(u_t, txy[:, 0], allow_unused=True) returns None (the
        slice txy[:, 0] is a new tensor that is not in the graph of u_t), that the same call
        without allow_unused raises, and that differentiating with respect to the full input
        tensor and indexing afterwards gives the second derivative;
  (ii)  applies the faulty and the correct residual operator of the wave equation u_tt = Lap u to
        an exact solution and to a non-solution (float64): the faulty operator returns 0 on both;
  (iii) trains a 3-50-50-1 tanh network with a loss made of the faulty residual, a data term that
        prescribes exp(-t) sin(pi x) sin(pi y) at the 10 000 i.i.d. points in [0,1] x [-1,1]^2, and
        400 zero-Dirichlet points, all at t = 0; the residual is replaced by zeros whenever autograd
        returns None; batch 64, Adam 1e-3, seed 0, 150 passes = 23 550 optimiser steps;
  (iv)  records the least-squares amplitude a(t) = <u_theta(t), S>/<S, S> of the mode
        S = sin(pi x) sin(pi y) and the relative L2 distance from the solution
        cos(sqrt(2) pi t) S of problem Wave2, per time slice and on the held-out space-time grid;
  (v)   draws figures/s7_pitfalls.pdf: (a) a(t) against exp(-t) and cos(sqrt(2) pi t);
        (b) mean |u| on the faces of the one-face 3-D Laplace problem of Section 7.2 for the five
        seeds stored in research_benchmark/results/oneface_laplace3d.json (no training).

The random-number calls are made in a fixed order (data, then network, then one permutation per
pass), so the run is deterministic on CPU with one thread.
Outputs: data/s7_pitfalls_autograd.json, figures/s7_pitfalls.pdf
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)
ROOT = ART if os.path.isdir(os.path.join(ART, "research_benchmark")) else os.path.dirname(ART)  # repository root
OUT_JSON = os.path.join(ART, "data", "s7_pitfalls_autograd.json")
OUT_FIG = os.path.join(ART, "figures", "s7_pitfalls.pdf")
ONEFACE = os.path.join(ROOT, "research_benchmark", "results", "oneface_laplace3d.json")

PI = math.pi
OMEGA = math.sqrt(2.0) * PI
SEED, N_TRAIN, PASSES, BATCH, LR = 0, 10000, 150, 64, 1e-3


# ------------------------------------------------------------------ the two residual patterns
def residual_faulty(u, txy):
    """The faulty pattern: second derivatives by autograd with respect to a slice of the input."""
    g = torch.autograd.grad(u, txy, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = g[:, 0]
    n_none = 0
    h = torch.autograd.grad(u_t, txy[:, 0], grad_outputs=torch.ones_like(u_t),
                            create_graph=True, allow_unused=True)[0]
    if h is None:
        n_none += 1
        u_tt = torch.zeros_like(u_t)
    else:
        u_tt = h[:, 0]
    lap = torch.zeros_like(u_t)
    for i in (1, 2):
        gi = torch.autograd.grad(g[:, i], txy[:, i], grad_outputs=torch.ones_like(u_t),
                                 create_graph=True, allow_unused=True)[0]
        if gi is None:
            n_none += 1
        else:
            lap = lap + gi
    return u_tt - lap, n_none


def residual_correct(u, txy):
    """Differentiate with respect to the full input tensor, index afterwards."""
    g = torch.autograd.grad(u.sum(), txy, create_graph=True)[0]
    r = 0.0
    for i, sign in ((0, 1.0), (1, -1.0), (2, -1.0)):
        h = torch.autograd.grad(g[:, i].sum(), txy, create_graph=True)[0]
        assert h is not None
        r = r + sign * h[:, i]
    return r


def mode(x, y):
    return torch.sin(PI * x) * torch.sin(PI * y)


def target_data(txy):                      # what the data term prescribes at every (t, x, y)
    return mode(txy[:, 1], txy[:, 2]) * torch.exp(-txy[:, 0])


class Net(torch.nn.Module):                # 3-50-50-1 tanh network
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 50)
        self.fc2 = torch.nn.Linear(50, 50)
        self.fc3 = torch.nn.Linear(50, 1)

    def forward(self, txy):
        return self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(txy)))))


# ------------------------------------------------------------------ experiment
def run():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    out = {"protocol": dict(seed=SEED, n_train=N_TRAIN, passes=PASSES, batch=BATCH, lr=LR,
                            network="3-50-50-1 tanh", optimiser="Adam", threads=1,
                            torch=torch.__version__, numpy=np.__version__)}

    # data, boundary set (all at t = 0) and network, in a fixed order
    txy_all = torch.rand(N_TRAIN, 3)
    txy_all[:, 1:3] = 2 * txy_all[:, 1:3] - 1
    s = torch.linspace(-1, 1, 100)
    bx = torch.cat([s, torch.ones(100), torch.linspace(1, -1, 100), -torch.ones(100)])
    by = torch.cat([-torch.ones(100), s, torch.ones(100), torch.linspace(1, -1, 100)])
    b_txy = torch.stack([torch.zeros(400), bx, by], dim=1)
    net = Net()

    # (i) the None gradient, on the untrained network and the first 64 training points
    p = txy_all[:64].clone().requires_grad_(True)
    u = net(p)
    g = torch.autograd.grad(u, p, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = g[:, 0]
    h_slice = torch.autograd.grad(u_t, p[:, 0], grad_outputs=torch.ones_like(u_t),
                                  create_graph=True, allow_unused=True)[0]
    try:
        torch.autograd.grad(u_t, p[:, 0], grad_outputs=torch.ones_like(u_t), create_graph=True)
        raised = None
    except RuntimeError as e:
        raised = str(e).split(".")[0]
    r_faulty, n_none = residual_faulty(u, p)
    r_true = residual_correct(u, p)
    out["autograd_demo"] = dict(
        slice_grad_is_None=h_slice is None, none_returns_per_call=n_none,
        without_allow_unused_raises=raised,
        max_abs_residual_faulty=float(r_faulty.abs().max()),
        residual_faulty_requires_grad=bool(r_faulty.requires_grad),
        max_abs_residual_correct=float(r_true.detach().abs().max()),
        note="untrained network, first 64 training points")
    print("(i)", json.dumps(out["autograd_demo"]), flush=True)

    # (ii) operator unit test in float64: an exact solution and a non-solution
    gen = torch.Generator().manual_seed(12345)          # separate stream: does not touch training
    q = torch.rand(2000, 3, generator=gen, dtype=torch.float64)
    q[:, 1:3] = 2 * q[:, 1:3] - 1
    tests = {}
    for name, fn in (("exact_solution_cos", lambda z: mode(z[:, 1], z[:, 2]) * torch.cos(OMEGA * z[:, 0])),
                     ("non_solution_exp", target_data)):
        z = q.clone().requires_grad_(True)
        rc, _ = residual_faulty(fn(z).unsqueeze(1), z)
        z = q.clone().requires_grad_(True)
        rt = residual_correct(fn(z), z)
        tests[name] = dict(max_abs_faulty=float(rc.detach().abs().max()),
                           max_abs_correct=float(rt.detach().abs().max()))
    tests["analytic_max_of_non_solution_residual"] = 1 + 2 * PI ** 2
    out["operator_unit_test_float64"] = tests
    print("(ii)", json.dumps(tests), flush=True)

    # (iii) training with the faulty residual in the loss
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    b_val = torch.zeros(400)
    hist, steps, max_pde, none_total = [], 0, 0.0, 0
    t0 = time.time()
    for ep in range(PASSES):
        perm = torch.randperm(N_TRAIN)
        acc = 0.0
        for i in range(0, N_TRAIN, BATCH):
            opt.zero_grad()
            batch = txy_all[perm[i:i + BATCH]]
            batch.requires_grad_(True)
            u = net(batch)
            res, n_none = residual_faulty(u, batch)
            pde = torch.mean(res ** 2)
            ic = torch.mean((u.squeeze() - target_data(batch)) ** 2)
            bc = torch.mean((net(b_txy).squeeze() - b_val) ** 2)
            loss = pde + ic + bc
            loss.backward()
            opt.step()
            steps += 1
            none_total += n_none
            max_pde = max(max_pde, float(pde))
            acc += loss.item() * batch.size(0)
        hist.append(acc / N_TRAIN)
        if ep % 25 == 0 or ep == PASSES - 1:
            print(f"pass {ep:4d}  loss {hist[-1]:.4e}  {time.time() - t0:6.1f}s", flush=True)
    out["training"] = dict(steps=steps, none_returns=none_total, autograd_calls=3 * steps,
                           max_pde_term_over_all_steps=max_pde,
                           loss_first_pass=hist[0], loss_last_pass=hist[-1], loss_per_pass=hist)

    # (iv) evaluation
    net.eval()
    xs = np.linspace(-1, 1, 101)
    X, Y = np.meshgrid(xs, xs)
    S = np.sin(PI * X) * np.sin(PI * Y)

    def slice_metrics(t):
        pts = torch.tensor(np.stack([t * np.ones(X.size), X.ravel(), Y.ravel()], -1)).float()
        with torch.no_grad():
            U = net(pts).numpy().reshape(X.shape).astype(np.float64)
        ex, tg = math.cos(OMEGA * t) * S, math.exp(-t) * S
        on_bnd = max(abs(U[0]).max(), abs(U[-1]).max(), abs(U[:, 0]).max(), abs(U[:, -1]).max())
        return dict(t=float(t), amplitude=float((U * S).sum() / (S * S).sum()),
                    exp_minus_t=math.exp(-t), cos_omega_t=math.cos(OMEGA * t),
                    rel_L2_vs_exact=float(np.linalg.norm(U - ex) / max(np.linalg.norm(ex), 1e-300)),
                    rel_L2_vs_target=float(np.linalg.norm(U - tg) / np.linalg.norm(tg)),
                    abs_L2_vs_exact=float(np.sqrt(np.mean((U - ex) ** 2))),
                    max_abs_on_boundary=float(on_bnd))

    out["slices"] = [slice_metrics(t) for t in np.round(np.linspace(0, 3, 301), 10)]
    out["slices_note"] = ("101 x 101 grid on [-1,1]^2 per time; the script samples t in [0,1] only, "
                          "so t > 1 is outside the sampled window")

    # held-out space-time grid of problem W2: 21 x 81 x 81 on [0,1] x [-1,1]^2
    tt, xx, yy = np.meshgrid(np.linspace(0, 1, 21), np.linspace(-1, 1, 81), np.linspace(-1, 1, 81), indexing="ij")
    P = np.stack([tt.ravel(), xx.ravel(), yy.ravel()], -1)
    with torch.no_grad():
        Uh = net(torch.tensor(P).float()).numpy().ravel().astype(np.float64)
    Sh = np.sin(PI * P[:, 1]) * np.sin(PI * P[:, 2])
    ex, tg = Sh * np.cos(OMEGA * P[:, 0]), Sh * np.exp(-P[:, 0])
    z = torch.tensor(P[::7]).float().requires_grad_(True)       # true residual of the trained network
    r_true = residual_correct(net(z).squeeze(1), z).detach().numpy()
    out["heldout_21x81x81"] = dict(
        rel_L2_network_vs_exact=float(np.linalg.norm(Uh - ex) / np.linalg.norm(ex)),
        rel_L2_network_vs_target=float(np.linalg.norm(Uh - tg) / np.linalg.norm(tg)),
        rel_L2_target_vs_exact=float(np.linalg.norm(tg - ex) / np.linalg.norm(ex)),
        rms_true_residual_of_network=float(np.sqrt(np.mean(r_true ** 2))),
        rms_true_residual_points=int(r_true.size))
    out["train_wall_s_not_a_timing_claim"] = time.time() - t0
    print("(iv)", json.dumps(out["heldout_21x81x81"]), flush=True)
    for k in (0, 50, 100, 200, 300):
        print("    ", json.dumps(out["slices"][k]), flush=True)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1)
    return out


# ------------------------------------------------------------------ figure
def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, HERE)
    from figstyle import apply_style, panel_label, BLUE, ORANGE, INK, INK2, GRID   # style shared by all figures
    apply_style(**{"lines.linewidth": 1.6})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.3, 2.75), gridspec_kw=dict(width_ratios=[1.3, 1.0], wspace=0.27))

    # (a) mode amplitude
    sl = out["slices"]
    t = np.array([s["t"] for s in sl])
    a = np.array([s["amplitude"] for s in sl])
    ax.axvspan(1.0, 3.0, ymin=0.0, ymax=2.5 / 3.25, color="#f3f2ee", lw=0, zorder=0)
    ax.axhline(0.0, color=INK2, lw=0.6, zorder=1)
    tf = np.linspace(0, 3, 601)
    ax.plot(tf, np.cos(OMEGA * tf), color=ORANGE, lw=1.4, label=r"solution of Wave2: $\cos(\sqrt{2}\,\pi t)$", zorder=2)
    ax.plot(tf, np.exp(-tf), color=INK2, lw=1.2, ls=(0, (4, 2)), label=r"data term: $\mathrm{e}^{-t}$", zorder=3)
    ax.plot(t, a, color=BLUE, lw=1.8, label=r"network trained with the zero residual: $a(t)$", zorder=4)
    k = [0, 100, 200, 300]
    ax.plot(t[k], a[k], ls="none", marker="o", ms=4.5, color=BLUE, mec="white", mew=0.8, zorder=5, clip_on=False)
    for i, off, ha in zip(k, [(5, 5), (-5, -11), (0, 7), (-3, 7)], ["left", "right", "center", "right"]):
        ax.annotate(f"{a[i]:.3f}", (t[i], a[i]), textcoords="offset points", xytext=off, ha=ha, fontsize=7, color=INK)
    ax.text(2.0, -1.21, "outside the sampled window", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xlim(0, 3)
    ax.set_ylim(-1.25, 2.0)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"amplitude of $\sin(\pi x)\sin(\pi y)$")
    ax.yaxis.set_label_coords(-0.165, 0.385)
    ax.spines["left"].set_bounds(-1.25, 1.0)
    ax.legend(loc="upper left", handlelength=2.2, borderaxespad=0.2, labelspacing=0.3)
    panel_label(ax, "(a)")

    # (b) one-face 3-D Laplace problem: values on the unconstrained faces, five seeds
    rows = json.load(open(ONEFACE))["oneface_laplace3d"]["rows"]
    faces = ["mean_abs_u_face_y0", "mean_abs_u_face_ypi", "mean_abs_u_face_z0", "mean_abs_u_face_zpi"]
    datum = 4 / PI ** 2                                   # mean |sin y cos z| on the constrained face
    bx.axhline(datum, color=INK2, lw=1.0, ls=(0, (4, 2)), zorder=1, label=r"mean $|$datum$|$ on the face $x=\pi$")
    bx.axhline(0.0, color=ORANGE, lw=1.6, zorder=2, label="problem Lap3 (zero)")
    for r in rows:
        sd = r["seed"]
        others = [r[k_] for k_ in faces]
        v = r["mean_abs_u_face_x0"]
        bx.plot([sd, sd], [0, v], color=BLUE, lw=1.0, zorder=3)
        bx.plot([sd] * 4, others, ls="none", marker="o", ms=3.8, color="#9a9994", mec="white", mew=0.4, zorder=4,
                label=r"faces $y=0$, $y=\pi$, $z=0$, $z=\pi$" if sd == 0 else None)
        bx.plot([sd], [v], ls="none", marker="o", ms=5.5, color=BLUE, mec="white", mew=0.8, zorder=5,
                label="face $x=0$" if sd == 0 else None)
        left = sd == 1                                    # keep the labels of seeds 1 and 2 apart
        bx.annotate(f"{v:.2f}", (sd, v), textcoords="offset points", xytext=(-6, -2.5) if left else (6, -2.5),
                    ha="right" if left else "left", fontsize=7, color=INK)
    bx.set_xlim(-0.5, 4.75)
    bx.set_ylim(-0.15, 4.6)
    bx.set_yticks([0, 1, 2, 3])
    bx.set_xticks(range(5))
    bx.set_xlabel("seed")
    bx.set_ylabel(r"mean $|u_\theta|$ on the face")
    bx.yaxis.set_label_coords(-0.105, 0.33)
    bx.spines["left"].set_bounds(-0.15, 3.0)
    h, l = bx.get_legend_handles_labels()
    order = [l.index("face $x=0$"), 2, 0, 1]
    bx.legend([h[i] for i in order], [l[i] for i in order], loc="upper left", handlelength=2.0,
              borderaxespad=0.2, labelspacing=0.3)
    bx.grid(axis="x", visible=False)
    panel_label(bx, "(b)")

    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    if "--preview" in sys.argv:                          # optional raster copy for a quick look
        fig.savefig(os.path.join(ART, "data", "s7_pitfalls_preview.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("figure:", OUT_FIG)


if __name__ == "__main__":
    if "--passes" in sys.argv:                           # the same run with another number of passes; no figure
        PASSES = int(sys.argv[sys.argv.index("--passes") + 1])
        OUT_JSON = os.path.join(ART, "data", f"s7_pitfalls_autograd_{PASSES}passes.json")
        run()
    else:
        if "--plot" in sys.argv:
            result = json.load(open(OUT_JSON))
        else:
            result = run()
        plot(result)
