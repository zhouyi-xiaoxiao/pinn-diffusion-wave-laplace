"""Plotting style and problem names shared by every figure script of the article.

One font family (DejaVu Sans, with the matching mathtext set) and one set of sizes for all
figures; figures are drawn at their final width (6.3 in = text width) with text of 7 pt or more.
Colour convention: Adam blue, Adam -> L-BFGS orange (Sections 4, 5); Deep Ritz blue, PINN orange
(Section 6).  PROBLEMS is the order in which the low-dimensional problems appear in every
table and figure of results; PNAME maps the identifiers of the code to the names of the article.
"""
import matplotlib

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, DARK, GREY = "#2a78d6", "#eb6834", "#9c3d12", "#6f6e6a"

PROBLEMS = ["heat1d", "laplace2d", "wave1d", "wave2d", "laplace3d"]
PNAME = {"heat1d": "Heat", "laplace2d": "Lap2", "wave1d": "Wave1", "wave2d": "Wave2", "laplace3d": "Lap3",
         "laplace3d_smooth": "Lap3s", "laplace": "LapD", "poisson": "PoiD"}


def apply_style(**overrides):
    rc = {
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 400,
        "pdf.fonttype": 42, "lines.linewidth": 1.4, "legend.frameon": False,
    }
    rc.update(overrides)
    matplotlib.rcParams.update(rc)


def panel_label(ax, text, x=0.0, y=1.03, **kw):
    """One style for the panel labels '(a)', '(b)', ... of every figure: plain weight, 8 pt, left-aligned above
    the top-left corner of the axes.  `text` may continue with a short title, e.g. '(a) Heat, 10 seeds'."""
    opts = dict(transform=ax.transAxes, fontsize=8, fontweight="normal", ha="left", va="bottom", color=INK)
    opts.update(kw)
    return ax.text(x, y, text, **opts)
