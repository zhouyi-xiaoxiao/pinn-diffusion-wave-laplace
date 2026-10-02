"""Names, colours and paths for the Section 4 figures; the plotting style itself is the one
shared by all figures (code/figstyle.py).

Colour convention of the article: Adam blue, Adam -> L-BFGS orange.  Collocation strategies,
where they are the colour variable (budget figure), use three other hues; marker shape is the
secondary encoding.  Both sets pass the all-pairs colour-vision checks of the palette validator
used for the research figures.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_style, panel_label, INK, INK2, GRID, PNAME, PROBLEMS  # noqa: E402,F401  (re-exported)

HERE = os.path.dirname(os.path.abspath(__file__))
ARTICLE = os.path.dirname(HERE)
ROOT = ARTICLE if os.path.isdir(os.path.join(ARTICLE, "research_benchmark")) else os.path.dirname(ARTICLE)  # repository root
BENCH = os.path.join(ROOT, "research_benchmark")
FIGDIR = os.path.join(ARTICLE, "figures")
DATADIR = os.path.join(ARTICLE, "data")

ARM_COL = {"adam": "#2a78d6", "adam_lbfgs": "#eb6834"}
ARM_LAB = {"adam": "Adam", "adam_lbfgs": "Adam → L-BFGS"}
ARM_MARK = {"adam": "o", "adam_lbfgs": "s"}

# benchmark identifiers -> names used in the article: PNAME and PROBLEMS come from figstyle
PLONG = {"heat1d": "Heat: heat, 1-D", "laplace2d": "Lap2: Laplace, 2-D", "wave2d": "Wave2: wave, 2-D",
         "laplace3d": "Lap3: Laplace, 3-D", "wave1d": "Wave1: two-mode wave, 1-D"}
STRATS = ["grid", "random", "resample", "sobol", "rad"]
SLAB = {"grid": "grid (cell-centred)", "random": "random", "resample": "resample", "sobol": "Sobol'", "rad": "RAD"}
STRAT_COL = {"grid": "#4a3aa7", "random": "#1baf7a", "sobol": "#eda100"}
STRAT_MARK = {"grid": "o", "random": "s", "sobol": "D"}
