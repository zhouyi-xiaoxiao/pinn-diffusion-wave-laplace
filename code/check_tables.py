"""Check that the tables of the article and of its Supplementary Material whose rows were pasted from generated
files still agree with them.

    python code/check_tables.py        (no training; run the table scripts first)

Most tables of results are written into sections/*.tex by their scripts, between the comment lines
"%% BEGIN GENERATED <name>" and "%% END GENERATED <name>".  The rows of the remaining generated tables
(Section 6 and Supplementary Section S5: the two error tables, the budget table, the cost table and the
penalty-bias table) were pasted from data/s6_highdim_tables.tex and data/s6_highdim_robin_bias_table.tex.
This script verifies that every row of each generated file occurs in the section files (main text and
supplement together), ignoring white space, and also runs the --check mode of code/s5_hard_tables.py.
It exits with an error if a row is missing.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)


def norm(x):
    return re.sub(r"\s+", "", x)


import glob
ALL = norm("".join(open(p, encoding="utf8").read() for p in sorted(glob.glob(os.path.join(ART, "sections", "*.tex")))))
bad = 0
for gen in ("s6_highdim_tables.tex", "s4_lowdim_tables.tex", "s4_lowdim_gridcontrol_table.tex",
            "s4_lowdim_timeslices_table.tex", "s5_hard_tables.tex", "s6_highdim_robin_bias_table.tex",
            "s6_highdim_bound_tables.tex", "s6_highdim_remedies_tables.tex"):
    rows = [l.strip() for l in open(os.path.join(ART, "data", gen), encoding="utf8") if l.strip() and not l.lstrip().startswith("%")]
    missing = [r for r in rows if norm(r) not in ALL]
    bad += len(missing)
    print(f"{'ok  ' if not missing else 'FAIL'} data/{gen}: {len(rows) - len(missing)} of {len(rows)} rows found in sections/*.tex")
    for r in missing:
        print("     missing:", r[:140])
rc = subprocess.run([sys.executable, os.path.join(HERE, "s5_hard_tables.py"), "--check"]).returncode
if bad or rc:
    sys.exit("TABLE CHECK FAILED")
print("ALL TABLE CHECKS PASSED")
