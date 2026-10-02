"""Helper shared by the table scripts: write generated LaTeX rows into a section file.

A section file marks a generated block with the two comment lines
    %% BEGIN GENERATED <name>
    %% END GENERATED <name>
`inject(section, name, text)` replaces what stands between them; if the named section file does not
hold the markers, the one file of sections/*.tex that does is used.  With check=True nothing is
written and the function returns whether the block already equals `text`; the table scripts
use this (option --check) to detect a table that no longer matches the stored results.
"""
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)


def _find(section, name):
    """The section file that holds the markers of `name`: the named file if it has them, otherwise
    the one file of sections/*.tex that does (tables move between the article and the supplement)."""
    a = f"%% BEGIN GENERATED {name}"
    path = os.path.join(ART, "sections", section)
    if os.path.exists(path) and a in open(path, encoding="utf8").read():
        return path
    hits = [p for p in sorted(glob.glob(os.path.join(ART, "sections", "*.tex")))
            if a in open(p, encoding="utf8").read()]
    assert len(hits) <= 1, f"marker for {name} found in several section files: {hits}"
    return hits[0] if hits else None


def inject(section, name, text, check=False):
    path = _find(section, name)
    if path is None:
        return None
    s = open(path, encoding="utf8").read()
    a, b = f"%% BEGIN GENERATED {name}", f"%% END GENERATED {name}"
    assert a in s and b in s, f"marker for {name} missing in {os.path.relpath(path, ART)}"
    i = s.index("\n", s.index(a)) + 1
    j = s.index(b)
    new = text.rstrip("\n") + "\n"
    if check:
        return s[i:j] == new
    if s[i:j] != new:
        open(path, "w", encoding="utf8").write(s[:i] + new + s[j:])
    return True
