#!/bin/sh
# Build main.pdf and supplement.pdf, which refer to each other through xr-hyper, and copy main.pdf to
# paper.pdf, the name under which the links of the supplement find the article.
#   sh build_pdfs.sh [folder]      (default: the folder of this script, the root of the repository)
# Runs pdflatex/bibtex in the order that resolves every cross-document reference, then checks the
# logs: no error, no undefined reference or citation, no multiply defined label.
set -e
DIR="${1:-$(cd "$(dirname "$0")" && pwd)}"
cd "$DIR"
# pdflatex and bibtex must be on the PATH (TeX Live 2025 was used)
L="pdflatex -interaction=nonstopmode -halt-on-error"
$L supplement >/dev/null
$L main >/dev/null
bibtex main >/dev/null
bibtex supplement >/dev/null
for i in 1 2 3; do
  $L supplement >/dev/null
  $L main >/dev/null
done
fail=0
for d in main supplement; do
  if grep -E "^! |undefined|multiply defined|Rerun to get" "$d.log" | grep -v "^Package rerunfilecheck" >/dev/null; then
    echo "$d: problems in the log:"; grep -n -E "^! |undefined|multiply defined|Rerun to get" "$d.log" | head -20; fail=1
  fi
  if grep -E "Warning--|error message" "$d.blg" >/dev/null 2>&1; then
    echo "$d: bibtex warnings:"; grep -E "Warning--|error message" "$d.blg" | head; fail=1
  fi
  pages=$(grep -o "Output written on $d.pdf ([0-9]* pages" "$d.log" | grep -o "[0-9]* pages")
  echo "$d.pdf: $pages"
done
cp main.pdf paper.pdf
exit $fail
