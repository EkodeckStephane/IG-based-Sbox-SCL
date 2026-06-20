# Boundary Information Geometry for S-box Side-Channel Leakage

This repository contains the manuscript sources, reproducibility scripts, and
generated figures/results for:

**Boundary Information Geometry for S-box Side-Channel Leakage: Walsh
Covariance, Active Fisher Structure, and CPA Belief Dynamics**

The revised manuscript is formatted for Springer Nature's `sn-jnl` class for
submission to *Cybersecurity*.

## Contents

- `main.tex` - main LaTeX manuscript; by default it builds a clean reading
  PDF without line numbers.
- `main_review.tex` - Springer review build that enables line numbering while
  reusing the same manuscript source.
- `sections/` and `appendices/` - manuscript source sections.
- `references.bib` - bibliography database.
- `sn-jnl.cls` and `sn-mathphys-num.bst` - Springer Nature LaTeX class and
  bibliography style required for local compilation.
- `figures/` - generated PDF figures used in the manuscript.
- `gen_figures.py` - regenerates the Walsh spectra and diagnostic figures.
- `compute_curvatures.py` - computes active covariance eigenspaces and
  curvature diagnostics.
- `verify_curvature.py` - sanity-checks the active curvature diagnostic.
- `main.pdf` - compiled clean manuscript PDF.
- `main_review.pdf` - compiled review/submission PDF with line numbering.
- `supplementary_reproducibility.tex` / `supplementary_reproducibility.pdf` -
  supplementary document describing the reproducibility protocol.
- `supplementary_material.zip` - compact archive of reproducibility scripts and
  Python requirements.

## Reproducibility

Install the Python requirements:

```bash
python -m pip install -r requirements.txt
```

Regenerate the computational outputs:

```bash
python gen_figures.py
python compute_curvatures.py
python verify_curvature.py
```

Compile the manuscript:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Compile the review/submission PDF with line numbering:

```bash
pdflatex -interaction=nonstopmode main_review.tex
bibtex main_review
pdflatex -interaction=nonstopmode main_review.tex
pdflatex -interaction=nonstopmode main_review.tex
```

## Notes

The repository does not currently assign an open-source license. Reuse rights
should therefore be clarified before public redistribution beyond inspection
and review.
