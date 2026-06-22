Dear Editor,

Please consider our manuscript, "Boundary Information Geometry for S-box
Side-Channel Leakage: Walsh Covariance, Active Fisher Structure, and CPA Belief
Dynamics", for publication in *Cybersecurity*.

The manuscript develops a reproducible framework for analysing the local
S-box distributions that appear in correlation power analysis of lightweight
block ciphers.  It connects three objects that are central to side-channel
security evaluation: singular key-induced S-box laws, regular Fisher geometry
in Walsh coordinates, and Bayesian posterior dynamics over key hypotheses.

The main contribution is a security-oriented diagnostic workflow.  The paper
derives closed-form boundary covariance structure for bijective S-boxes,
identifies the active Fisher subspace forced by bijectivity, and links Gaussian
Hamming-weight CPA to pairwise likelihood gaps that govern posterior-odds
growth.  This gives a precise way to separate baseline invariants from
key-difference effects that influence CPA discrimination.

The numerical section applies the framework to PRESENT, GIFT, and SKINNY
S-boxes.  The experiments report Walsh spectra, boundary covariance ranks,
local Walsh/Fisher key-separation proxies, pairwise Hamming-weight leakage
gaps, projected active-curvature diagnostics, and Monte Carlo CPA simulations.
The CPA simulations translate the pairwise-gap theory into empirical success
probability and true-key-rank curves under the stated leakage model.

The submission is fully reproducible.  The repository provides the LaTeX
sources, generated figures, review PDF with line numbering, supplementary
reproducibility document, scripts, CSV simulation results, and a compact
supplementary archive.  The manuscript cites the archived GitHub tag used for
the submitted version.

We believe the article fits *Cybersecurity* because it advances side-channel
analysis methodology for lightweight cryptography, combines mathematical
modelling with reproducible security diagnostics, and provides an explicit
bridge between information geometry and CPA key-recovery behavior.

Sincerely,

Stéphane Gaël R. Ekodeck  
on behalf of all authors
