# Supplementary Reproducibility Package

This package contains the scripts and dependency list used to regenerate the
numerical outputs in the manuscript.

## Commands

Run from the project root:

```bash
python gen_figures.py
python simulate_cpa.py
python compute_curvatures.py
python verify_curvature.py
```

`gen_figures.py` regenerates the Walsh and local geometric figures.
`simulate_cpa.py` regenerates the CPA Monte Carlo figure and the CSV summary.
`compute_curvatures.py` computes the active covariance and curvature values.
`verify_curvature.py` performs an independent curvature sanity check.

## Expected Outputs

- `figures/fig1_wht_spectra.pdf`
- `figures/fig2_wht_histogram.pdf`
- `figures/fig3_fr_distance_matrix.pdf`
- `figures/fig4_per_trace_info.pdf`
- `figures/fig5_fr_vs_hw.pdf`
- `figures/fig6_cpa_simulation.pdf`
- `cpa_simulation_results.csv`

The CPA simulation uses seed `20260622`, Gaussian noise standard deviation
`sigma=2`, and `3000` Monte Carlo trials per S-box.
