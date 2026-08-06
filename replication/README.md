# Convergent Abatement, Divergent Prices — Replication Package

Replication materials for:

> Md Moinul Islam and Matthew A. Oehlschlaeger,
> "Convergent Abatement, Divergent Prices: What a Binding Feasibility
> Constraint Hides in the Social Cost of Carbon."

All results are produced with PyDICE-2023 (an open-source Python
reimplementation of DICE-2023). The experiment is deterministic; every table
and figure in the paper regenerates from the code here. There are no
proprietary or third-party datasets — all data are model-generated.

## Contents

```
run_all.py                                   reproduces the whole paper (one command)
run_experiment.py                            full experiment; writes every result CSV to results/
convergent_abatement_FULL_experiment.ipynb   the same experiment as a notebook (reference copy)
figs/
  fig_mechanism.py                           Figure 1 (schematic; no data input)
  fig_wedge.py                               Figure 2 (reads the two trajectory files below)
  fig_scatter.py                             Figure 3 (values from Tables 4 and 2)
results/                                      generated result files (provided for inspection)
```

## Requirements

- Python 3.11
- numpy, pandas, scipy, numba, matplotlib, seaborn
- A LaTeX install is needed ONLY to render figure fonts in Computer Modern
  (set `USETEX=True` at the top of each figure script). Results do not need LaTeX.

```
pip install numpy pandas scipy numba matplotlib seaborn
```

## One-command reproduction

```
python run_all.py
```

This runs `run_experiment.py` (writing all CSVs into `results/`), then builds
the three figures (`fig_mechanism.pdf`, `fig_wedge.pdf`, `fig_scatter.pdf`).
Approximate runtime: [XX] minutes on [machine]. To run the pieces separately,
run `python run_experiment.py` first, then each `figs/fig_*.py`.

## What produces each table and figure

`run_experiment.py` writes all of the CSVs below into `results/`.

| Output in paper | Produced by | Result file |
|---|---|---|
| Table 1 — damage specifications | `run_experiment.py` (Part A parameters) | — (parameters) |
| Table 2 — discount-rate sweep | `run_experiment.py` (Part B) | `results/summary.csv` |
| Table 3 — ramp dispersion | `run_experiment.py` (Part B ramp experiment) | `results/ramp_ceiling_experiment_clean/ramp_summary.csv`, `.../ramp_case_dispersion_summary.csv` |
| Table 4 — main results | `run_experiment.py` (Part B) | `results/table1_main_results_clean.csv`, `results/summary.csv` |
| Table 5 — binding diagnostic + welfare | `run_experiment.py` (Part B, Part D) | `results/ramp_ceiling_experiment_clean/standard_dice2023_constraint_diagnostic_table.csv`, `results/welfare_cost_consumption_equivalent.csv` |
| Table 6 — deferral-wedge decomposition | `run_experiment.py` (Part C) | `results/nu_wedge_summary.csv`, `results/wedge_split_final.csv` |
| Table A1 — damage curvature | `run_experiment.py` (Part D) | `results/damage_curvature_along_optimal_path.csv` |
| Table A2 — SCC growth rates | `run_experiment.py` (Part D) | `results/scc_growth_rates.csv` |
| Table A3 — wedge convergence | `run_experiment.py` (Part C) | `results/nu_wedge_summary.csv` |
| Tables A4–A5 — variance decomposition | `run_experiment.py` (Part B) | `results/variance_decomposition_corrected_updated.csv`, `results/variance_decomposition_groupB.csv` |
| SCC pulse-stability check | `run_experiment.py` (Part D) | `results/scc_pulse_stability.csv` |
| Carbon-cycle (IRF) audit | `run_experiment.py` (Part D) | `results/ramp_ceiling_experiment_clean/irf_alpha_audit.csv` |
| Figure 1 — mechanism schematic | `figs/fig_mechanism.py` | — (schematic; no data) |
| Figure 2 — observed gap over time | `figs/fig_wedge.py` | `results/ramp_ceiling_experiment_clean/runs/ramp005_standard_dice2023_hs_high.csv`, `.../ramp007_standard_dice2023_dietz_stern.csv` |
| Figure 3 — SCC vs. control rate | `figs/fig_scatter.py` | values from Tables 4 and 2 (embedded) |

## Software and toolboxes

Numba JIT-compiles the forward simulation; SciPy (`minimize` SLSQP, `brentq`)
solves the constrained optimisation and the pulse root-finding; Richardson
extrapolation is used for the SCC pulse and for the one-sided envelope
estimate of the ramp multiplier.

## Citation / DOI

Archived at Zenodo: DOI [10.5281/zenodo.21816356
Aug 6, 2026].

## License

See the repository `LICENSE`.
