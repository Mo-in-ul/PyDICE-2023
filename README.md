# PyDICE-2023

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20006871.svg)](https://doi.org/10.5281/zenodo.20006871)
[![Launch Dashboard](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Mo-in-ul/PyDICE-2023/main?urlpath=voila/render/dice/notebooks/dashboard.ipynb)
[![CI](https://github.com/Mo-in-ul/PyDICE-2023/actions/workflows/test.yml/badge.svg)](https://github.com/Mo-in-ul/PyDICE-2023/actions/workflows/test.yml)


**Validated Python implementation of the DICE-2023 integrated assessment model.**

PyDICE-2023 is an open-source Python reimplementation of DICE-2023 (Barrage & Nordhaus, 2024) designed for high-throughput programmatic evaluation. The central contribution is a reformulation of the implicit carbon cycle saturation parameter α(t) as a co-optimization variable, enabling full Numba JIT compilation. A complete 405-year trajectory evaluates in **0.035 milliseconds**, making Monte Carlo uncertainty analysis and reinforcement learning workflows computationally tractable on standard hardware.

---

## Features

- Full Numba JIT compilation — 0.035 ms per trajectory, ~28,500 evaluations per second
- Validated against GAMS reference solutions across nine primary policy scenarios
- Welfare within 0.03% of GAMS; given the GAMS control paths, temperatures identical to GAMS
- 1,000-sample Monte Carlo uncertainty analysis over economic and physical parameters
- Gymnasium-compatible reinforcement learning environment
- Richardson-extrapolated finite-difference Social Cost of Carbon
- Explicit infeasibility detection for the 1.5°C scenario with best-effort trajectory

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Mo-in-ul/PyDICE-2023.git
cd PyDICE-2023
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, NumPy, SciPy, Numba, Pandas, Matplotlib, Gymnasium

---
## Quick Start

```python
import numpy as np
from dice import Dice2023Model

# Run the cost-benefit optimal scenario (scenario 9)
model = Dice2023Model(num_times=81, scenario=9)
x_opt, output, result_meta = model.run_model()

# Output array columns: EIND, ECO2, CO2PPM, TATM, Y, ...
# See dump_state() for full column list
years = np.arange(2020, 2020 + 5 * 81, 5)[:81]
model.dump_state(years, output, './results/dice2023_state.csv', scenario=9)

print(f"Peak temperature: {output[:, 3].max():.3f} °C")
print(f"SCC in 2025:      ${output[1, 39]:.2f} / tCO2")
```

---

## Scenarios

| Scenario number | Description |
|:-:|:--|
| 1 | 1% pure rate of social time preference |
| 2 | 2% pure rate of social time preference |
| 3 | 3% pure rate of social time preference |
| 4 | 4% pure rate of social time preference |
| 5 | 5% pure rate of social time preference |
| 6 | 1.5°C temperature limit (infeasible — best-effort trajectory returned) |
| 7 | 2.0°C temperature limit |
| 8 | Paris Agreement prescribed emissions schedule |
| 9 | Cost-benefit optimal |
| 10 | Baseline (no emissions control) |

---
## Validation Summary

Validation against GAMS reference solutions (Barrage & Nordhaus, 2023) across nine primary scenarios.
Welfare gap is (PyDICE − GAMS) / |GAMS| using the PyDICE welfare function on each consumption path;
ΔT is the largest temperature difference over 2020–2215 between the PyDICE optimum and the GAMS solution.

| Scenario | Welfare gap (%) | Max ΔT (°C) | Iterations | Max IRF residual |
|:--|--:|--:|--:|--:|
| cb_optimal | 0.000 | 0.000 | 71 | 3.00 × 10⁻⁷ |
| 2°C limit | +0.001 | 0.001 | 63 | 9.64 × 10⁻⁷ |
| Paris | 0.000 | 0.000 | 68 | 2.16 × 10⁻⁹ |
| Baseline | 0.000 | 0.000 | 56 | 3.80 × 10⁻⁷ |
| 1% discount | +0.030 | 0.001 | 103 | 1.22 × 10⁻⁴ |
| 2% discount | +0.008 | 0.003 | 136 | 3.83 × 10⁻⁶ |
| 3% discount | 0.000 | 0.009 | 87 | 2.95 × 10⁻⁸ |
| 4% discount | 0.000 | 0.000 | 83 | 1.17 × 10⁻⁷ |
| 5% discount | +0.002 | 0.168 | 79 | 1.66 × 10⁻⁸ |

Given the GAMS control and savings paths, the PyDICE forward model reproduces the GAMS
temperature path to within 0.002 °C in every scenario above. The remaining ΔT for 5% discounting
comes from the far tail (after about 2140), where the objective is nearly flat: GAMS stops
below the control-rate ceiling, while PyDICE goes to the ceiling and attains slightly higher welfare.

**1.5°C scenario.** No control path meets 1.5°C while capital accumulation holds with equality:
maximum abatement with the lowest allowed savings rate peaks at 1.632°C, and the best-effort
trajectory reaches 1.633°C. The GAMS reference solution reaches 1.5°C only because its capital
stock falls from 295 to about 2 in 2025, far below the depreciation path, which amounts to
free disposal of capital. Replaying the GAMS 1.5°C emissions path through DFAIR, with α(t)
solved from the IRF condition period by period, reproduces the GAMS temperatures exactly, so
the carbon cycle is not the source of the gap.

**Known open issue.** The pulse-based SCC in 2025 is 13–20% below the GAMS shadow-price
SCC in the unconstrained runs (for example $51 vs $59 in the cost-benefit run). This is unaffected by the corrections
above and is under investigation.

---

## Repository Structure

<img width="1472" height="840" alt="image" src="https://github.com/user-attachments/assets/425cdf48-0916-4950-9ec9-8e4895c355f5" />


---

## Reproducing Manuscript Results

All figures and tables in the companion paper can be reproduced by running the notebooks in order:

```bash
cd dice/notebooks
jupyter lab
```

---

## Monte Carlo Uncertainty Analysis

The 1,000-sample Latin hypercube sweep over five parameters (ψ₂, ρ, σ_C, σ_T, E_land0) takes approximately 4,648 seconds on a single CPU core. Results confirm that SCC uncertainty is overwhelmingly determined by discount rate and damage function assumptions, with physical carbon cycle parameters contributing negligibly (R² = 0.88–0.96).

---

## Reinforcement Learning Environment

```python
# The Gymnasium environment wraps PyDICE-2023 for RL training
# Observation: [K̃, T̃_AT, M̃_AT, C̃CATOT, t̃]  (5-dimensional, normalized)
# Action:      [μ(t), S(t)]                     (2-dimensional, continuous)
# Episodes execute at 4,129 steps per second (19.1 ms per episode)
```

---


## Citation

If you use PyDICE-2023 in your research, please cite:

```bibtex
@article{Islam2025pydice,
  title   = {{PyDICE-2023}: Validated {Python} Tools for
             Climate-Economy Integrated Assessment},
  author  = {Islam, Md Moinul and Oehlschlaeger, Matthew A.},
  journal = {Environmental Modelling \& Software},
  year    = {2025},
  doi     = {10.5281/zenodo.20006871},
  url     = {https://github.com/Mo-in-ul/PyDICE-2023}
}
```

---

## References

Barrage, L. & Nordhaus, W. D. (2023). *Policies, Projections, and the Social Cost of Carbon: Results from the DICE-2023 Model.* NBER Working Paper 31112.

Barrage, L. & Nordhaus, W. D. (2024). *Policies, Projections, and the Social Cost of Carbon.* PNAS, 121(13).

Millar, R. J. et al. (2017). Emission budgets and pathways consistent with limiting warming to 1.5°C. *Nature Geoscience*, 10, 741–747.


---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

© 2025 Md Moinul Islam, Matthew A. Oehlschlaeger — Rensselaer Polytechnic Institute

## Contact

**Md Moinul Islam** — Graduate Student, Rensselaer Polytechnic Institute  
islamm10@rpi.edu
