# PyDICE-2023

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19461660.svg)](https://doi.org/10.5281/zenodo.19461660)
[![Launch Dashboard](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Mo-in-ul/PyDICE-2023/main?urlpath=voila/render/dice/notebooks/dashboard.ipynb)

**Validated Python implementation of the DICE-2023 integrated assessment model.**

PyDICE-2023 is an open-source Python reimplementation of DICE-2023 (Barrage & Nordhaus, 2024) designed for high-throughput programmatic evaluation. The central contribution is a reformulation of the implicit carbon cycle saturation parameter α(t) as a co-optimization variable, enabling full Numba JIT compilation. A complete 405-year trajectory evaluates in **0.035 milliseconds**, making Monte Carlo uncertainty analysis and reinforcement learning workflows computationally tractable on standard hardware.

---

## Features

- Full Numba JIT compilation — 0.035 ms per trajectory, ~28,500 evaluations per second
- Validated against GAMS reference solutions across nine primary policy scenarios
- Welfare gaps below 0.05% and temperature agreement within 0.19°C against GAMS
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
from dice.model import Dice2023Model

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
| 1 | 5% pure rate of social time preference |
| 2 | 4% pure rate of social time preference |
| 3 | 3% pure rate of social time preference |
| 4 | 2% pure rate of social time preference |
| 5 | 1% pure rate of social time preference |
| 6 | 1.5°C temperature limit (infeasible — best-effort trajectory returned) |
| 7 | 2.0°C temperature limit |
| 8 | Paris Agreement prescribed emissions schedule |
| 9 | Cost-benefit optimal |
| 10 | Baseline (no emissions control) |

---
## Validation Summary

Validation against GAMS reference solutions (Barrage & Nordhaus, 2023) across nine primary scenarios:

| Scenario | Welfare gap (%) | Max ΔT (°C) | Iterations | Max IRF residual |
|:--|--:|--:|--:|--:|
| cb_optimal | −0.018 | 0.089 | 75 | 7.84 × 10⁻¹⁰ |
| 2°C limit | −0.031 | 0.159 | 64 | 9.79 × 10⁻⁷ |
| Paris | −0.018 | 0.016 | 57 | 5.26 × 10⁻⁶ |
| Baseline | −0.029 | 0.025 | 57 | 3.83 × 10⁻⁸ |
| 1% discount | −0.022 | 0.094 | 80 | 5.08 × 10⁻¹⁰ |
| 2% discount | −0.045 | 0.149 | 90 | 9.25 × 10⁻⁷ |
| 3% discount | −0.035 | 0.087 | 90 | 2.96 × 10⁻⁸ |
| 4% discount | −0.023 | 0.064 | 91 | 1.72 × 10⁻⁷ |
| 5% discount | −0.017 | 0.181 | 106 | 1.21 × 10⁻⁴ |

The 1.5°C scenario is physically infeasible under sequential DFAIR dynamics.
Maximum abatement yields a peak of 1.742°C; the best-effort trajectory achieves 1.628°C
(irreducible gap: 0.114°C). See the companion paper for full diagnostics.

---

## Repository Structure

<img width="1472" height="840" alt="image" src="https://github.com/user-attachments/assets/425cdf48-0916-4950-9ec9-8e4895c355f5" />


---

## Reproducing Manuscript Results

All figures and tables in the companion paper can be reproduced by running the notebooks in order:

```bash
cd notebooks
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
  doi     = {10.5281/zenodo.19461660},
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
