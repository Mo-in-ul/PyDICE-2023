# PyDICE-2023

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A validated Python reimplementation of the **DICE-2023** (Dynamic Integrated Climate-Economy) model by Nordhaus & Barrage (2024). Reproduces the official GAMS reference solution across 10 policy scenarios with welfare gaps below 0.05% for all primary scenarios.

---

## Features

- Numba-JIT compiled forward simulation for fast repeated evaluation
- Co-optimization of the carbon cycle saturation parameter α(t) for GAMS compatibility
- Richardson extrapolation for Social Cost of Carbon (SCC) computation
- Full validation against official GAMS reference across 10 scenarios

---

## Installation
```bash
git clone https://github.com/Mo-in-ul/PyDICE-2023.git
cd PyDICE-2023
pip install -r requirements.txt
```

---

## Scenarios

| Scenario | Description |
|---|---|
| `baseline` | No climate policy |
| `cb_optimal` | Cost-benefit optimal |
| `alt_damage` | Alternative damage function |
| `discount_1pct` – `discount_5pct` | Varying pure time preference rates |
| `temp_2c` | 2°C temperature constraint |
| `temp_1_5c` | 1.5°C temperature constraint |

---

## Validation

Validated against the official GAMS reference (Nordhaus & Barrage, 2024):

| Metric | Result |
|---|---|
| Welfare gap (9 primary scenarios) | < 0.05% |
| Max temperature deviation | 0.23°C (temp_1_5c scenario) |

Full validation details are in the [`validation/`](validation/) folder.

---

## Citation

If you use PyDICE-2023 in your research, please cite:
```bibtex
@software{islam2026pydice,
  author    = {Islam, Md Moinul},
  title     = {PyDICE-2023: A Validated Open-Source Python Implementation
               of the DICE-2023 Integrated Assessment Model},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://github.com/Mo-in-ul/PyDICE-2023}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

**Md Moinul Islam** — Graduate Student, Rensselaer Polytechnic Institute  
islamm10@rpi.edu
