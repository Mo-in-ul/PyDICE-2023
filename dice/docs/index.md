# PyDICE-2023

**A validated open-source Python reimplementation of the DICE-2023 integrated assessment model.**

PyDICE-2023 reproduces the official GAMS reference solution by Nordhaus & Barrage (2024) across 10 policy scenarios, with welfare gaps below 0.05% for all primary scenarios. It is designed for researchers who need a fast, scriptable, and extensible DICE implementation in Python.

[View on GitHub](https://github.com/Mo-in-ul/PyDICE-2023) · [Cite this software](https://github.com//Mo-in-ul/PyDICE-2023/blob/main/CITATION.cff) · [Paper (under review)]()

---

## Why PyDICE-2023?

The official DICE-2023 model is implemented in GAMS, a proprietary optimization language. PyDICE-2023 provides a fully open, Python-native alternative that:

- Runs without a GAMS license
- Integrates naturally with NumPy, SciPy, and machine learning libraries
- Enables stochastic experiments, sensitivity analysis, and RL-based policy research
- Is validated to match GAMS output to within 0.05% welfare for 9 of 10 scenarios

---

## Quick Start

```bash
pip install -r requirements.txt
```

```python
from dice.model import Dice2023Model

model = Dice2023Model(scenario="cb_optimal")
results = model.optimize()
print(f"Welfare:  {results.welfare:.4f}")
print(f"SCC 2020: ${results.scc[1]:.2f}")
```

---

## Validation

PyDICE-2023 is validated against the official GAMS reference across 10 scenarios:

| Scenario   | Welfare Gap | Max Temp Deviation |
|------------|-------------|--------------------|                 
| baseline   | < 0.01%     |      < 0.01°C      |
| cb_optimal | < 0.01%     |      < 0.05°C      |
| alt_damage | < 0.05%     |      < 0.10°C      |
| rate 1%–5% | < 0.05%     |      < 0.15°C      |
| temp_2c    | < 0.05%     |      < 0.10°C      |
| temp_1_5c  | < 0.05%     |        0.23°C*     |

*The temp_1_5c deviation is a documented IRF timing artifact; see the paper for details.

Full validation figures are available in the [repository](https://github.com/islamm10/pydice2023/tree/main/validation/figures).

---

## Citation

```bibtex
@software{islam2025pydice,
  author    = {Islam, Md Moinul},
  title     = {PyDICE-2023: A Validated Open-Source Python Implementation of the DICE-2023 Integrated Assessment Model},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://github.com/islamm10/pydice2023}
}
```

---

## Contact

Md Moinul Islam — Graduate Student, Rensselaer Polytechnic Institute  
islamm10@rpi.edu
