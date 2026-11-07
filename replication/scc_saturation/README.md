# PyDICE-2023, release 3.0: replication package for the SCC saturation-feedback paper

Python implementation of the DICE-2023 integrated assessment model (Barrage and Nordhaus 2024), verified against
the ten published GAMS runs, with an exact decomposition of the social cost of carbon (SCC) into a direct part
and the carbon-sink saturation feedback.

This release contains the code and results behind the manuscript
"How Much of the Social Cost of Carbon Comes from Carbon-Sink Saturation? Evidence from PyDICE-2023"
(Md Moinul Islam and Matthew A. Oehlschlaeger).

## Layout

- `notebook/dice2023_corrected.py`: the model (Numba forward model, SLSQP optimization, pulse SCC).
- `notebook/gams_reference/`: GAMS outputs of the published runs, used for verification.
- `paper_build/dice_jax.py`: the same forward model in JAX, for exact derivatives.
- `paper_build/ipopt_model.py`, `mc_ift.py`: IPOPT solves with Lagrange multipliers.
- `paper_build/scc_decomposition.py`: the exact SCC decomposition (see below).
- `paper_build/*.py`: one script per experiment, table and figure; `results/` holds their output.
- `paper_build/tests/`: regression tests against GAMS and against the reported numbers.
- `paper_build/run_paper.sh`: reproduces every number in the manuscript, in dependency order.
- `paper_build/fair_own_weights.py`, `jacobian_check.py`: the robustness checks of Appendices C and D (FaIR v2 sets with
  their own baseline welfare weights; the diagonal of the implicit-function Jacobian).

## Quick start

```bash
pip install -r requirements.txt
cd paper_build
python scc_decomposition.py      # 2025: SCC 59.07 = direct 51.36 + feedback 7.70 (share 13.04%)
python -m pytest -q tests        # about 10 seconds
```

```python
from common import scenario_params, gams_controls
from scc_decomposition import scc_decomposition

_, p = scenario_params(9)                 # cost-benefit optimum, GAMS run settings
MIU, S, _ = gams_controls(9, p)           # published control paths (1-based)
out = scc_decomposition(p, MIU, S)        # alpha solved from the IRF condition
out['total'][1], out['direct'][1], out['feedback'][1]     # 2025, 2019 $/tCO2
```

`total` equals the multiplier SCC when no state bound binds, and the pulse SCC with alpha re-solved after the
pulse. `direct` equals a pulse SCC with alpha held at its optimized path.

## Reproducing the paper

`paper_build/run_paper.sh` runs every step. The Monte Carlo (500 IPOPT solves) takes about one hour of processor
time; the gap attribution about one hour; everything else minutes. `run_all.sh` is the full history of the build,
including the companion analysis of the elasticity of marginal utility.

## Numerical notes

- The lower bound on the saturation factor alpha is 0.1, as in the GAMS program, both as the Numba model's default
  and in every solver bound, so the forward-model clamp and the solver bounds agree
  (`tests/test_alpha_floor.py`). Earlier versions of the Numba model defaulted to 0.2; on deep-removal paths, where
  alpha falls below 0.2, that changes results.
- The Numba forward model clamps atmospheric temperature to [0.01, 20] C. With the GAMS stability bound of 0.5 C
  the clamp never binds. Runs that lower the temperature bound to 0.01 C (the Monte Carlo, Experiments A and B)
  should be solved with IPOPT on the unclamped JAX model (`mc_fast.evaluate_ipopt`, `mc_ift.solve_draw`), where the
  bound is a constraint; with SLSQP the bound then acts as a floor on the state instead. `clamp_check.py` and
  `slsqp_disagree.py` document where this matters.

## Versions

Results were computed on Linux with Python 3.13 and the package versions in `requirements.txt`.
The earlier release v2.0 (Zenodo 10.5281/zenodo.20006871) predates the corrections and the analysis in this
release; its reported validation gaps no longer apply.

## Licence

MIT. See `LICENSE`.
