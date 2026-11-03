# Elasticity of marginal utility and the SCC in DICE-2023: replication package

Replication materials for:

> Md Moinul Islam and Matthew A. Oehlschlaeger,
> "Precautionary discounting reverses the elasticity–SCC relationship in DICE-2023."
> Manuscript submitted to *Economics Letters*.

The letter shows that DICE-2023's precautionary discount term can reverse the usual effect of the elasticity of marginal utility φ on the optimal social cost of carbon. With other parameters at their published values, the 2025 SCC falls from \$59.1 at φ = 0.95 to a minimum of \$54.4 near φ = 1.30 and then rises.

Every number, the figure and the table in the letter regenerate from the code here. All data are model-generated; the only inputs are the published GAMS reference solutions in `dice/validation/gams_reference/`.

## Contents

```
dice2023_corrected.py   reference DICE-2023 model (NumPy), checked against the GAMS output
dice_jax.py             differentiable (JAX) copy of the forward model
ipopt_model.py          IPOPT solve of the cost-benefit optimum and the multiplier SCC
common.py               GAMS session settings, including the 0.5 °C lower bound on temperature
eta_letter.py           elasticity settings shared by all experiments
house_style.py          figure style
blockV/                 numerical verification: brute-force SCC, multi-start, horizon (PROTOCOL.md, REPORT.md)
stage2/                 the experiments behind the letter (s2_* to s8_*), see REPORT.md
  results/              saved results (JSON) of every run used in the letter
blockV/results/         saved results of the verification runs
letter/
  make_letter_assets.py builds Figure 1 and Table 1 from the saved results
  fig_letter_main.pdf, tab_letter.tex   the figure and table as submitted
REPRODUCE.md            run order, times, settings and the checks to expect
requirements.txt        Python packages
```

## Requirements

Python 3.11 or later, with the packages in `requirements.txt`. cyipopt needs the IPOPT library; the simplest route is conda:

```
conda install -c conda-forge cyipopt
pip install -r requirements.txt
```

## Quick check

From `stage2/`:

```python
import s2_core as S
pr = S.problem(81, 0.95)                    # published DICE-2023, phi = 0.95
o, x = S.solve(pr, pr.start_from_gams())
print(o['scc_2025'])                        # 59.07, the published 2025 SCC
```

With `0.95` replaced by `1.3` the result is 54.36, the minimum reported in the letter.

## Rebuilding the figure and table only

The saved results are included, so Figure 1 and Table 1 can be rebuilt without re-solving the model:

```
cd letter
python make_letter_assets.py
```

To re-run the experiments themselves, follow `REPRODUCE.md`.
