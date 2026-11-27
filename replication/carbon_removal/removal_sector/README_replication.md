# Replication package: anticipated carbon removal in DICE-2023

## Layout
The three folders must sit side by side:
- `removal_sector/`: removal model (`removal3.py`, `removal3_stoch.py`), run scripts (`p3_*.py`), `replicate.py`, `requirements.txt`, and `results/` (only `p3_runs.json` and `p3_hedge_line.json` are needed, as starting points for the reduced-space solver).
- `paper_build/`: `common.py` (published GAMS session settings) and `dice_jax.py` (DICE-2023 equations in JAX).
- `notebook/`: `dice2023_corrected.py` (corrected DICE-2023) and `gams_reference/*.xlsx` (published GAMS solutions, used for the base-model check and as starting values).

## Environment
Python 3.13.18; package versions are pinned in `requirements.txt`. IPOPT with the MUMPS linear solver, via cyipopt 1.7.0 (the reported numbers were computed with IPOPT 3.11.9 and re-verified with IPOPT 3.14.20) (`conda install -c conda-forge cyipopt=1.7.0` is the simplest route). All runs use IPOPT tolerance 1e-10 (acceptable 1e-8), adaptive barrier, no bound relaxation; the reduced-space solver uses SciPy L-BFGS-B.

## Running
    cd removal_sector
    python replicate.py          # 24 checks, about 3 minutes
    python replicate.py --full   # adds learning in 2070 (reduced space)

Each check re-solves the model from scratch and compares the result with the value reported in the paper. The script writes `replication_report.md`.

## What is checked
1. The corrected DICE-2023 forward model reproduces the published GAMS temperature path on the GAMS controls.
2. No-removal and central SCC and peak warming (published and 2 percent calibrations), the welfare value of removal, and the rise in peak warming when removal arrives in 2110.
3. The two-state model (removal from 2070 or never, p = 0.5, learned 2050): SCC and the value of hedging against the optimist and pessimist plans.
4. The premium over the expected-capacity benchmark and its exact decomposition into the Jensen gap and the policy-path effect, including the check that the two-state SCC equals the probability-weighted state SCCs along the common policy.

## Units
Prices are 2019 US dollars, the base of DICE-2023. SCC values are dollars per tonne of CO2 in the year stated. Welfare differences are utility differences divided by the marginal welfare of consumption in 2025 in the no-removal run, i.e. trillions of 2019 dollars of 2025 consumption. Storage values are dollars per tonne either in the year stated (current value) or discounted to 2025 with the run's own marginal welfare of consumption.
