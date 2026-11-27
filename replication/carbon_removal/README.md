# Replication package: uncertain carbon removal in DICE-2023

Code and results for the manuscript "When Does Uncertain Carbon Removal Matter for Optimal Climate Policy?" (Md Moinul Islam and Matthew A. Oehlschlaeger), submitted to Economic Analysis and Policy.

The model extends the corrected DICE-2023 (Barrage and Nordhaus 2024) with three carbon removal families (biological, geological and mineral), finite storage, and uncertainty about removal capacity resolved at a known date. It is written in JAX and solved with IPOPT (cyipopt). Learning in 2070 uses a reduced-space solver (L-BFGS-B outer problem, IPOPT in each state).

## Layout

The three folders must sit side by side, because the model finds the other two through relative paths.

- `removal_sector/`
  - `removal3.py`: the deterministic model with removal.
  - `removal3_stoch.py`: the two-state model.
  - `removal3_nstate.py`: the n-state model.
  - `p3_*.py` to `p8_*.py`: the experiment scripts.
  - `make_ms_*.py`, `make_jensen_table.py`, `make_p3d_figs.py`, `make_p4_figs.py`: the tables and figures.
  - `replicate*.py`: the replication checks.
  - `results/`: the result files used in the paper.
- `paper_build/`
  - `common.py`: the published GAMS session settings.
  - `dice_jax.py`: the DICE-2023 equations in JAX.
- `notebook/`
  - `dice2023_corrected.py`: the corrected DICE-2023.
  - `gams_reference/`: the published GAMS solutions, used for verification and starting values.

## Quick check

    conda install -c conda-forge cyipopt=1.7.0
    pip install -r removal_sector/requirements.txt
    cd removal_sector
    python replicate.py        # 24 checks, about 3 minutes
    python replicate_p5.py     # revision checks, including learning in 2070 (about 5 minutes)

Each check re-solves the model from scratch and compares the result with the value reported in the paper. `removal_sector/README_replication.md` lists what is checked, the solver settings and the units.

These checks show that the documented code and environment reproduce the reported numbers. They are not an independent verification of the implementation.

## Licence

MIT, as for the rest of the repository.
