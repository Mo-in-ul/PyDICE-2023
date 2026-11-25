# GAMS cross-check (NEOS)

Checks that the elasticity results are a property of the original DICE-2023 GAMS program, not of the Python implementation.

`make_neos.py` builds four standalone GAMS files from the published `DICE2023-b-4-3-10.gms` (October 2023, from the DICE website). The two model include files are inlined, only the optimal run is kept (its three solves and the GAMS SCC formula), and the only change to the model is `elasmu` = 0.95, 1.30, 1.50 or 1.75. `solprint` is off and the scenario output files are dropped, since NEOS returns only the listing. The generated `.gms` files are included here.

The files were solved on the NEOS Server (CONOPT4, GAMS 54.7) on 24 November 2026. Listings for phi = 0.95, 1.30 and 1.50 are included (`.lst`); for phi = 1.75 (NEOS job 25198403) the values were read from the results page and are recorded in `comparison.md`.

`compare_neos.py` compares the listings with `../stage2/results/s2_sigma_main.json` (the as-coded curve of Fig. 1a) and writes `comparison.md`.

Result: 2025 SCC within 0.004%, 2050 SCC within 0.012%, SCC path 2025-2100 within 0.08%, peak warming and objective identical, at all four values.

To rerun: `python make_neos.py PATH_TO/GAMS-DICE2023-b-4-3-10`, submit each `.gms` at https://neos-server.org/neos/solvers/nco:CONOPT/GAMS.html, save the listings here as `dice2023_eta<phi>.lst`, then `python compare_neos.py`.
