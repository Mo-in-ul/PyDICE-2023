# Reproducing "Precautionary discounting reverses the elasticity–SCC relationship in DICE-2023"

All paths are relative to `replication/eta_letter/`.

This file lists the steps that regenerate every number, the figure and the table of the letter from scratch. Each step writes JSON results; nothing is computed inside the LaTeX files.

## Environment

These are the versions used for the submitted results. Newer versions should work, but the numbers were produced with these.

| Component | Version used |
|---|---|
| Python | 3.13 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| jax | 0.12.0.1 (64-bit enabled by `dice_jax`) |
| cyipopt | 1.7.0, with IPOPT and MUMPS |
| matplotlib | 3.11.2 |

The model code is in this folder:
- `dice2023_corrected.py` (parameters and the reference model);
- `dice_jax.py` (the differentiable model);
- `ipopt_model.py` (IPOPT solve and multiplier SCC);
- `common.py` (GAMS session settings, incl. TATM.lo = 0.5);
- `eta_letter.py` (elasticity settings).

All runs start from the published GAMS optimum, which is read through `common.py`.

## Run order

Run from `replication/eta_letter/` unless stated otherwise. The run times are rough figures for a 2-core machine.

| Step | Command | Produces | Used in the letter for | Time |
|---|---|---|---|---|
| 1 | `cd blockV && python v14_grid.py` | `blockV/results/v14_grid.json` | Fig. 1a (term fixed); Table 1C (no-floor SCC); bound facts | 20 min |
| 2 | `cd blockV && python v2_bruteforce.py none` (also `0.01`, `0.5`) | `blockV/results/v2_bf_*.json` | Sect. 3 verification (largest gap 0.0003%) | 60 min |
| 3 | `cd blockV && python v3_multistart.py` | `blockV/results/v3_multistart.json` | Sect. 3 multi-start statement | 20 min |
| 4 | `cd blockV && python v5_horizon.py` | `blockV/results/v5_horizon.json` | Sect. 4.4 (φ = 2 doubles from 2420 to 2720) | 40 min |
| 5 | `cd stage2 && python s2_sigma.py main` (then `caps`, `horizon`) | `stage2/results/s2_sigma_*.json` | Fig. 1a (as coded); Sect. 4.3 (σ, cap); Table 1C (horizon) | 60 min |
| 6 | `cd stage2 && python s2_rho.py 80` | `stage2/results/s2_rho_h80.json` | Sect. 5 (ρ < 0 from φ ≈ 1.05); needed by step 9 | 30 min |
| 7 | `cd stage2 && python s3_noprec.py` | `stage2/results/s3_noprec.json` | Fig. 1a (no precautionary term) | 15 min |
| 8 | `cd stage2 && python s7_rhophi.py && python s7_rhophi_095.py && python s7_rhophi_h.py` | `stage2/results/s7_rhophi*.json` | Fig. 1b; Sect. 4.3 (turning points, horizon) | 40 min |
| 9 | `cd stage2 && python s5_constrained.py` | `stage2/results/s5_constrained.json` | Table 1C (floor term) | 30 min |
| 10 | `cd stage2 && python s7_channels.py` | `stage2/results/s7_channels.json` | Table 1B (re-optimized channels) | 10 min |
| 11 | `cd stage2 && python s7_split.py` | `stage2/results/s7_split.json` | Table 1A | 20 min |
| 12 | `cd stage2 && python s8_checks.py` | `stage2/results/s8_checks.json` | Fig. 1a (fixed policy); Table 1B (fixed policy); turning point 1.283 | 40 min |
| 13 | `cd letter && python make_letter_assets.py` | `fig_letter_main.pdf`, `tab_letter.tex` | Figure 1, Table 1 | 1 min |

**Settings.** Set the environment variable `S2_MAXIT` (IPOPT iteration cap) to 5000 for steps 7, 8 and 10, and to 20000 for steps 11 and 12. This matches the submitted runs.

**Restarting.** Every script except `s8_checks.py` saves after each solve and skips points that are already done, so an interrupted step can be restarted. `s8_checks.py` recomputes everything on each run.

## Checks to expect

| Check | Expected value |
|---|---|
| φ = 0.95, GAMS bound: 2025 SCC | 59.07 (the published DICE-2023 value) |
| `s8_checks.json`, `A.zero_0.0125` | ≈ 1.283 |
| `s7_split.json`: `direct` against `direct_analytic` | within 1% |
| `s5_constrained.json`: `resid_pct` | below 0.01% |
