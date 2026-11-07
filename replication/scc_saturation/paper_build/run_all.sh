#!/usr/bin/env bash
# Reproduce every number in paper_build/results and paper_build/tex, in dependency order.
# Requirements: numpy, scipy, pandas, openpyxl, numba, matplotlib, seaborn, jax, cyipopt (IPOPT).
# Run from paper_build/. Times are for one core; steps 2 and 3 can run in parallel.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

# 1. SCC on the GAMS control paths: GAMS vs responsive alpha vs fixed alpha (about 3 min)
python3 scc_gams_controls.py            > logs/scc_gams.log 2>&1

# 2. Validation: model as coded (run_validation) and with the GAMS session settings (run_faithful)
for s in 1 2 3 4 5 6 7 8 9 10; do python3 run_validation.py $s > logs/val_$s.log 2>&1; done
for s in 1 2 3 4 5 7 8 9 10;   do python3 run_faithful.py   $s > logs/faith_$s.log 2>&1; done
python3 run_faithful.py 1 all gams      > logs/faith_1w.log 2>&1   # 1% run warm-started from GAMS

# 3. The SCC checks (see SCC_VALIDATION.md)
python3 check1_pipeline.py              > logs/check1.log 2>&1   # SCC on PyDICE's own optimum
python3 check3_ift.py                   > logs/check3.log 2>&1   # implicit-function-theorem SCC (JAX)
python3 check2_pulse_sweep.py           > logs/check2.log 2>&1   # pulse size and solver tolerance (needs check3)
python3 check5_multiplier_scc.py        > logs/check5.log 2>&1   # GAMS multiplier SCC with IPOPT (about 2 min)
python3 check4_envelope.py              > logs/check4.log 2>&1   # brute-force re-optimization
python3 check6_tlo_effect.py            > logs/check6.log 2>&1   # effect of the TATM >= 0.5 bound on the SCC

# 4. Paper experiments (see LAB_NOTEBOOK in the project and the docstring of each script)
python3 decomposition.py a 9 4 5               > logs/decomp_a.log 2>&1   # cumulative + leave-one-out (about 1 h)
python3 decomposition.py b --cum-only 8 10 7 2 3 1 > logs/decomp_b.log 2>&1   # cumulative only (1% run is slow)
python3 mechanism.py                    > logs/mechanism.log 2>&1   # pulse responses, alpha fixed vs responding
python3 fig_feedback_share.py           > logs/fig_share.log 2>&1   # feedback share by pulse year (needs check3)
python3 gym_replay.py                   > logs/gym.log 2>&1         # Gymnasium environment replay
python3 onepointfive.py                 > logs/onepointfive.log 2>&1 # 1.5 C cap sweep with and without disposal
python3 policy_fixed_alpha.py           > logs/policy.log 2>&1      # planner who ignores the feedback
python3 mc_fast.py 0 500                > logs/mc_0_500.log 2>&1 &  # Monte Carlo, 1000 draws in two lanes
python3 mc_fast.py 500 1000             > logs/mc_500_1000.log 2>&1
wait
python3 mc_ift.py 0 1000 --every 50     > logs/mc_ift.log 2>&1      # IPOPT validation of 20 draws

# 5. LaTeX macros and table fragments
python3 make_tables.py

# ---- v3 additions (31 Oct 2026) ----
python fair_benchmark.py            # FaIR v2 benchmark (Table fair)
python formula_check.py             # two-step mechanism, sign rule (Table channels)
python eta_sweep.py                 # elasticity of marginal utility sweep (Table eta)
python ecs_mechanism.py             # policy-fixed vs re-optimized share (Table ecsmech)
python mc_v2.py 0 250 & python mc_v2.py 250 500; wait          # Monte Carlo v2, SLSQP cross-check
python mc_v2_ipopt.py 0 250 & python mc_v2_ipopt.py 250 500; wait   # Monte Carlo v2, IPOPT (primary)
python mc_v2_polish.py              # IPOPT re-solve of SLSQP failures + 20-draw validation
python make_tables_v3.py            # v3 table fragments + results/mc2_summary.json
python fig_mc2.py                   # appendix figures (house style)

# ---- gap-closing round (31 Oct 2026) ----
python fair_opt.py                  # DICE-2023 re-optimized with the FaIR v2 carbon cycle (Table fairopt)
python sobol.py                     # Sobol indices on a GP surrogate of the Monte Carlo (Table sobol)
python fig_feedback_share.py; python mechanism.py; python onepointfive.py --plot-only   # main figures, house style
# companion letter (eta)
python eta_letter.py grid; python eta_letter.py horizon; python eta_letter.py nofloor; python eta_polish.py
python letter_analysis.py           # letter figures, table and summary -> ../letter/
