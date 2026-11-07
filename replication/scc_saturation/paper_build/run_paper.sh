#!/usr/bin/env bash
# Reproduce every number, table and figure of the manuscript "How Much of the Social Cost of Carbon Comes from
# Carbon-Sink Saturation? Evidence from PyDICE-2023" (pydice2023_v6_submission_revised.tex). Run from paper_build/.
# Times are rough, one core of a 2.8 GHz Xeon. Lines ending in & can run in parallel.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
python3 -c "import sys, numpy, scipy, numba, jax, cyipopt, fair, matplotlib, pandas; print(sys.version); [print(m.__name__, getattr(m, '__version__', '?')) for m in (numpy, scipy, numba, jax, cyipopt, fair, matplotlib, pandas)]" > logs/versions.txt

# 1. Verification against the ten published GAMS runs (Table 2; Appendix A)
for s in 1 2 3 4 5 7 8 9 10; do python3 run_faithful.py $s > logs/faith_$s.log 2>&1; done   # ~10 min
python3 run_faithful.py 1 all gams      > logs/faith_1w.log 2>&1                         # 1% run, warm start
python3 decomposition.py a 9 4 5        > logs/decomp_a.log 2>&1                         # gap attribution (~1 h)
python3 decomposition.py b --cum-only 8 10 7 2 3 1 > logs/decomp_b.log 2>&1

# 2. SCC on the GAMS policy paths and numerical checks (Section 4, Appendix B)
python3 scc_gams_controls.py            > logs/scc_gams.log 2>&1
python3 check3_ift.py                   > logs/check3.log 2>&1   # exact decomposition, all scenarios
python3 check2_pulse_sweep.py           > logs/check2.log 2>&1
python3 check5_multiplier_scc.py        > logs/check5.log 2>&1
python3 check4_envelope.py              > logs/check4.log 2>&1
python3 check6_tlo_effect.py            > logs/check6.log 2>&1
python3 exp_d_benchmark.py              > logs/exp_d.log 2>&1    # Table 3, accuracy and cost

# 3. Size, carbon-cycle substitution and sign (Section 5, Appendices C and D)
python3 fair_benchmark.py               > logs/fair.log 2>&1
python3 exp_c_fair_range.py             > logs/exp_c.log 2>&1    # CMIP6 FaIR v2 calibrations
python3 fair_audit.py                   > logs/fair_audit.log 2>&1  # units, d alpha channels, r0 re-anchoring
python3 fair_opt.py                     > logs/fair_opt.log 2>&1
python3 fair_own_weights.py             > logs/fair_own.log 2>&1   # each FaIR set with its own baseline welfare weights (Table C)
python3 formula_check.py                > logs/channels.log 2>&1
python3 mechanism.py                    > logs/mechanism.log 2>&1
python3 exp_a_removal_sign.py           > logs/exp_a.log 2>&1    # net-negative emissions and the sign change

# 4. Policy response and uncertainty (Section 6, Appendix E)
python3 exp_b_physical_policy.py        > logs/exp_b.log 2>&1    # physical / abatement / savings split (IPOPT)
python3 exp_b_order.py                  > logs/exp_b_order.log 2>&1  # same split, savings replaced first
python3 exp_e_robust.py 1               > logs/exp_e1.log 2>&1       # high ECS under three temperature-bound treatments
python3 exp_e_robust.py 2               > logs/exp_e2.log 2>&1       # sigma_C, sigma_T: fixed vs re-optimized
python3 policy_fixed_alpha.py           > logs/policy.log 2>&1
python3 mc_v2.py 0 250 > logs/mc2a.log 2>&1 & python3 mc_v2.py 250 500 > logs/mc2b.log 2>&1; wait               # SLSQP cross-check
python3 mc_v2_ipopt.py 0 250 > logs/ipa.log 2>&1 & python3 mc_v2_ipopt.py 250 500 > logs/ipb.log 2>&1; wait     # primary (~1 h)
python3 mc_v2_polish.py                 > logs/polish.log 2>&1
python3 sobol.py                        > logs/sobol.log 2>&1
python3 slsqp_disagree.py               > logs/slsqp_disagree.log 2>&1  # draws where SLSQP and IPOPT differ
python3 clamp_check.py                  > logs/clamp_check.log 2>&1     # where the Numba 0.01 C clamp is active
python3 fallback_check.py               > logs/fallback_check.log 2>&1  # the seven SLSQP draws, IPOPT warm start
python3 fallback_diag.py                > logs/fallback_diag.log 2>&1   # their feasibility, KKT residuals, exclusion
python3 sobol_bootstrap.py              > logs/sobol_boot.log 2>&1      # bootstrap refits of the Sobol surrogate
python3 jacobian_check.py mc            > logs/jacobian.log 2>&1   # diagonal of dR/dalpha, Eq. (16); needs the MC results; re-solves 29 draws (~15 min)

# 5. Appendices F and G
python3 onepointfive.py                 > logs/onepointfive.log 2>&1
python3 gym_replay.py                   > logs/gym.log 2>&1

# 6. Tables, macros and figures
python3 make_tables.py && python3 make_tables_v3.py && python3 make_tables_v4.py && python3 mc_bound_check.py && python3 make_tables_v6.py && python3 make_tables_submission.py
python3 fig_mc2.py && python3 fig_v4.py && python3 fig_v6.py && python3 fig_feedback_share.py
python3 overlap_check.py                # collision check of every manuscript figure (expects TOTAL 0)
python3 -m pytest -q tests
