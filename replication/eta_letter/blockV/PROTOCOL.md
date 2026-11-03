# Block V: numerical verification protocol (written before running the experiments)

Scope: the η letter (`letter/eta_letter.tex`). Baseline code (`eta_letter.py`, `ipopt_model.py`,
`dice_jax.py`) and baseline results (`results/eta_letter_*.json`) are not modified. All Block V code
lives in `paper_build/blockV/` and imports the baseline modules. Results go to `blockV/results/`.

## What the baseline does (from code inspection)

* Model: `eta_letter.params(N, η, variant)`. Every parameter at its DICE-2023 value except
  `elasmu = η` (1.0001 at η = 1) and the precautionary term
  `rprecaut(t) = -0.5 · siggc1² · 5 · min(t-1, 47) · η_p²`, with η_p = η ("dice"), 0.95 ("fixed"), 0 ("noprec").
  Savings rate fixed at 0.28 from period 38 (2205). MIU(1) fixed. Horizon N = 81 (2420).
* Lower temperature bound: the letter's "dice" runs use **TATM ≥ 0.01**, and "dice_nofloor" uses
  TATM ≥ −10 (no bound). **The GAMS program uses TATM.lo = 0.5** (`common.gams_session`, flag `tlo`).
  The letter text calls 0.01 "the model's lower temperature bound". This is audited in V4.
* Solver: IPOPT 3.x via cyipopt 1.7, exact first and second derivatives from JAX. Decision variables:
  MIU(1..N), S(1..N) and α(1..N) (the FaIR α is a decision variable tied down by the IRF equality,
  as in GAMS). Options: tol 1e-10, acceptable_tol 1e-8, mu_strategy adaptive, bound_relax_factor 0,
  honor_original_bounds, MUMPS, objective scaled by 1/|U(x0)|. The baseline grid uses continuation
  (each η warm-started from the previous η's solution).
* SCC: multiplier (envelope) SCC, SCC(t) = −1000 · [dV/dε_E(t)] / [dV/dε_C(t)], where
  dV/dε = ∂f/∂ε + λ'∂g/∂ε at the IPOPT optimum. ε_E(t) is an additive term in the emissions equation
  (GtCO₂/yr), ε_C(t) one in the consumption equation (trillion $/yr). Includes shadow prices of
  any binding state bound.

## Independent SCC verification (V2)

1. **Re-optimization (welfare perturbation).** For t ∈ {2025, 2050}, solve the full planner problem
   again with ε_E(t) = ±h_E and, separately, ε_C(t) = ±h_C, everything else unchanged, warm-started
   from the optimum (primal and dual). V = optimal welfare.
   dV/dε ≈ Richardson [4·D(h/2) − D(h)]/3 of central differences D(h) = [V(+h) − V(−h)]/(2h).
   h_E ∈ {0.1, 0.05} GtCO₂/yr, h_C ∈ {0.006, 0.003} trillion $/yr (about the same money value at
   SCC ≈ $60/t). SCC_BF = −1000 · dV/dε_E / dV/dε_C. This uses no multipliers.
2. **Fixed-policy pulse (simulation SCC).** MIU and S held at their optimal values; α re-solved
   from the IRF equality; temperature bound removed. Same differencing. This is the IAM
   simulation definition. It equals the multiplier SCC by the envelope theorem only when no
   inequality on a state binds. A gap therefore measures the shadow value of a binding bound.
3. Grid: η ∈ {0.5, 0.95, 1.28, 1.5, 1.75, 2.0}. Temperature bound: none (the letter's nofloor runs),
   plus 0.01 and 0.5 where they bind.

## Solver convergence and starting points (V3)

η ∈ {0.95, 1.28, 1.5, 1.75, 2.0}, no bound. Starts: GAMS DICE-2023 optimum; PyDICE default
simulation start; three random starts (MIU, S multiplied by U(0.8, 1.2), clipped to bounds;
α from the GAMS path); continuation solution from the letter's grid. Report status, iterations,
U, SCC(2025).

## Temperature-bound audit (V4)

η = 0.50, 0.55, …, 2.00 (31 values) × TATM.lo ∈ {0.5 (GAMS), 0.01 (letter), none}, N = 81, each
from the GAMS start. Report SCC(2025), min T, years at the bound, Σ|λ_T|.

## Terminal horizon (V5)

N ∈ {61, 81, 101, 121, 141} (horizon 2320 … 2720) × η ∈ {0.5, 0.95, 1.25, 1.5, 1.6, 1.75, 1.9, 2.0},
"dice" discounting, no bound, plus the "fixed" counterfactual at a subset. Report SCC(2025),
peak T and the relative change between successive horizons. Statements are finite-horizon only.

## Acceptance criteria (fixed in advance)

| check | criterion |
|---|---|
| solver | IPOPT status 0 (or 1 = acceptable, flagged); scaled KKT stationarity < 1e-8 on free variables |
| IRF | max \|IRF residual\| < 1e-8 |
| reproduction | \|SCC − manuscript\| < 0.05% (rounding in the text aside); peak T within 0.005 °C |
| multiplier vs re-optimization | relative gap < 0.1%, and Richardson estimates from h and h/2 agree to < 0.05% (otherwise "unresolved", not "fail") |
| multiplier vs fixed-policy pulse | < 0.1% when no state bound binds; a larger gap is expected, and reported, when one binds |
| multi-start | U equal to 1e-8 relative; SCC equal to 0.01% |
| horizon | reported as relative changes; no pass/fail cut-off (the user asked for qualitative judgement) |
