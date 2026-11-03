# Block V report: numerical verification of the η letter

Protocol fixed before the runs: `PROTOCOL.md`. Code: `v14_grid.py`, `v2_bruteforce.py`, `v3_multistart.py`, `v5_horizon.py`, `diag_alpha.py`, analysis in `analyze_v14.py` and `analyze_all.py`. Results: `results/*.json`. Baseline code, baseline results and the manuscript were not changed.

SCC in 2019 $/tCO₂ for 2025 unless stated. "No bound" means TATM ≥ −10.

## 1. Manuscript against reproduced values

All 138 grid points were solved again from the GAMS start, without the continuation used in the baseline. The largest difference from the baseline grid is 0.0015% on the SCC and 0.002 °C on peak warming.

| Quantity in the letter | Letter | Reproduced | Verdict |
|---|---|---|---|
| SCC at φ = 0.5 | $77.3 | 77.32 | match |
| Minimum | $54.8 at φ ≈ 1.28 | 54.855 at 1.285 | match |
| SCC back at the default value | at φ = 1.56 | 1.564 | match |
| Default SCC | $59.1 | 59.07 | match |
| SCC at φ = 1.75 | $69.6 | 69.58 | match |
| SCC at φ = 2.0 | $140 | 140.27 | match (but see §5) |
| Precautionary term fixed, φ = 0.5 → 2.0 | $100 → $27 | 100.42 → 26.77 | match |
| No precautionary term, φ = 0.5 → 2.0 | $72 → $25 | 71.50 → **24.46** | **rounding error: should read $24** |
| Term's contribution at φ = 0.95 and φ = 2 | $12, $116 | 12.34, 115.81 | match |
| Peak warming: default, φ = 1.75, φ = 2 | 2.60, 2.05, 1.80 °C | 2.601, 2.050, 1.800 | match |
| Peak warming, term fixed at φ = 2 | 3.26 °C | 3.264 | match |
| Cumulative CO₂ 2020–2100: default → φ = 2 | 2,610 → 980 Gt | 2,608 → 981 | match |
| Lower bound first binds | φ ≥ 1.6 | 1.6 (0.01 °C bound) | match for 0.01, **but GAMS uses 0.5 °C** (§4) |
| Lowest temperature without the bound | −0.12 °C | −0.115 (2420 horizon) | match at N = 81 only; −0.175 at longer horizons |
| Horizon table (2320/2420/2520) | Table 2 | identical to rounding | match |
| "Up to φ = 1.75 the SCC changes by less than 2%" | | 1.8% then **2.15%** at φ = 1.75 | **slightly overstated** |

## 2. Independent SCC verification

The multiplier SCC was compared with two methods that do not use the multipliers:
- **Re-optimization.** Re-solve the full planner problem with ε_E(t) = ±h and ε_C(t) = ±h. Take Richardson-extrapolated central differences of optimal welfare, with h_E = 0.1 and 0.05 GtCO₂/yr and h_C = 0.006 and 0.003 trillion $/yr.
- **Fixed-policy pulse.** Hold MIU and S fixed, re-solve α, and remove the temperature bound.

| φ | Bound | Year | Multiplier | Re-optimization | Gap | Fixed-policy pulse | Gap |
|---|---|---|---|---|---|---|---|
| 0.5 | none | 2025 / 2050 | 77.323 / 144.699 | 77.323 / 144.699 | <2e-6% | 77.323 / 144.699 | 0.00% |
| 0.95 | none | 2025 / 2050 | 59.067 / 124.541 | 59.067 / 124.541 | <2e-6% | same | 0.00% |
| 1.28 | none | 2025 / 2050 | 54.852 / 127.807 | 54.852 / 127.807 | <1e-6% | same | 0.00% |
| 1.5 | none | 2025 / 2050 | 57.198 / 145.413 | 57.198 / 145.413 | 3e-5% | same | 0.00% |
| 1.75 | none | 2025 / 2050 | 69.573 / 199.655 | 69.572 / 199.654 | 3e-4% | 69.573 / 199.655 | 0.00% |
| 2.0 | none | 2025 / 2050 | 140.266 / 461.892 | 140.266 / 461.890 | 3e-4% | 176.35 / 595.45 | 26% / 29% (invalid, see below) |
| 1.75 | 0.01 | 2025 / 2050 | 70.110 / 201.540 | 70.110 / 201.540 | 2e-4% | 73.84 / 214.99 | 5.3% / 6.7% |
| 2.0 | 0.01 | 2025 / 2050 | 156.12 / 520.24 | 156.12 / 520.24 | 7e-5% | 210.37 / 718.17 | 35% / 38% |
| 1.1 | 0.5 (GAMS) | 2025 / 2050 | 55.940 / 122.798 | 55.940 / 122.798 | <1e-6% | 56.40 / 124.20 | 0.8% / 1.1% |
| 1.28 | 0.5 (GAMS) | 2025 / 2050 | 54.440 / 126.484 | 54.440 / 126.484 | <1e-6% | 55.42 / 129.65 | 1.8% / 2.5% |
| 1.5 | 0.5 (GAMS) | 2025 / 2050 | 56.261 / 142.251 | 56.261 / 142.251 | 1e-5% | 60.84 / 157.95 | 8.1% / 11.0% |
| 2.0 | 0.5 (GAMS) | 2025 / 2050 | 109.55 / 357.26 | 109.55 / 357.26 | 1e-4% | 708.5 / 2567 | 547% / 619% |

Step-size consistency, comparing the estimates from h and h/2, is below 2e-4% everywhere.

- **The multiplier SCC passes.** In every case it equals the derivative of optimal welfare to at least six significant figures.
- **The fixed-policy pulse matches only where no state bound binds.** It equals the multiplier SCC whenever no inequality on a state binds, which covers φ ≤ 1.75 with no bound. Where a bound binds, the gap is the bound's shadow value.
- **The φ = 2, no-bound pulse value is not a valid estimate.** Two of its four perturbed solves are infeasible (IPOPT status 2). At φ = 2 the carbon-cycle factor α sits at its GAMS lower bound of 0.1 in the last five or six periods, 2395 to 2420, with positive multipliers. A one-tonne pulse with policy held fixed would push α below that floor.

## 3. Solver and constraint diagnostics

- **Fresh grid solves.** All 138 reach IPOPT status 0, except the no-precautionary-term runs at φ = 1.75 and 2.0. Those reach status 1 (acceptable) after about 4,000 iterations, and their values still agree with the baseline to 0.0015%. The largest scaled KKT stationarity residual is 4e-7, and the largest IRF residual is 1e-8.
- **Multi-start.** I used six starts at each φ: the GAMS optimum, the PyDICE default, three random starts (MIU and S scaled by U(0.8, 1.2)) and continuation from the neighbouring φ.

| φ | Welfare spread (relative) | SCC spread |
|---|---|---|
| 0.95 | 2e-12 | 2e-7% |
| 1.28 | 3e-12 | 1e-7% |
| 1.5 | 1e-12 | 5e-5% |
| 1.75 | 5e-13 | 3e-6% |
| 2.0 | 2e-9 | 1e-3% |

  No alternative local optimum was found. At φ = 2 the optimum is flatter.

- **Bounds that bind.**
  - MIU is at its upper bound in 45 periods at the default, which is normal for DICE-2023.
  - S is fixed from 2205, by design.
  - The MAT floor of 10 GtC is never active; the minimum is above 480 GtC.
  - α is at its 0.1 lower bound only at φ = 2, in the final periods for N = 81 and N = 101.
- **End-of-horizon behaviour at φ = 2.** Emissions jump from 0.5 GtCO₂/yr in 2405 to 59 GtCO₂/yr in 2420.

## 4. Temperature lower bound

The GAMS program bounds temperature at 0.5 °C, not 0.01 °C as the letter states. The table compares the three treatments.

| φ | GAMS 0.5 °C | Letter's 0.01 °C | No bound | Lowest T, no bound |
|---|---|---|---|---|
| 0.95 | 59.07 | 59.07 | 59.07 | 0.525 |
| 1.05 | **56.76** (binds) | 57.01 | 57.01 | 0.474 |
| 1.3 | 54.36 | 54.86 | 54.86 | 0.300 |
| 1.5 | 56.26 | 57.20 | 57.20 | 0.109 |
| 1.6 | 59.42 | 59.93 (binds) | 60.41 | 0.003 |
| 1.75 | 68.70 | 70.11 | 69.58 | −0.115 |
| 1.9 | 85.27 | 96.47 | 93.73 | −0.111 |
| 2.0 | 109.55 | 156.12 | 140.27 | −0.086 |

| Bound treatment | Minimum SCC | φ at the minimum | SCC back at default |
|---|---|---|---|
| GAMS 0.5 °C | $54.36 | 1.30 | φ = 1.59 |
| No bound | $54.86 | 1.285 | φ = 1.564 |

- **Below φ = 1.05 the bound does not matter.**
- **From φ = 1.05 to 1.75 it moves the SCC by 0.4 to 2%.** The U-shape, the location of its minimum and the point where the SCC returns to its default are essentially unchanged.
- **At φ ≥ 1.9 the bound treatment dominates the result.** At φ = 2 the SCC ranges from $110 to $156 across the three treatments, and the ordering is not monotone in the bound's tightness.
- **Under the GAMS bound the multiplier SCC partly prices the bound itself.** This happens from φ = 1.05; the fixed-policy pulse gap is 1.8% at φ = 1.28 and 8% at φ = 1.5.

## 5. Terminal horizon

SCC in 2025, with "dice" discounting and no bound.

| φ | 2320 | 2420 | 2520 | 2620 | 2720 | Last step |
|---|---|---|---|---|---|---|
| 0.5 | 77.14 | 77.32 | 77.35 | 77.35 | 77.35 | 0.00% |
| 0.95 | 58.84 | 59.07 | 59.11 | 59.12 | 59.13 | 0.00% |
| 1.25 | 54.38 | 54.91 | 55.08 | 55.14 | 55.17 | 0.04% |
| 1.5 | 55.99 | 57.20 | 58.08 | 58.65 | 59.04 | 0.66% |
| 1.6 | 58.99 | 60.41 | 62.18 | 63.54 | 64.61 | 1.7% |
| 1.75 | 68.34 | 69.58 | 71.07 | 71.25 | 71.63 | 0.5% |
| 1.9 | 89.94 | 93.73 | 97.11 | 102.04 | 109.15 | 7.0% (rising) |
| 2.0 | 127.84 | 140.27 | 160.26 | 202.59 | 291.47 | 44% (rising) |
| Term fixed, φ = 0.5 / 1.5 / 2.0 | | | | | | 0.04% / 0% / 0% |

How the SCC behaves as the horizon is extended:

| Range of φ | Behaviour |
|---|---|
| φ ≤ 1.25 | Converged |
| φ = 1.5 to 1.75 | Changes shrink but are still 0.5 to 2% per century: about 3% at φ = 1.5 and 1.75, and 7% at φ = 1.6, between 2420 and 2720 |
| φ ≥ 1.9 | Growing changes; no convergence within 2720 |

- **The counterfactual with the precautionary term fixed converges at every φ.** So the horizon dependence comes from the precautionary term.
- **The minimum stays near φ ≈ 1.25 to 1.3 at every horizon.** The return to the default SCC moves from φ = 1.56 at the 2420 horizon to about 1.50 at 2720.
- **The lowest temperature without the bound reaches −0.175 °C at longer horizons.**

## 6. Problems found

1. **Lower temperature bound misdescribed.** The letter calls 0.01 °C "the model's lower temperature bound". DICE-2023 GAMS uses 0.5 °C, which binds from φ = 1.05. The letter's runs are therefore not the GAMS program with a different φ in the range 1.05 to 1.55 either; there the bound is silently relaxed. The effect is small, at most 2%, but it has to be stated.
2. **φ = 2 results are not properties of DICE-2023.** These are the $140 SCC, the $116 precautionary contribution, 1.80 °C peak warming and 980 Gt of emissions. They depend on the horizon (+108% from 2420 to 2720), on the temperature-bound convention ($110 to $156), and at N = 81 on α hitting its floor in the final periods. The abstract and conclusion quote "$140 at 2".
3. **Horizon sentence slightly overstated.** "Less than 2% up to 1.75" is 2.15% at φ = 1.75 for 2420 → 2520, and drift continues at 1.5 to 1.75 beyond 2520.
4. **"$25" should be "$24".**
5. **"At most 0.12 °C below preindustrial" holds only at the 2420 horizon.**
6. **Methodological point.** At high φ under any lower temperature bound, the GAMS multiplier SCC includes the bound's shadow value. A simulation SCC (fixed-policy pulse) would then differ, by 1.8 to 8% at φ = 1.28 to 1.5 under the GAMS bound.

## 7. Assessment

- **The falling side and the minimum are verified.** The SCC falls from $77 at φ = 0.5 to about $55 at φ ≈ 1.3, by three independent methods, from every starting point, under all three bound conventions, and at every horizon from 2320 to 2720.
- **The rise from the minimum back to the default SCC at φ ≈ 1.5 to 1.6 is verified.** It is robust to starting point and bound convention. Its exact location moves by about 0.06 with the horizon.
- **The rise to about $70 at φ = 1.75 is verified numerically.** It is mildly horizon-dependent, about 3% over three centuries.
- **The values at φ ≥ 1.9 are not robust.** They should be presented as the region where the finite-horizon model stops giving a well-defined SCC, not as point estimates.

**Verdict: the U-shape over φ ∈ [0.5, 1.75] is credible. The headline "$140 at φ = 2" is not.**
