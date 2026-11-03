# Stage 2 report: when does the elasticity–SCC reversal occur?

**Setup.** Isolated code is in `paper_build/stage2/` (`s2_core.py`, `s2_sigma.py`, `s2_sigma_ext.py`, `s2_rho.py`, `s2_rho_horizon.py`). The baseline code and the manuscript were not changed.

**Baseline.** The GAMS model with only the named parameters changed. This includes the GAMS lower temperature bound of 0.5 °C, with runs without the bound as a robustness check. Unless stated otherwise:
- the horizon is N = 81 (ending 2420);
- σ = 1%, cap = 235 years, ρₑ = 2.634%;
- all SCCs are for 2025, in 2019 $/tCO₂.

**Literature basis for σ.** B&N define σ as the standard deviation of the average growth rate of consumption per person, and give no source for it. Two literature estimates are close to 1%:
- Christensen, Gillingham and Nordhaus (2018) report an SD of 1.1 percentage points for growth to 2100.
- Müller, Stock and Watson put the 16th–84th percentile range of 100-year world per capita growth at 1.06–2.96%.

The grid of 0.5 to 1.5% therefore brackets the literature value. The 235-year cap appears only in the program code.

## 1. Growth uncertainty σ

| σ | SCC at φ = 0.5 / 0.95 / 1.3 / 1.75 | Minimum (φ, SCC) | SCC back at its φ = 0.95 value |
|---|---|---|---|
| 0.5% | 72.9 / 49.1 / 38.6 / 30.3 | none up to φ = 3.0 (19.7 at 3.0) | never |
| 0.75% | 74.6 / 52.6 / 43.2 / 36.3 | about 2.2, $33.7 | above 3 or never |
| **1.0% (DICE)** | 77.3 / 59.1 / 54.4 / 68.7 | **1.30, $54.4** | 1.59 |
| 1.25% | 80.9 / 71.9 / 95.5 / 1,152 | 0.87, $71.5 | 0.95 |
| 1.5% | 86.2 / 103.2 / 569 / 1.6 million | 0.64, $84.2 | 0.95 |

All main-grid solves converged (IPOPT status 0, or 1 = acceptable).

## 2. Precautionary cap (σ = 1%)

| Cap | SCC at φ = 0.5 / 0.95 / 1.3 / 1.5 / 1.75 | Minimum |
|---|---|---|
| 100 years | 76.0 / 55.7 / 47.7 / 44.8 / 42.6 | none, falls throughout |
| **235 years (DICE)** | 77.3 / 59.1 / 54.4 / 56.3 / 68.7 | 1.30 |
| No cap within the horizon | 77.5 / 59.6 / 63.3 / 110 / 1,745 | 1.13, $58.0 |

## 3. Holding the near-term discount rate fixed

**Method.** For each φ, ρₑ is chosen so that the consumption discount rate from 2020 over h years equals its value at φ = 0.95. I used three windows: h = 50, 80 and 100. In the code ρ and β enter only through ρₑ, so each required ρₑ is reported two ways: as ρ with β = 0.5, and as β with ρ = 0.1%. Every root-find converged, with a residual below 1e-7.

| h (target R) | SCC at φ = 0.5 / 0.95 / 1.3 / 1.5 / 1.75 | ρ (β fixed) at 0.5 / 1.75 | ρ turns negative at | β (ρ fixed) range |
|---|---|---|---|---|
| 50 years (4.28%) | 53.0 / 59.1 / 72.0 / 89.0 / 141.5 | +0.78% / −0.64% | φ ≈ 1.05 | 0.64 → 0.35 |
| 80 years (4.07%) | 56.5 / 59.1 / 65.8 / 76.1 / 107.0 | +0.65% / −0.40% | φ ≈ 1.06 | 0.61 → 0.40 |
| 100 years (3.94%) | 59.0 / 59.1 / 62.4 / 68.7 / 88.0 | +0.57% / −0.18% | φ ≈ 1.10 | 0.59 → 0.44 |

The horizon checks for h = 80 are below; they cover N = 61 to 121, ending 2320 to 2620.

| φ | 2320 | 2420 | 2520 | 2620 |
|---|---|---|---|---|
| 0.95 | 58.8 | 59.1 | 59.1 | 59.1 |
| 1.25 | 63.8 | 64.1 | 64.3 | 64.3 |
| 1.4 | 69.3 | 70.0 | 70.4 | 70.7 |
| 1.5 | 75.4 | 76.1 | 77.0 | 77.8 |
| 1.75 | 103.0 | 107.0 | 116.1 | 141.3 |

With near-term discounting held fixed, the SCC never falls as φ rises: it is flat to rising from φ = 0.5. This holds for every window, and is horizon-stable up to about φ = 1.4. In DICE-2023, therefore, the textbook "higher φ, lower SCC" effect works entirely through near-term discounting.

Holding that rate through pure time preference needs ρ < 0 from φ ≈ 1.05–1.1. Holding it through the climate beta keeps β between 0.35 and 0.64.

## 4. Horizon sensitivity and the long-run discount rate

After the cap, the forward utility discount rate is constant at about (1+ρₑ)(1 − ½φ²σ²·cap) − 1. It is zero at

φ₀ = √(2ρₑ / ((1+ρₑ) σ² · cap)).

The table compares φ₀ with the minimum and with the first φ at which the SCC changes by more than 1% between the 2420 and 2620 horizons. The φ values checked were 0.5, 0.75, 1.0, 1.15, 1.3, 1.5 and 1.75.

| Case | φ₀ | φ at the minimum | Horizon sensitivity first exceeds 1% at | Breakdown (non-convergence or negative SCC at 2620) |
|---|---|---|---|---|
| σ = 0.5% | 2.96 | none up to 3.0 | none up to 2.8 | none up to 2.8 (acceptable or iteration-limit status above 2.3) |
| σ = 0.75% | 1.97 | about 2.2 | none up to 1.75 (2620 check above 1.75 pending) | |
| σ = 1.0% | 1.48 | 1.30 | 1.75 (+5.7%); 1.5: +0.9% | |
| σ = 1.25% | 1.18 | 0.87 | 1.15 (+1.3%); 1.3: +6.8% | 1.75 |
| σ = 1.5% | 0.99 | 0.64 | 1.0 (+4.0%) | 1.3 |
| Cap 100 years | 2.27 | none up to 1.75 | none | |
| No cap | 1.13 | 1.13 | 1.0 (+2.0%); 1.15: +11% | 1.5 |
| Recalibrated, h = 80 | about 1.37 | none (rising throughout) | 1.5 (+2.2%); 1.4: +1.0% | |

What the table shows:
- **φ₀ predicts where horizon sensitivity begins.** In every case it starts within about 0.1–0.15 of φ₀. No case was run in which this fails.
- **The minimum lies below φ₀ in every case except σ = 0.75%.** Across cases it falls between 0.6 and 1.1 times φ₀. The ratio is not constant, so neither the minimum nor φ₀ follows a simple scaling law in σ, which is consistent with your instruction to test that relation rather than impose it.
- **In the default calibration the rise from the minimum is real but short.** The SCC goes from $54.4 at 1.30 to $56.3 at 1.5, which is horizon-stable to 0.9%. Above about 1.5 the level drifts with the horizon, while the qualitative rise persists.
- **No bound against the 0.5 °C bound.** Removing the bound changes the SCC by less than 2% wherever the result is horizon-stable. In the unstable region the two differ widely, as expected.

## 5. Answers to the Stage 2 questions

1. **Does the U exist at other economically reasonable σ?** Not within φ ≤ 1.75 for σ ≤ 0.75%. At 0.75% a shallow minimum appears near φ ≈ 2.2, and at 0.5% there is none up to φ = 3. It does exist for σ ≥ 1%, with the minimum moving to lower φ as σ rises. Above σ = 1% the rising side lies mostly in the region where the finite-horizon SCC does not settle.
2. **Does it survive recalibration to a fixed near-term discount rate?** It turns into something stronger: with near-term discounting held fixed, the SCC rises with φ from the start.
3. **When does it occur?** The SCC turns up shortly before φ reaches the value at which DICE-2023's long-run utility discount rate becomes zero. That depends jointly on ρₑ, σ and the cap.

## 6. Implication for the paper

The reversal is not a general property of the extended Ramsey rule. It is a near-boundary phenomenon of DICE-2023's calibration. With σ = 1%, the 235-year cap and ρₑ = 2.6%, the boundary φ₀ ≈ 1.5 sits inside the commonly used range of φ, so the U shows up there.

A more general and defensible headline would combine three findings:
- In DICE-2023 the elasticity lowers the SCC only through near-term discounting.
- The precautionary term pushes the other way, with a strength set by φ²σ²·cap.
- Where φ²σ²·cap approaches 2ρₑ, the SCC turns up and then stops being horizon-robust.

**Open items:**
- The 2620-horizon check for σ = 0.75% at φ > 1.75 is still running; it is needed to confirm the minimum near 2.2.
- The damage-period decomposition, which would explain the mechanism.
- Re-solving the letter's figures under the chosen framing.
