"""
benchmark/timing_benchmark.py
==============================
Measures PyDICE-2023 forward-pass speed and reports the Numba JIT speedup
over a pure-Python reference implementation.

Usage
-----
    python benchmark/timing_benchmark.py

Output
------
Prints a summary table and writes results to benchmark/timing_results.csv.
"""

import timeit
import csv
import os
import numpy as np

# ── Import PyDICE-2023 ──────────────────────────────────────────────────────
from dice import LoadParams, diceTrajectory


# ── Pure-Python reference (no Numba) ────────────────────────────────────────
def diceTrajectory_python(p, MIU, S, Alpha):
    """Uncompiled Python forward pass — used only as a timing baseline."""
    N = p['num_periods']
    TATM   = np.zeros(N + 1); TATM[1]   = p['tatm0']
    MAT    = np.zeros(N + 1); MAT[1]    = p['mat0']
    K      = np.zeros(N + 1); K[1]      = p['k0']
    C      = np.zeros(N + 1)
    Y      = np.zeros(N + 1)
    CCATOT = np.zeros(N + 1); CCATOT[1] = p['CumEmiss0']
    RES0 = p['res00']; RES1 = p['res10']
    RES2 = p['res20']; RES3 = p['res30']
    TBOX1 = p['tbox10']; TBOX2 = p['tbox20']
    F_GHG = p['F_GHGabate2020']

    YGROSS1  = p['eco2Param'][1] * (K[1] ** p['gama'])
    DAMFRAC1 = p['a1'] * TATM[1] + p['a2base'] * TATM[1] ** p['a3']
    Y[1] = YGROSS1 * (1 - DAMFRAC1) - p['cost1tot'][1] * (MIU[1] ** p['expcost2']) * YGROSS1
    I    = S[1] * Y[1]
    C[1] = Y[1] - I
    PERIODU = np.zeros(N + 1)

    for i in range(2, N + 1):
        miu_i   = min(max(MIU[i],   0.0),           p['miuup'][i])
        s_i     = min(max(S[i],     p['sLBounds'][i]), p['sUBounds'][i])
        alpha_i = max(Alpha[i], p['AlphaLowerBound'])

        K[i]      = (1 - p['dk']) ** p['tstep'] * K[i - 1] + p['tstep'] * I
        YGROSS    = p['eco2Param'][i] * (K[i] ** p['gama'])
        ECO2      = p['sigma'][i] * YGROSS * (1 - miu_i) + p['eland'][i]
        CCATOT[i] = CCATOT[i - 1] + ECO2 * p['tstep'] / 3.667
        F_GHG     = p['Fcoef2'] * F_GHG + p['Fcoef1'] * p['CO2E_GHGabateB'][i] * (1 - miu_i)
        inflow    = ECO2 / 3.667

        exp0 = np.exp(-p['tstep'] / (p['tau0'] * alpha_i))
        exp1 = np.exp(-p['tstep'] / (p['tau1'] * alpha_i))
        exp2 = np.exp(-p['tstep'] / (p['tau2'] * alpha_i))
        exp3 = np.exp(-p['tstep'] / (p['tau3'] * alpha_i))
        RES0 = p['emshare0'] * p['tau0'] * alpha_i * inflow * (1 - exp0) + RES0 * exp0
        RES1 = p['emshare1'] * p['tau1'] * alpha_i * inflow * (1 - exp1) + RES1 * exp1
        RES2 = p['emshare2'] * p['tau2'] * alpha_i * inflow * (1 - exp2) + RES2 * exp2
        RES3 = p['emshare3'] * p['tau3'] * alpha_i * inflow * (1 - exp3) + RES3 * exp3

        MAT[i]  = p['mateq'] + RES0 + RES1 + RES2 + RES3
        FORC    = (p['fco22x'] * np.log(MAT[i] / p['mateq']) / np.log(2)
                   + p['F_Misc'][i] + F_GHG)
        TBOX1   = (TBOX1 * np.exp(-p['tstep'] / p['d1'])
                   + p['teq1'] * FORC * (1 - np.exp(-p['tstep'] / p['d1'])))
        TBOX2   = (TBOX2 * np.exp(-p['tstep'] / p['d2'])
                   + p['teq2'] * FORC * (1 - np.exp(-p['tstep'] / p['d2'])))
        TATM[i] = min(max(TBOX1 + TBOX2, 0.01), 20.0)

        DAMFRAC  = p['a1'] * TATM[i] + p['a2base'] * TATM[i] ** p['a3']
        ABATE    = p['cost1tot'][i] * (miu_i ** p['expcost2']) * YGROSS
        Y[i]     = YGROSS * (1 - DAMFRAC) - ABATE
        I        = s_i * Y[i]
        C[i]     = Y[i] - I
        CPC      = 1000.0 * C[i] / p['L'][i]
        PERIODU[i] = ((CPC ** (1 - p['elasmu'])) - 1) / (1 - p['elasmu']) - 1

    W = (p['tstep'] * p['scale1']
         * np.sum(PERIODU[1:] * p['L'][1:] * p['RR'][1:])
         + p['scale2'])
    return W


# ── Main benchmark ───────────────────────────────────────────────────────────
def run_benchmark(n_numba: int = 1000, n_python: int = 50) -> dict:
    p = LoadParams(81)
    N = p['num_periods']

    MIU   = np.full(N + 1, 0.5);  MIU[1]   = p['miu1']
    S     = np.full(N + 1, 0.25)
    Alpha = np.full(N + 1, p['a0'])

    # ── Warm up Numba (first call triggers JIT compilation) ─────────────────
    print("Warming up Numba JIT (first call compiles — this takes ~20s) ...")
    diceTrajectory(p, MIU.copy(), S.copy(), Alpha.copy())
    print("  JIT compilation complete.\n")

    # ── Time Numba (compiled) ────────────────────────────────────────────────
    t_start = timeit.default_timer()
    for _ in range(n_numba):
        diceTrajectory(p, MIU.copy(), S.copy(), Alpha.copy())
    t_numba_ms = (timeit.default_timer() - t_start) / n_numba * 1000

    evals_per_sec = 1000 / t_numba_ms

    # ── Time pure Python ─────────────────────────────────────────────────────
    print("Timing pure-Python reference ...")
    t_start = timeit.default_timer()
    for _ in range(n_python):
        diceTrajectory_python(p, MIU.copy(), S.copy(), Alpha.copy())
    t_python_ms = (timeit.default_timer() - t_start) / n_python * 1000

    speedup = t_python_ms / t_numba_ms

    return {
        'n_numba_calls'    : n_numba,
        'n_python_calls'   : n_python,
        't_numba_ms'       : t_numba_ms,
        't_python_ms'      : t_python_ms,
        'evals_per_second' : evals_per_sec,
        'speedup'          : speedup,
    }


def print_results(r: dict) -> None:
    print("=" * 55)
    print("PyDICE-2023  —  Forward-Pass Timing Benchmark")
    print("=" * 55)
    print(f"  Numba compiled   : {r['t_numba_ms']:.4f} ms / call"
          f"  ({r['n_numba_calls']} calls)")
    print(f"  Pure Python      : {r['t_python_ms']:.2f} ms / call"
          f"  ({r['n_python_calls']} calls)")
    print(f"  Evaluations/sec  : {r['evals_per_second']:,.0f}")
    print(f"  Speedup (Numba)  : {r['speedup']:.0f}x")
    print(f"  Log10(speedup)   : {np.log10(r['speedup']):.2f}")
    print("=" * 55)
    print(f"\nManuscript quote (Section 3.2):")
    print(f"  'A complete 405-year trajectory evaluates in "
          f"{r['t_numba_ms']:.3f} ms ({r['evals_per_second']:,.0f} "
          f"evaluations per second), a {round(r['speedup'] / 10) * 10:.0f}x "
          f"speedup over the pure-Python baseline.'")


def save_csv(r: dict, path: str = "benchmark/timing_results.csv") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=r.keys())
        writer.writeheader()
        writer.writerow(r)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    results = run_benchmark(n_numba=1000, n_python=50)
    print_results(results)
    save_csv(results)
