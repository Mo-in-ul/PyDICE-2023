"""One-command reproduction of the headline results of paper 2 (anticipated carbon removal in DICE-2023).

    python replicate.py            # core checks, about 3 minutes (165 s on the reference machine)
    python replicate.py --full     # adds the reduced-space solution with learning in 2070 (about 1 hour)

Every check re-solves the model from scratch; nothing is read from results/. Each line compares the recomputed value
with the reference value reported in the paper and passes if the difference is within the stated tolerance.
A report is written to replication_report.md. Requires the three folders hidden/, paper_build/ and notebook/
(with notebook/gams_reference/*.xlsx) side by side, and the packages in requirements.txt."""
import os, sys, time, platform
import numpy as np, jax.numpy as jnp
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'paper_build')); sys.path.insert(0, os.path.join(HERE, '..', 'notebook'))
import common, dice2023_corrected as dm
import removal3 as R3, removal3_stoch as ST
from p3_stoch import det_solve, iy

FULL = '--full' in sys.argv
REF70 = (6659.014363352142, 60.73989294131984)     # reduced-space optimum, learning in 2070, published calibration, p = 0.5
rows = []


def check(name, value, ref, tol):
    ok = abs(value - ref) <= tol
    rows.append((name, value, ref, tol, ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}: {value:.6g} (reference {ref:.6g}, tolerance {tol:g})", flush=True)


def solve(sc, cal=None, allow='', fixed=None):
    pr = R3.Problem(sc, cal=cal or {}, allow=allow)
    if fixed is not None:
        pr.fixed = fixed
    x, info, r, d, _ = pr.solve(x0=None if fixed is None else fixed[0]); return x, pr.summary(x, info, r, d)


t0 = time.time()
# 1. Base model: the corrected DICE-2023 forward model reproduces the published GAMS solution (scenario 9)
p, b = common.gams_session(9); g = dm.load_gams(9); T, res = dm.model_only_check(p, g)
check('Base model: max |T - T_GAMS| on GAMS controls (C)', float(np.max(np.abs(T - g['TATM'][:p['num_periods']]))), 0.0, 1e-8)

# 2. Deterministic results (published calibration = 9, 2 percent = 2)
A70 = dict(startN=2070.0, startG=2070.0, startM=2070.0)
REF = {'pub_none': (63.0093, 2.4728), 'd2_none': (282.1484, 1.8572), 'pub_NGM': (58.0355, 2.2738), 'd2_NGM': (210.9790, 1.8446)}
sol = {}
for key, (sc, allow) in {'pub_none': (9, ''), 'd2_none': (2, ''), 'pub_NGM': (9, 'NGM'), 'd2_NGM': (2, 'NGM')}.items():
    x, s = solve(sc, allow=allow); sol[key] = (x, s)
    check(f'{key}: SCC 2025 ($/tCO2)', s['scc2025'], REF[key][0], 1e-3); check(f'{key}: peak warming (C)', s['peakT'], REF[key][1], 1e-4)
# welfare value of removal, trillion 2019 dollars of 2025 consumption
for c in ('pub', 'd2'):
    v = (sol[f'{c}_NGM'][1]['U'] - sol[f'{c}_none'][1]['U']) / (sol[f'{c}_none'][1]['lamC2025'] / 5)
    check(f'{c}: value of removal (T$)', v, {'pub': 30.96, 'd2': 283.23}[c], 0.01)
# late access raises the peak
_, s = solve(9, cal=dict(startN=2110.0, startG=2110.0, startM=2110.0), allow='NGM')
check('pub: peak with removal from 2110 minus no-removal peak (C)', s['peakT'] - sol['pub_none'][1]['peakT'], 0.1052, 1e-4)

# 3. Two-state model, removal from 2070 or never, p = 0.5, learned 2050
for c, sc in (('pub', 9), ('d2', 2)):
    xS, sS = solve(sc, cal=A70, allow='NGM'); xF, sF = sol[f'{c}_none']
    k = iy(2050); lam = sF['lamC2025'] / 5
    m = ST.TwoState(sc, A70, 'NGM', {}, '', 0.5, 2050)
    x, info, scc, _ = m.solve(x0=m.start(xS, xF, common_from='S')); EU = -m.objective(x)
    _, oS = solve(sc, cal=A70, allow='NGM', fixed=(xS, k)); _, oF = solve(sc, allow='', fixed=(xS, k))
    _, pS = solve(sc, cal=A70, allow='NGM', fixed=(xF, k))
    EUo = 0.5 * oS['U'] + 0.5 * oF['U']; EUp = 0.5 * pS['U'] + 0.5 * sF['U']
    check(f'{c}: two-state SCC 2025', float(scc[2]), {'pub': 60.7419, 'd2': 248.4375}[c], 2e-3)
    vo, vp = (EU - EUo) / lam, (EU - EUp) / lam
    check(f'{c}: value of hedging vs optimist (T$), learned 2050', vo, {'pub': 0.00861, 'd2': 0.04289}[c], 5e-4)
    check(f'{c}: value of hedging vs pessimist (T$), learned 2050', vp, {'pub': 0.00843, 'd2': 0.04337}[c], 5e-4)
    # 4. premium over the expected-capacity benchmark and its exact decomposition
    CAP = dict(RN_inf=5.0, RG_inf=17.0, RM_inf=4.0)
    _, sE = solve(sc, cal=dict(A70, **{kk: 0.5 * v for kk, v in CAP.items()}), allow='NGM')
    prem = float(scc[2]) - sE['scc2025']
    check(f'{c}: premium over expected-capacity SCC (%)', 100 * prem / sE['scc2025'], {'pub': 1.582, 'd2': 7.68}[c], 0.01)
    aS, aF = [np.asarray(z) for z in m.assemble(jnp.asarray(x))]
    cs = []
    for cal_, al_, xa in ((A70, 'NGM', aS), ({}, '', aF)):
        pr = R3.Problem(sc, cal=cal_, allow=al_); pr.fixed = (xa, k); pr.fixed_blocks = range(5)
        xx, inf_, rr, dd, _ = pr.solve(x0=xa); cs.append(pr.summary(xx, inf_, rr, dd)['scc2025'])
    jensen = 0.5 * sS['scc2025'] + 0.5 * sF['scc2025'] - sE['scc2025']
    check(f'{c}: weighting residual relative to SCC, SCC_2s - E[SCC_s along common path]', (float(scc[2]) - 0.5 * cs[0] - 0.5 * cs[1]) / float(scc[2]), 0.0, 1e-5)
    check(f'{c}: Jensen-gap share of premium', jensen / prem, 1.0, 0.02)

if FULL:   # 5. learning in 2070, reduced-space solution
    import p3_reduced                          # starts from the segment solution in results/p3_hedge_line.json
    o = p3_reduced.run('U1_pub_2070', 0.5, 'segment')
    lam = sol['pub_none'][1]['lamC2025'] / 5
    check('pub, learned 2070: two-state SCC 2025 (reduced space)', o['scc2025'], REF70[1], 2e-3)
    check('pub, learned 2070: expected welfare, T$ relative to reference', (o['EU'] - REF70[0]) / lam, 0.0, 1e-3)

with open('replication_report.md', 'w') as f:
    f.write(f"# Replication report\n\nPython {platform.python_version()}, {platform.platform()}, run time {time.time() - t0:.0f} s\n\n")
    f.write('| Check | Recomputed | Reference | Tolerance | Result |\n|---|---|---|---|---|\n')
    for n, v, r_, tl, ok in rows:
        f.write(f"| {n} | {v:.6g} | {r_:.6g} | {tl:g} | {'pass' if ok else 'FAIL'} |\n")
print(f"{sum(r[-1] for r in rows)}/{len(rows)} checks passed; report in replication_report.md")
