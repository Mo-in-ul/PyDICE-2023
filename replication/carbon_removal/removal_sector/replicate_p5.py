"""One-command reproduction of headline results of the second review (11 Nov 2026). Companion to replicate.py and
replicate_p4.py.

    python replicate_p5.py        # about 5 minutes

Every check re-solves from scratch. The one exception is the last check, which re-solves the two states around the stored
reduced-space optimum for learning in 2070 (results/p3_reduced.json) instead of searching for it again. Reference values are those in the research record
(p2_report/removal_split_report.tex, Sections on discounting, horizon and low-temperature damages, robustness and
policy regret). Writes replication_report_p5.md."""
import os, sys, time, platform
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
import numpy as np
import removal3_stoch as ST
import dice2023_corrected as dm
from p4_common import det, capcal, VAR, A70, iy
from common import gams_session
from p5_regret import fixed, cum

rows = []


def check(name, value, ref, tol):
    ok = abs(value - ref) <= tol
    rows.append((name, value, ref, tol, ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}: {value:.6g} (reference {ref:.6g}, tolerance {tol:g})", flush=True)


def gap_pct(c):
    g = 0.5 * (c[0.0] + c[1.0]) - c[0.5]
    return 100 * g / c[0.5]


t0 = time.time()

# 1. Policy regret, published calibration, learned 2050, p = 0.5 (Table tab:regret)
k = iy(2050)
calS, calF = capcal(1.0), {}
xS, sS, _ = det(9, calS, 'NGM'); xF, sF, _ = det(9, calF, '')
lam = sF['lamC2025'] / 5
xE, sE, _ = det(9, capcal(0.5), 'NGM')
a = fixed(9, calS, 'NGM', xE, k); b = fixed(9, calF, '', xE, k)
m = ST.TwoState(9, calS, 'NGM', calF, '', 0.5, 2050)
x, info, scc, _ = m.solve(x0=m.start(xS, xF, common_from='S'))
s2 = m.summary(x, info, scc)
check('Regret of the expected-capacity plan, published, 2050, p = 0.5 (T$)', (s2['EU'] - 0.5 * (a['U'] + b['U'])) / lam, 0.0013770, 0.0001)
check('Two-state SCC 2025, published, 2050, p = 0.5 ($/t)', float(scc[2]), 60.7419, 0.002)
check('Abatement before learning, expected-capacity plan (GtCO2)', cum(a, k), 356.579, 0.05)

# 2. Discounting split: published session, utility discount 2%, elasticity 0.95 and 0.001, hard caps
for eta, ref in ((0.95, 0.6975), (0.001, 7.6747)):
    def mk():
        p, bb = gams_session(9); dm.apply_disc_prstp(p, prstp_value=0.02, elasmu_value=eta); return p, bb
    c = {s: det(9, capcal(s) if s else {}, 'NGM' if s else '', mk())[1]['scc2025'] for s in (0.0, 0.5, 1.0)}
    check(f'Premium (Jensen gap), utility discount 2%, elasticity {eta}, caps (%)', gap_pct(c), ref, 0.01)

# 3. Injection rate halved, published, hard caps
c = {s: det(9, dict(A70, RN_inf=5.0 * s, RG_inf=8.5 * s, RM_inf=4.0 * s) if s else {}, 'NGM' if s else '')[1]['scc2025'] for s in (0.0, 0.5, 1.0)}
check('Premium, geological injection 8.5 GtCO2/yr, published, caps (%)', gap_pct(c), 0.7771, 0.01)

# 4. Smooth low-temperature damages (marginal damage zero at 0.5 C), 2%, no storage limit
base = dict(VAR['D'], dam_van_t1=1.0)
c = {s: det(2, capcal(s, base) if s else dict(base), 'NGM' if s else '')[1]['scc2025'] for s in (0.0, 0.5, 1.0)}
check('Premium, damages vanishing at 0.5 C, 2%, no storage limit (%)', gap_pct(c), 9.7360, 0.02)

# 5. Regret of the expected-capacity plan, 2% calibration, learning in 2070, p = 0.5 (Table tab:main, the largest regret reported).
# The stochastic optimum is the stored reduced-space candidate (results/p3_reduced.json, reached from two starting points);
# both states are re-solved here with its controls before 2070 fixed, so the check confirms the welfare of that candidate.
import json
import removal3 as R3
k70 = iy(2070); calS2, calF2 = capcal(1.0), {}
xS2, _, prS2 = det(2, calS2, 'NGM'); xF2, sF2, _ = det(2, calF2, '')
lam2 = sF2['lamC2025'] / 5; N2 = prS2.N
rr = json.load(open(os.path.join('results', 'p3_reduced.json')))['U1_d2_2070|0.5|segment']
idx = np.concatenate([np.arange(b * N2, b * N2 + k70) for b in range(5)])
base = np.array(xF2, float); base[idx] = rr['xcommon']
def fixed_ctrl(cal, allow, xc):
    pr = R3.Problem(2, cal=cal, allow=allow); pr.fixed_blocks = range(5); pr.fixed = (np.asarray(xc), k70)
    x, info, r, d, _ = pr.solve(x0=np.asarray(xc)); return pr.summary(x, info, r, d)
EUs = 0.5 * fixed_ctrl(calS2, 'NGM', base)['U'] + 0.5 * fixed_ctrl(calF2, '', base)['U']
xE2, _, _ = det(2, capcal(0.5), 'NGM')
EUe = 0.5 * fixed(2, calS2, 'NGM', xE2, k70)['U'] + 0.5 * fixed(2, calF2, '', xE2, k70)['U']
check('Regret of the expected-capacity plan, 2%, 2070, p = 0.5 (T$)', (EUs - EUe) / lam2, 0.5252, 0.002)

npass = sum(r[4] for r in rows)
with open('replication_report_p5.md', 'w') as f:
    f.write(f"# Replication report, second review\n\n{npass} of {len(rows)} checks pass. Run time {time.time() - t0:.0f} s. "
            f"Python {platform.python_version()}, numpy {np.__version__}.\n\n| Check | Value | Reference | Tolerance | Result |\n| --- | --- | --- | --- | --- |\n")
    for n, v, r, t, ok in rows:
        f.write(f"| {n} | {v:.6g} | {r:.6g} | {t:g} | {'PASS' if ok else 'FAIL'} |\n")
print(f"{npass}/{len(rows)} pass")
sys.exit(0 if npass == len(rows) else 1)
