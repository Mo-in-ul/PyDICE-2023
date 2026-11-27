"""One-command reproduction of the revision-analysis results of 11 Nov 2026 (p4 runs). Companion to replicate.py.

    python replicate_p4.py        # about 10 minutes

Every check re-solves from scratch; nothing is read from results/. Reference values are those reported in
ANALYSIS_2026-11-11.md and in the analysis record. Writes replication_report_p4.md."""
import os, sys, time, platform, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
import numpy as np
import removal3_stoch as ST, removal3_nstate as NS
from p4_common import det, capcal, VAR, A70, session

rows = []


def check(name, value, ref, tol):
    ok = abs(value - ref) <= tol
    rows.append((name, value, ref, tol, ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}: {value:.6g} (reference {ref:.6g}, tolerance {tol:g})", flush=True)


t0 = time.time()
# A1, A2: Jensen gap at 2%, abatement free and control rate held at the no-removal path
for v, ref_free, ref_held in (('D', 17.553, 19.714), ('A', 17.719, 20.824)):
    x0n, s0, _ = det(2, dict(VAR[v]), '')
    free = {s: det(2, capcal(s, VAR[v]), 'NGM')[1]['scc2025'] for s in (0.5, 1.0)}
    held = {}
    for s in (0.5, 1.0):
        xf, _, _ = det(2, capcal(s, VAR[v]), 'NGM'); N = len(x0n) // 6; xs = np.array(xf); xs[:N] = np.asarray(x0n)[:N]
        held[s] = det(2, capcal(s, VAR[v]), 'NGM', fix=x0n, x0=xs)[1]['scc2025']
    g = lambda c: 0.5 * (s0['scc2025'] + c[1.0]) - c[0.5]
    lab = 'no storage limit' if v == 'D' else 'hard caps'
    check(f'A2 2%, {lab}: Jensen gap, abatement re-optimised ($/t)', g(free), ref_free, 0.01)
    check(f'A1 2%, {lab}: Jensen gap, control rate held ($/t)', g(held), ref_held, 0.01)

# B3: engineered-only failure, published, removal from 2070, p = 0.5, learned 2050
xS, sS, _ = det(9, A70, 'NGM'); xF, sF, _ = det(9, A70, 'N')
_, sE, _ = det(9, dict(A70, RG_inf=8.5, RM_inf=2.0), 'NGM')
m = ST.TwoState(9, A70, 'NGM', A70, 'N', 0.5, 2050)
x, info, scc, _ = m.solve(x0=m.start(xS, xF, common_from='S'))
check('B3 published, engineered-only failure: premium ($/t)', float(scc[2]) - sE['scc2025'], 0.7826, 0.002)

# B1: three-state lottery {0, 0.5, 1} with probabilities (0.25, 0.5, 0.25), published, hard caps
d = {s: det(9, *(({}, '') if s == 0 else (capcal(s), 'NGM'))) for s in (0.0, 0.5, 1.0)}
m3 = NS.NState(9, [({}, '', 0.25), (capcal(0.5), 'NGM', 0.5), (capcal(1.0), 'NGM', 0.25)], 2050)
x, info, scc, _ = m3.solve(m3.start([d[s][0] for s in (0.0, 0.5, 1.0)], common_from=1))
check('B1 published, three-state lottery: premium ($/t)', float(scc[2]) - d[0.5][1]['scc2025'], 0.4730, 0.002)
check('B1 published, three-state lottery: premium minus Jensen gap ($/t)',
      float(scc[2]) - (0.25 * d[0.0][1]['scc2025'] + 0.5 * d[0.5][1]['scc2025'] + 0.25 * d[1.0][1]['scc2025']), -0.0004, 0.002)

# B5: value of learning in 2040 rather than 2070, published, p = 0.5 (2070 = reduced-space optimum of replicate.py)
lam = d[0.0][1]['lamC2025'] / 5; xS, _, _ = det(9, A70, 'NGM'); xF = d[0.0][0]
m = ST.TwoState(9, A70, 'NGM', {}, '', 0.5, 2040)
x, info, scc, _ = m.solve(x0=m.start(xS, xF, common_from='S'))
check('B5 published: value of learning in 2040 rather than 2070 (T$)', (-m.objective(x) - 6659.014363352142) / lam, 0.0334, 0.001)

# C2: elasticity 1.5 with discount factors re-matched (kappa 0.5069) gives the published SCC
_, s15, _ = det(9, {}, '', session(9, elasmu=1.5, rr_kappa=0.5069))
check('C2 elasticity 1.5, re-matched: no-removal SCC 2025 ($/t)', s15['scc2025'], 63.0093, 0.02)

# C1: AR6 carbon-cycle parameters (separate process: p4_carbon replaces the carbon-cycle residual at import)
code = ("import p4_carbon as C; from p4_common import det; A=dict(startN=2110.0,startG=2110.0,startM=2110.0);"
        "a=det(9,{},'',C.sess_ar6(9))[1]; b=det(9,A,'NGM',C.sess_ar6(9))[1]; print('RESULT', max(b['paths']['T'][:37]) - max(a['paths']['T'][:37]))")
proc = subprocess.run([sys.executable, '-W', 'ignore', '-c', code], capture_output=True, text=True, cwd=HERE)
res = [l for l in proc.stdout.splitlines() if l.startswith('RESULT')]
if res:
    check('C1 published, AR6 parameters: change in peak to 2200, removal from 2110 (C)', float(res[0].split()[1]), 0.131, 0.002)
else:                                            # report the failure as a FAIL row instead of stopping the script
    print('C1 subprocess failed:\n' + proc.stderr[-2000:], flush=True)
    check('C1 published, AR6 parameters: change in peak to 2200, removal from 2110 (C) [subprocess failed]', float('nan'), 0.131, 0.002)

with open('replication_report_p4.md', 'w') as f:
    f.write(f"# Replication report, revision analysis (p4)\n\nPython {platform.python_version()}, {platform.platform()}, run time {time.time() - t0:.0f} s\n\n")
    f.write('| Check | Recomputed | Reference | Tolerance | Result |\n|---|---|---|---|---|\n')
    for n, v, r_, tl, ok in rows:
        f.write(f"| {n} | {v:.6g} | {r_:.6g} | {tl:g} | {'pass' if ok else 'FAIL'} |\n")
print(f"{sum(r[-1] for r in rows)}/{len(rows)} checks passed; report in replication_report_p4.md")
