"""Risk aversion separated from intertemporal substitution (paper 2, v0.8). Results: results/p3_ez.json (resumable).
Case U1 (removal from 2070 or never), learned in 2050. The two states' post-learning welfare is combined with an
Epstein-Zin-type certainty equivalent with relative risk aversion g (removal3_stoch.TwoState(gamma=g)); g = elasmu (0.95 at the published
calibration, 0.001 at the 3 and 2 percent calibrations) is expected utility. For each g and p: SCC in 2025, abatement before learning, and the welfare of the optimist and
pessimist plans evaluated under the same objective (their post-learning policies are optimal in each state under any g)."""
import os, sys, json
import numpy as np
import removal3 as R3, removal3_stoch as ST
from p3_stoch import det_solve, SC, CASES, iy

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'results', 'p3_ez.json')
RUNS = json.load(open(os.path.join(HERE, 'results', 'p3_runs.json')))
MC = json.load(open(os.path.join(HERE, 'results', 'p3_mech.json')))
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
calS, aS, calF, aF = CASES['U1']; t = 2050; k = iy(t)
for c in (sys.argv[1:] or ('pub', 'd2')):
    sc = SC[c]; lam = RUNS[f'{c}_none']['lamC2025'] / 5
    xS, _ = det_solve(sc, calS, aS); xF, _ = det_solve(sc, calF, aF)
    plans = {}
    for lab, xc in (('optimist', xS), ('pessimist', xF)):
        xa, _ = det_solve(sc, calS, aS, fixed=(xc, k)); xb, _ = det_solve(sc, calF, aF, fixed=(xc, k))
        plans[lab] = (np.asarray(xa), np.asarray(xb))
    for g in (None, 2.0, 5.0, 10.0):
        for p in (0.25, 0.5, 0.75):
            key = f'{c}|{g}|{p}'
            if key in res:
                continue
            m = ST.TwoState(sc, calS, aS, calF, aF, p, t, gamma=g)
            x, info, scc, dt = m.solve(x0=m.start(xS, xF, common_from='S' if p >= 0.5 else 'F'))
            s = m.summary(x, info, scc); EU = -m.objective(x)
            ev = {}
            for lab, (xa, xb) in plans.items():
                cS, tS = m.split(xa); _, tF = m.split(xb)
                ev[lab] = -m.objective(np.concatenate([cS, tS, tF]))
            res[key] = {'status': s['status'], 'scc2025': s['scc2025'], 'obj': EU, 'abt_to_res': s['S']['cumABT_to_res'],
                        'voh_opt_T': (EU - ev['optimist']) / lam, 'voh_pes_T': (EU - ev['pessimist']) / lam,
                        'ce': MC['A_' + c]['curve'][str(p)]['scc2025'] if 'A_' + c in MC and str(p) in MC['A_' + c]['curve'] else None,
                        'peakF': s['F']['peakT'], 'peakS': s['S']['peakT']}
            json.dump(res, open(OUT, 'w'))
            r = res[key]
            print(f"{c} g={g} p={p}: st{r['status']} SCC {r['scc2025']:.3f} CE {r['ce']} abt {r['abt_to_res']:.1f} VoH opt {r['voh_opt_T']:.4f} pes {r['voh_pes_T']:.4f} {dt:.0f}s", flush=True)
print('done')
