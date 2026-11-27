"""Diagnostics for the reduced-space two-state solutions (paper 2, v0.7). Results: results/p3_diag.json.
For every reduced-space solution: number of outer variables, best expected welfare and the spread across starting points,
the scaled projected gradient, and the largest constraint and bound violation of the two inner solutions re-solved at the
reported outer variables. For learning in 2050 the reduced-space optimum is compared with the full-space (joint) optimum."""
import os, json
import numpy as np
import removal3 as R3
import p3_reduced as PR
from p3_stoch import SC, CASES, iy

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'results', 'p3_diag.json')
RD = json.load(open(os.path.join(HERE, 'results', 'p3_reduced.json')))
ST = json.load(open(os.path.join(HERE, 'results', 'p3_stoch.json')))
RUNS = json.load(open(os.path.join(HERE, 'results', 'p3_runs.json')))
import p3_prec
CASES.update(p3_prec.CASES)
groups = {}
for tag, r in RD.items():
    key, p, st = tag.split('|'); groups.setdefault((key, p), []).append((st, r))
res = {}
for (key, p), lst in sorted(groups.items()):
    case, c, t = key.rsplit('_', 2); t = int(t); k = iy(t); sc = SC[c]
    calS, allowS, calF, allowF = CASES[case]
    lam = RUNS[f'{c}_none']['lamC2025'] / 5
    best = max(lst, key=lambda z: z[1]['EU'])[1]
    EUs = [z[1]['EU'] for z in lst]
    viol = []
    for cal, allow, ref in ((calS, allowS, 'S'), (calF, allowF, 'F')):
        S = PR.State(sc, cal, allow, k); xd, *_ = S.pr.solve(); N = S.pr.N
        idx = np.concatenate([np.arange(b * N, b * N + k) for b in range(PR.NCTRL)])
        xf = np.array(xd, float); xf[idx] = np.asarray(best['xcommon'])
        mask = np.zeros(S.pr.nv, bool); mask[idx] = True; S.mask = mask; S.xref = np.array(xd)
        u, g, stt, (x, info, r, d) = S.solve(xf)
        S.pr.fixed = None; xl, xu, gl, gu = S.pr.bounds(); gg = S.pr.constraints(x)
        viol.append(max(float(np.max(gl - gg)), float(np.max(gg - gu)), float(np.max(xl - x)), float(np.max(x - xu)), 0.0))
        nfree = int(np.sum(xu[idx] - xl[idx] > 1e-12)) if ref == 'F' else None
    out = {'n_outer': nfree, 'n_starts': len(lst), 'EU': best['EU'], 'start_spread_T': (max(EUs) - min(EUs)) / lam if len(lst) > 1 else None,
           'proj_grad_T': best['proj_grad_max_T'], 'max_violation': max(viol), 'starts': [z[0] for z in lst]}
    if t == 2050 and key in ST and p in ST[key]['stoch']:
        out['joint_minus_reduced_T'] = (ST[key]['stoch'][p]['EU'] - best['EU']) / lam
    res[f'{key}|{p}'] = out; json.dump(res, open(OUT, 'w'))
    print(key, p, {a: (round(b, 7) if isinstance(b, float) else b) for a, b in out.items()}, flush=True)
print('done')
