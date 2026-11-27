"""Solver-tolerance check for small welfare differences (paper 2, v0.6). Results: results/p3_tolcheck.json.
For U1 learned in 2050, p = 0.5 (published and 2 percent): the value of hedging against the optimist and pessimist plans,
recomputed with IPOPT tolerances 1e-8, 1e-10 and 1e-12 for every solve involved."""
import os, json
import numpy as np
import removal3 as R3, removal3_stoch as ST
from p3_stoch import SC, CASES, iy

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'results', 'p3_tolcheck.json')
RUNS = json.load(open(os.path.join(HERE, 'results', 'p3_runs.json')))
res = {}
calS, allowS, calF, allowF = CASES['U1']; t = 2050; k = iy(t); p = 0.5


def det(sc, cal, allow, tol, fixed=None):
    pr = R3.Problem(sc, cal=cal, allow=allow)
    if fixed is not None:
        pr.fixed = fixed
    x, info, r, d, dt = pr.solve(x0=None if fixed is None else fixed[0], tol=tol)
    return x, float(r['UTILITY']), int(info['status'])


for c in ('pub', 'd2'):
    sc = SC[c]; lam = RUNS[f'{c}_none']['lamC2025'] / 5
    for tol in (1e-8, 1e-10, 1e-12):
        xS, uS, s1 = det(sc, calS, allowS, tol); xF, uF, s2 = det(sc, calF, allowF, tol)
        _, oS, s3 = det(sc, calS, allowS, tol, (xS, k)); _, oF, s4 = det(sc, calF, allowF, tol, (xS, k))
        _, pS, s5 = det(sc, calS, allowS, tol, (xF, k)); _, pF, s6 = det(sc, calF, allowF, tol, (xF, k))
        m = ST.TwoState(sc, calS, allowS, calF, allowF, p, t)
        x, info, scc, dt = m.solve(x0=m.start(xS, xF, common_from='S'), tol=tol)
        EU = -m.objective(x)
        r = {'EU': EU, 'voh_opt': (EU - p * oS - (1 - p) * oF) / lam, 'voh_pes': (EU - p * pS - (1 - p) * pF) / lam,
             'scc2025': float(scc[2]), 'status': [s1, s2, s3, s4, s5, s6, int(info['status'])]}
        res[f'{c}|{tol:g}'] = r
        print(c, tol, {kk: (round(v, 6) if isinstance(v, float) else v) for kk, v in r.items()}, flush=True)
        json.dump(res, open(OUT, 'w'))
print('done')
