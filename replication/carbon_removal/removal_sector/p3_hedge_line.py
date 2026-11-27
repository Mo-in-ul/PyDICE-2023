"""Two-state policy restricted to the segment between the optimist and pessimist plans (paper 2, Path A).

The joint two-state problem with learning in 2070 does not converge reliably. This script computes a restricted version:
before the resolution date the planner chooses controls (control rate, removal, savings) on the segment
x(l) = l x_S + (1 - l) x_F between the deterministic plans of the success state (l = 1, the optimist) and the failure
state (l = 0, the pessimist); the carbon-cycle variable is solved freely. After resolution each state is re-optimised.
W_S(l) and W_F(l) do not depend on p, so they are computed on a grid once per case; for each p the best l is found on a
cubic interpolant and then evaluated exactly. The result is a lower bound on the two-state optimum. Running the same
procedure for learning in 2050, where the joint problem converges, measures how close the bound is.
Results: results/p3_hedge_line.json (resumable)."""
import os, sys, json
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
import removal3 as R3
from p3_stoch import SC, CASES, iy

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'results', 'p3_hedge_line.json')
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
PS = (0.1, 0.25, 0.5, 0.75, 0.9)


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def solve_state(sc, cal, allow, xc, k, nctrl):
    pr = R3.Problem(sc, cal=cal, allow=allow); pr.fixed = (xc, k); pr.fixed_blocks = range(nctrl)
    x, info, r, d, dt = pr.solve(x0=xc)
    s = pr.summary(x, info, r, d); P = s['paths']
    return {'U': s['U'], 'status': s['status'], 'scc2025': s['scc2025'], 'peakT': s['peakT'], 'peakT_year': s['peakT_year'],
            'abt_to_res': float(5 * np.array(P['ABT'])[:k].sum()), 'miu2045': float(P['MU'][iy(2045)])}


for key in sys.argv[1:]:
    case, c, t = key.split('_'); t = int(t); calS, allowS, calF, allowF = CASES[case]; sc = SC[c]; k = iy(t)
    res = load(); out = res.get(key, {'grid': {}, 'p': {}})
    prS = R3.Problem(sc, cal=calS, allow=allowS); xS, *_ = prS.solve()
    prF = R3.Problem(sc, cal=calF, allow=allowF); xF, *_ = prF.solve()
    N = prS.N; nctrl = 5                                     # blocks: MU, RN, RG, RM, savings; block 5 (alpha) left free

    def plan(l):
        xc = np.array(xF, float).copy()
        for b in range(nctrl):
            sl = slice(b * N, b * N + k); xc[sl] = l * np.asarray(xS)[sl] + (1 - l) * np.asarray(xF)[sl]
        lo, hi, _, _ = prF.bounds()                          # failure-state bounds are the binding ones before t_res
        for b in range(nctrl):
            sl = slice(b * N, b * N + k); xc[sl] = np.clip(xc[sl], lo[sl], hi[sl])
        return xc

    for l in GRID:
        if str(l) in out['grid']:
            continue
        xc = plan(l)
        out['grid'][str(l)] = {'S': solve_state(sc, calS, allowS, xc, k, nctrl), 'F': solve_state(sc, calF, allowF, xc, k, nctrl)}
        g = out['grid'][str(l)]
        print(f"{key} l={l}: WS {g['S']['U']:.5f} WF {g['F']['U']:.5f} st {g['S']['status']}{g['F']['status']} peakS {g['S']['peakT']:.3f} peakF {g['F']['peakT']:.3f}", flush=True)
        res = load(); res[key] = out; json.dump(res, open(OUT, 'w'))
    ls = np.array(GRID); WS = np.array([out['grid'][str(l)]['S']['U'] for l in GRID]); WF = np.array([out['grid'][str(l)]['F']['U'] for l in GRID])
    for p in PS:
        if str(p) in out['p']:
            continue
        cs = CubicSpline(ls, p * WS + (1 - p) * WF)
        r = minimize_scalar(lambda l: -cs(l), bounds=(0, 1), method='bounded'); lstar = float(r.x)
        xc = plan(lstar)
        a = solve_state(sc, calS, allowS, xc, k, nctrl); b = solve_state(sc, calF, allowF, xc, k, nctrl)
        EU = p * a['U'] + (1 - p) * b['U']
        best_grid = float(np.max(p * WS + (1 - p) * WF))
        out['p'][str(p)] = {'l': lstar, 'EU': EU, 'EU_interp': float(cs(lstar)), 'EU_best_grid': best_grid, 'S': a, 'F': b,
                            'EU_optimist': float(p * WS[-1] + (1 - p) * WF[-1]), 'EU_pessimist': float(p * WS[0] + (1 - p) * WF[0])}
        print(f"{key} p={p}: l* {lstar:.3f} EU {EU:.5f} (interp {float(cs(lstar)):.5f}, best grid {best_grid:.5f}) peakS {a['peakT']:.3f} peakF {b['peakT']:.3f}", flush=True)
        res = load(); res[key] = out; json.dump(res, open(OUT, 'w'))
print('done')
