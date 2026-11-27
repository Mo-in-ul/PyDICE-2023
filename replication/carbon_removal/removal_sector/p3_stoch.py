"""Two-state uncertainty experiments (paper 2, Path A). Results: results/p3_stoch.json (resumable).

U1: all three removal families become available in 2070 with probability p, otherwise never. Resolution in 2030, 2050 or
    2070. p in {0.1, 0.25, 0.5, 0.75, 0.9}; p = 0 and p = 1 are the deterministic runs.
U2: biological removal is certain (available from 2025 in both states); geological and mineral removal follow the
    central schedule with probability p, otherwise never. Resolution in 2050 (neither is used before 2050 in any plan).
For each (case, calibration, resolution date) the script also evaluates two rule-of-thumb plans: the optimist plans as if
removal were certain until the resolution date, the pessimist as if it were impossible; after resolution both re-optimise
optimally in each state. Their expected welfare is p W_S + (1-p) W_F with W_S, W_F from deterministic re-optimisations."""
import os, sys, json
import numpy as np
import removal3 as R3, removal3_stoch as ST
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'results', 'p3_stoch.json')
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
YR = R3.YR[:81]; iy = lambda y: int(np.where(YR == y)[0][0])
SC = {'pub': 9, 'd3': 3, 'd2': 2}
A70 = dict(startN=2070.0, startG=2070.0, startM=2070.0)
NEVER_GM = dict(startG=3000.0, startM=3000.0)
CASES = {'U1': (A70, 'NGM', {}, ''), 'U2': ({}, 'NGM', NEVER_GM, 'NGM')}
PLAN = [('U1', c, t) for c in SC for t in (2030, 2050, 2070)] + [('U2', c, 2050) for c in ('pub', 'd3', 'd2')]
PS = (0.1, 0.25, 0.5, 0.75, 0.9)


def det_solve(sc, cal, allow, fixed=None):
    pr = R3.Problem(sc, cal=cal, allow=allow)
    if fixed is not None:
        pr.fixed = fixed
    x, info, r, d, dt = pr.solve(x0=None if fixed is None else fixed[0])
    return x, pr.summary(x, info, r, d)


def short(s):
    P = s['paths']
    return {'U': s['U'], 'status': s['status'], 'scc2025': s['scc2025'], 'peakT': s['peakT'], 'peakT_year': s['peakT_year'],
            'T2100': s['T2100'], 'miu2050': s['miu2050'], 'MU': P['MU'], 'T': P['T'], 'ABT': P['ABT'], 'cumABT_2100': s['cumABT_2100']}


if __name__ == '__main__':
    only = sys.argv[1:]
    for case, c, t in PLAN:
        key = f'{case}_{c}_{t}'
        if key in res or (only and not any(o in key for o in only)):
            continue
        calS, allowS, calF, allowF = CASES[case]; sc = SC[c]; k = iy(t)
        out = {'case': case, 'cal': c, 't_res': t}
        xS, sS = det_solve(sc, calS, allowS); xF, sF = det_solve(sc, calF, allowF)
        out['det_S'] = short(sS); out['det_F'] = short(sF)
        # rule-of-thumb plans
        for lab, xc in (('optimist', xS), ('pessimist', xF)):
            _, a = det_solve(sc, calS, allowS, fixed=(xc, k)); _, b = det_solve(sc, calF, allowF, fixed=(xc, k))
            out[lab] = {'S': short(a), 'F': short(b)}
        out['stoch'] = {}
        xprev = None
        for p in (PS if t < 2070 else (0.25, 0.5, 0.75)):
            m = ST.TwoState(sc, calS, allowS, calF, allowF, p, t)
            x0 = m.start(xS, xF, common_from='S' if p >= 0.5 else 'F') if xprev is None else xprev
            x, info, scc, dt = m.solve(x0=x0, tol=1e-10 if t < 2070 else 1e-8, max_iter=1500 if t >= 2070 else 5000)
            xprev = x
            s = m.summary(x, info, scc)
            out['stoch'][str(p)] = s
            print(f"{key} p={p}: st{s['status']} EU {s['EU']:.5f} SCC25 {s['scc2025']:.2f} peakS {s['S']['peakT']:.3f} peakF {s['F']['peakT']:.3f} "
                  f"abt_to_res {s['S']['cumABT_to_res']:.0f} miu2045 {s['S']['miu2045']:.3f} {dt:.0f}s", flush=True)
        res[key] = out
        cur = json.load(open(OUT)) if os.path.exists(OUT) else {}
        cur[key] = out; json.dump(cur, open(OUT, 'w'))
    print('done', len(res))
    