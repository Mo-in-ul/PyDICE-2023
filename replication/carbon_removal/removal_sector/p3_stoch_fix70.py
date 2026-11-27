"""Re-solve the two-state runs with learning in 2070 from feasible starting points. The optimist and pessimist plans,
with each state re-optimised after 2070, are feasible for the two-state problem; the solver starts from whichever has
the higher expected welfare. A solution that does not improve on that start is replaced by the start (reported as a
lower bound). Results are merged into results/p3_stoch.json under 'stoch' with flag 'source'."""
import os, sys, json
import numpy as np, jax.numpy as jnp
import removal3 as R3, removal3_stoch as ST
sys.argv, _a = sys.argv[:1], sys.argv[1:]
from p3_stoch import det_solve, SC, CASES, iy, OUT  # noqa: E402  (all keys done, so the import solves nothing)
sys.argv += _a
for key in sys.argv[1:]:
    case, c, t = key.split('_'); t = int(t); calS, allowS, calF, allowF = CASES[case]; sc = SC[c]; k = iy(t)
    xS, _ = det_solve(sc, calS, allowS); xF, _ = det_solve(sc, calF, allowF)
    plans = {}
    for lab, xc in (('optimist', xS), ('pessimist', xF)):
        xa, a = det_solve(sc, calS, allowS, fixed=(xc, k)); xb, b = det_solve(sc, calF, allowF, fixed=(xc, k))
        plans[lab] = (xa, xb, a['U'], b['U'])
    cur = json.load(open(OUT)); out = cur[key]; out['stoch'] = {}; out['feasible_plans'] = []
    for p in ((0.25, 0.5, 0.75) if t == 2070 else (0.1, 0.25, 0.5, 0.75, 0.9)):
        m = ST.TwoState(sc, calS, allowS, calF, allowF, p, t)
        xl, xu, gl, gu = m.bounds(); cand = {}
        for lab, (xa, xb, ua, ub) in plans.items():
            cS, tS = m.split(np.asarray(xa)); _, tF = m.split(np.asarray(xb)); x0 = np.concatenate([cS, tS, tF])
            gg = m.constraints(x0); viol = max(float(np.max(xl - x0)), float(np.max(x0 - xu)), float(np.max(gl - gg)), float(np.max(gg - gu)), 0.0)
            if viol < 1e-7:
                cand[lab] = (x0, -m.objective(x0))
        best = max(cand, key=lambda l: cand[l][1]); x0, EU0 = cand[best]; clip = gv = 0.0
        x, info, scc, dt = m.solve(x0=x0, tol=1e-8, max_iter=1500, mu_strategy='monotone', mu_init=1e-6, bound_push=1e-8, bound_frac=1e-8, acceptable_iter=0)
        s = m.summary(x, info, scc)
        if s['EU'] < EU0 - 1e-6:
            s = m.summary(x0, {'status': -99}, np.full(m.N + 1, np.nan)); s['source'] = 'start:' + best
        else:
            s['source'] = 'solved from ' + best
        s['EU_start'] = EU0
        out['stoch'][str(p)] = s
        print(f"{key} p={p}: st{s['status']} EU {s['EU']:.5f} start {EU0:.5f} ({best}) SCC25 {s['scc2025']:.2f} peakS {s['S']['peakT']:.3f} peakF {s['F']['peakT']:.3f} {s['source']} clip {clip:.1e} gviol {gv:.1e} {dt:.0f}s", flush=True)
    cur = json.load(open(OUT)); cur[key] = out; json.dump(cur, open(OUT, 'w'))
print('done')
