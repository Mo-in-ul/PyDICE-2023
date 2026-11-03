"""V1 + V4: every grid point solved from the GAMS start (no continuation).
dice x tlo in {0.5, 0.01, None}; fixed and noprec with the letter's tlo = 0.01."""
import sys, numpy as np, bv_core as B, bv_io as IO
ETA = np.round(np.arange(0.5, 2.0001, 0.05), 2)
jobs = [('dice', t, e) for t in (0.01, None, 0.5) for e in ETA] + [('fixed', 0.01, e) for e in ETA] + \
       [('noprec', 0.01, e) for e in (0.5, 0.75, 0.95, 1.25, 1.5, 1.75, 2.0)]
out = IO.load('v14_grid.json') or []
done = {(o['variant'], o['tlo'], o['eta']) for o in out}
for v, t, e in jobs:
    if (v, t, float(e)) in done: continue
    pr = B.problem(81, e, v, t); x0 = pr.start_from_gams()
    x, info, n, dt = B.solve(pr, x0)
    if info['status'] not in (0, 1):     # one retry with more iterations, still from the GAMS start
        x, info, n2, dt2 = B.solve(pr, x0, max_iter=20000); n += n2; dt += dt2
    d = B.diagnostics(pr, x, info)
    out.append(IO.row(81, e, v, t, 'gams', d, n, dt)); IO.save(out, 'v14_grid.json')
    print(v, t, e, d['status'], n, round(d['scc'][2], 3), round(d['minT'], 3), len(d['Tlo_binding_years']), flush=True)
