"""V3: sensitivity to the starting point. No temperature bound (letter's high-eta setting)."""
import numpy as np, bv_core as B, bv_io as IO, os, sys, json
sys.path.insert(0, '..')
ETAS = (0.95, 1.28, 1.5, 1.75, 2.0)
out = IO.load('v3_multistart.json') or []
done = {(o['eta'], o['start']) for o in out}
for e in ETAS:
    pr0 = B.problem(81, e, 'dice', None)
    xl, xu, _, _ = pr0.bounds(); N = pr0.N
    xg = pr0.start_from_gams()
    starts = {'gams': xg, 'default': pr0.start_default()}
    for s in range(3):
        rng = np.random.default_rng(100 + s); x = xg.copy()
        x[:2 * N] *= rng.uniform(0.8, 1.2, 2 * N); starts[f'random{s}'] = np.clip(x, xl, xu)
    # continuation from the next-lower eta of the same run, as in the baseline grid
    if e > 0.95:
        lo = round(e - 0.05, 2)
        prl = B.problem(81, lo, 'dice', None); xc, _, _, _ = B.solve(prl, prl.start_from_gams())
        starts['continuation'] = xc
    for name, x0 in starts.items():
        if (e, name) in done: continue
        pr = B.problem(81, e, 'dice', None)
        x, info, n, dt = B.solve(pr, x0, max_iter=20000)
        d = B.diagnostics(pr, x, info)
        out.append(IO.row(81, e, 'dice', None, name, d, n, dt)); IO.save(out, 'v3_multistart.json')
        print(e, name, d['status'], n, repr(d['U']), round(d['scc'][2], 5), flush=True)
