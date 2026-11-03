"""V5: terminal-horizon sensitivity. N = 61..141; continuation in eta within each N (start_default for eta first)."""
import numpy as np, bv_core as B, bv_io as IO
NS = (61, 81, 101, 121, 141)
ETAS = (0.5, 0.95, 1.25, 1.5, 1.6, 1.75, 1.9, 2.0)
jobs = [(N, e, 'dice') for N in NS for e in ETAS] + [(N, e, 'fixed') for N in NS for e in (0.5, 1.5, 2.0)]
out = IO.load('v5_horizon.json') or []
done = {(o['N'], o['eta'], o['variant']) for o in out}
prev = {}
for N, e, v in jobs:
    if (N, e, v) in done: continue
    pr = B.problem(N, e, v, None)
    x0 = prev.get((N, v))
    if x0 is None:
        x0 = pr.start_from_gams() if N == 81 else pr.start_default()
    x, info, n, dt = B.solve(pr, x0, max_iter=20000)
    d = B.diagnostics(pr, x, info)
    prev[(N, v)] = x
    r = IO.row(N, e, v, None, 'continuation', d, n, dt)
    r['scc_path'] = d['scc'][1:].tolist(); r['T_path'] = d['T'].tolist()
    out.append(r); IO.save(out, 'v5_horizon.json')
    print(N, e, v, d['status'], n, round(d['scc'][2], 3), round(d['peakT'], 3), round(d['minT'], 3), flush=True)
