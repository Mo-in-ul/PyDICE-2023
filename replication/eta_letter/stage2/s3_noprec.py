"""No-precaution SCC (sigma = 0) on the full phi grid, GAMS bound, N = 81. Independent of sigma and cap."""
import numpy as np, s2_core as S
ETA = np.round(np.concatenate([np.arange(0.5, 1.7501, 0.05), np.arange(1.8, 3.0001, 0.1)]), 2)
out = S.load('s3_noprec.json') or []; done = {o['eta'] for o in out}
x = None
for e in ETA:
    pr = S.problem(81, float(e), sigma=0.0)
    if x is None: x = pr.start_from_gams()
    if float(e) in done: continue
    o, xn = S.solve(pr, x)
    if o['status'] in (0, 1): x = xn
    o['eta'] = float(e); o.pop('lamC'); out.append(o); S.save(out, 's3_noprec.json')
    print('noprec', e, o['status'], round(o['scc_2025'], 3), flush=True)
