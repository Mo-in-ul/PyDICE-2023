"""Extend phi beyond 1.75 for low sigma, where no minimum appears in [0.5, 1.75]. N = 81 and 121, GAMS bound."""
import numpy as np, s2_core as S
ETA = np.round(np.arange(1.8, 3.0001, 0.1), 2)
name = 's2_sigma_ext.json'
out = S.load(name) or []
done = {(o['N'], o['sigma'], o['eta']) for o in out}
main = S.load('s2_sigma_main.json')
for N in (81, 121):
    for s in (0.005, 0.0075):
        x = None
        for e in ETA:
            pr = S.problem(N, float(e), s)
            if x is None: x = pr.start_from_gams() if N == 81 else pr.start_default()
            if (N, s, float(e)) in done: continue
            o, xn = S.solve(pr, x)
            if o['status'] in (0, 1): x = xn
            o.update({'sigma': s, 'eta': float(e)}); o.pop('lamC'); out.append(o); S.save(out, name)
            print('ext', N, s, e, o['status'], round(o['scc_2025'], 3), round(o['peakT'], 3), round(o['minT'], 3), flush=True)
