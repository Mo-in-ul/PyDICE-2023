"""Horizon checks for the recalibrated runs: same rho_e(phi) as the N = 81 calibration, horizons N = 61, 101, 121."""
import sys, numpy as np, s2_core as S
h = int(sys.argv[1]) if len(sys.argv) > 1 else 80
cal = {o['eta']: o['rho_e'] for o in S.load(f's2_rho_h{h}.json')}
name = f's2_rho_h{h}_horizon.json'
out = S.load(name) or []
done = {(o['N'], o['eta']) for o in out}
for N in (61, 101, 121):
    x = None
    for e in (0.5, 0.75, 0.95, 1.1, 1.25, 1.4, 1.5, 1.6, 1.75):
        pr = S.problem(N, e, rho_e=cal[e])
        if x is None: x = pr.start_default()
        if (N, e) in done: continue
        o, xn = S.solve(pr, x)
        if o['status'] in (0, 1): x = xn
        o.update({'eta': e, 'rho_e': cal[e], 'h': h}); o.pop('lamC'); out.append(o); S.save(out, name)
        print('rhoH', N, e, o['status'], round(o['scc_2025'], 3), round(o['minT'], 3), flush=True)
