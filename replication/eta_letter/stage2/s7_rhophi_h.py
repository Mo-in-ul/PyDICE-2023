"""Horizon check of the rho-phi map near the SCC minimum: N = 121 at phi in {phi*-0.2, phi*, phi*+0.2} (grid values)."""
import numpy as np, s2_core as S
PHIMIN = {0.0: 1.3, 0.001: 1.3, 0.0025: 1.3, 0.005: 1.4, 0.01: 1.5, 0.015: 1.6}
out = S.load('s7_rhophi_h.json') or []; done = {(o['rho'], o['eta']) for o in out}
for rho, pm in PHIMIN.items():
    re = float(np.exp(rho + 0.5 * 0.05) - 1); x = None
    for e in (round(pm - 0.2, 2), pm, round(pm + 0.2, 2)):
        pr = S.problem(121, e, rho_e=re)
        if x is None: x = pr.start_default()
        if (rho, e) in done: continue
        o, xn = S.solve(pr, x)
        if o['status'] in (0, 1): x = xn
        o.update({'rho': rho, 'eta': e}); o.pop('lamC'); out.append(o); S.save(out, 's7_rhophi_h.json')
        print('rph', rho, e, o['status'], round(o['scc_2025'], 3), flush=True)
