"""Joint (rho, phi) map of the 2025 SCC in DICE-2023 (sigma = 1%, cap 235 y, beta = 0.5, GAMS 0.5 C floor, N = 81).
rho_e = exp(rho + beta*pi) - 1. Continuation in phi within each rho."""
import numpy as np, s2_core as S
RHOS = (0.0, 0.001, 0.0025, 0.005, 0.01, 0.015)
ETAS = np.round(np.arange(0.5, 1.8001, 0.1), 2)
name = 's7_rhophi.json'; out = S.load(name) or []; done = {(o['rho'], o['eta']) for o in out}
for rho in RHOS:
    re = float(np.exp(rho + 0.5 * 0.05) - 1); x = None
    for e in ETAS:
        pr = S.problem(81, float(e), rho_e=re)
        if x is None: x = pr.start_from_gams()
        if (rho, float(e)) in done: continue
        o, xn = S.solve(pr, x)
        if o['status'] in (0, 1): x = xn
        o.update({'rho': rho, 'eta': float(e), 'rho_e': re, 'kappa': e ** 2 * 1e-4 * 235 * (1 + re) / (2 * re)}); o.pop('lamC')
        out.append(o); S.save(out, name)
        print('rp', rho, e, o['status'], round(o['scc_2025'], 3), 'k %.2f' % o['kappa'], 'floor' if o['Tlo_binds'] else '', flush=True)
