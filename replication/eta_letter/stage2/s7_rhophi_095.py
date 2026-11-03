"""Exact phi = 0.95 runs for each rho of the joint map (replaces the spline value used for normalization)."""
import numpy as np, s2_core as S
out = S.load('s7_rhophi_095.json') or []; done = {o['rho'] for o in out}
for rho in (0.0, 0.001, 0.0025, 0.005, 0.01, 0.015):
    if rho in done: continue
    re = float(np.exp(rho + 0.025) - 1); pr = S.problem(81, 0.95, rho_e=re)
    o, x = S.solve(pr, pr.start_from_gams()); o.update({'rho': rho, 'eta': 0.95, 'rho_e': re}); o.pop('lamC')
    out.append(o); S.save(out, 's7_rhophi_095.json'); print('r095', rho, o['status'], round(o['scc_2025'], 4), flush=True)
