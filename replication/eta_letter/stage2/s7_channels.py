"""Utility curvature (phi_u) x precautionary elasticity (phi_p): 3 x 3 design, sigma = 1%, cap 235 y, default rho_e.
The precautionary term uses phi_p through sigma_eff = sigma * phi_p / phi_u (rprecaut = -0.5 sigma_eff^2 m phi_u^2).
Run with the GAMS floor and without it."""
import numpy as np, s2_core as S
LEV = (0.95, 1.3, 1.75)
name = 's7_channels.json'; out = S.load(name) or []; done = {(o['phi_u'], o['phi_p'], o['tlo']) for o in out}
for tlo in (0.5, None):
    for pu in LEV:
        x = None
        for pp in LEV:
            pr = S.problem(81, pu, sigma=0.01 * pp / pu, tlo=tlo)
            if x is None: x = pr.start_from_gams()
            if (pu, pp, tlo) in done: continue
            o, xn = S.solve(pr, x)
            if o['status'] in (0, 1): x = xn
            o.update({'phi_u': pu, 'phi_p': pp, 'tlo': tlo}); o.pop('lamC'); out.append(o); S.save(out, name)
            print('ch', tlo, pu, pp, o['status'], round(o['scc_2025'], 3), round(o['peakT'], 3), 'floor' if o['Tlo_binds'] else '', flush=True)
