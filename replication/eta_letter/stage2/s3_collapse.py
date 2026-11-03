"""Collapse test: does the precautionary multiplier M = SCC / SCC_noprec, and the onset of horizon
sensitivity, line up when plotted against kappa = phi^2 sigma^2 cap_years / (2 rho_e / (1 + rho_e))?"""
import numpy as np, s2_core as S
from scipy.interpolate import interp1d
RE0 = 0.0263409
kap = lambda e, s, cy, re=RE0: e ** 2 * s ** 2 * cy / (2 * re / (1 + re))
NP = {o['eta']: o['scc_2025'] for o in S.load('s3_noprec.json')}
rows = []
for o in S.load('s2_sigma_main.json') + S.load('s2_sigma_caps.json') + (S.load('s2_sigma_ext.json') or []):
    if o.get('N', 81) != 81 or o.get('tlo', 0.5) != 0.5 or o['eta'] not in NP: continue
    cap = o.get('cap', 47); cy = min(cap, 80) * 5
    rows.append(('sig%.4f_cap%d' % (o['sigma'], cy), o['eta'], kap(o['eta'], o['sigma'], cy), o['scc_2025'] / NP[o['eta']], o['scc_2025']))
for o in S.load('s2_rho_h80.json'):
    rows.append(('recal_h80', o['eta'], kap(o['eta'], 0.01, 235, o['rho_e']), o['scc_2025'] / NP[o['eta']], o['scc_2025']))
S.save([dict(case=r[0], eta=r[1], kappa=r[2], M=r[3], scc=r[4]) for r in rows], 's3_collapse.json')
grid = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
print('M = SCC/SCC_noprec at given kappa (interpolated within each case)')
print('case'.ljust(20), ' '.join(f'{k:6.1f}' for k in grid))
for c in sorted({r[0] for r in rows}):
    r = sorted([x for x in rows if x[0] == c], key=lambda x: x[2]); k = np.array([x[2] for x in r]); M = np.array([x[3] for x in r])
    f = interp1d(k, M, bounds_error=False)
    print(c.ljust(20), ' '.join(f'{float(f(g)):6.2f}' if k.min() <= g <= k.max() else '     -' for g in grid))
