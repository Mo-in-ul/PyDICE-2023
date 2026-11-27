"""Nesting and accounting tests for removal3.py.
1. Removal off: must reproduce the no-removal runs (SCC 2025 63.0093 / 107.4322 / 282.1484).
2. v2 central reproduced: biological + geological only, no leakage, geological capacity 20 (EW inside it), must give
   SCC 59.4065 and peak 2.2799 (removal_split central).
3. Carbon accounting with leakage: cumulative removal - stock - cumulative leakage = 0 every period."""
import json, numpy as np, removal3 as R3
out = {}
for sc, ref in [(9, 63.009274), (3, 107.432197), (2, 282.148382)]:
    pr = R3.Problem(sc, allow='')
    x, info, r, d, dt = pr.solve(); s = pr.summary(x, info, r, d)
    print('none', sc, s['status'], round(s['scc2025'], 4), 'ref', ref, flush=True); out[f'none_{sc}'] = s['scc2025']
pr = R3.Problem(9, allow='NG', cal={'delta0': 0.0, 'RG_inf': 20.0})
x, info, r, d, dt = pr.solve(); s = pr.summary(x, info, r, d)
print('v2 central', s['status'], round(s['scc2025'], 4), round(s['peakT'], 4), 'ref 59.4065 2.2799', flush=True)
out['v2_central'] = [s['scc2025'], s['peakT']]
for gam in (0.0, 0.5):
    pr = R3.Problem(9, allow='NGM', cal={'gamma': gam})
    x, info, r, d, dt = pr.solve(); s = pr.summary(x, info, r, d); P = s['paths']
    RN, SN, LN = map(np.array, (P['RN'], P['SN'], P['LEAK']))
    # stock at start of period i = sum of removal before i - leakage before i
    bal = np.array([5 * RN[:i].sum() - SN[i] - 5 * LN[:i].sum() for i in range(len(RN))])
    print('accounting gamma', gam, 'max |imbalance| GtCO2', float(np.abs(bal).max()), 'status', s['status'], flush=True)
    out[f'balance_gamma{gam}'] = float(np.abs(bal).max())
json.dump(out, open('results/p3_nesting.json', 'w'), indent=1)
