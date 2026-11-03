import json, numpy as np
from bv_io import load
g = load('v14_grid.json')
base = json.load(open('../results/eta_letter_grid.json')) + json.load(open('../results/eta_letter_nofloor.json'))
B = {(o['variant'], o['eta']): o for o in base if o['N'] == 81}
V = {(o['variant'], o['tlo'], o['eta']): o for o in g}
print('max status', max(o['status'] for o in g), 'max kkt', max(o['kkt'] for o in g), 'max irf', max(o['irf_max'] for o in g))
print([ (o['variant'],o['tlo'],o['eta'],o['status'],o['iters']) for o in g if o['status']!=0])
# baseline (continuation) vs fresh start
worst = []
for (v, e), o in B.items():
    tlo = None if v == 'dice_nofloor' else 0.01
    vv = 'dice' if v == 'dice_nofloor' else v
    n = V.get((vv, tlo, e))
    if n: worst.append((abs(n['scc_2025'] / o['scc_2025'] - 1), v, e, o['scc_2025'], n['scc_2025'], abs(n['peakT'] - o['peak_T'])))
worst.sort(reverse=True); print('max rel diff vs baseline', worst[:4]); print('max dT', max(w[5] for w in worst))
# letter composite: dice tlo .01 below 1.6, nofloor from 1.6
comp = {e: (V[('dice', 0.01, e)] if e < 1.6 else V[('dice', None, e)]) for e in np.round(np.arange(0.5, 2.0001, 0.05), 2)}
for e in (0.5, 0.95, 1.25, 1.3, 1.5, 1.55, 1.6, 1.75, 2.0):
    c = comp[e]; print('comp', e, round(c['scc_2025'], 3), round(c['peakT'], 3), round(c['cumE_2100']))
for v in ('fixed', 'noprec'):
    for e in (0.5, 0.95, 2.0):
        o = V.get((v, 0.01, e)); print(v, e, round(o['scc_2025'], 3), round(o['peakT'], 3))
# minimum & crossing, for each bound treatment, cubic interpolation on fine grid
from scipy.interpolate import CubicSpline
for tlo in (0.01, None, 0.5):
    es = np.round(np.arange(0.5, 2.0001, 0.05), 2)
    if tlo == 0.01: s = np.array([comp[e]['scc_2025'] for e in es]); lab = 'letter composite'
    else: s = np.array([V[('dice', tlo, e)]['scc_2025'] for e in es]); lab = str(tlo)
    cs = CubicSpline(es, s); f = np.linspace(0.5, 2, 15001); y = cs(f); i = y.argmin()
    back = f[(f > f[i]) & (y >= V[('dice', 0.01, 0.95)]['scc_2025'])][0]
    bind = [e for e in es if V[('dice', tlo, e)]['Tlo_mult_sum'] > 1e-8] if tlo is not None else []
    print(lab, 'min', round(f[i], 3), round(y[i], 3), 'back to default at', round(back, 3), 'bound binds from', bind[:1])
for e in (1.75, 1.9, 2.0):
    print('tlo .5 peakT', e, round(V[('dice', 0.5, e)]['peakT'], 3), 'nobound', round(V[('dice', None, e)]['peakT'], 3), 'minT nobound', round(V[('dice', None, e)]['minT'], 3), V[('dice', None, e)]['alpha_at_lo_hi'])
