import json, numpy as np, s2_core as S
from scipy.interpolate import CubicSpline
d = S.load('s2_sigma_main.json') or []
for s in (0.005, 0.0075, 0.01, 0.0125, 0.015):
    r = sorted([o for o in d if o['sigma'] == s], key=lambda o: o['eta'])
    if len(r) < 4: continue
    e = np.array([o['eta'] for o in r]); y = np.array([o['scc_2025'] for o in r])
    st = {o['status'] for o in r}; bind = [o['eta'] for o in r if o['Tlo_binds']]
    cs = CubicSpline(e, y); f = np.linspace(e[0], e[-1], 20001); yy = cs(f); i = yy.argmin()
    s095 = cs(0.95); above = f[(f > f[i]) & (yy >= s095)]
    print(f"sigma {s*100:.2f}%  n={len(r)} status {st}  SCC@0.5 {y[0]:.2f} @0.95 {s095:.2f} min {yy[i]:.2f} at {f[i]:.3f}  back-to-0.95 at {above[0] if len(above) else None}  SCC@{e[-1]} {y[-1]:.2f}  bound binds from {bind[:1]}  peakT 0.95->{e[-1]}: {r[list(e).index(0.95)]['peakT']:.2f}->{r[-1]['peakT']:.2f}")
