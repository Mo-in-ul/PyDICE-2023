"""Use of removal by capacity scale (paper 2, v0.7): how much removal the deterministic success economy actually uses when
capacity is scaled by s. Saturation (use growing less than proportionally with capacity) is the candidate source of the
convexity of the SCC in capacity. Results: results/p3_util.json."""
import os, sys, json
import numpy as np
import removal3 as R3
from p3_stoch import SC
from p3_mech import VAR, A70, CAP

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'results', 'p3_util.json')
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
YR = R3.YR[:81]
for item in sys.argv[1:]:
    v, c = item.split('_'); sc = SC[c]
    for s in (0.25, 0.5, 0.75, 1.0):
        key = f'{item}|{s}'
        if key in res:
            continue
        pr = R3.Problem(sc, cal=dict(A70, **VAR[v], **{k: s * val for k, val in CAP.items()}), allow='NGM')
        x, info, r, d, dt = pr.solve(); sm = pr.summary(x, info, r, d); P = sm['paths']
        R = 5 * (np.array(P['RN']) + np.array(P['RG']) + np.array(P['RM']))
        cap = 5 * (np.array(pr.P['RbarN'][1:]) + np.array(pr.P['RbarG'][1:]) + np.array(pr.P['RbarM'][1:]))
        cap = np.where(YR >= 2070, cap, 0.0)
        res[key] = {'status': sm['status'], 'scc2025': sm['scc2025'], 'R_2200': float(R[YR <= 2200].sum()), 'R_all': float(R.sum()),
                    'cap_2200': float(cap[YR <= 2200].sum()), 'minT': float(np.min(P['T'][10:]))}
        json.dump(res, open(OUT, 'w'))
        print(key, {a: round(b, 2) if isinstance(b, float) else b for a, b in res[key].items()}, flush=True)
print('done')
