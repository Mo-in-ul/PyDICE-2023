"""Multiple starting points for the deterministic runs behind the SCC-capacity curves (paper 2, v0.8).
Results: results/p3_multistart.json."""
import os, json
import numpy as np
import removal3 as R3
from p3_mech import VAR, A70, CAP
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'results', 'p3_multistart.json'); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for v, c, sc in (('A', 'pub', 9), ('D', 'pub', 9), ('A', 'd2', 2), ('D', 'd2', 2)):
    base = VAR[v]; starts = {}
    if all(f'{v}_{c}|{s}' in res for s in (0.5, 0.75)):
        continue
    for s0 in (0.25, 1.0):
        pr = R3.Problem(sc, cal=dict(A70, **base, **{k: s0 * val for k, val in CAP.items()}), allow='NGM'); starts[s0], *_ = pr.solve()
    for s in (0.5, 0.75):
        out = {}
        for lab, x0 in (('default', None), ('from_s0.25', starts[0.25]), ('from_s1', starts[1.0])):
            pr = R3.Problem(sc, cal=dict(A70, **base, **{k: s * val for k, val in CAP.items()}), allow='NGM')
            x, info, r, d, dt = pr.solve(x0=x0); sm = pr.summary(x, info, r, d)
            out[lab] = {'U': sm['U'], 'scc2025': sm['scc2025'], 'status': sm['status']}
        res[f'{v}_{c}|{s}'] = out; json.dump(res, open(OUT, 'w'))
        print(v, c, s, {k: (round(o['U'], 6), round(o['scc2025'], 4), o['status']) for k, o in out.items()}, flush=True)
print('done')
