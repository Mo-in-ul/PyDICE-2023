"""Re-solve eta_letter points where IPOPT hit its iteration limit: warm start from the solution at the
neighbouring eta (0.05 lower on the grid, 0.25 lower in the horizon runs) and allow 20000 iterations.
Replaces the records in results/eta_letter_grid.json and results/eta_letter_horizon.json."""
import json, os
import numpy as np
import eta_letter as L
from common import RES, save_json

for what, step in (('grid', 0.05), ('horizon', 0.25)):
    f = os.path.join(RES, f'eta_letter_{what}.json'); d = json.load(open(f))
    for i, o in enumerate(d):
        if o['status'] in (0, 1):
            continue
        N, e, v = o['N'], o['eta'], o['variant']
        e0 = round(e - step, 2) if not (what == 'horizon' and e == 1.75) else 1.5
        _, x0 = L.run(N, e0, v, None if N == 81 else None) if False else (None, None)
        p, b = L.params(N, e0, v)
        L.IM.gams_session = lambda sc, flags=None, _pb=(p, b): _pb
        start = L.IM.Problem(9).start_from_gams() if N == 81 else L.IM.Problem(9).start_default()
        _, x0 = L.run(N, e0, v, start)
        new, _ = L.run(N, e, v, x0, max_iter=20000, tol=1e-9)
        print(what, N, e, v, 'old', o['status'], round(o['scc_2025'], 3), 'new', new['status'], round(new['scc_2025'], 3), flush=True)
        if new['status'] in (0, 1):
            d[i] = new
    save_json(d, f'eta_letter_{what}.json')
