"""Check 1: full pipeline. SCC on PyDICE's own optimized path vs GAMS.

scc_gams_controls.py evaluates the SCC on the GAMS control paths, which tests the SCC
definition. This check uses the controls PyDICE found itself (run_faithful.py, SLSQP),
with the SCC from recoverAllVars (alpha re-solved with each pulse). It needs the
results/faith_scen<N>.npz files. Scenario 6 (1.5 C) is excluded: infeasible without
capital disposal.
Writes results/check1_pipeline.json.
"""
import os
import numpy as np
from common import dm, save_json, RES

YEARS = (2025, 2030, 2050, 2100, 2150)

if __name__ == '__main__':
    allr = {}
    for sc in (1, 2, 3, 4, 5, 7, 8, 9, 10):
        f = os.path.join(RES, f'faith_scen{sc}.npz')
        if not os.path.exists(f):
            print('missing', f); continue
        out = np.load(f)['output']
        g = dm.load_gams(sc)
        o = {'scenario': sc, 'years': list(YEARS), 'pydice': [], 'gams': [], 'dev_pct': []}
        for y in YEARS:
            i = (y - 2020) // 5
            a, b = float(out[i, 39]), float(g['SCC'][i])
            o['pydice'].append(a); o['gams'].append(b); o['dev_pct'].append(100 * (a / b - 1))
        allr[str(sc)] = o
        print(f"S{sc:2d} " + '  '.join(f"{y}: {a:8.2f} vs {b:8.2f} ({d:+.3f}%)" for y, a, b, d in zip(YEARS, o['pydice'], o['gams'], o['dev_pct'])))
    save_json(allr, 'check1_pipeline.json')
