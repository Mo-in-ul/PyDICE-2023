"""Injection-rate bottleneck (second review, point 6): geological removal capacity halved (8.5 instead of 17 GtCO2/yr),
other families unchanged; Jensen gap at p = 0.5 with hard caps and without storage limits. Results: results/p5_tests.json
under 'inj:CAL'. Usage: python p5_inj.py pub d2"""
import sys
from p4_common import *

OUT = 'p5_tests.json'
if __name__ == '__main__':
    for c in sys.argv[1:]:
        item = f'inj:{c}'; out = load(OUT).get(item, {}); sc = SC[c]
        for v in ('A', 'D'):
            for s in (0.0, 0.5, 1.0):
                key = f'{v}|{s}'
                if key in out:
                    continue
                cal = dict(VAR[v]) if s == 0.0 else dict(A70, **VAR[v], RN_inf=5.0 * s, RG_inf=8.5 * s, RM_inf=4.0 * s)
                x, r, pr = det(sc, cal, '' if s == 0.0 else 'NGM')
                out[key] = rec(r, pr); save(OUT, item, out)
            cc = {s: out[f'{v}|{s}']['scc2025'] for s in (0.0, 0.5, 1.0)}; g = 0.5 * (cc[0.0] + cc[1.0]) - cc[0.5]
            print(f"{item} {v}: Jensen gap {g:+.4f} ({100 * g / cc[0.5]:.2f}%)", flush=True)
    print('done')
