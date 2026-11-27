"""Elasticity of marginal utility (C2). Results: results/p4_eta.json (resumable).

Published calibration with the elasticity of marginal utility raised from 0.95 to 1.5 and 2. Two versions: discount
factors as published (the consumption discount rate rises with the elasticity), and discount factors rescaled,
RR(t)^kappa, with kappa chosen so that the no-removal SCC in 2025 equals the published $63.01 ("re-matched").
For each: SCC by capacity (0, 0.25, 0.5, 0.75, 1; removal from 2070; hard caps and no storage limit) and the change in
peak warming with removal from 2110.
Usage: python p4_eta.py 1.5 2.0"""
import sys
import numpy as np
from scipy.optimize import brentq
from p4_common import *

OUT = 'p4_eta.json'; TARGET = 63.0093

if __name__ == '__main__':
    A = lambda y: dict(startN=float(y), startG=float(y), startM=float(y))
    for eta in [float(a) for a in sys.argv[1:]]:
        out = load(OUT).get(str(eta), {})
        if 'kappa' not in out:
            f = lambda kap: det(9, {}, '', session(9, elasmu=eta, rr_kappa=kap))[1]['scc2025'] - TARGET
            out['kappa'] = float(brentq(f, 0.02, 1.0, xtol=1e-4)); save(OUT, str(eta), out)
            print(f"eta {eta}: kappa {out['kappa']:.4f}", flush=True)
        for lab, kap in (('asis', 1.0), ('matched', out['kappa'])):
            o = out.get(lab, {})
            runs = {'none': ({}, ''), 'a2110': (A(2110), 'NGM'), 'a2070': (A(2070), 'NGM')}
            runs.update({f'A_{s}': (capcal(s), 'NGM') for s in (0.25, 0.5, 0.75)})
            runs.update({f'D_{s}': (capcal(s, VAR['D']), 'NGM') for s in (0.25, 0.5, 0.75, 1.0)})
            for name, (cal, allow) in runs.items():
                if name in o:
                    continue
                x, s, pr = det(9, cal, allow, session(9, elasmu=eta, rr_kappa=kap))
                o[name] = rec(s, pr); out[lab] = o; save(OUT, str(eta), out)
                print(f"eta {eta} {lab} {name}: st{s['status']} SCC {s['scc2025']:.3f} peak {s['peakT']:.4f} ({s['peakT_year']})", flush=True)
    print('done')
