"""Future-year SCC: SCC(t) paths over phi at sigma = 1%, cap 235 y, as coded.
Usage: python s4_future.py N tlo   (tlo: none | 0.5). Continuation in phi from the GAMS (N=81) or default start."""
import sys, numpy as np, s2_core as S
N = int(sys.argv[1]); tlo = None if sys.argv[2] == 'none' else float(sys.argv[2])
ETA = np.round(np.arange(0.5, 1.7501, 0.05), 2)
name = f's4_future_N{N}_tlo{sys.argv[2]}.json'
out = S.load(name) or []; done = {o['eta'] for o in out}
x = None
for e in ETA:
    pr = S.problem(N, float(e), tlo=tlo)
    if x is None: x = pr.start_from_gams() if N == 81 else pr.start_default()
    if float(e) in done: continue
    o, xn = S.solve(pr, x)
    if o['status'] in (0, 1): x = xn
    o.update({'eta': float(e), 'tlo': tlo}); o.pop('lamC'); out.append(o); S.save(out, name)
    print('fut', N, sys.argv[2], e, o['status'], round(o['scc_2025'], 3), round(o['scc_2100'], 2), flush=True)
