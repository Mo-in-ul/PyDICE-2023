"""Growth uncertainty and cap. Usage: python s2_sigma.py part   (part = main | caps | horizon | nobound)"""
import sys, numpy as np, s2_core as S
part = sys.argv[1]
ETA = np.round(np.arange(0.5, 1.7501, 0.05), 2)
SIGS = (0.005, 0.0075, 0.01, 0.0125, 0.015)
if part == 'main':
    series = [(81, s, S.CAP0, 0.5, ETA) for s in SIGS]
elif part == 'caps':
    series = [(81, 0.01, c, 0.5, ETA) for c in (20, 80)]
elif part == 'horizon':
    series = [(121, s, S.CAP0, 0.5, (0.5, 0.75, 1.0, 1.15, 1.3, 1.5, 1.75)) for s in SIGS] + \
             [(121, 0.01, c, 0.5, (0.5, 0.75, 1.0, 1.15, 1.3, 1.5, 1.75)) for c in (20, 80)]
elif part == 'nobound':
    series = [(81, s, S.CAP0, None, (0.5, 0.75, 0.95, 1.15, 1.3, 1.5, 1.75)) for s in SIGS]
name = f's2_sigma_{part}.json'
out = S.load(name) or []
done = {(o['N'], o['sigma'], o['cap'], o['tlo'], o['eta']) for o in out}
for N, s, c, tlo, etas in series:
    x = None
    for e in etas:
        pr = S.problem(N, float(e), s, c, None, tlo)
        if x is None:
            x = pr.start_from_gams() if N == 81 else pr.start_default()
        if (N, s, c, tlo, float(e)) in done:
            continue
        o, xn = S.solve(pr, x)
        if o['status'] not in (0, 1):     # retry once from the GAMS / default start
            o, xn = S.solve(pr, pr.start_from_gams() if N == 81 else pr.start_default())
        if o['status'] in (0, 1):
            x = xn
        o.update({'sigma': s, 'cap': c, 'tlo': tlo, 'eta': float(e)})
        out.append(o); S.save(out, name)
        print(part, N, s, c, tlo, e, o['status'], o['iters'], round(o['scc_2025'], 3), round(o['peakT'], 3), round(o['minT'], 3), flush=True)
