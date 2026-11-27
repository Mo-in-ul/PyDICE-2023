"""Direction A: anticipation experiments (paper 2, model v3). Results: results/p3_anticip.json.

For each calibration:
  P0   no removal ever (perfect foresight).
  PA   removal (all three families) available from 2070, known in 2020 (perfect foresight).
  PAx  plan PA until 2070, then removal turns out to be unavailable: re-optimise from 2070 with decisions before 2070
       held at PA. Abatement before 2070 differs from P0 only through anticipation, since no removal happens before 2070.
  P0a  plan P0 until 2050, then removal is announced as available from 2070: re-optimise from 2050.
  PC   central (removal from 2025); PCx: plan PC until 2050, then removal stops: re-optimise from 2050."""
import os, json
import numpy as np
import removal3 as R3
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'results', 'p3_anticip.json')
YR = R3.YR[:81]
iy = lambda y: int(np.where(YR == y)[0][0])
SC = {'pub': 9, 'd3': 3, 'd2': 2}
ALL70 = dict(startN=2070.0, startG=2070.0, startM=2070.0)


def solve(sc, allow, cal=None, fixed=None):
    pr = R3.Problem(sc, cal=cal or {}, allow=allow)
    if fixed is not None:
        pr.fixed = fixed
    x, info, r, d, dt = pr.solve(x0=None if fixed is None else fixed[0])
    s = pr.summary(x, info, r, d)
    return pr, x, s


def keep(s, extra):
    P = s['paths']; MU = np.array(P['MU']); ABT = np.array(P['ABT'])
    o = {k: s[k] for k in ('status', 'U', 'scc2025', 'scc2030', 'scc2050', 'miu2030', 'miu2050', 'peakT', 'peakT_year', 'T2100', 'T2200',
                           'cumN', 'cumG', 'cumM', 'cumABT_2100', 'cumNCO2_2100', 'cumE_2020_2100', 'lamC2025')}
    o['cumABT_to2065'] = float(5 * ABT[:iy(2065) + 1].sum()); o['cumABT_to2045'] = float(5 * ABT[:iy(2045) + 1].sum())
    o['miu2040'] = float(MU[iy(2040)]); o['miu2060'] = float(MU[iy(2060)]); o['miu2065'] = float(MU[iy(2065)])
    o['T_path'] = P['T']; o['MU_path'] = P['MU']; o['E_path'] = P['E']
    o.update(extra)
    return o


res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for c, sc in SC.items():
    if c in res:
        continue
    out = {}
    pr0, x0, s0 = solve(sc, '')
    out['P0'] = keep(s0, {})
    prA, xA, sA = solve(sc, 'NGM', ALL70)
    out['PA'] = keep(sA, {})
    _, _, sAx = solve(sc, '', None, fixed=(xA, iy(2070)))
    out['PAx'] = keep(sAx, {})
    _, _, s0a = solve(sc, 'NGM', ALL70, fixed=(x0, iy(2050)))
    out['P0a'] = keep(s0a, {})
    prC, xC, sC = solve(sc, 'NGM')
    out['PC'] = keep(sC, {})
    _, _, sCx = solve(sc, 'NGM', dict(startN=3000.0, startG=3000.0, startM=3000.0), fixed=(xC, iy(2050)))
    out['PCx'] = keep(sCx, {})
    res[c] = out
    json.dump(res, open(OUT, 'w'))
    for k, o in out.items():
        print(f"{c} {k:4s} st{o['status']} SCC25 {o['scc2025']:7.2f} peakT {o['peakT']:.3f}@{o['peakT_year']} T2100 {o['T2100']:.2f} "
              f"abt<2065 {o['cumABT_to2065']:.0f} miu2050 {o['miu2050']:.3f} miu2065 {o['miu2065']:.3f} U {o['U']:.4f}", flush=True)
print('done')
