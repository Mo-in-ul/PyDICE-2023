"""Mechanism tests for the displacement channel (tasks A1-A3, C3, B8). Results: results/p4_mech.json (resumable).

All cases: removal from 2070, capacity scaled by s in [0, 1]; the 2025 SCC of each deterministic economy gives the Jensen
gap p SCC(1) + (1-p) SCC(0) - SCC(p), which is the premium up to the policy-path effect (Appendix B of the manuscript).

Usage: python p4_mech.py KIND:VAR_CAL[:ARG] ...
  curve:D_d2        free optimum on a grid of eighths (A2), with the periods in which the control rate is at its bounds
  fixmu:D_d2        control rate held at the no-removal optimum in every period; savings and removal re-optimised (A1)
  ceil:D_d2         published DICE ceiling on the control rate (up to 1.10 from 2120) instead of the cap at one (A3)
  abc:D_d2:0.5      abatement cost multiplied by ARG (C3 and its mirror image)
  disc:D_d4         other discount rates (d1, d4, d5; d2, d3 and pub are in curve) on the scales 0, 0.5, 1 (B8)
  two:D_d2:KIND[:ARG]  two-state SCC, p = 0.5, learned 2050, for the same variant (checks premium = Jensen gap)"""
import sys
import numpy as np
import removal3_stoch as ST
from p4_common import *

OUT = 'p4_mech.json'
EIGHTHS = tuple(i / 8 for i in range(9)); QUARTERS = (0.0, 0.25, 0.5, 0.75, 1.0); HALVES = (0.0, 0.5, 1.0)


def run(item):
    parts = item.split(':'); kind, vc = parts[0], parts[1]; arg = parts[2:] ; v, c = vc.split('_'); sc = SC[c]; base = VAR[v]
    out = load(OUT).get(item, {})
    if kind == 'two':
        sub = arg[0]; mult = float(arg[1]) if len(arg) > 1 else 1.0
        if 'p0.5' in out:
            return
        sess = (lambda: session(sc, abcost=mult)) if sub == 'abc' else (lambda: None)
        up = 'dice' if sub == 'ceil' else None
        xS, sS, _ = det(sc, capcal(1.0, base), 'NGM', sess(), up); xF, sF, _ = det(sc, dict(base), '', sess(), up)
        _, sE, _ = det(sc, capcal(0.5, base), 'NGM', sess(), up)
        m = ST.TwoState(sc, capcal(1.0, base), 'NGM', dict(base), '', 0.5, 2050, session=sess())
        if up == 'dice':
            m.prS.up = dice_ceiling(m.prS); m.prF.up = dice_ceiling(m.prF)
        x, info, scc, dt = m.solve(x0=m.start(xS, xF, common_from='S'))
        o = {'status': int(info['status']), 'scc2s': float(scc[2]), 'sccK': sS['scc2025'], 'scc0': sF['scc2025'], 'sccE': sE['scc2025'],
             'EU': -float(m.objective(x))}
        o['premium'] = o['scc2s'] - o['sccE']; o['jensen'] = 0.5 * o['sccK'] + 0.5 * o['scc0'] - o['sccE']
        out['p0.5'] = o; save(OUT, item, out)
        print(f"{item}: st{o['status']} SCC2s {o['scc2s']:.3f} premium {o['premium']:+.3f} ({100 * o['premium'] / o['sccE']:.2f}%) Jensen {o['jensen']:+.3f} {dt:.0f}s", flush=True)
        return
    scales = {'curve': EIGHTHS, 'fixmu': QUARTERS, 'ceil': QUARTERS, 'abc': QUARTERS, 'disc': HALVES}[kind]
    mult = float(arg[0]) if kind == 'abc' else 1.0
    sess = (lambda: session(sc, abcost=mult)) if kind == 'abc' else (lambda: None)
    up = 'dice' if kind == 'ceil' else None
    x0n = None
    for s in scales:
        if str(s) in out:
            continue
        if kind == 'fixmu':
            if x0n is None:
                x0n, _, _ = det(sc, dict(base), '')                       # no-removal optimum: its control rate is imposed
            if s == 0.0:
                x, r, pr = det(sc, dict(base), '', fix=x0n)
            else:
                xfree, _, _ = det(sc, capcal(s, base), 'NGM')             # start from the free optimum at this capacity
                N = len(x0n) // 6; xs = np.array(xfree); xs[:N] = np.asarray(x0n)[:N]
                x, r, pr = det(sc, capcal(s, base), 'NGM', fix=x0n, x0=xs)
        elif s == 0.0:
            x, r, pr = det(sc, dict(base), '', sess(), up)
        else:
            x, r, pr = det(sc, capcal(s, base), 'NGM', sess(), up)
        out[str(s)] = rec(r, pr); save(OUT, item, out)
        o = out[str(s)]
        print(f"{item} s={s}: st{o['status']} SCC {o['scc2025']:.4f} ABT2100 {o['cumABT_2100']:.0f} R2200 {o['cumR_2200']:.0f} "
              f"at max from {o['first_at_max']} ({o['n_at_max_2200']} periods) ramp to {o['last_on_ramp']} mu_max {o['mu_max']:.3f}", flush=True)


if __name__ == '__main__':
    for item in sys.argv[1:]:
        run(item)
    print('done')
