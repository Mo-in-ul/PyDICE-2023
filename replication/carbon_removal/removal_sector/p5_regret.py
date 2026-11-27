"""Policy regret of planning on expected removal capacity (P0, review of 11 Nov 2026). Results: results/p5_regret.json.

Case U1: all removal from 2070 with capacity K (probability p) or never; outcome learned in t_r. Five strategies share
the same lottery and re-optimise in each state after t_r:
  stochastic   optimal under uncertainty (joint solve for 2050; reduced-space optimum from p3_reduced.json for 2070)
  expected     plans before t_r as if capacity were pK for sure (deterministic optimum at capacity pK)
  optimist     as if capacity were K for sure;  pessimist  as if removal never came
  perfect      state-specific optimum from the start (upper bound)
Reported per strategy: expected welfare, regret against the stochastic optimum (trillion 2019 dollars of 2025
consumption), cumulative abatement before t_r, the SCC the plan would set in 2025, peak warming if removal fails.
Usage: python p5_regret.py pub:2050 pub:2070 d2:2050 d2:2070 pub:2050:noceil d2:2050:noceil"""
import sys
import numpy as np
import removal3 as R3, removal3_stoch as ST
from p4_common import *

OUT = 'p5_regret.json'
RED = load('p3_reduced.json')


def fixed(sc, cal, allow, xc, k):
    pr = R3.Problem(sc, cal=cal, allow=allow); pr.fixed = (np.asarray(xc), k)
    x, info, r, d, _ = pr.solve(x0=np.asarray(xc)); return pr.summary(x, info, r, d)


def cum(s, k):
    return float(5 * np.array(s['paths']['ABT'])[:k].sum())


if __name__ == '__main__':
    for item in sys.argv[1:]:
        parts = item.split(':'); c, t = parts[0], int(parts[1]); noceil = 'noceil' in parts
        sc = SC[c]; k = iy(t); base = dict(ramp_set=False) if noceil else {}
        out = load(OUT).get(item, {})
        calS, calF = capcal(1.0, base), dict(base)
        xS, sS, _ = det(sc, calS, 'NGM'); xF, sF, _ = det(sc, calF, '')
        lam = sF['lamC2025'] / 5
        plan = {}
        for lab, xc in (('optimist', xS), ('pessimist', xF)):
            a = fixed(sc, calS, 'NGM', xc, k); b = fixed(sc, calF, '', xc, k)
            plan[lab] = {'US': a['U'], 'UF': b['U'], 'abt': cum(a, k), 'peakF': b['peakT'], 'stS': a['status'], 'stF': b['status']}
        plan['optimist']['scc'] = sS['scc2025']; plan['pessimist']['scc'] = sF['scc2025']
        for p in (0.25, 0.5, 0.75):
            if str(p) in out:
                continue
            xE, sE, _ = det(sc, capcal(p, base), 'NGM')
            a = fixed(sc, calS, 'NGM', xE, k); b = fixed(sc, calF, '', xE, k)
            E = {'US': a['U'], 'UF': b['U'], 'abt': cum(a, k), 'peakF': b['peakT'], 'scc': sE['scc2025'], 'stS': a['status'], 'stF': b['status']}
            if t == 2070:
                rr = RED[f'U1_{c}_2070|{p}|segment']
                st = {'EU': rr['EU'], 'abt': rr['abt_to_res'], 'peakF': rr['peakF'], 'scc': rr['scc2025'], 'status': f"{rr['status_S']}/{rr['status_F']}"}
            elif noceil and c == 'd2':                       # joint solve stops below feasible plans here: reduced space
                import p3_stoch, p3_reduced
                p3_stoch.CASES['U1nc'] = p3_reduced.CASES['U1nc'] = (dict(A70, ramp_set=False), 'NGM', dict(ramp_set=False), '')
                rr = p3_reduced.run(f'U1nc_{c}_2050', p, 'segment')
                st = {'EU': rr['EU'], 'abt': rr['abt_to_res'], 'peakF': rr['peakF'], 'scc': rr['scc2025'], 'status': f"reduced {rr['status_S']}/{rr['status_F']}"}
            else:
                m = ST.TwoState(sc, calS, 'NGM', calF, '', p, t)
                x, info, scc, dt = m.solve(x0=m.start(xS, xF, common_from='S' if p >= 0.5 else 'F'))
                s2 = m.summary(x, info, scc)
                st = {'EU': s2['EU'], 'abt': s2['S']['cumABT_to_res'], 'peakF': s2['F']['peakT'], 'scc': s2['scc2025'], 'status': s2['status']}
            EU = lambda d: p * d['US'] + (1 - p) * d['UF']
            o = {'lam': lam, 'stochastic': st, 'expected': dict(E, EU=EU(E)), 'optimist': dict(plan['optimist'], EU=EU(plan['optimist'])),
                 'pessimist': dict(plan['pessimist'], EU=EU(plan['pessimist'])), 'perfect': {'EU': p * sS['U'] + (1 - p) * sF['U']}}
            for lab in ('expected', 'optimist', 'pessimist'):
                o[lab]['regret'] = (st['EU'] - o[lab]['EU']) / lam
            o['perfect']['gain'] = (o['perfect']['EU'] - st['EU']) / lam
            out[str(p)] = o; save(OUT, item, out)
            print(f"{item} p={p}: regret expected {o['expected']['regret']:+.5f} optimist {o['optimist']['regret']:+.5f} pessimist {o['pessimist']['regret']:+.5f} "
                  f"PI gain {o['perfect']['gain']:.5f} | abt stoch {st['abt']:.1f} exp {E['abt']:.1f} opt {plan['optimist']['abt']:.1f} pes {plan['pessimist']['abt']:.1f} "
                  f"| SCC stoch {st['scc']:.2f} exp {E['scc']:.2f} | peakF stoch {st['peakF']:.3f} exp {E['peakF']:.3f} st {st['status']}", flush=True)
    print('done')
