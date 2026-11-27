"""Engineered-only failure (B3). Results: results/p4_partial.json (resumable).

E70: all removal becomes available in 2070 if it succeeds; if it fails, only biological removal does (from 2070).
E25: the central schedule; biological removal is certain from 2025; geological and mineral removal follow the central
     schedule with probability p, otherwise never (case U2 of p3_stoch.py, solved again here with the benchmark added).
Learned in 2050. For each p: two-state SCC, premium over the expected-capacity benchmark (biological capacity full,
geological and mineral capacity scaled by p), Jensen gap, value of hedging against the optimist and pessimist plans,
and the weight on the optimist plan (cumulative abatement to 2045).
Usage: python p4_partial.py CASE_CAL ...  e.g. E70_pub E25_d2"""
import sys
import numpy as np
import removal3 as R3, removal3_stoch as ST
from p4_common import *

OUT = 'p4_partial.json'
NEVER_GM = dict(startG=3000.0, startM=3000.0)
CASES = {'E70': (dict(A70), 'NGM', dict(A70), 'N', dict(A70)), 'E25': ({}, 'NGM', dict(NEVER_GM), 'NGM', {})}


def fixed(sc, cal, allow, xc, k):
    pr = R3.Problem(sc, cal=cal, allow=allow); pr.fixed = (xc, k)
    x, info, r, d, _ = pr.solve(x0=xc); return pr.summary(x, info, r, d)


if __name__ == '__main__':
    for item in sys.argv[1:]:
        case, c = item.split('_'); sc = SC[c]; k = iy(2050)
        calS, allowS, calF, allowF, calB = CASES[case]
        out = load(OUT).get(item, {})
        xS, sS, _ = det(sc, calS, allowS); xF, sF, _ = det(sc, calF, allowF); _, s0, _ = det(sc, {}, '')
        lam = s0['lamC2025'] / 5
        plans = {}
        for lab, xc in (('opt', xS), ('pes', xF)):
            a = fixed(sc, calS, allowS, xc, k); b = fixed(sc, calF, allowF, xc, k)
            plans[lab] = (a, b)
        for p in (0.25, 0.5, 0.75):
            if str(p) in out:
                continue
            _, sE, _ = det(sc, dict(calB, RG_inf=p * CAP['RG_inf'], RM_inf=p * CAP['RM_inf']), 'NGM')
            m = ST.TwoState(sc, calS, allowS, calF, allowF, p, 2050)
            x, info, scc, dt = m.solve(x0=m.start(xS, xF, common_from='S' if p >= 0.5 else 'F'))
            s = m.summary(x, info, scc); EU = s['EU']
            EUo = p * plans['opt'][0]['U'] + (1 - p) * plans['opt'][1]['U']; EUp = p * plans['pes'][0]['U'] + (1 - p) * plans['pes'][1]['U']
            EUpi = p * sS['U'] + (1 - p) * sF['U']
            ab = lambda q: float(5 * np.array(q['paths']['ABT'])[:k].sum())
            aO, aP, a2 = ab(sS), ab(sF), s['S']['cumABT_to_res']
            o = {'status': s['status'], 'st_plans': [plans[l][j]['status'] for l in ('opt', 'pes') for j in (0, 1)], 'scc2s': s['scc2025'],
                 'sccS': sS['scc2025'], 'sccF': sF['scc2025'], 'sccE': sE['scc2025'], 'scc_none': s0['scc2025'],
                 'premium': s['scc2025'] - sE['scc2025'], 'jensen': p * sS['scc2025'] + (1 - p) * sF['scc2025'] - sE['scc2025'],
                 'voh_opt': (EU - EUo) / lam, 'voh_pes': (EU - EUp) / lam, 'loss_opt_vs_pi': (EUpi - EUo) / lam, 'loss_2s_vs_pi': (EUpi - EU) / lam,
                 'weight_opt': (a2 - aP) / (aO - aP) if abs(aO - aP) > 1e-9 else None, 'abt_opt': aO, 'abt_pes': aP, 'abt_2s': a2,
                 'peakS': s['S']['peakT'], 'peakF': s['F']['peakT'], 'value_S': (sS['U'] - s0['U']) / lam, 'value_F': (sF['U'] - s0['U']) / lam}
            out[str(p)] = o; save(OUT, item, out)
            print(f"{item} p={p}: st{o['status']} plans {o['st_plans']} SCC {o['scc2s']:.3f} E {o['sccE']:.3f} premium {o['premium']:+.4f} "
                  f"({100 * o['premium'] / o['sccE']:.2f}%) Jensen {o['jensen']:+.4f} VoH opt {o['voh_opt']:.4f} pes {o['voh_pes']:.4f} "
                  f"weight {o['weight_opt']} values S {o['value_S']:.1f} F {o['value_F']:.1f} {dt:.0f}s", flush=True)
    print('done')
