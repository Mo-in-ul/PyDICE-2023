"""Precautionary-pricing tests and partial-success cases (paper 2, v0.6). Results: results/p3_prec.json (resumable).

Two-state cases, all learned in 2050 (where the joint two-state problem converges):
  U1q   : removal from 2070 or never, quadratic damages (as U1 of p3_stoch, recomputed for a common benchmark set)
  U1c   : the same with cubic damages, a2 T^3 matched to DICE-2023 damages at 2.5 C
  P_half: removal from 2070 at full capacity (S) or half capacity (F)
  P_eff : removal from 2070 with full effectiveness (S) or effectiveness 0.5 (F)
  D_tail: removal as in the central case and certain; damages as published (S) or twice as large (F)
For each case and p the script stores the two deterministic plans, the optimist and pessimist plans, the two-state policy,
and a certainty-equivalent deterministic economy (capacities, effectiveness or damage level set to their expected value).
The precautionary premium is SCC(two-state) - SCC(certainty equivalent)."""
import os, sys, json
import numpy as np
import removal3 as R3, removal3_stoch as ST
from p3_stoch import det_solve, short, SC, iy

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'results', 'p3_prec.json')
A70 = dict(startN=2070.0, startG=2070.0, startM=2070.0); NEVER = dict(startN=3000.0, startG=3000.0, startM=3000.0)
CUB = dict(dam_exp=3.0, dam_tref=2.5)
CAP = dict(RN_inf=5.0, RG_inf=17.0, RM_inf=4.0)
half = {k: v / 2 for k, v in CAP.items()}
EFF0 = 0.5


def ce(case, p):
    """Certainty-equivalent deterministic calibration (expected capacity, effectiveness or damage level)."""
    if case in ('U1q', 'U1c'):
        d = dict(A70, **{k: p * v for k, v in CAP.items()})
        return dict(d, **CUB) if case == 'U1c' else d, 'NGM'
    if case == 'P_half':
        return dict(A70, **{k: (p + (1 - p) / 2) * v for k, v in CAP.items()}), 'NGM'
    if case == 'P_eff':
        e = p + (1 - p) * EFF0
        return dict(A70, effN=e, effG=e, effM=e), 'NGM'
    if case == 'D_tail':
        return dict(dam_mult=p + (1 - p) * 2.0), 'NGM'


CASES = {'U1q': (A70, 'NGM', {}, ''), 'U1c': (dict(A70, **CUB), 'NGM', dict(CUB), ''),
         'P_half': (A70, 'NGM', dict(A70, **half), 'NGM'),
         'P_eff': (A70, 'NGM', dict(A70, effN=EFF0, effG=EFF0, effM=EFF0), 'NGM'),
         'D_tail': ({}, 'NGM', dict(dam_mult=2.0), 'NGM')}
PS = {'D_tail': (0.9, 0.75, 0.5)}
T_RES = 2050

if __name__ == '__main__':
    todo = sys.argv[1:]                         # e.g. U1c_pub P_half_d2
    for item in todo:
        case, c = item.rsplit('_', 1); sc = SC[c]; k = iy(T_RES)
        res = json.load(open(OUT)) if os.path.exists(OUT) else {}
        out = res.get(item, {'case': case, 'cal': c, 't_res': T_RES, 'stoch': {}, 'ce': {}})
        calS, allowS, calF, allowF = CASES[case]
        xS, sS = det_solve(sc, calS, allowS); xF, sF = det_solve(sc, calF, allowF)
        out['det_S'] = short(sS); out['det_F'] = short(sF)
        if 'optimist' not in out:
            for lab, xc in (('optimist', xS), ('pessimist', xF)):
                _, a = det_solve(sc, calS, allowS, fixed=(xc, k)); _, b = det_solve(sc, calF, allowF, fixed=(xc, k))
                out[lab] = {'S': short(a), 'F': short(b)}
        for p in PS.get(case, (0.1, 0.25, 0.5, 0.75, 0.9)):
            if str(p) in out['stoch']:
                continue
            m = ST.TwoState(sc, calS, allowS, calF, allowF, p, T_RES)
            x, info, scc, dt = m.solve(x0=m.start(xS, xF, common_from='S' if p >= 0.5 else 'F'))
            s = m.summary(x, info, scc); out['stoch'][str(p)] = s
            cc, ca = ce(case, p); _, sc_ = det_solve(sc, cc, ca); out['ce'][str(p)] = short(sc_)
            EUo = p * out['optimist']['S']['U'] + (1 - p) * out['optimist']['F']['U']; EUp = p * out['pessimist']['S']['U'] + (1 - p) * out['pessimist']['F']['U']
            print(f"{item} p={p}: st{s['status']} SCC {s['scc2025']:.3f} CE {sc_['scc2025']:.3f} lin {p * sS['scc2025'] + (1 - p) * sF['scc2025']:.3f} "
                  f"EU-max(opt,pes) {s['EU'] - max(EUo, EUp):.5f} abt {s['S']['cumABT_to_res']:.1f} {dt:.0f}s", flush=True)
            res = json.load(open(OUT)) if os.path.exists(OUT) else {}
            res[item] = out; json.dump(res, open(OUT, 'w'))
        res = json.load(open(OUT)) if os.path.exists(OUT) else {}
        res[item] = out; json.dump(res, open(OUT, 'w'))
    print('done')
