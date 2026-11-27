"""Mechanism tests for the precautionary SCC premium (paper 2, v0.7). Results: results/p3_mech.json (resumable).

Case U1 (all removal from 2070 with probability p, otherwise never), learned in 2050. For each storage/cost variant and
calibration: (1) the SCC of deterministic economies with removal capacity scaled by s in {0, 0.25, 0.5, 0.75, 1}, which
gives the Jensen gap p SCC(1) + (1-p) SCC(0) - SCC(p); (2) the two-state SCC at p in {0.25, 0.5, 0.75} and its premium
over the certainty-equivalent SCC(p) and the probability-weighted SCC.
Variants:  A hard caps (central)   B soft limits   C caps x4 (1,200 / 4,000 Gt)   D no storage limit
           E no storage limit and weak cost convexity (b = 10 instead of 100)
           F hard caps, damage curvature above 2 C (slope-matched): dam_g = 2.25 (damages +25% at 3 C)
           G hard caps, cubic damages matched in level only at 2.5 C (the v0.6 test, for comparison)"""
import os, sys, json
import numpy as np
import removal3 as R3, removal3_stoch as ST
from p3_stoch import det_solve, short, SC

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'results', 'p3_mech.json')
A70 = dict(startN=2070.0, startG=2070.0, startM=2070.0)
CAP = dict(RN_inf=5.0, RG_inf=17.0, RM_inf=4.0)
VAR = {'A': {}, 'B': dict(capN=None, capG=None, sN=200.0, sG=200.0, nu=4.0), 'C': dict(capN=1200.0, capG=4000.0),
       'D': dict(capN=None, capG=None), 'E': dict(capN=None, capG=None, bN=10.0, bG=10.0, bM=10.0),
       'F': dict(dam_g=2.25, dam_t0=2.0), 'H': dict(capN=None, capG=None, tlo_set=-3.0), 'I': dict(tlo_set=-3.0), 'J': dict(dam_g=2.25, dam_t0=1.5), 'G': dict(dam_exp=3.0, dam_tref=2.5),
       # v0.8 robustness: non-CO2 mitigation fixed at the no-removal path (K, L), no near-term control ceiling (M, N),
       # access 2050 / 2090 (P50, P90), capacity x0.5 / x2 (Qh, Q2), leakage 1%, low effectiveness, capacity investment,
       # Howard-Sterner-level damages (x3.3)
       'K': dict(ncfix=True), 'L': dict(capN=None, capG=None, ncfix=True), 'M': dict(ramp_set=False), 'N': dict(capN=None, capG=None, ramp_set=False),
       'P50': dict(startN=2050.0, startG=2050.0, startM=2050.0), 'P90': dict(startN=2090.0, startG=2090.0, startM=2090.0),
       'Qh': dict(capmult=0.5), 'Q2': dict(capmult=2.0), 'Lk': dict(delta0=0.01), 'Ef': dict(effN=0.7, effG=0.8, effM=0.6),
       'Iv': dict(invest=True), 'HS': dict(dam_mult=3.3)}
SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
PS = (0.25, 0.5, 0.75)


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def save(key, out):
    r = load(); r[key] = out; json.dump(r, open(OUT, 'w'))


if __name__ == '__main__':
    for item in sys.argv[1:]:                     # e.g. A_pub D_d2
        v, c = item.split('_'); sc = SC[c]; base = VAR[v]
        out = load().get(item, {'curve': {}, 'stoch': {}})
        base = dict(base); cm = float(base.pop('capmult', 1.0))
        if base.pop('ncfix', False):                    # non-CO2 abatement follows the no-removal optimum of this calibration
            import jax.numpy as jnp
            pr0 = R3.Problem(sc, cal=base, allow=''); x0, *_ = pr0.solve()
            mu0 = np.array(pr0.unpack(jnp.asarray(x0))[0]); mu0[1] = pr0.p['miu1']; base['munc'] = mu0.tolist()
        calS = dict(A70, **base, **{k: cm * val for k, val in CAP.items()}); calF = dict(base)   # failure state: never available
        xs = {}
        for s in SCALES:
            if str(s) in out['curve']:
                continue
            if s == 0.0:
                x, r = det_solve(sc, calF, '')
            else:
                x, r = det_solve(sc, dict(calS, **{k: s * cm * val for k, val in CAP.items()}), 'NGM')
            out['curve'][str(s)] = short(r); save(item, out)
            print(f"{item} s={s}: st{r['status']} SCC {r['scc2025']:.3f} U {r['U']:.5f}", flush=True)
        xS, _ = det_solve(sc, calS, 'NGM'); xF, _ = det_solve(sc, calF, '')
        for p in PS:
            if str(p) in out['stoch']:
                continue
            m = ST.TwoState(sc, calS, 'NGM', calF, '', p, 2050)
            x, info, scc, dt = m.solve(x0=m.start(xS, xF, common_from='S' if p >= 0.5 else 'F'))
            s = m.summary(x, info, scc); cv = out['curve']
            ce = None
            if str(p) in cv:
                ce = cv[str(p)]['scc2025']
            lin = p * cv['1.0']['scc2025'] + (1 - p) * cv['0.0']['scc2025']
            out['stoch'][str(p)] = {'status': s['status'], 'scc2025': s['scc2025'], 'EU': s['EU'], 'ce': ce, 'lin': lin,
                                    'abt_to_res': s['S']['cumABT_to_res'], 'peakS': s['S']['peakT'], 'peakF': s['F']['peakT']}
            save(item, out)
            print(f"{item} p={p}: st{s['status']} SCC {s['scc2025']:.3f} CE {ce:.3f} lin {lin:.3f} premCE {(s['scc2025'] - ce) / ce * 100:+.2f}% {dt:.0f}s", flush=True)
    print('done')
