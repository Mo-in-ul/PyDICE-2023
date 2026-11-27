"""Exact decomposition of the precautionary premium (paper 2). Results: results/p3_jensen.json (resumable).

Premium over the expected-capacity benchmark:  SCC_2s - SCC(pK)
  = [p SCC(K) + (1-p) SCC(0) - SCC(pK)]                          Jensen gap (deterministic optima)
  + p [SCC_S^c - SCC(K)] + (1-p) [SCC_F^c - SCC(0)]              policy-path effect
  + [SCC_2s - p SCC_S^c - (1-p) SCC_F^c]                         weighting residual
SCC_s^c is the SCC of state s when the periods before learning are held at the two-state common policy and the state
re-optimises afterwards. Under expected utility with consumption common before learning, the marginal welfare of
consumption in a common period is the same in both states, so the residual should be zero up to solver tolerance.
Case U1 (removal from 2070 with probability p, otherwise never), learned in 2050, as in p3_mech."""
import os, sys, json
import numpy as np, jax.numpy as jnp
import removal3 as R3, removal3_stoch as ST
from p3_stoch import det_solve, SC, iy
from p3_mech import VAR, CAP, A70

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'results', 'p3_jensen.json')
T_RES = 2050


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


if __name__ == '__main__':
    for item in sys.argv[1:]:                                  # e.g. A_pub_0.5
        if item in load():
            continue
        v, c, p = item.split('_'); p = float(p); sc = SC[c]; k = iy(T_RES)
        base = dict(VAR[v]); cm = float(base.pop('capmult', 1.0))
        if base.pop('ncfix', False):
            pr0 = R3.Problem(sc, cal=base, allow=''); x0, *_ = pr0.solve()
            mu0 = np.array(pr0.unpack(jnp.asarray(x0))[0]); mu0[1] = pr0.p['miu1']; base['munc'] = mu0.tolist()
        calS = dict(A70, **base, **{kk: cm * val for kk, val in CAP.items()}); calF = dict(base)
        calE = dict(A70, **base, **{kk: p * cm * val for kk, val in CAP.items()})
        xS, sK = det_solve(sc, calS, 'NGM'); xF, s0 = det_solve(sc, calF, ''); _, sE = det_solve(sc, calE, 'NGM')
        m = ST.TwoState(sc, calS, 'NGM', calF, '', p, T_RES)
        x, info, scc, dt = m.solve(x0=m.start(xS, xF, common_from='S' if p >= 0.5 else 'F'))
        aS, aF = [np.asarray(z) for z in m.assemble(jnp.asarray(x))]
        cS, cF = None, None
        for lab, cal_, al_, xa in (('S', calS, 'NGM', aS), ('F', calF, '', aF)):
            pr = R3.Problem(sc, cal=cal_, allow=al_); pr.fixed = (xa, k); pr.fixed_blocks = range(5)   # controls only; carbon-cycle state free
            xx, inf_, rr, dd, _ = pr.solve(x0=xa); s_ = pr.summary(xx, inf_, rr, dd)
            if lab == 'S': cS = s_
            else: cF = s_
        S2, K_, Z_, E_, Sc, Fc = float(scc[2]), sK['scc2025'], s0['scc2025'], sE['scc2025'], cS['scc2025'], cF['scc2025']
        jensen = p * K_ + (1 - p) * Z_ - E_; pathS = p * (Sc - K_); pathF = (1 - p) * (Fc - Z_); resid = S2 - p * Sc - (1 - p) * Fc
        # welfare check: the re-solved states must reproduce the two-state expected welfare
        EUc = p * cS['U'] + (1 - p) * cF['U']; EU2 = -m.objective(x)
        out = dict(p=p, status=int(info['status']), statusS=cS['status'], statusF=cF['status'], scc2s=S2, sccK=K_, scc0=Z_, sccE=E_,
                   sccSc=Sc, sccFc=Fc, premium=S2 - E_, jensen=jensen, pathS=pathS, pathF=pathF, resid=resid, EU2=EU2, EUc=EUc)
        r = load(); r[item] = out; json.dump(r, open(OUT, 'w'), indent=1)
        print(f"{item}: st{out['status']}/{cS['status']}/{cF['status']} prem {S2 - E_:+.4f} = J {jensen:+.4f} + pS {pathS:+.4f} + pF {pathF:+.4f} + res {resid:+.2e}"
              f"  dEU {EU2 - EUc:+.2e} {dt:.0f}s", flush=True)
    print('done')
