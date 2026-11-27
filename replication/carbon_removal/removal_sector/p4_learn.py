"""Value of learning by date (B5) and uncertain arrival date (B7). Results: results/p4_learn.json (resumable).

learn:CAL:YEAR   removal from 2070 or never, p = 0.5 (and 0.25, 0.75), outcome learned in YEAR (joint solve).
                 Expected welfare is compared across learning dates outside this script.
timing:CAL       removal arrives in 2070 (probability p) or in 2110, learned in 2050. Benchmarks: the two deterministic
                 plans, the probability-weighted SCC, and a known arrival in 2090 (the mean date for p = 0.5)."""
import sys
import numpy as np
import removal3 as R3, removal3_stoch as ST
from p4_common import *

OUT = 'p4_learn.json'
A = lambda y: dict(startN=float(y), startG=float(y), startM=float(y))


def fixed(sc, cal, allow, xc, k):
    pr = R3.Problem(sc, cal=cal, allow=allow); pr.fixed = (xc, k)
    x, info, r, d, _ = pr.solve(x0=xc); return pr.summary(x, info, r, d)


if __name__ == '__main__':
    for item in sys.argv[1:]:
        parts = item.split(':'); kind, c = parts[0], parts[1]; sc = SC[c]
        out = load(OUT).get(item, {})
        if kind == 'learn':
            t = int(parts[2]); calS, alS, calF, alF = A70, 'NGM', {}, ''
        else:
            t = 2050; calS, alS, calF, alF = A(2070), 'NGM', A(2110), 'NGM'
        k = iy(t)
        xS, sS, _ = det(sc, calS, alS); xF, sF, _ = det(sc, calF, alF); _, s0, _ = det(sc, {}, '')
        lam = s0['lamC2025'] / 5
        plans = {lab: (fixed(sc, calS, alS, xc, k), fixed(sc, calF, alF, xc, k)) for lab, xc in (('opt', xS), ('pes', xF))}
        s90 = det(sc, A(2090), 'NGM')[1] if kind == 'timing' else None
        xprev = None
        for p in (0.5, 0.25, 0.75):
            if str(p) in out:
                continue
            m = ST.TwoState(sc, calS, alS, calF, alF, p, t)
            x0 = m.start(xS, xF, common_from='S' if p >= 0.5 else 'F')
            x, info, scc, dt = m.solve(x0=x0)
            s = m.summary(x, info, scc); EU = s['EU']
            EUo = p * plans['opt'][0]['U'] + (1 - p) * plans['opt'][1]['U']; EUp = p * plans['pes'][0]['U'] + (1 - p) * plans['pes'][1]['U']
            EUpi = p * sS['U'] + (1 - p) * sF['U']
            ab = lambda q: float(5 * np.array(q['paths']['ABT'])[:k].sum())
            aO, aP, a2 = ab(sS), ab(sF), s['S']['cumABT_to_res']
            o = {'status': s['status'], 'EU': EU, 'EUo': EUo, 'EUp': EUp, 'EUpi': EUpi, 'lam': lam, 'scc2s': s['scc2025'], 'sccS': sS['scc2025'],
                 'sccF': sF['scc2025'], 'scc_none': s0['scc2025'], 'voh_opt': (EU - EUo) / lam, 'voh_pes': (EU - EUp) / lam,
                 'loss_2s_vs_pi': (EUpi - EU) / lam, 'weight_opt': (a2 - aP) / (aO - aP) if abs(aO - aP) > 1e-9 else None,
                 'peakS': s['S']['peakT'], 'peakF': s['F']['peakT'], 'peak_detS': sS['peakT'], 'peak_detF': sF['peakT'], 'peak_none': s0['peakT'],
                 'peakF_opt': plans['opt'][1]['peakT'], 'peakS_pes': plans['pes'][0]['peakT'],
                 'st_plans': [plans[l][j]['status'] for l in ('opt', 'pes') for j in (0, 1)]}
            if s90 is not None:
                o['scc2090'] = s90['scc2025']; o['peak2090'] = s90['peakT']
            out[str(p)] = o; save(OUT, item, out)
            print(f"{item} p={p}: st{o['status']} EU {EU:.6f} SCC {o['scc2s']:.3f} (S {o['sccS']:.3f} F {o['sccF']:.3f}) VoH opt {o['voh_opt']:.4f} pes {o['voh_pes']:.4f} "
                  f"loss vs PI {o['loss_2s_vs_pi']:.4f} weight {o['weight_opt']} peaks S {o['peakS']:.3f} F {o['peakF']:.3f} {dt:.0f}s", flush=True)
    print('done')
