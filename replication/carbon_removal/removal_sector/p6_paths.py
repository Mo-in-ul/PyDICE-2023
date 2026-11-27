"""Control-rate and abatement paths by planning strategy, for Figure 3 of the journal manuscript. No new experiment:
re-solves the regret cases of p5_regret.py (2% calibration, removal from 2070 or never, p = 0.5, learned 2050 and
2070) and keeps the full paths. Stochastic optimum: joint solve for 2050; for 2070 the common controls of the
reduced-space optimum in results/p3_reduced.json, with each state re-solved after learning. Results: results/p6_paths.json.
Usage: python p6_paths.py d2:2050 d2:2070"""
import sys
import numpy as np
import removal3_stoch as ST
from p4_common import det, capcal, SC, iy, load, save
from p5_regret import fixed
import removal3 as R3


def fixed_ctrl(sc, cal, allow, xc, k):
    """Re-solve with only the control blocks of the first k periods fixed (the carbon-cycle variable stays free),
    as p3_reduced.State does."""
    pr = R3.Problem(sc, cal=cal, allow=allow); pr.fixed_blocks = range(NCTRL); pr.fixed = (np.asarray(xc), k)
    x, info, r, d, _ = pr.solve(x0=np.asarray(xc)); return pr.summary(x, info, r, d)

OUT = 'p6_paths.json'
NCTRL = 5


def keep(s):
    P = s['paths']
    return {'MU': [float(v) for v in P['MU']], 'ABT': [float(v) for v in P['ABT']], 'T': [float(v) for v in P['T']],
            'peakT': s['peakT'], 'U': s['U']}


if __name__ == '__main__':
    p = 0.5
    for item in sys.argv[1:]:
        c, t = item.split(':'); t = int(t); sc = SC[c]; k = iy(t)
        calS, calF = capcal(1.0), {}
        xS, sS, prS = det(sc, calS, 'NGM'); xF, sF, _ = det(sc, calF, '')
        xE, sE, _ = det(sc, capcal(p), 'NGM')
        N = prS.N
        lo, hi, _, _ = prS.bounds()
        out = {'years': [2025 + 5 * i for i in range(len(sS['paths']['MU']))], 'ceiling': [float(v) for v in np.asarray(hi)[1:N + 1]],
               'k': k, 'optimist': keep(sS), 'pessimist': keep(sF)}
        out['expected_S'] = keep(fixed(sc, calS, 'NGM', xE, k)); out['expected_F'] = keep(fixed(sc, calF, '', xE, k))
        if t == 2050:
            m = ST.TwoState(sc, calS, 'NGM', calF, '', p, t)
            x, info, scc, _ = m.solve(x0=m.start(xS, xF, common_from='S'))
            s2 = m.summary(x, info, scc)
            for lab in 'SF':
                out[f'stochastic_{lab}'] = {'MU': list(map(float, s2[lab]['MU'])), 'ABT': list(map(float, s2[lab]['ABT'])),
                                            'T': list(map(float, s2[lab]['T'])), 'peakT': s2[lab]['peakT'], 'U': s2[lab]['U']}
        else:
            rr = load('p3_reduced.json')[f'U1_{c}_2070|{p}|segment']
            idx = np.concatenate([np.arange(b * N, b * N + k) for b in range(NCTRL)])
            base = np.array(xF, float); base[idx] = rr['xcommon']
            out['stochastic_S'] = keep(fixed_ctrl(sc, calS, 'NGM', base, k)); out['stochastic_F'] = keep(fixed_ctrl(sc, calF, '', base, k))
            out['check_EU'] = {'reduced': rr['EU'], 'recomputed': p * out['stochastic_S']['U'] + (1 - p) * out['stochastic_F']['U']}
        save(OUT, item, out)
        cum = lambda d: 5 * sum(d['ABT'][:k])
        print(item, 'cum abatement to learning: stoch', round(cum(out['stochastic_S']), 1), 'expected', round(cum(out['expected_S']), 1),
              'opt', round(cum(out['optimist']), 1), 'pes', round(cum(out['pessimist']), 1), out.get('check_EU', ''), flush=True)
    print('done')
