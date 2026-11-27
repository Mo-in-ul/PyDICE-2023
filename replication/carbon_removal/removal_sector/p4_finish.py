"""Open items of the revision analysis (11 Nov 2026, second pass). Results: results/p4_finish.json (resumable).

fix60     2%, removal from 2070 or never, learned 2060, p = 0.25 and 0.75: the first pass ended with IPOPT status 1.
          Re-solved warm-started from the p = 0.5 solution, at tol 1e-10 and, if needed, 1e-8.
disc:R    SCC-capacity curve (0, 0.25, 0.5, 0.75, 1; removal from 2070) at a constant utility discount rate R
          (0.025 or 0.035), hard caps (A) and no storage limit (D). Built like the DICE-2023 alternative discounting runs:
          the session of the neighbouring DISC run (2% for 2.5%, 3% for 3.5%) with its RR rebuilt at rate R by
          dice2023_corrected.apply_disc_prstp (elasticity 0.001). k0 and the carbon-price cap are those of that DISC run.
wide4:C   four-state lottery {0, 0.25, 0.75, 1}, equal chances, learned 2050 (C = A_pub or A_d2), run alone for memory.
Usage: python p4_finish.py fix60 | disc:0.025 | disc:0.035 | wide4:A_pub"""
import sys
import numpy as np
import removal3_stoch as ST, removal3_nstate as NS
import dice2023_corrected as dm
from p4_common import *

OUT = 'p4_finish.json'


def disc_session(rate):
    base = 2 if rate < 0.03 else 3
    p, b = gams_session(base)
    dm.apply_disc_prstp(p, prstp_value=rate)
    return base, (p, b)


if __name__ == '__main__':
    for item in sys.argv[1:]:
        out = load(OUT).get(item, {})
        if item == 'fix60':
            sc, k = 2, iy(2060)
            xS, sS, _ = det(sc, A70, 'NGM'); xF, sF, _ = det(sc, {}, '')
            m5 = ST.TwoState(sc, A70, 'NGM', {}, '', 0.5, 2060)
            x5, info5, _, _ = m5.solve(x0=m5.start(xS, xF, common_from='S'))
            out['p0.5_status'] = int(info5['status'])
            for p in (0.25, 0.75):
                if str(p) in out and out[str(p)]['status'] == 0:
                    continue
                best = None
                for tol in (1e-10, 1e-8):
                    m = ST.TwoState(sc, A70, 'NGM', {}, '', p, 2060)
                    x, info, scc, dt = m.solve(x0=x5, tol=tol, max_iter=8000)
                    s = m.summary(x, info, scc)
                    best = {'status': s['status'], 'tol': tol, 'EU': s['EU'], 'scc2s': s['scc2025'], 'peakF': s['F']['peakT'],
                            'abt_to_res': s['S']['cumABT_to_res'], 'time': dt}
                    print(f"fix60 p={p} tol {tol}: st{s['status']} EU {s['EU']:.6f} SCC {s['scc2025']:.3f} {dt:.0f}s", flush=True)
                    if s['status'] == 0:
                        break
                out[str(p)] = best; save(OUT, item, out)
        elif item.startswith('disc:'):
            rate = float(item.split(':')[1]); base, sess = disc_session(rate)
            for v in ('A', 'D'):
                for s in (0.0, 0.25, 0.5, 0.75, 1.0):
                    key = f'{v}|{s}'
                    if key in out:
                        continue
                    cal, allow = (dict(VAR[v]), '') if s == 0.0 else (capcal(s, VAR[v]), 'NGM')
                    x, r, pr = det(base, cal, allow, disc_session(rate)[1])
                    out[key] = rec(r, pr); out[key]['k0'] = float(pr.p['k0']); save(OUT, item, out)
                    print(f"{item} {key}: st{r['status']} SCC {r['scc2025']:.4f} peak {r['peakT']:.4f}", flush=True)
        elif item.startswith('wide4:'):
            if 'premium' in out:
                continue
            v, c = item.split(':')[1].split('_'); sc = SC[c]; base = VAR[v]
            scales, probs = (0.0, 0.25, 0.75, 1.0), (0.25, 0.25, 0.25, 0.25)
            spec = lambda s: (dict(base), '') if s == 0.0 else (capcal(s, base), 'NGM')
            dets = {s: det(sc, *spec(s)) for s in sorted(set(scales) | {0.5})}
            m = NS.NState(sc, [spec(s) + (p,) for s, p in zip(scales, probs)], 2050)
            x, info, scc, dt = m.solve(m.start([dets[s][0] for s in scales], common_from=3))
            lin = float(sum(p * dets[s][1]['scc2025'] for s, p in zip(scales, probs))); sE = dets[0.5][1]['scc2025']
            out = {'status': int(info['status']), 'scc_ns': float(scc[2]), 'sccE': sE, 'premium': float(scc[2]) - sE, 'jensen': lin - sE, 'time': dt}
            save(OUT, item, out)
            print(f"{item}: st{out['status']} premium {out['premium']:+.4f} ({100 * out['premium'] / sE:.2f}%) Jensen {out['jensen']:+.4f} {dt:.0f}s", flush=True)
    print('done')
