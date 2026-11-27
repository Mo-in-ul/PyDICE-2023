"""Multi-start checks for the new runs (task C5). Results: results/p4_multistart.json (resumable).

Each check re-solves a reported case from a different starting point and records the difference in the reported quantity.
  det_free   2%, no storage limit, capacity 0.5 and 1.0, abatement free: start from the no-removal optimum instead of the
             default start (GAMS control path).
  det_held   2%, no storage limit and hard caps, capacity 0.5 and 1.0, control rate held: start from the default start
             with the control block overwritten, instead of from the free optimum.
  two_E70    engineered-only failure, 2%, p = 0.5: common periods taken from the failure state instead of the success state.
  two_learn  2%, learned 2040, p = 0.5: common periods from the failure state.
  ns_narrow  2%, no storage limit, capacity 0.25 or 0.75: common periods from the other state.
Usage: python p4_multistart.py det_free det_held two_E70 two_learn ns_narrow"""
import sys
import numpy as np
import removal3_stoch as ST, removal3_nstate as NS
from p4_common import *

OUT = 'p4_multistart.json'
REF_M = load('p4_mech.json'); REF_P = load('p4_partial.json'); REF_L = load('p4_learn.json'); REF_N = load('p4_nstate.json')

if __name__ == '__main__':
    for item in sys.argv[1:]:
        out = load(OUT).get(item, {})
        if item == 'det_free':
            x0n, _, _ = det(2, dict(VAR['D']), '')
            for s in (0.5, 1.0):
                if str(s) in out:
                    continue
                x, r, _ = det(2, capcal(s, VAR['D']), 'NGM', x0=x0n)
                ref = REF_M['curve:D_d2'][str(s)]['scc2025']
                out[str(s)] = {'status': r['status'], 'scc': r['scc2025'], 'ref': ref, 'diff': r['scc2025'] - ref}; save(OUT, item, out)
                print(f"{item} s={s}: st{r['status']} SCC {r['scc2025']:.5f} ref {ref:.5f} diff {r['scc2025'] - ref:+.2e}", flush=True)
        elif item == 'det_held':
            for v in ('D', 'A'):
                x0n, _, _ = det(2, dict(VAR[v]), '')
                for s in (0.5, 1.0):
                    key = f'{v}|{s}'
                    if key in out:
                        continue
                    import removal3 as R3
                    pr0 = R3.Problem(2, cal=capcal(s, VAR[v]), allow='NGM'); xs = np.array(pr0.start(None)); N = pr0.N
                    xs[:N] = np.asarray(x0n)[:N]
                    x, r, _ = det(2, capcal(s, VAR[v]), 'NGM', fix=x0n, x0=xs)
                    ref = REF_M[f'fixmu:{v}_d2'][str(s)]['scc2025']
                    out[key] = {'status': r['status'], 'scc': r['scc2025'], 'ref': ref, 'diff': r['scc2025'] - ref}; save(OUT, item, out)
                    print(f"{item} {key}: st{r['status']} SCC {r['scc2025']:.5f} ref {ref:.5f} diff {r['scc2025'] - ref:+.2e}", flush=True)
        elif item in ('two_E70', 'two_learn'):
            if 'scc' in out:
                continue
            if item == 'two_E70':
                calS, alS, calF, alF, t, ref = dict(A70), 'NGM', dict(A70), 'N', 2050, REF_P['E70_d2']['0.5']['scc2s']
            else:
                calS, alS, calF, alF, t, ref = dict(A70), 'NGM', {}, '', 2040, REF_L['learn:d2:2040']['0.5']['scc2s']
            xS, _, _ = det(2, calS, alS); xF, _, _ = det(2, calF, alF)
            m = ST.TwoState(2, calS, alS, calF, alF, 0.5, t)
            x, info, scc, dt = m.solve(x0=m.start(xS, xF, common_from='F'))
            out = {'status': int(info['status']), 'scc': float(scc[2]), 'ref': ref, 'diff': float(scc[2]) - ref, 'EU': -m.objective(x)}
            if item == 'two_learn':
                out['EU_ref'] = REF_L['learn:d2:2040']['0.5']['EU']; out['EU_diff'] = out['EU'] - out['EU_ref']
            save(OUT, item, out)
            print(f"{item}: st{out['status']} SCC {out['scc']:.5f} ref {ref:.5f} diff {out['diff']:+.2e}", flush=True)
        elif item == 'ns_narrow':
            if 'scc' in out:
                continue
            base = VAR['D']; scales = (0.25, 0.75)
            xs = [det(2, capcal(s, base), 'NGM')[0] for s in scales]
            m = NS.NState(2, [(capcal(s, base), 'NGM', 0.5) for s in scales], 2050)
            x, info, scc, dt = m.solve(m.start(xs, common_from=1))
            ref = REF_N['narrow:D_d2']['scc_ns']
            out = {'status': int(info['status']), 'scc': float(scc[2]), 'ref': ref, 'diff': float(scc[2]) - ref}; save(OUT, item, out)
            print(f"{item}: st{out['status']} SCC {out['scc']:.5f} ref {ref:.5f} diff {out['diff']:+.2e}", flush=True)
    print('done')
