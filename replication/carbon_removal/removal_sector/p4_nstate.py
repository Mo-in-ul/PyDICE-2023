"""Does the Jensen gap of the deterministic SCC-capacity curve give the premium for other capacity lotteries? (B1, B2)

Removal from 2070, learned 2050. For each lottery the script solves the n-state problem and compares its SCC premium over
the expected-capacity benchmark with the Jensen gap read off the deterministic curve, E[SCC(K)] - SCC(E[K]).
Usage: python p4_nstate.py LOTTERY:VAR_CAL ...   Results: results/p4_nstate.json (resumable).
  check   {0, 1} with probabilities (0.5, 0.5): must reproduce removal3_stoch.TwoState
  three   {0, 0.5, 1} with probabilities (0.25, 0.5, 0.25)            mean 0.5
  narrow  {0.25, 0.75} with probabilities (0.5, 0.5)                  mean 0.5
  skew    {0, 0.5, 1} with probabilities (0.5, 0.25, 0.25)            mean 0.375
  wide4   {0, 0.25, 0.75, 1} with probabilities 0.25 each             mean 0.5"""
import sys
import numpy as np
import removal3_nstate as NS
from p4_common import *

OUT = 'p4_nstate.json'
LOT = {'check': ((0.0, 1.0), (0.5, 0.5)), 'three': ((0.0, 0.5, 1.0), (0.25, 0.5, 0.25)), 'narrow': ((0.25, 0.75), (0.5, 0.5)),
       'skew': ((0.0, 0.5, 1.0), (0.5, 0.25, 0.25)), 'wide4': ((0.0, 0.25, 0.75, 1.0), (0.25, 0.25, 0.25, 0.25))}

if __name__ == '__main__':
    for item in sys.argv[1:]:
        if item in load(OUT):
            continue
        lot, vc = item.split(':'); v, c = vc.split('_'); sc = SC[c]; base = VAR[v]
        scales, probs = LOT[lot]; mean = float(np.dot(scales, probs))
        spec = lambda s: (dict(base), '') if s == 0.0 else (capcal(s, base), 'NGM')
        dets = {}
        for s in sorted(set(scales) | {mean}):
            cal, allow = spec(s); x, r, _ = det(sc, cal, allow); dets[s] = (x, r)
        m = NS.NState(sc, [spec(s) + (p,) for s, p in zip(scales, probs)], 2050)
        x, info, scc, dt = m.solve(m.start([dets[s][0] for s in scales], common_from=int(np.argmax(probs))))
        sccE = dets[mean][1]['scc2025']; lin = float(sum(p * dets[s][1]['scc2025'] for s, p in zip(scales, probs)))
        o = {'scales': scales, 'probs': probs, 'mean': mean, 'status': int(info['status']), 'scc_ns': float(scc[2]), 'sccE': sccE,
             'scc_det': {str(s): dets[s][1]['scc2025'] for s in dets}, 'premium': float(scc[2]) - sccE, 'jensen': lin - sccE,
             'EU': -float(m.objective(x)), 'time': dt}
        save(OUT, item, o)
        print(f"{item}: st{o['status']} SCC {o['scc_ns']:.4f} benchmark {sccE:.4f} premium {o['premium']:+.4f} ({100 * o['premium'] / sccE:.2f}%) "
              f"Jensen gap {o['jensen']:+.4f} path {o['premium'] - o['jensen']:+.4f} {dt:.0f}s", flush=True)
    print('done')
