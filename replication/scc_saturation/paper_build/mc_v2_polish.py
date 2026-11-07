"""Monte Carlo v2, second pass with IPOPT (mc_ift.evaluate, multiplier SCC + IFT split).
  * every draw where SLSQP did not report success is re-solved with IPOPT; the IPOPT result replaces
    it when IPOPT converges (status 0 or 1) and the IFT total matches the multiplier SCC within 0.1%;
  * 20 successful SLSQP draws, evenly spaced, are re-solved as a check on the SLSQP results.
Writes results/mc2_polish/polish.json and results/mc2_polish/validate.json."""
import os, glob, json
import numpy as np
from common import save_json, RES
import mc_v2 as V
import mc_ift as M

if __name__ == '__main__':
    R = {}
    for f in sorted(glob.glob(os.path.join(RES, 'mc2', '*.json'))):
        for o in json.load(open(f)):
            R[o['draw']] = o
    X = V.design()
    from ipopt_model import Problem
    xg = Problem(9).start_from_gams()
    os.makedirs(os.path.join(RES, 'mc2_polish'), exist_ok=True)
    bad = [k for k in sorted(R) if not R[k].get('success')]
    out = []
    for k in bad:
        o, _ = M.evaluate(X[k], xg)
        o['draw'] = k; o['solver'] = 'ipopt'
        o['success'] = bool(o['status'] in (0, 1) and not o['flag'])
        out.append(o); save_json(out, 'mc2_polish/polish.json')
        print('polish', k, o['status'], o['success'], round(o['share_2025'], 4), flush=True)
    good = [k for k in sorted(R) if R[k].get('success')]
    val = []
    for k in good[::max(1, len(good) // 20)][:20]:
        o, _ = M.evaluate(X[k], xg)
        s = R[k]
        val.append({'draw': k, 'status': o['status'], 'scc_2025_slsqp': s['scc_2025'], 'scc_2025_ipopt': o['scc_2025'],
                    'share_2025_slsqp': s['share_2025'], 'share_2025_ipopt': o['share_2025'],
                    'scc_rel_diff_pct': 100 * (s['scc_2025'] / o['scc_2025'] - 1),
                    'share_diff_pp': 100 * (s['share_2025'] - o['share_2025'])})
        save_json(val, 'mc2_polish/validate.json')
        print('validate', val[-1], flush=True)
