"""The seven Monte Carlo draws where IPOPT from the DICE-2023 start did not converge use the SLSQP solution, whose
SCC is the fixed-control decomposition total (mc_fast.evaluate). Proposition 1 makes it equal to the multiplier SCC
when no state bound binds. Check: re-solve each with IPOPT warm-started from the SLSQP solution, and report the
IPOPT status, the multiplier SCC and its difference from the SLSQP decomposition total, the lowest temperature and
alpha on the path. Writes results/fallback_check.json."""
import json, glob, os
import numpy as np
from common import RES, save_json
import mc_v2 as V, mc_fast as Fs, mc_ift as M

if __name__ == '__main__':
    I = {}; S = {}
    for f in glob.glob(os.path.join(RES, 'mc2_ipopt', '*.json')):
        for o in json.load(open(f)): I[o['draw']] = o
    for f in glob.glob(os.path.join(RES, 'mc2', '*.json')):
        for o in json.load(open(f)): S[o['draw']] = o
    fb = sorted(k for k in I if I[k].get('status') not in (0, 1))
    X = V.design()
    from ipopt_model import Problem
    xg = Problem(9).start_from_gams()
    out = []
    for k in fb:
        p, b = V.draw_params(X[k])
        res, xs, irf = Fs.solve_slsqp(p, b, xg)
        o, x = M.evaluate(X[k], xs)
        r = {'draw': k, 'ipopt_status_cold': I[k].get('status'), 'ipopt_status_warm': o['status'],
             'scc_2025_slsqp_ift': S[k]['scc_2025'], 'scc_2025_ipopt_mult': o['scc_2025'],
             'diff_pct': 100 * (S[k]['scc_2025'] / o['scc_2025'] - 1),
             'share_2025_slsqp': S[k]['share_2025'], 'share_2025_ipopt': o['share_2025'],
             'ift_vs_mult_pct_2025': o['ift_vs_mult_pct_2025'], 'min_T': o['min_T'], 'alpha_min': o['alpha_min'], 'flag': o['flag']}
        out.append(r); print(json.dumps({q: (round(v, 6) if isinstance(v, float) else v) for q, v in r.items()}), flush=True)
        save_json(out, 'fallback_check.json')
    print('FB_DONE')
