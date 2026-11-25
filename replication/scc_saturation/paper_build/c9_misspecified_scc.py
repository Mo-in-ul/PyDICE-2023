"""v10: SCCs on the path chosen by the planner who ignores the feedback (final-round referee comment 5.3).

Re-runs the fixed point of policy_fixed_alpha.py (optimal alpha as the start) and reports, for 2030, on the optimal path
and on the fixed-point path: the full (responsive) SCC, the fixed-alpha SCC (direct part) and the carbon price.
Writes results/c9_misspecified_scc.json.
"""
import numpy as np
from common import dm, save_json
from ipopt_model import solve
from scc_decomposition import scc_decomposition
from c5_fixed_alpha_starts import fixed_point

N = 81
out = {}
for sc in (9, 4):
    opt = solve(sc, tol=1e-11)
    p = opt['problem'].p
    a0 = np.concatenate([[0.0], opt['x'][2 * N:3 * N]])
    fp = fixed_point(sc, a0, opt['x'].copy(), p)
    MIUo = np.concatenate([[0.0], opt['x'][:N]]); So = np.concatenate([[0.0], opt['x'][N:2 * N]])
    r = solve(sc, tol=1e-11, fix_alpha=fp['alpha'], x0=opt['x'])
    MIUf = np.concatenate([[0.0], r['x'][:N]]); Sf = np.concatenate([[0.0], r['x'][N:2 * N]])
    do = scc_decomposition(p, MIUo, So); df = scc_decomposition(p, MIUf, Sf, alpha=fp['alpha'])
    k = 2  # 2030
    price = lambda M: float(p['PBACKTIME'][3] * M[3] ** (p['expcost2'] - 1))
    out[str(sc)] = {'optimal': {'scc_full': float(do['total'][k]), 'scc_fixed_alpha': float(do['direct'][k]), 'price': price(MIUo)},
                    'ignored': {'scc_full': float(df['total'][k]), 'scc_fixed_alpha': float(df['direct'][k]), 'price': price(MIUf)}}
    print(sc, out[str(sc)], flush=True)
save_json(out, 'c9_misspecified_scc.json')
