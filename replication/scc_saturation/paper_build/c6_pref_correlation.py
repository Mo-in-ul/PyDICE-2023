"""v8: Monte Carlo with correlated preference parameters (referee comment 7).

The main Monte Carlo draws the pure rate of time preference and the elasticity of marginal utility independently.
Here the first n draws of the same Latin hypercube are re-paired so that the two have a rank correlation of +r or -r
(the elasticity values are permuted, so both marginal distributions are unchanged), and every draw is re-solved with
IPOPT and decomposed. Reported: median and 5th-95th percentiles of the 2025 share and SCC, against the same n draws
with independent pairing (taken from the main run, results/mc2_ipopt*).
Usage: python c6_pref_correlation.py r n     (e.g. 0.5 200 and -0.5 200). Writes results/c6_pref_corr_<r>.json.
"""
import sys, os
import numpy as np
from scipy.stats import norm
from common import save_json, load_json, RES
import mc_v2 as V, mc_fast as F

r = float(sys.argv[1]); n = int(sys.argv[2])
X = V.design()[:n].copy()
i_rho, i_eta = V.KEYS.index('rho'), V.KEYS.index('elasmu')
rng = np.random.default_rng(7)
rk = np.argsort(np.argsort(X[:, i_rho]))
z1 = norm.ppf((rk + 1) / (n + 1))
z2 = r * z1 + np.sqrt(1 - r ** 2) * rng.standard_normal(n)
eta_sorted = np.sort(X[:, i_eta])
X[:, i_eta] = eta_sorted[np.argsort(np.argsort(z2))]
from scipy.stats import spearmanr
rho_s = float(spearmanr(X[:, i_rho], X[:, i_eta])[0])

from ipopt_model import Problem
xg = Problem(9).start_from_gams()
name = f'c6_pref_corr_{r:+.1f}.json'
out = {'r_target': r, 'spearman_achieved': rho_s, 'n': n, 'draws': []}
for k in range(n):
    try:
        o, _ = F.evaluate_ipopt(X[k], xg)
        out['draws'].append({'k': k, 'success': o['success'], 'share_2025': o['share_2025'], 'scc_2025': o['scc_2025'],
                             'rho': float(X[k, i_rho]), 'elasmu': float(X[k, i_eta])})
    except Exception as e:
        out['draws'].append({'k': k, 'error': repr(e)})
    if k % 10 == 0 or k == n - 1:
        ok = [d for d in out['draws'] if d.get('success')]
        sh = np.array([d['share_2025'] for d in ok]); sc = np.array([d['scc_2025'] for d in ok])
        out['summary'] = {'n_ok': len(ok), 'share_pct': [float(np.percentile(100 * sh, q)) for q in (5, 50, 95)],
                          'scc': [float(np.percentile(sc, q)) for q in (5, 50, 95)]}
        save_json(out, name)
        print(k, out['summary'], flush=True)
