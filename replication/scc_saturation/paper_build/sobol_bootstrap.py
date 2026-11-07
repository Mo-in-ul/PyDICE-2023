"""Stability of the Sobol ranking of the 2025 feedback share (surrogate uncertainty). The 500 draws are resampled
with replacement 100 times; each time the Gaussian-process surrogate of sobol.py is refitted (fixed kernel form,
hyperparameters re-optimized) and the Sobol indices recomputed on 2^12 Saltelli base samples. Reports the 5th-95th
percentile of S1 and ST per parameter and how often each ordering of psi2 and ECS occurs.
Writes results/sobol_bootstrap.json."""
import numpy as np, warnings
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Ck
from SALib.sample import sobol as ssample
from SALib.analyze import sobol as sanalyze
from common import save_json
import mc_v2 as V
from fig_mc2 import load

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    U = qmc.LatinHypercube(d=len(V.KEYS), seed=2026).random(n=500)
    rec = {o['draw']: o for o in load()}; idx = np.array(sorted(rec))
    Ux = U[idx]; y = np.array([100 * rec[k]['share_2025'] for k in idx])
    problem = {'num_vars': len(V.KEYS), 'names': V.KEYS, 'bounds': [[0, 1]] * len(V.KEYS)}
    S = ssample.sample(problem, 2 ** 12, calc_second_order=False, seed=7)
    rng = np.random.default_rng(11); B = 100
    S1 = np.zeros((B, len(V.KEYS))); ST = np.zeros_like(S1)
    for b in range(B):
        j = rng.integers(0, len(y), len(y))
        k = Ck(1.0) * RBF(length_scale=np.ones(len(V.KEYS)), length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(1e-3, (1e-8, 1e1))
        gp = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=0, random_state=b).fit(Ux[j], y[j])
        a = sanalyze.analyze(problem, gp.predict(S), calc_second_order=False, seed=7)
        S1[b], ST[b] = a['S1'], a['ST']
        if b % 10 == 0:
            print(b, dict(zip(V.KEYS, np.round(S1[b], 3))), flush=True)
    iE, iP = V.KEYS.index('ecs'), V.KEYS.index('psi2')
    out = {'B': B, 'S1': {k: {'p5': float(np.percentile(S1[:, i], 5)), 'p50': float(np.median(S1[:, i])), 'p95': float(np.percentile(S1[:, i], 95))} for i, k in enumerate(V.KEYS)},
           'ST': {k: {'p5': float(np.percentile(ST[:, i], 5)), 'p50': float(np.median(ST[:, i])), 'p95': float(np.percentile(ST[:, i], 95))} for i, k in enumerate(V.KEYS)},
           'frac_psi2_above_ecs_S1': float(np.mean(S1[:, iP] > S1[:, iE])),
           'frac_rho_first_S1': float(np.mean(np.argmax(S1, 1) == V.KEYS.index('rho'))),
           'max_sum_carbon_S1': float(np.max(S1[:, V.KEYS.index('irC')] + S1[:, V.KEYS.index('irT')]))}
    save_json(out, 'sobol_bootstrap.json'); print(out)
