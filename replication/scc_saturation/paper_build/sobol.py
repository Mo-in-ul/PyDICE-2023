"""Variance-based (Sobol) sensitivity of the feedback share and the SCC, Monte Carlo v2.

The 500 Latin hypercube draws are not a Saltelli design, so the indices are computed on a surrogate:
a Gaussian process (scikit-learn, anisotropic RBF + white noise) fitted on the unit-cube coordinates
of the draws (the Latin hypercube before the inverse-CDF transform, so the Sobol indices are with
respect to the sampled distributions). Surrogate accuracy is reported as 10-fold cross-validated R^2.
Sobol first-order (S1) and total (ST) indices from SALib (Saltelli sampling, 2^13 base samples) with
bootstrap 95% intervals. Outputs: share of the SCC in 2025 and 2100 (%) and log SCC in 2025.
Writes results/sobol.json and tex/tab_sobol.tex."""
import os, json, glob
import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Ck
from sklearn.model_selection import cross_val_score, KFold
from SALib.sample import sobol as ssample
from SALib.analyze import sobol as sanalyze
from common import RES, HERE, save_json
import mc_v2 as V
from fig_mc2 import load

if __name__ == '__main__':
    U = qmc.LatinHypercube(d=len(V.KEYS), seed=2026).random(n=500)
    X = V.design()
    rec = {o['draw']: o for o in load()}
    idx = sorted(rec)
    assert np.allclose([[rec[k]['params'][kk] for kk in V.KEYS] for k in idx], X[idx])
    Ux = U[idx]
    outs = {'share_2025': np.array([100 * rec[k]['share_2025'] for k in idx]),
            'share_2100': np.array([100 * rec[k]['share_2100'] for k in idx]),
            'logscc_2025': np.array([np.log(rec[k]['scc_2025']) for k in idx])}
    problem = {'num_vars': len(V.KEYS), 'names': V.KEYS, 'bounds': [[0, 1]] * len(V.KEYS)}
    S = ssample.sample(problem, 2 ** 13, calc_second_order=False, seed=7)
    res = {}
    for name, y in outs.items():
        k = Ck(1.0) * RBF(length_scale=np.ones(len(V.KEYS)), length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(1e-3, (1e-8, 1e1))
        gp = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=2, random_state=0)
        cv = cross_val_score(gp, Ux, y, cv=KFold(10, shuffle=True, random_state=0), scoring='r2')
        gp.fit(Ux, y)
        Y = gp.predict(S)
        a = sanalyze.analyze(problem, Y, calc_second_order=False, seed=7)
        res[name] = {'cv_r2_mean': float(cv.mean()), 'cv_r2_min': float(cv.min()),
                     'S1': dict(zip(V.KEYS, a['S1'].tolist())), 'S1_conf': dict(zip(V.KEYS, a['S1_conf'].tolist())),
                     'ST': dict(zip(V.KEYS, a['ST'].tolist())), 'ST_conf': dict(zip(V.KEYS, a['ST_conf'].tolist())),
                     'sum_S1': float(a['S1'].sum())}
        print(name, 'CV R2', round(cv.mean(), 3), {kk: (round(a['S1'][j], 3), round(a['ST'][j], 3)) for j, kk in enumerate(V.KEYS)}, flush=True)
    save_json(res, 'sobol.json')
    lab = {'rho': '$\\rho$', 'elasmu': '$\\varphi$', 'ecs': 'ECS', 'psi2': '$\\psi_2$', 'a3': 'Damage exponent',
           'irC': '$\\sigma_C$', 'irT': '$\\sigma_T$', 'eland0': '$E_{\\mathrm{land}}(0)$'}
    L = []
    for kk in V.KEYS:
        L.append(lab[kk] + ' & ' + ' & '.join(f"{res[n]['S1'][kk]:.2f} & {res[n]['ST'][kk]:.2f}" for n in ('share_2025', 'share_2100', 'logscc_2025')) + ' \\\\')
    L.append('\\midrule')
    L.append('Sum of $S_1$ & ' + ' & '.join(f"\\multicolumn{{2}}{{c}}{{{res[n]['sum_S1']:.2f}}}" for n in ('share_2025', 'share_2100', 'logscc_2025')) + ' \\\\')
    L.append('Surrogate CV $R^2$ & ' + ' & '.join(f"\\multicolumn{{2}}{{c}}{{{res[n]['cv_r2_mean']:.3f}}}" for n in ('share_2025', 'share_2100', 'logscc_2025')) + ' \\\\')
    s = ("\\begin{tabular}{lrrrrrr}\n\\toprule\n & \\multicolumn{2}{c}{Share, 2025} & \\multicolumn{2}{c}{Share, 2100} & \\multicolumn{2}{c}{log SCC, 2025} \\\\\n"
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\nParameter & $S_1$ & $S_T$ & $S_1$ & $S_T$ & $S_1$ & $S_T$ \\\\\n\\midrule\n"
         + '\n'.join(L) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(HERE, 'tex', 'tab_sobol.tex'), 'w').write(s)
