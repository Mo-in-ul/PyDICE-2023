"""Appendix figures for Monte Carlo v2 in the house style (Times 8 pt, ticks in, Elsevier widths).
fig_mc2_share.pdf: 2025 feedback share against each of the eight parameters.
fig_mc2_scc_fan.pdf: SCC percentiles over time, with the DICE-2023 optimal SCC."""
import os, glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE, RES, dm
import mc_v2 as V

BLUE, ORANGE = '#2166ac', '#d6604d'
MM = 1 / 25.4
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'font.size': 8,
                     'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'pdf.fonttype': 42, 'savefig.dpi': 500,
                     'axes.linewidth': 0.6, 'mathtext.fontset': 'stix'})
LAB = {'rho': r'Time preference $\rho$', 'elasmu': r'Elasticity $\varphi$', 'ecs': r'ECS ($^\circ$C)', 'psi2': r'Damage coefficient $\psi_2$',
       'a3': 'Damage exponent', 'irC': r'$\sigma_C$ (yr/GtC)', 'irT': r'$\sigma_T$ (yr/$^\circ$C)', 'eland0': r'$E_{\mathrm{land}}(0)$ (GtCO$_2$/yr)'}


def load():
    R = {}
    for f in glob.glob(os.path.join(RES, 'mc2_ipopt', '*.json')):
        for o in json.load(open(f)):
            if o.get('status') in (0, 1):
                R[o['draw']] = o
    for f in glob.glob(os.path.join(RES, 'mc2', '*.json')):
        for o in json.load(open(f)):
            if o['draw'] not in R and o.get('success'):
                R[o['draw']] = o
    return [R[k] for k in sorted(R)]


if __name__ == '__main__':
    ok = load()
    summ = json.load(open(os.path.join(RES, 'mc2_summary.json')))
    P = np.array([[o['params'][k] for k in V.KEYS] for o in ok]); sh = np.array([100 * o['share_2025'] for o in ok])
    fig, axs = plt.subplots(2, 4, figsize=(190 * MM, 85 * MM), sharey=True)
    for j, (a, k) in enumerate(zip(axs.ravel(), V.KEYS)):
        a.scatter(P[:, j], sh, s=2, color='0.55', alpha=0.6, lw=0)
        a.axvline(V.DICE[k], color=ORANGE, lw=0.8, ls=':')
        a.set_xlabel(LAB[k])
        a.set_title(f"{summ['2025']['coef_pp_per_sd'][k]:+.2f} pp per s.d.", fontsize=7)
        if k == 'rho':
            a.set_xscale('log')
    for a in axs[:, 0]:
        a.set_ylabel('Feedback share, 2025 (%)')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'figs', 'fig_mc2_share.pdf')); plt.close(fig)

    # SCC fan: the IPOPT records carry 2025/2050/2100 only; use those three years
    yrs = np.array([2025, 2050, 2100])
    scc = np.array([[o[f'scc_{y}'] for y in yrs] for o in ok])
    q = np.percentile(scc, [5, 25, 50, 75, 95], axis=0)
    g = dm.load_gams(9)
    fig, ax = plt.subplots(figsize=(90 * MM, 60 * MM))
    ax.fill_between(yrs, q[0], q[4], color=BLUE, alpha=0.15, lw=0, label='5th to 95th percentile')
    ax.fill_between(yrs, q[1], q[3], color=BLUE, alpha=0.3, lw=0, label='25th to 75th percentile')
    ax.plot(yrs, q[2], color=BLUE, lw=1.2, marker='o', ms=2.5, label='Median')
    ax.plot(yrs, [g['SCC'][(y - 2020) // 5] for y in yrs], color=ORANGE, ls='--', lw=1.0, marker='s', ms=2.5, label='DICE-2023 optimum')
    ax.set_yscale('log'); ax.set_xlabel('Year of pulse'); ax.set_ylabel(r'SCC (2019 \$/tCO$_2$)')
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'figs', 'fig_mc2_scc_fan.pdf')); plt.close(fig)
    print('figures written', len(ok))
