"""Monte Carlo figures and summary from results/mc/fast_*.json.

fig_mc_scc_fan.pdf      : SCC (alpha responsive) by year, median and 5-95 / 25-75 bands, with the
                          DICE-2023 GAMS optimal SCC for reference
fig_mc_share.pdf        : 2025 feedback share against each sampled parameter, DICE calibration marked
results/mc_regression.json : standardized linear regression of the 2025, 2050 and 2100 shares on
                          the five parameters (coefficients in percentage points per s.d., R^2)
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import dm, load_json, save_json, RES, HERE

KEYS = ['psi2', 'rho', 'irC', 'irT', 'eland0']
LABELS = {'psi2': r'Damage coefficient $\psi_2$', 'rho': r'Time preference $\rho$', 'irC': r'IRF carbon sensitivity',
          'irT': r'IRF temperature sensitivity', 'eland0': r'Land emissions 2020 (GtCO$_2$/yr)'}
DICE = {'psi2': 0.003467, 'rho': 0.001, 'irC': 0.019, 'irT': 4.165, 'eland0': 5.9}


def load():
    """All draws; where the GAMS stability bound TATM >= 0.5 bound, the rerun without it
    (results/mc/nofloor_*.json) replaces the original draw."""
    R = {}
    for f in sorted(glob.glob(os.path.join(RES, 'mc', 'fast_*.json'))):
        for o in load_json(os.path.join('mc', os.path.basename(f))):
            if 'error' not in o:
                R[o['draw']] = o
    for f in sorted(glob.glob(os.path.join(RES, 'mc', 'nofloor_*.json'))):
        for o in load_json(os.path.join('mc', os.path.basename(f))):
            if 'error' not in o:
                R[o['draw']] = o
    return [R[k] for k in sorted(R)]


if __name__ == '__main__':
    R = load(); ok = [o for o in R if o['success']]
    P = np.array([[o['params'][k] for k in KEYS] for o in ok])
    reg = {'n_draws': len(R), 'n_success': len(ok), 'n_floor_removed': sum(bool(o.get('floor_removed')) for o in R),
           'n_floor_still_binding': sum(bool(o['Tlo_binds']) for o in R if not o.get('floor_removed'))}
    for y in (2025, 2050, 2100):
        sh = np.array([100 * o[f'share_{y}'] for o in ok])
        X = np.c_[np.ones(len(sh)), (P - P.mean(0)) / P.std(0)]
        b = np.linalg.lstsq(X, sh, rcond=None)[0]
        reg[str(y)] = {'coef_pp_per_sd': dict(zip(KEYS, b[1:].tolist())), 'R2': float(1 - np.var(sh - X @ b) / np.var(sh)),
                       'p5': float(np.percentile(sh, 5)), 'p50': float(np.median(sh)), 'p95': float(np.percentile(sh, 95)),
                       'min': float(sh.min()), 'max': float(sh.max()), 'share_negative': float(np.mean(sh < 0))}
    save_json(reg, 'mc_regression.json')
    print(reg)

    # fan chart
    scc = np.array([o['scc_path'] for o in ok]); yrs = 2020 + 5 * np.arange(scc.shape[1])
    k = (yrs >= 2025) & (yrs <= 2150)
    q = np.percentile(scc[:, k], [5, 25, 50, 75, 95], axis=0)
    g = dm.load_gams(9)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.fill_between(yrs[k], q[0], q[4], color='#1f5fa8', alpha=0.12, lw=0, label='5-95th percentile')
    ax.fill_between(yrs[k], q[1], q[3], color='#1f5fa8', alpha=0.25, lw=0, label='25-75th percentile')
    ax.plot(yrs[k], q[2], color='#1f5fa8', lw=1.8, label='Median')
    ax.plot(yrs[k], g['SCC'][:len(yrs)][k], color='0.3', ls='--', lw=1.2, label='DICE-2023 optimal (GAMS)')
    ax.set_xlabel('Year'); ax.set_ylabel(r'SCC (2019 US\$/tCO$_2$)'); ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, f'{len(ok)} Latin hypercube draws, cost-benefit optimal, alpha responsive', fontsize=7, color='0.4')
    fig.tight_layout(rect=[0, 0.04, 1, 1]); fig.savefig(os.path.join(HERE, 'figs', 'fig_mc_scc_fan.pdf')); plt.close(fig)

    # share vs parameters
    sh = np.array([100 * o['share_2025'] for o in ok])
    fig, axs = plt.subplots(1, 5, figsize=(13, 2.9), sharey=True)
    for j, (a, key) in enumerate(zip(axs, KEYS)):
        a.scatter(P[:, j], sh, s=3, color='0.55', alpha=0.5, lw=0)
        a.axvline(DICE[key], color='#1f5fa8', lw=1, ls=':')
        a.set_xlabel(LABELS[key], fontsize=8); a.grid(alpha=0.25)
        c = reg['2025']['coef_pp_per_sd'][key]
        a.set_title(f'{c:+.2f} pp per s.d.', fontsize=8)
    axs[0].set_ylabel('Feedback share, SCC 2025 (%)', fontsize=8)
    fig.text(0.01, 0.01, f'{len(ok)} draws; dotted line: DICE-2023 calibration; titles: standardized regression coefficient (R$^2$ = {reg["2025"]["R2"]:.2f})', fontsize=7, color='0.4')
    fig.tight_layout(rect=[0.01, 0.05, 1, 1]); fig.savefig(os.path.join(HERE, 'figs', 'fig_mc_share.pdf')); plt.close(fig)
