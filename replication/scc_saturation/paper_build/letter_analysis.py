"""Figures, table and numbers for the letter on the elasticity of marginal utility in DICE-2023.
Inputs: results/eta_letter_grid.json, results/eta_letter_horizon.json (eta_letter.py).
Outputs (letter/): fig_eta_scc.pdf, fig_eta_rates.pdf, tab_horizon.tex, numbers_letter.tex; results/letter_summary.json.

Spot consumption discount rate between 2020 and 2020 + h (h = 5(t-1)):
    R(h) = -(1/h) ln[lambda_C(t) / lambda_C(1)],
lambda_C the marginal utility of consumption (GAMS cc.m) along the optimum. In DICE-2023
    R(h) ~ rho_eff + eta * gbar(h) - 0.5 eta^2 sigma^2 min(h, 235),
gbar(h) average growth of consumption per person, so dR/deta < 0 when eta > eta*(h) = gbar(h) / (sigma^2 min(h, 235)).
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import RES, HERE, load_json, save_json

OUT = os.path.join(os.path.dirname(HERE), 'letter')
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GREY = '#2166ac', '#d6604d', '0.45'
MM = 1 / 25.4
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'font.size': 8,
                     'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'pdf.fonttype': 42, 'savefig.dpi': 500,
                     'axes.linewidth': 0.6, 'mathtext.fontset': 'stix'})
SIG2 = 0.01 ** 2


def spot(o):
    r = np.array(o['rate_path'])                 # 5-year forward rates, per year
    h = 5 * np.arange(1, len(r) + 1)
    return h, np.cumsum(np.log1p(r) * 5) / h        # continuously compounded spot rate


if __name__ == '__main__':
    G = [o for o in load_json('eta_letter_grid.json') if o['status'] in (0, 1) and np.isfinite(o['scc_2025'])]
    H = [o for o in load_json('eta_letter_horizon.json') if o['status'] in (0, 1)]
    # The lower temperature bound (0.01 C) binds in the 'dice' runs from eta = 1.6 on; there the multiplier SCC
    # carries its shadow value. The 'dice_nofloor' runs (eta >= 1.45) drop the bound; they replace the 'dice'
    # runs wherever they exist (identical below 1.6).
    NF = [o for o in load_json('eta_letter_nofloor.json') if o['status'] in (0, 1)]
    nf = {(o['N'], o['eta']) for o in NF}
    floor_bind = sorted(o['eta'] for o in G if o['variant'] == 'dice' and o['min_T'] < 0.0101)
    G = [o for o in G if not (o['variant'] == 'dice' and (o['N'], o['eta']) in nf)] + [dict(o, variant='dice') for o in NF if o['N'] == 81]
    H = [o for o in H if not (o['variant'] == 'dice' and (o['N'], o['eta']) in nf)] + [dict(o, variant='dice') for o in NF if o['N'] != 81]
    by = lambda L, v, N=81: sorted([o for o in L if o['variant'] == v and o['N'] == N], key=lambda o: o['eta'])
    dice, fixed, nop = by(G, 'dice'), by(G, 'fixed'), by(G, 'noprec')
    e_d = np.array([o['eta'] for o in dice]); s_d = np.array([o['scc_2025'] for o in dice])
    i_min = int(np.argmin(s_d))
    # quadratic refinement of the minimum
    j = slice(max(i_min - 2, 0), i_min + 3)
    c = np.polyfit(e_d[j], s_d[j], 2); e_star = -c[1] / (2 * c[0]); s_star = np.polyval(c, e_star)
    # where the SCC returns to its eta = 0.95 value
    s095 = s_d[np.argmin(np.abs(e_d - 0.95))]
    above = np.where((e_d > e_star) & (s_d > s095))[0]
    e_back = float(np.interp(s095, s_d[i_min:], e_d[i_min:])) if len(above) else None
    # threshold eta*(h) from the consumption path at eta = 0.95
    o95 = [o for o in dice if abs(o['eta'] - 0.95) < 1e-9][0]
    cpc = np.array(o95['cpc']); hh = 5 * np.arange(1, len(cpc))
    gbar = np.log(cpc[1:] / cpc[0]) / hh
    eta_thr = gbar / (SIG2 * np.minimum(hh, 235))
    summ = {'floor_binds_from': floor_bind[0] if floor_bind else None, 'nofloor_minT': min(o['min_T'] for o in NF),
            'eta_min': float(e_star), 'scc_min': float(s_star), 'eta_back_to_095': e_back, 'scc_095': float(s095),
            'grid': [(o['eta'], o['scc_2025'], o['peak_T']) for o in dice],
            'fixed': [(o['eta'], o['scc_2025'], o['peak_T']) for o in fixed],
            'noprec': [(o['eta'], o['scc_2025'], o['peak_T']) for o in nop],
            'eta_thr': {int(h): float(v) for h, v in zip(hh, eta_thr) if h in (50, 80, 100, 130, 150, 200, 235)},
            'gbar': {int(h): float(v) for h, v in zip(hh, gbar) if h in (50, 100, 150, 200, 235)}}
    for e in (0.5, 0.95, 1.5, 2.0):
        o = [x for x in dice if abs(x['eta'] - e) < 1e-9][0]; h, R = spot(o)
        summ[f'spot_{e}'] = {int(k): float(v) for k, v in zip(h, R) if k in (5, 50, 100, 150, 200, 235, 300, 400)}
        summ[f'fwd_{e}'] = {int(5 * (k + 1)): float(v) for k, v in enumerate(o['rate_path']) if 5 * (k + 1) in (5, 50, 100, 150, 200, 230, 240, 300)}
    hz = {}
    for N in (61, 81, 101):
        L = H if N != 81 else G
        for v in ('dice', 'fixed'):
            for o in by(L, v, N):
                if o['eta'] in (0.5, 0.75, 0.95, 1.25, 1.5, 1.75, 2.0):
                    hz.setdefault(v, {}).setdefault(o['eta'], {})[N] = o['scc_2025']
    summ['horizon'] = {v: {str(e): d for e, d in m.items()} for v, m in hz.items()}
    save_json(summ, 'letter_summary.json')

    # Figure 1: SCC and peak warming against eta
    fig, ax = plt.subplots(1, 2, figsize=(190 * MM, 62 * MM))
    for L, lab, col, ls in ((dice, 'As in DICE-2023', BLUE, '-'), (fixed, 'Precautionary term held at $\\varphi = 0.95$', ORANGE, '--'),
                            (nop, 'No precautionary term', GREY, ':')):
        e = [o['eta'] for o in L]
        ax[0].plot(e, [o['scc_2025'] for o in L], color=col, ls=ls, lw=1.1, label=lab)
        ax[1].plot(e, [o['peak_T'] for o in L], color=col, ls=ls, lw=1.1, label=lab)
    ax[0].set_ylabel(r'Optimal SCC in 2025 (2019 \$/tCO$_2$)'); ax[0].set_ylim(0, 150)
    ax[1].set_ylabel(r'Optimal peak warming ($^\circ$C)')
    for a in ax:
        a.axvline(0.95, color='0.7', lw=0.6, zorder=0); a.set_xlabel(r'Elasticity of marginal utility $\varphi$')
    ax[0].legend(frameon=False, loc='lower left')
    ax[0].text(0.02, 0.96, '(a)', transform=ax[0].transAxes, va='top'); ax[1].text(0.02, 0.96, '(b)', transform=ax[1].transAxes, va='top')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_eta_scc.pdf')); plt.close(fig)

    # Figure 2: spot discount rates and the threshold eta*(h)
    fig, ax = plt.subplots(1, 2, figsize=(190 * MM, 62 * MM))
    cols = {0.5: '0.65', 0.95: GREY, 1.5: ORANGE, 2.0: BLUE}
    for e in (0.5, 0.95, 1.5, 2.0):
        o = [x for x in dice if abs(x['eta'] - e) < 1e-9][0]; h, R = spot(o)
        ax[0].plot(2020 + h, 100 * R, color=cols[e], lw=1.1, label=f'$\\varphi = {e}$')
    ax[0].axhline(0, color='0.7', lw=0.6); ax[0].set_xlabel('Year'); ax[0].set_ylabel('Discount rate from 2020 (%/yr)'); ax[0].set_ylim(0, 6)
    ax[0].legend(frameon=False, loc='upper right')
    ax[1].plot(2020 + hh, eta_thr, color=BLUE, lw=1.1)
    ax[1].axhline(0.95, color='0.7', lw=0.6); ax[1].set_ylim(0, 4)
    ax[1].set_xlabel('Year'); ax[1].set_ylabel(r'Threshold $\varphi^*(h)$')
    ax[1].text(2230, 2.6, r'$\partial R/\partial\varphi < 0$ above the curve', fontsize=7)
    ax[0].text(0.02, 0.96, '(a)', transform=ax[0].transAxes, va='top'); ax[1].text(0.02, 0.96, '(b)', transform=ax[1].transAxes, va='top')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_eta_rates.pdf')); plt.close(fig)

    # Table: horizon robustness
    rows = []
    for e in (0.5, 0.95, 1.25, 1.5, 1.75, 2.0):
        d = hz['dice'][e]; f = hz['fixed'][e]
        rows.append(f"{e:.2f} & " + ' & '.join(f"{d[N]:.0f}" if N in d else '--' for N in (61, 81, 101)) + ' & ' + ' & '.join(f"{f[N]:.0f}" if N in f else '--' for N in (61, 81, 101)) + ' \\\\')
    tab = ("\\begin{tabular}{lrrrrrr}\n\\toprule\n & \\multicolumn{3}{c}{As in DICE-2023} & \\multicolumn{3}{c}{Precautionary term fixed} \\\\\n"
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n$\\varphi$ & 2320 & 2420 & 2520 & 2320 & 2420 & 2520 \\\\\n\\midrule\n" + '\n'.join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(OUT, 'tab_horizon.tex'), 'w').write(tab)
    print(json.dumps({k: v for k, v in summ.items() if k not in ('grid', 'fixed', 'noprec')}, indent=1))
