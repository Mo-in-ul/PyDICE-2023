"""Mechanism of the saturation feedback, and why its share of the SCC changes sign.

On the GAMS control paths (scenarios 9 optimal, 8 Paris, 10 baseline), add a pulse of
h GtCO2/yr in year y0 and follow the responses per unit of pulse:
  fixed alpha      : alpha held at its baseline path
  responsive alpha : alpha re-solved from the IRF condition (solve_alpha)
The feedback contribution to any variable is responsive minus fixed. Reported for pulses in
2025, 2100 and 2150: d alpha, d MAT, d TATM, the four carbon boxes, and the baseline path of
emissions and of the boxes, so the sign of the feedback can be read against them.
Writes results/mechanism.json and figs/fig_mechanism.pdf (paper figure).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from house_style import BLUE, ORANGE, DOUBLE, MM, panel_label
from common import dm, save_json, scenario_params, gams_controls, HERE

H = 0.1
PULSES = (2025, 2100, 2150)


def responses(sc):
    _, p = scenario_params(sc)
    MIU, S, g = gams_controls(sc, p)
    N = p['num_periods']
    a0, _ = dm.solve_alpha(p, MIU, S)
    base = dm.diceTrajectory(p, MIU, S, a0)
    out = {'scenario': sc, 'years': (2020 + 5 * np.arange(N)).tolist(),
           'base': {'ECO2': g['ECO2'][:N].tolist(), 'MAT': base[3][1:].tolist(), 'TATM': base[4][1:].tolist(),
                    'alpha': a0[1:].tolist(), 'RES': [base[k][1:].tolist() for k in (11, 12, 13, 14)]},
           'pulses': {}}
    for y0 in PULSES:
        i = (y0 - 2020) // 5 + 1
        pulse = np.zeros(N + 1); pulse[i] = H
        a_r, _ = dm.solve_alpha(p, MIU, S, pulse=pulse, alpha_start=a0, first_period=i)
        rr = dm.diceTrajectory(p, MIU, S, a_r, pulse)
        rf = dm.diceTrajectory(p, MIU, S, a0, pulse)
        d = lambda run, k: ((run[k][1:] - base[k][1:]) / H).tolist()
        out['pulses'][str(y0)] = {
            'dalpha': ((a_r[1:] - a0[1:]) / H).tolist(),
            'resp': {'MAT': d(rr, 3), 'TATM': d(rr, 4), 'C': d(rr, 1), 'RES': [d(rr, k) for k in (11, 12, 13, 14)]},
            'fixed': {'MAT': d(rf, 3), 'TATM': d(rf, 4), 'C': d(rf, 1), 'RES': [d(rf, k) for k in (11, 12, 13, 14)]}}
    return out


def summarize(o):
    yrs = np.array(o['years'])
    s = {}
    for y0, d in o['pulses'].items():
        fb_T = np.array(d['resp']['TATM']) - np.array(d['fixed']['TATM'])
        fb_M = np.array(d['resp']['MAT']) - np.array(d['fixed']['MAT'])
        da = np.array(d['dalpha'])
        k = yrs <= 2300
        s[y0] = {'dalpha_max': float(da[k].max()), 'dalpha_min': float(da[k].min()),
                 'fbT_max': float(fb_T[k].max()), 'fbT_min': float(fb_T[k].min()),
                 'fbT_year_max': int(yrs[k][np.argmax(fb_T[k])]), 'fbT_year_min': int(yrs[k][np.argmin(fb_T[k])]),
                 'fbMAT_min': float(fb_M[k].min()), 'fbMAT_max': float(fb_M[k].max())}
    E = np.array(o['base']['ECO2']); neg = yrs[E < 0]
    s['first_net_negative_year'] = int(neg[0]) if len(neg) else None
    R = np.array(o['base']['RES'])
    s['first_year_box_negative'] = {f'box{j}': (int(yrs[R[j] < 0][0]) if np.any(R[j] < 0) else None) for j in range(4)}
    return s


def figure(o9, path):
    yrs = np.array(o9['years']); k = yrs <= 2250
    fig, ax = plt.subplots(1, 3, figsize=(DOUBLE, 60 * MM))
    for y0, ls in (('2025', '-'), ('2150', '--')):
        d = o9['pulses'][y0]
        ax[0].plot(yrs[k], np.array(d['dalpha'])[k], color='0.2', ls=ls, label=f'pulse in {y0}')
        for lab, col in (('fixed', ORANGE), ('resp', BLUE)):
            ax[1].plot(yrs[k], np.array(d[lab]['MAT'])[k], color=col, ls=ls)
            ax[2].plot(yrs[k], 1000 * np.array(d[lab]['TATM'])[k], color=col, ls=ls)
    ax[0].set_ylabel(r'Change in $\alpha$'); ax[0].legend(frameon=False, loc='lower right')
    ax[1].set_ylabel('Change in atmospheric carbon (GtC)'); ax[2].set_ylabel(r'Change in temperature ($10^{-3}$ $^\circ$C)')
    ax[1].plot([], [], color=ORANGE, label=r'$\alpha$ fixed'); ax[1].plot([], [], color=BLUE, label=r'$\alpha$ responds')
    ax[1].legend(frameon=False, loc='upper right')
    for a in ax:
        a.axhline(0, color='0.7', lw=0.6); a.set_xlabel('Year')
    for a, l in zip(ax, '(a) (b) (c)'.split()):
        panel_label(a, l)
    fig.tight_layout()
    fig.savefig(path); plt.close(fig)


if __name__ == '__main__':
    allr = {}
    for sc in (9, 8, 10):
        o = responses(sc); o['summary'] = summarize(o); allr[str(sc)] = o
        print(sc, o['summary'], flush=True)
    save_json(allr, 'mechanism.json')
    os.makedirs(os.path.join(HERE, 'figs'), exist_ok=True)
    figure(allr['9'], os.path.join(HERE, 'figs', 'fig_mechanism.pdf'))
