"""v8: Figure 5 of the ERE manuscript (fixed-policy vs policy response), as fig_v6.py with the first bar of panel (d)
renamed from 'physical' to 'fixed policy' (referee comment 3). Writes figs/fig_v8_policy.pdf."""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from common import dm, load_json, gams_session, gams_controls, HERE
from house_style import BLUE, ORANGE, DOUBLE, MM


def panel_label(ax, s):
    ax.text(-0.02, 1.03, s, transform=ax.transAxes, ha='right', va='bottom', fontsize=8, weight='bold')


FIG = os.path.join(HERE, 'figs')
GREY = '0.45'

# ---------------- Figure 4: physical vs policy response ----------------
B = load_json('exp_b_physical_policy.json'); base = B['baseline']
fig, A4 = plt.subplots(2, 2, figsize=(DOUBLE, 112 * MM)); axs = A4.ravel()
LINES = (('policy fixed at DICE-2023 optimum', 'fixed', '0.35', '--'), ('policy re-optimized', 'reopt', BLUE, '-'))
for ax, key, xl, scale in ((axs[0], 'ecs', 'Equilibrium climate sensitivity ($^\\circ$C)', 1.0), (axs[1], 'psi2', 'Damage coefficient $\\psi_2$ ($\\times 10^{-3}$)', 1e3)):
    R = [r for r in B['rows'] if r['param'] == key]; x = np.array([r['value'] for r in R]) * scale
    for lab, k_, col, ls in LINES:
        yv = [100 * r[k_]['share_2025'] for r in R]
        ax.plot(x, yv, color=col, ls=ls, lw=1.6 if k_ == 'reopt' else 1.2, marker='o', ms=2.5, label=lab)
        for xi, yi in ((x[0], yv[0]), (x[-1], yv[-1])):
            ax.annotate(f'{yi:.1f}', (xi, yi), textcoords='offset points', xytext=(-5 if xi == x[0] else 5, 7 if k_ == 'fixed' else 2), ha='right' if xi == x[0] else 'left', va='center', fontsize=6.3, color=col)
    ax.set_xlim(x[0] - 0.2 * (x[-1] - x[0]), x[-1] + 0.14 * (x[-1] - x[0]))
    ax.axhline(100 * base['share_2025'], color=GREY, lw=0.5, ls=':')
    ax.set_xlabel(xl); ax.set_ylabel('Feedback share (%)')
axs[0].legend(frameon=False, fontsize=6.5, loc='lower left'); panel_label(axs[0], '(a)'); panel_label(axs[1], '(b)')
R = [r for r in B['rows'] if r['param'] == 'ecs']; x = [r['value'] for r in R]
for lab, k_, col, ls in LINES:
    yv = [r[k_]['feedback_2025'] for r in R]
    axs[2].plot(x, yv, color=col, ls=ls, lw=1.6 if k_ == 'reopt' else 1.2, marker='o', ms=2.5, label=lab)
    axs[2].annotate(f'\\${yv[-1]:.2f}', (x[-1], yv[-1]), textcoords='offset points', xytext=(5, 0), ha='left', va='center', fontsize=6.3, color=col)
    axs[2].annotate(f'\\${yv[0]:.2f}', (x[0], yv[0]), textcoords='offset points', xytext=(-5, 0), ha='right', va='center', fontsize=6.3, color=col)
axs[2].set_xlim(1.35, 5.75)
axs[2].axhline(base['feedback_2025'], color=GREY, lw=0.5, ls=':')
axs[2].set_xlabel('Equilibrium climate sensitivity ($^\\circ$C)'); axs[2].set_ylabel('Feedback (2019 \\$/tCO$_2$)')
axs[2].legend(frameon=False, fontsize=6.5, loc='upper left'); panel_label(axs[2], '(c)')
# (d) waterfall of the exact split at ECS 5
ax = axs[3]; r5 = [r for r in R if abs(r['value'] - 5.0) < 1e-9][0]; sp = r5['split']; b0 = 100 * base['share_2025']
steps = [('DICE-2023', b0, None), ('fixed\npolicy', 100 * sp['physical'], 1), ('abatement', 100 * sp['abatement'], 1), ('savings', 100 * sp['savings'], 1), ('ECS 5$^\\circ$C,\nre-optimized', 100 * r5['reopt']['share_2025'], None)]
lev = 0.0; y0 = 8.0
for j, (n, v, kind) in enumerate(steps):
    if kind is None:
        ax.bar(j, v - y0, bottom=y0, color='0.65', width=0.55); lev = v
        ax.text(j, v + 0.15, f'{v:.1f}%', ha='center', va='bottom', fontsize=6.6)
    else:
        ax.bar(j, v, bottom=lev, color=ORANGE if v > 0 else BLUE, width=0.55)
        ax.plot([j - 0.6, j - 0.275], [lev, lev], color='0.5', lw=0.5, ls=':')
        ax.text(j, max(lev, lev + v) + 0.15, ('+' if v > 0 else '$-$') + f'{abs(v):.2f}', ha='center', va='bottom', fontsize=6.6)
        lev += v
ax.set_xticks(range(len(steps))); ax.set_xticklabels([s_[0] for s_ in steps], fontsize=6.5)
ax.set_ylim(y0, 16.5); ax.set_ylabel('Feedback share (%)'); panel_label(ax, '(d)')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_v8_policy.pdf')); plt.close(fig)
