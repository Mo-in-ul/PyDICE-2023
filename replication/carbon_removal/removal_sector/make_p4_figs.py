"""Diagnostic figure for the mechanism tests (p4_mech.py): SCC against removal capacity with abatement free and with the
control rate held at its no-removal path. Output: p2_report/assets/fig_p4_mechanism.png and .pdf. Palette as in
make_p3d_figs.py. A working figure for the research record, not a manuscript figure."""
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, 'results', 'p4_mech.json')))
C1, C2 = '#2a78d6', '#eb6834'; INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d3'
YR = 2020 + 5 * np.arange(81)
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'DejaVu Sans'})


def curve(key):
    c = d[key]; s = sorted(c, key=float)
    x = np.array([float(v) for v in s]); y = np.array([c[v]['scc2025'] for v in s])
    fl = np.array([min(c[v]['T']) <= 0.5005 for v in s])
    return x, y, fl


fig, axs = plt.subplots(2, 2, figsize=(9.2, 6.6))
for row, (cal, lab) in enumerate((('pub', 'Published calibration'), ('d2', '2% discount rate'))):
    for col, (v, vlab) in enumerate((('A', 'hard storage caps'), ('D', 'no storage limit'))):
        ax = axs[row, col]
        x, y, fl = curve(f'curve:{v}_{cal}'); xf, yf, flf = curve(f'fixmu:{v}_{cal}')
        ax.plot([0, 1], [y[0], y[-1]], ':', color=INK2, lw=1); ax.plot([0, 1], [yf[0], yf[-1]], ':', color=INK2, lw=1)
        ax.plot(x, y, '-o', color=C1, lw=2, ms=5, label='abatement re-optimised')
        ax.plot(xf, yf, '-s', color=C2, lw=2, ms=5, label='control rate held at no-removal path')
        if fl.any():
            ax.plot(x[fl], y[fl], 'o', ms=9, mfc='none', mec=INK, mew=1.2, label='run reaches the 0.5 °C lower bound')
        if flf.any():
            ax.plot(xf[flf], yf[flf], 's', ms=9, mfc='none', mec=INK, mew=1.2)
        g = 0.5 * (y[0] + y[-1]) - y[x == 0.5][0]; gf = 0.5 * (yf[0] + yf[-1]) - yf[xf == 0.5][0]
        ax.set_title(f'{lab}, {vlab}', fontsize=9.5, loc='left', color=INK)
        ax.text(0.03, 0.06, f'Jensen gap at half capacity:\nre-optimised {g:.2f}, held {gf:.2f} US\\$/t', transform=ax.transAxes, fontsize=8, color=INK2, va='bottom')
        ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xlabel('Removal capacity (share of central)')
        if col == 0:
            ax.set_ylabel('SCC in 2025 (US\\$ per tCO$_2$)')
axs[0, 1].legend(frameon=False, fontsize=8, loc='upper right')

fig.tight_layout()
out = os.path.join(HERE, 'p2_report', 'assets'); os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, 'fig_p4_mechanism.png'), dpi=200); fig.savefig(os.path.join(out, 'fig_p4_mechanism.pdf'))
print('saved')
