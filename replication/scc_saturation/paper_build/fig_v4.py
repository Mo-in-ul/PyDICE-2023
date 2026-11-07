"""Main-text figures for the v4 manuscript (ERE). House style (house_style.py), one y-axis per panel.
  fig_v4_arch.pdf      PyDICE-2023 framework: forward models, solvers, analysis outputs
  fig_v4_decomp.pdf    the 2025 SCC split into direct part and saturation feedback, five scenarios
  fig_v4_sign.pdf      (a) feedback share by pulse year; (b) net emissions and net flux into the sinks, optimum
  fig_v4_policy.pdf    physical vs policy response (Experiment B): ECS and damage coefficient
  fig_v4_mc.pdf        (a) distribution of the 2025 share over 500 draws; (b) Sobol indices
Data: results/check3_ift.json, exp_b_physical_policy.json, mc2*, sobol.json; GAMS paths via common.py."""
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
import fig_mc2 as FM

FIG = os.path.join(HERE, 'figs')
GREY = '0.45'

# ---------------- Figure 1: architecture ----------------
fig, ax = plt.subplots(figsize=(DOUBLE, 96 * MM)); ax.set_xlim(0, 100); ax.set_ylim(0, 54); ax.axis('off')


def box(x, y, w, h, title, lines, fc='white', ec='0.25'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h - 2.0, title, ha='center', va='top', fontsize=8.4, weight='bold')
    ax.text(x + w / 2, y + h - 6.6, '\n'.join(lines), ha='center', va='top', fontsize=7.2, linespacing=1.35)


def arrow(x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', color='0.25', lw=0.7))


def note(x, y, txt, **kw):
    ax.text(x, y, txt, fontsize=6.8, color='0.25', **kw)


box(0.5, 14, 15.5, 34, 'Inputs', ['DICE-2023', 'parameters', '', 'GAMS run', 'settings', '', 'Policy paths', '$\\mu(t)$, $S(t)$'])
box(22, 30, 23, 22, 'Forward model', ['(Numba)', 'economy, emissions,', 'DFAIR carbon cycle,', '$\\alpha(t)$ a decision', 'variable, welfare'], fc='#eef3f9')
box(22, 2, 23, 22, 'Forward model', ['(JAX)', 'same equations,', 'automatic', 'differentiation,', 'agrees to $10^{-11}$'], fc='#eef3f9')
box(53, 30, 17, 22, 'SLSQP', ['243 variables,', '81 IRF', 'equalities,', 'fast optimum'])
box(53, 2, 17, 22, 'IPOPT', ['exact gradients,', 'Lagrange', 'multipliers,', 'multiplier SCC'])
box(78, 30, 21.5, 22, 'Verification', ['GAMS paths and', 'SCCs, ten runs;', 'regression tests'], fc='#fbefec')
box(78, 2, 21.5, 22, 'Analysis', ['exact SCC split', '(direct + feedback);', 'FaIR v2 swap;', 'Monte Carlo,', 'policy runs'], fc='#fbefec')
arrow(16.4, 41, 21.6, 41); arrow(16.4, 20, 21.6, 13)
arrow(45.4, 41, 52.6, 41); note(49, 42.2, 'objective,\nconstraints', ha='center', va='bottom')
arrow(45.4, 13, 52.6, 13); note(49, 14.2, 'derivatives', ha='center', va='bottom')
arrow(70.4, 41, 77.6, 41); arrow(70.4, 13, 77.6, 13)
arrow(61.5, 29.6, 61.5, 24.4); note(62.6, 27, 'optimum', va='center')
arrow(33.5, 29.6, 33.5, 24.4); note(34.6, 27, 'cross-check', va='center')
fig.savefig(os.path.join(FIG, 'fig_v4_arch.pdf'), bbox_inches='tight'); plt.close(fig)

# ---------------- Figure 2: the 2025 SCC split ----------------
ift = load_json('check3_ift.json')
NAMES = [(9, 'Cost-benefit\noptimal'), (4, '4%\ndiscounting'), (5, '5%\ndiscounting'), (8, 'Paris'), (10, 'Baseline')]
fig, ax = plt.subplots(figsize=(DOUBLE * 0.62, 62 * MM))
for j, (sc, lab) in enumerate(NAMES):
    o = ift[str(sc)]; d = o['direct_all'][1]; f = o['feedback_all'][1]
    ax.bar(j, d, width=0.6, color='0.75', edgecolor='white', linewidth=0.8)
    ax.bar(j, f, bottom=d, width=0.6, color=BLUE, edgecolor='white', linewidth=0.8)
    ax.text(j, d + f + 1.5, f'\\${d + f:.1f}', ha='center', va='bottom', fontsize=6.8)
    ax.text(j, d + f / 2, f'{100 * f / (d + f):.1f}%', ha='center', va='center', fontsize=6.4, color='white')
ax.set_xticks(range(len(NAMES))); ax.set_xticklabels([l for _, l in NAMES], fontsize=6.8)
ax.set_ylabel('SCC in 2025 (2019 \\$/tCO$_2$)')
ax.bar(0, 0, color='0.75', label='direct part ($\\alpha$ held fixed)'); ax.bar(0, 0, color=BLUE, label='saturation feedback')
ax.legend(frameon=False, loc='upper left'); ax.set_ylim(0, max(ift[str(s)]['total_all'][1] for s, _ in NAMES) * 1.22)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_v4_decomp.pdf')); plt.close(fig)

# ---------------- Figure 3: sign change ----------------
fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 64 * MM))
ax = axs[0]
for sc, col, ls, lab in ((9, BLUE, '-', 'Cost-benefit optimal'), (4, '0.5', '--', '4% discounting'), (5, '0.7', '--', '5% discounting'),
                         (8, ORANGE, '-', 'Paris'), (10, '0.3', '-', 'Baseline')):
    o = ift[str(sc)]; yrs = np.array(o['years_all']); sh = 100 * np.array(o['share_all']); k = (yrs >= 2025) & (yrs <= 2200)
    ax.plot(yrs[k], sh[k], color=col, ls=ls, lw=1.4 if sc == 9 else 1.0, label=lab)
ax.axhline(0, color=GREY, lw=0.6); ax.set_xlabel('Year of emissions pulse'); ax.set_ylabel('Feedback share of the SCC (%)')
ax.legend(frameon=False, fontsize=6.5, loc='upper right'); panel_label(ax, '(a)')
# (b) net emissions and net flux into the sinks along the optimum
import jax.numpy as jnp, dice_jax as dj
p, b = gams_session(9); MIU, S, g = gams_controls(9, p); alpha, _ = dm.solve_alpha(p, MIU, S)
tr = dj.trajectory(dj.pack_params(p), jnp.asarray(MIU), jnp.asarray(S), jnp.asarray(alpha))
E = np.asarray(tr['ECO2'][1:]); MAT = np.asarray(tr['MAT'][1:]); yrs = 2020 + 5 * np.arange(81)
Ec = E / 3.667; phi = Ec[1:] - np.diff(MAT) / 5.0; yphi = yrs[1:]
ax = axs[1]; k = yrs <= 2250; kp = yphi <= 2250
ax.plot(yrs[k], Ec[k], color='0.3', lw=1.2, label='net emissions')
ax.plot(yphi[kp], phi[kp], color=BLUE, lw=1.4, label='net flux into ocean and land, $\\Phi$')
ax.axhline(0, color=GREY, lw=0.6)
for yv, txt in ((int(yrs[E < 0][0]), 'net-negative\nemissions'), (int(yphi[phi < 0][0]), '$\\Phi < 0$')):
    ax.axvline(yv, color=GREY, lw=0.6, ls=':'); ax.text(yv + 2, ax.get_ylim()[1] * 0.95 if txt.startswith('net') else ax.get_ylim()[1] * 0.55, f'{txt}\n{yv}', fontsize=6.3, color='0.25', va='top')
ax.set_xlabel('Year'); ax.set_ylabel('GtC per year'); ax.legend(frameon=False, fontsize=6.5, loc='lower left'); panel_label(ax, '(b)')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_v4_sign.pdf')); plt.close(fig)
json.dump({'first_net_negative': int(yrs[E < 0][0]), 'first_phi_negative': int(yphi[phi < 0][0])}, open(os.path.join(HERE, 'results', 'fig_v4_sign.json'), 'w'))

# ---------------- Figure 4: physical vs policy response ----------------
B = load_json('exp_b_physical_policy.json'); base = B['baseline']
fig, A4 = plt.subplots(2, 2, figsize=(DOUBLE, 112 * MM)); axs = A4.ravel()
LINES = (('policy fixed at baseline optimum', 'fixed', '0.35', '--'), ('fully re-optimized', 'reopt', BLUE, '-'),
         ('abatement only re-optimized', 'mu_only', ORANGE, 'none'))
for ax, key, xl, scale in ((axs[0], 'ecs', 'Equilibrium climate sensitivity ($^\\circ$C)', 1.0), (axs[1], 'psi2', 'Damage coefficient $\\psi_2$ ($\\times 10^{-3}$)', 1e3)):
    R = [r for r in B['rows'] if r['param'] == key]; x = np.array([r['value'] for r in R]) * scale
    for lab, k_, col, ls in LINES:
        ax.plot(x, [100 * r[k_]['share_2025'] for r in R], color=col, ls=ls, lw=1.4, marker='x' if k_ == 'mu_only' else 'o', ms=3.5 if k_ == 'mu_only' else 2.5, mew=0.8, label=lab)
    ax.axhline(100 * base['share_2025'], color=GREY, lw=0.5, ls=':')
    ax.set_xlabel(xl); ax.set_ylabel('Feedback share, 2025 SCC (%)')
axs[0].legend(frameon=False, fontsize=6.5, loc='lower left'); panel_label(axs[0], '(a)'); panel_label(axs[1], '(b)')
R = [r for r in B['rows'] if r['param'] == 'ecs']; x = [r['value'] for r in R]
for lab, k_, col, ls in LINES[:2]:
    axs[2].plot(x, [r[k_]['feedback_2025'] for r in R], color=col, ls=ls, lw=1.4, marker='o', ms=2.5, label=lab)
axs[2].axhline(base['feedback_2025'], color=GREY, lw=0.5, ls=':')
axs[2].set_xlabel('Equilibrium climate sensitivity ($^\\circ$C)'); axs[2].set_ylabel('Feedback, 2025 (2019 \\$/tCO$_2$)')
axs[2].legend(frameon=False, fontsize=6.5, loc='upper left'); panel_label(axs[2], '(c)')
axs[3].plot(x, [r['fixed']['cumE_2100'] for r in R], color='0.35', ls='--', lw=1.4, marker='o', ms=2.5, label='policy fixed')
axs[3].plot(x, [r['reopt']['cumE_2100'] for r in R], color=BLUE, lw=1.4, marker='o', ms=2.5, label='re-optimized')
axs[3].set_xlabel('Equilibrium climate sensitivity ($^\\circ$C)'); axs[3].set_ylabel('Emissions 2020--2100 (GtCO$_2$)')
axs[3].legend(frameon=False, fontsize=6.5, loc='lower left'); panel_label(axs[3], '(d)')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_v4_policy.pdf')); plt.close(fig)

# ---------------- Figure 5: uncertainty ----------------
ok = FM.load(); sh = np.array([100 * o['share_2025'] for o in ok])
S = load_json('sobol.json')['share_2025']
fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 60 * MM), gridspec_kw={'width_ratios': [1, 1.15]})
ax = axs[0]; ax.hist(sh, bins=30, color='0.75', edgecolor='white', linewidth=0.5)
for q, ls in ((5, ':'), (50, '-'), (95, ':')):
    ax.axvline(np.percentile(sh, q), color='0.3', lw=0.8, ls=ls)
ax.axvline(13.04, color=BLUE, lw=1.4); ax.text(13.6, ax.get_ylim()[1] * 0.99, 'DICE-2023\ncalibration', color=BLUE, fontsize=6.4, va='top')
ax.set_xlabel('Feedback share of the 2025 SCC (%)'); ax.set_ylabel('Number of draws'); panel_label(ax, '(a)')
ax = axs[1]; keys = sorted(S['S1'], key=lambda k: -S['ST'][k]); y = np.arange(len(keys))[::-1]
ax.barh(y + 0.18, [S['S1'][k] for k in keys], height=0.34, color=BLUE, label='first order $S_1$')
ax.barh(y - 0.18, [S['ST'][k] for k in keys], height=0.34, color='0.75', label='total $S_T$')
ax.set_yticks(y); ax.set_yticklabels([FM.LAB[k] for k in keys], fontsize=6.6); ax.set_xlabel('Sobol index, 2025 share')
ax.legend(frameon=False, loc='lower right'); ax.grid(False); panel_label(ax, '(b)')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_v4_mc.pdf')); plt.close(fig)
print('v4 figures written; draws', len(sh), 'median', np.median(sh), 'p5', np.percentile(sh, 5), 'p95', np.percentile(sh, 95))
