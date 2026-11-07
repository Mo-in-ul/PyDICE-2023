"""Main-text figures for the v6 manuscript (ERE). House style (house_style.py), one y-axis per panel.
  fig_v6_arch.pdf      PyDICE-2023 framework: forward models, solvers, analysis outputs
  fig_v6_decomp.pdf    the 2025 SCC split into direct part and saturation feedback, five scenarios
  fig_v6_sign.pdf      (a) feedback share by pulse year; (b) net emissions and net flux into the sinks, optimum
  fig_v6_policy.pdf    physical vs policy response (Experiment B): ECS and damage coefficient
  fig_v6_mc.pdf        (a) distribution of the 2025 share over 500 draws; (b) Sobol indices
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

# ---------------- Figure (Appendix A): architecture ----------------
fig, ax = plt.subplots(figsize=(DOUBLE, 74 * MM)); ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis('off')


def box(x, y, w, h, title, lines, fc='white', ec='0.3'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.0', fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h - 1.6, title, ha='center', va='top', fontsize=8.2, weight='bold')
    ax.text(x + w / 2, y + h - 6.6, '\n'.join(lines), ha='center', va='top', fontsize=7.0, linespacing=1.3)


def arrow(x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', color='0.3', lw=0.7))


def note(x, y, txt, **kw):
    kw.setdefault('fontsize', 6.6); kw.setdefault('color', '0.3')
    ax.text(x, y, txt, **kw)


note(0.5, 47.6, 'Optimization', style='italic', fontsize=7.2)
note(0.5, 21.6, 'Derivatives and SCC', style='italic', fontsize=7.2)
box(1, 28, 16, 16, 'DICE-2023', ['parameters,', 'GAMS run', 'settings'])
box(25, 28, 19, 16, 'Numba model', ['forward recursion,', '0.09 ms per run'], fc='#eef3f9')
box(55, 28, 16, 16, 'SLSQP', ['243 variables,', '81 IRF', 'equalities'])
box(25, 2, 19, 16, 'JAX model', ['same equations,', 'exact derivatives'], fc='#eef3f9')
box(55, 2, 16, 16, 'IPOPT', ['multipliers,', 'multiplier SCC'])
box(80, 28, 19.5, 16, 'Verification', ['ten GAMS runs,', 'regression tests'], fc='#fbefec')
box(80, 2, 19.5, 16, 'SCC split', ['direct + feedback,', 'experiments,', 'Monte Carlo'], fc='#fbefec')
arrow(17.4, 36, 24.6, 36); arrow(17.4, 30, 24.6, 12)
arrow(44.4, 36, 54.6, 36); note(49.5, 37.2, 'objective', ha='center', va='bottom')
arrow(44.4, 10, 54.6, 10); note(49.5, 11.2, 'gradients', ha='center', va='bottom')
arrow(71.4, 36, 79.6, 36); arrow(71.4, 10, 79.6, 10); arrow(71.4, 15, 79.6, 30)
arrow(34.5, 27.6, 34.5, 18.4); note(35.6, 23, 'agree to $10^{-11}$', va='center')
arrow(63, 27.6, 63, 18.4); note(64.1, 23, 'warm start', va='center')
fig.savefig(os.path.join(FIG, 'fig_v6_arch.pdf'), bbox_inches='tight'); plt.close(fig)

# ---------------- Figure 2: the 2025 SCC split ----------------
ift = load_json('check3_ift.json')
NAMES = [(9, 'Cost-benefit\noptimal'), (4, '4%\ndiscounting'), (5, '5%\ndiscounting'), (8, 'Paris'), (10, 'Baseline')]
fig, ax = plt.subplots(figsize=(DOUBLE * 0.62, 62 * MM))
for j, (sc, lab) in enumerate(NAMES):
    o = ift[str(sc)]; d = o['direct_all'][1]; f = o['feedback_all'][1]
    ax.bar(j, d, width=0.6, color='0.75', edgecolor='white', linewidth=0.8)
    ax.bar(j, f, bottom=d, width=0.6, color=BLUE, edgecolor='white', linewidth=0.8)
    ax.text(j, d + f + 1.5, f'\\${d + f:.1f}', ha='center', va='bottom', fontsize=6.8)
    if f >= 6:
        ax.text(j, d + f / 2, f'{100 * f / (d + f):.1f}%', ha='center', va='center', fontsize=6.4, color='white')
    else:
        ax.text(j + 0.33, d + f / 2, f'{100 * f / (d + f):.1f}%', ha='left', va='center', fontsize=6.4, color=BLUE)
ax.set_xticks(range(len(NAMES))); ax.set_xticklabels([l for _, l in NAMES], fontsize=6.8)
ax.set_ylabel('SCC in 2025 (2019 \\$/tCO$_2$)')
ax.bar(0, 0, color='0.75', label='direct part ($\\alpha$ held fixed)'); ax.bar(0, 0, color=BLUE, label=f"saturation feedback (\\${ift['9']['feedback_all'][1]:.2f} in the optimum)")
ax.legend(frameon=False, loc='upper left'); ax.set_ylim(0, max(ift[str(s)]['total_all'][1] for s, _ in NAMES) * 1.32); ax.set_xlim(-0.55, len(NAMES) - 0.45)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_v6_decomp.pdf')); plt.close(fig)

# ---------------- Figure 3: sign change, and when the feedback acts ----------------
import fair_benchmark as FB
fig, A3 = plt.subplots(2, 2, figsize=(DOUBLE, 118 * MM)); axs = A3.ravel()
ax = axs[0]
for sc, col, ls, lab in ((9, BLUE, '-', 'Cost-benefit optimal'), (4, '0.5', '--', '4% discounting'), (5, '0.7', '--', '5% discounting'),
                         (8, ORANGE, '-', 'Paris'), (10, '0.3', '-', 'Baseline')):
    o = ift[str(sc)]; yrs = np.array(o['years_all']); sh = 100 * np.array(o['share_all']); k = (yrs >= 2025) & (yrs <= 2200)
    ax.plot(yrs[k], sh[k], color=col, ls=ls, lw=1.4 if sc == 9 else 1.0, label=lab)
o9 = ift['9']; y9 = np.array(o9['years_all']); s9 = np.array(o9['share_all']); yc = int(y9[(y9 >= 2025) & (s9 < 0)][0])
ax.axvline(yc, ymin=(-3.5 + 15) / 41.5, color=GREY, lw=0.6, ls=':'); ax.text(yc + 2, 25.5, f'optimum turns\nnegative: {yc}', fontsize=6.3, color='0.25', va='top', ha='left')
ax.axhline(0, color=GREY, lw=0.6); ax.set_xlabel('Year of emissions pulse'); ax.set_ylabel('Feedback share of the SCC (%)')
ax.set_ylim(-15, 26.5); ax.legend(frameon=False, fontsize=6.0, loc='lower left', handlelength=1.6, labelspacing=0.25, ncol=2, columnspacing=0.8); panel_label(ax, '(a)')
import jax.numpy as jnp, dice_jax as dj
p, b = gams_session(9); MIU, S, g = gams_controls(9, p); alpha, _ = dm.solve_alpha(p, MIU, S)
tr = dj.trajectory(dj.pack_params(p), jnp.asarray(MIU), jnp.asarray(S), jnp.asarray(alpha))
E = np.asarray(tr['ECO2'][1:]); MAT = np.asarray(tr['MAT'][1:]); yrs = 2020 + 5 * np.arange(81)
Ec = E / 3.667; phi = Ec[1:] - np.diff(MAT) / 5.0; yphi = yrs[1:]
ax = axs[1]; k = yrs <= 2250; kp = yphi <= 2250
ax.plot(yrs[k], Ec[k], color='0.3', lw=1.2, label='net emissions')
ax.plot(yphi[kp], phi[kp], color=BLUE, lw=1.4, label='net flux into the sinks, $\\Phi$')
ax.axhline(0, color=GREY, lw=0.6)
ax.set_ylim(-13.6, 16)
for yv, txt, yy, side in ((int(yrs[E < 0][0]), 'net-negative\nemissions', 15.6, 'right'), (int(yphi[phi < 0][0]), '$\\Phi < 0$', 4.6, 'left')):
    ax.axvline(yv, color=GREY, lw=0.6, ls=':'); ax.text(yv - 2 if side == 'right' else yv + 2, yy, f'{txt}\n{yv}', fontsize=6.3, color='0.25', va='top', ha=side)
ax.annotate('bound on $\\mu$ falls\nfrom 1.1 to 1.05', xy=(2201, Ec[(2200 - 2020) // 5]), xytext=(2146, -13.2), fontsize=6.2, color='0.3', ha='left', va='bottom',
            arrowprops=dict(arrowstyle='-', color='0.5', lw=0.5, shrinkA=0, shrinkB=1))
ax.set_xlabel('Year'); ax.set_ylabel('GtC per year'); ax.legend(frameon=False, fontsize=6.2, loc='upper right', handlelength=1.8); panel_label(ax, '(b)')
json.dump({'first_net_negative': int(yrs[E < 0][0]), 'first_phi_negative': int(yphi[phi < 0][0])}, open(os.path.join(HERE, 'results', 'fig_v4_sign.json'), 'w'))
# (c), (d): contribution of each damage year to the feedback for pulses in 2025 and 2120
p_, q_, MIUf, Sf, af, wf, lamCf, Ef, Tf, Fnf = FB.setup()
timing = {}
for ax, ypulse, lab in ((axs[2], 2025, '(c)'), (axs[3], 2120, '(d)')):
    i = (ypulse - 2020) // 5 + 1
    dT_dir, dT_fb = FB.dice_split(q_, MIUf, Sf, af, wf, lamCf, i)
    cf = -1000 * wf[1:] * dT_fb / lamCf[i]; cd = -1000 * wf[1:] * dT_dir / lamCf[i]
    kk = yrs <= 2300
    ax.bar(yrs[kk], cf[kk], width=4.0, color=np.where(cf[kk] >= 0, BLUE, ORANGE), lw=0)
    ax.axhline(0, color=GREY, lw=0.6); ax.axvline(ypulse, color=GREY, lw=0.6, ls=':')
    tot, scc_ = float(cf.sum()), float(cf.sum() + cd.sum())
    pos, neg = float(cf[cf > 0].sum()), float(cf[cf < 0].sum())
    by2100 = float(cf[yrs <= 2100].sum())
    timing[str(ypulse)] = {'feedback': tot, 'scc': scc_, 'positive': pos, 'negative': neg, 'by2100': by2100}
    money = lambda v: ('$-$' if v < 0 else '') + f'\\${abs(v):.2f}'
    txt = (f'pulse in {ypulse}\nfeedback {money(tot)} of SCC {money(scc_)}\n{100 * by2100 / tot:.0f}% from damages before 2100' if ypulse == 2025 else
           f'pulse in {ypulse}\nfeedback {money(tot)}\nof SCC {money(scc_)}\npositive +\\${pos:.2f}\nnegative {money(neg)}')
    ax.text(0.98 if ypulse == 2025 else 0.03, 0.96 if ypulse == 2025 else 0.04, txt, transform=ax.transAxes, ha='right' if ypulse == 2025 else 'left', va='top' if ypulse == 2025 else 'bottom', fontsize=6.4, color='0.2')
    ax.set_xlabel('Year in which the damage occurs'); ax.set_ylabel('Feedback contribution (\\$/tCO$_2$)'); panel_label(ax, lab)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_v6_sign.pdf')); plt.close(fig)
json.dump(timing, open(os.path.join(HERE, 'results', 'fig_v6_timing.json'), 'w'), indent=1)

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
steps = [('DICE-2023', b0, None), ('physical', 100 * sp['physical'], 1), ('abatement', 100 * sp['abatement'], 1), ('savings', 100 * sp['savings'], 1), ('ECS 5$^\\circ$C,\nre-optimized', 100 * r5['reopt']['share_2025'], None)]
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
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_v6_policy.pdf')); plt.close(fig)

# ---------------- Figure 5: uncertainty ----------------
ok = FM.load(); sh = np.array([100 * o['share_2025'] for o in ok])
S = load_json('sobol.json')['share_2025']
fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 66 * MM), gridspec_kw={'width_ratios': [1.1, 1]})
ax = axs[0]
scc = np.array([o['scc_2025'] for o in ok]); fbk = np.array([o['feedback_2025'] for o in ok]); ecs_ = np.array([o['params']['ecs'] for o in ok])
scp = ax.scatter(scc, fbk, c=ecs_, s=5, cmap='viridis', lw=0)
xs = np.geomspace(1.5, 700, 50)
for shv in (0.05, 0.10, 0.20):
    ax.plot(xs, shv * xs, color='0.6', lw=0.6, ls=':')
    ax.text(xs[-1] * 1.12, shv * xs[-1], f'{int(100 * shv)}%', fontsize=6.2, color='0.4', va='center', ha='left')
ax.plot(59.07, 7.70, marker='*', color=ORANGE, ms=9, mec='white', mew=0.5, ls='none', label='DICE-2023')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1.5, 2200); ax.set_ylim(0.05, 250)
ax.set_xlabel('SCC 2025 (2019 \\$/tCO$_2$)'); ax.set_ylabel('Feedback 2025 (2019 \\$/tCO$_2$)')
ax.legend(frameon=False, loc='upper left', fontsize=6.6); panel_label(ax, '(a)')
cax = ax.inset_axes([0.62, 0.1, 0.3, 0.035]); cb = fig.colorbar(scp, cax=cax, orientation='horizontal')
cb.ax.set_title('ECS ($^\\circ$C)', fontsize=6.2, pad=2); cb.ax.tick_params(labelsize=5.8, length=2)
ax = axs[1]; keys = sorted(S['S1'], key=lambda k: -S['ST'][k]); y = np.arange(len(keys))[::-1]
Bt = load_json('sobol_bootstrap.json')
def eb(key, k):
    v = S[key][k]; return [[max(0.0, v - Bt[key][k]['p5'])], [max(0.0, Bt[key][k]['p95'] - v)]]
ax.barh(y + 0.18, [S['S1'][k] for k in keys], height=0.34, color=BLUE, label='first order $S_1$')
ax.barh(y - 0.18, [S['ST'][k] for k in keys], height=0.34, color='0.75', label='total $S_T$')
for yy, k in zip(y, keys):
    ax.errorbar(S['S1'][k], yy + 0.18, xerr=eb('S1', k), fmt='none', ecolor='0.15', elinewidth=0.6, capsize=1.5)
    ax.errorbar(S['ST'][k], yy - 0.18, xerr=eb('ST', k), fmt='none', ecolor='0.15', elinewidth=0.6, capsize=1.5)
ax.set_yticks(y); ax.set_yticklabels([FM.LAB[k] for k in keys], fontsize=7.2); ax.set_xlabel('Sobol index, 2025 share')
ax.legend(frameon=False, loc='lower right'); ax.grid(False); panel_label(ax, '(b)')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_v6_mc.pdf')); plt.close(fig)
print('v6 figures written; draws', len(sh), 'median', np.median(sh), 'p5', np.percentile(sh, 5), 'p95', np.percentile(sh, 95),
      'share range of draws', sh.min(), sh.max(), 'scc range', scc.min(), scc.max())

# ---------------- Figure: carbon-cycle substitution as a dot plot ----------------
Aud = load_json('fair_audit.json')['sets']; Cc = load_json('exp_c_fair_range.json')['sets']
cm = sorted([n for n in Aud if n not in ('FaIRv2.0.0 default', 'FaIR v2.2 AR6 default')], key=lambda n: Cc[n]['share_2025'])
rows = cm + ['FaIRv2.0.0 default', 'FaIR v2.2 AR6 default']
fig, ax = plt.subplots(figsize=(DOUBLE * 0.62, 80 * MM))
for j, n in enumerate(rows):
    a, b_ = 100 * Cc[n]['share_2025'], 100 * Aud[n]['share_2025_reanchored']
    ax.plot([a, b_], [j, j], color='0.75', lw=1.0, zorder=1)
    ax.plot(a, j, 'o', color=BLUE, ms=4.5, zorder=2, label='as calibrated' if j == 0 else None)
    ax.plot(b_, j, 'o', mfc='white', mec=BLUE, ms=4.5, zorder=2, label='$r_0$ re-anchored to the DICE 2020 state' if j == 0 else None)
ax.axvline(13.04, color=ORANGE, lw=1.2, zorder=0)
ax.text(13.2, -1.05, 'DICE-2023\n(DFAIR) 13.0%', color=ORANGE, fontsize=6.4, va='bottom')
ax.axhline(len(cm) - 0.5, color='0.85', lw=0.6)
lab_ = {'FaIRv2.0.0 default': 'FaIR v2.0.0 default', 'FaIR v2.2 AR6 default': 'FaIR v2.2, AR6', 'CanESM5': 'CanESM5$^{*}$'}
ax.set_yticks(range(len(rows))); ax.set_yticklabels([lab_.get(r_, r_) for r_ in rows], fontsize=6.8)
ax.set_ylim(-1.3, len(rows) - 0.4); ax.set_xlim(5.5, 19.5)
ax.set_xlabel('Feedback share of the 2025 SCC (%)'); ax.legend(frameon=False, fontsize=6.3, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_v6_fair.pdf')); plt.close(fig)

# ---------------- Appendix: SCC distribution at the three computed pulse dates (replaces the fan chart) --------
fig, ax = plt.subplots(figsize=(DOUBLE * 0.5, 58 * MM))
yrs3 = (2025, 2050, 2100); G = load_json('scc_gams_controls.json')['9']
for j, y in enumerate(yrs3):
    v = np.array([o[f'scc_{y}'] for o in ok])
    q5, q25, q50, q75, q95 = np.percentile(v, [5, 25, 50, 75, 95])
    ax.plot([j, j], [q5, q95], color='0.45', lw=0.9)
    ax.add_patch(plt.Rectangle((j - 0.16, q25), 0.32, q75 - q25, fc='#c9d9ec', ec=BLUE, lw=0.8))
    ax.plot([j - 0.16, j + 0.16], [q50, q50], color=BLUE, lw=1.6)
    for qq in (q5, q95):
        ax.plot([j - 0.07, j + 0.07], [qq, qq], color='0.45', lw=0.9)
    ax.plot(j + 0.27, G['gams'][G['years'].index(y)], marker='s', ms=3.5, color=ORANGE, ls='none', label='DICE-2023 optimum' if j == 0 else None)
ax.set_yscale('log'); ax.set_xticks(range(3)); ax.set_xticklabels([str(y) for y in yrs3]); ax.set_xlim(-0.5, 2.6)
ax.set_xlabel('Year of pulse'); ax.set_ylabel('SCC (2019 \\$/tCO$_2$)')
ax.legend(frameon=False, loc='upper left', fontsize=6.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_v6_scc_dist.pdf')); plt.close(fig)

# ---------------- Appendix: the 1.5 C pathway, threshold marked ----------------
runs = [o for o in load_json('onepointfive.json')['runs'] if 'error' not in o and o['status'] in (0, 1)]
fig, ax2 = plt.subplots(1, 2, figsize=(DOUBLE, 62 * MM))
for disp, col, lab in ((True, BLUE, 'capital can be scrapped (GAMS)'), (False, ORANGE, 'equality accumulation')):
    rs = sorted([o for o in runs if o['disposal'] == disp], key=lambda o: o['cap'])
    ax2[0].plot([o['cap'] for o in rs], [o['U'] for o in rs], 'o-', color=col, label=lab, ms=2.5, lw=1.2)
    ax2[1].plot([o['cap'] for o in rs], [o['cpc2025'] for o in rs], 'o-', color=col, ms=2.5, lw=1.2)
for a in ax2:
    a.axvspan(a.get_xlim()[0], 1.632, color='0.92', zorder=0)
    a.axvline(1.632, color='0.3', ls='--', lw=1.0)
    a.set_xlabel('Temperature cap ($^\\circ$C)')
ax2[0].text(1.645, ax2[0].get_ylim()[0] + 0.06 * np.diff(ax2[0].get_ylim())[0], '1.632$^\\circ$C: lowest peak\nwith equality accumulation', fontsize=6.3, color='0.2')
ax2[0].text(2.58, ax2[0].get_ylim()[0] + 0.24 * np.diff(ax2[0].get_ylim())[0], 'shaded: infeasible without\ncapital scrapping', fontsize=6.3, color='0.35', va='bottom', ha='right')
panel_label(ax2[0], '(a)'); panel_label(ax2[1], '(b)')
ax2[0].set_ylabel('Welfare (DICE utility units)'); ax2[0].legend(frameon=False, loc='center right', fontsize=6.5)
ax2[1].set_ylabel('Consumption per person, 2025\n(thousand 2019 \\$)')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_v6_1p5c.pdf')); plt.close(fig)

# ---------------- Mechanism schematic (before Section 4) ----------------
fig, ax = plt.subplots(figsize=(DOUBLE, 64 * MM)); ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis('off')


def sbox(x, y, w, h, txt, fc='white', ec='0.3', bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.35,rounding_size=1.0', fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=7.4, weight='bold' if bold else 'normal', linespacing=1.3)


sbox(0.5, 21, 14, 10, 'Emissions\npulse in $t$', bold=True)
sbox(23, 35, 22, 11, 'More CO$_2$ in the\nreservoirs, $\\alpha$ fixed', fc='#eeeeee')
sbox(53, 35, 22, 11, 'Warming and\ndamages, $s \\geq t$', fc='#eeeeee')
sbox(23, 5, 22, 13, 'Carbon and temperature\nstates change;\n$\\alpha$ re-solved from IRF', fc='#e3ecf6')
sbox(53, 5, 22, 13, 'Persistence of the\nwhole stock changes;\nwarming and damages', fc='#e3ecf6')
sbox(81.5, 19, 18, 14, 'SCC $=$\ndirect $+$\nfeedback', bold=True)
arrow(14.9, 28, 22.6, 40); arrow(14.9, 24, 22.6, 12)
arrow(45.4, 40.5, 52.6, 40.5); arrow(45.4, 11.5, 52.6, 11.5)
arrow(75.4, 40.5, 81.1, 30); arrow(75.4, 11.5, 81.1, 22)
arrow(34, 34.6, 34, 18.4); ax.text(32.8, 26.5, 'stocks and $T_{\\mathrm{AT}}$\nenter the\nIRF target', fontsize=6.4, color='0.25', va='center', ha='right')
ax.annotate('', xy=(41, 18.6), xytext=(57, 18.6), arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.8, connectionstyle='arc3,rad=0.45'))
ax.text(49, 24.6, 'in later periods', fontsize=6.4, color=BLUE, ha='center')
ax.text(49, 48.5, 'direct term: $\\partial W/\\partial \\varepsilon_t$', fontsize=7.0, color='0.25', ha='center')
ax.text(49, 0.6, 'feedback term: $(\\partial W/\\partial \\alpha)\\,(d\\alpha/d\\varepsilon_t)$', fontsize=7.0, color=BLUE, ha='center')
fig.savefig(os.path.join(FIG, 'fig_v6_schematic.pdf'), bbox_inches='tight'); plt.close(fig)
print('appendix figures and schematic written')

# ---------------- Appendix D: the two steps of the feedback (v6 copy of mechanism.figure) ----------------
o9m = load_json('mechanism.json')['9']; yrs_m = np.array(o9m['years']); km = yrs_m <= 2250
fig, axm = plt.subplots(1, 3, figsize=(DOUBLE, 60 * MM))
for y0, ls in (('2025', '-'), ('2150', '--')):
    d_ = o9m['pulses'][y0]
    axm[0].plot(yrs_m[km], np.array(d_['dalpha'])[km], color='0.2', ls=ls, label=f'pulse in {y0}')
    for lab_m, col in (('fixed', ORANGE), ('resp', BLUE)):
        axm[1].plot(yrs_m[km], np.array(d_[lab_m]['MAT'])[km], color=col, ls=ls)
        axm[2].plot(yrs_m[km], 1000 * np.array(d_[lab_m]['TATM'])[km], color=col, ls=ls)
axm[0].set_ylabel('Change in $\\alpha$'); axm[0].legend(frameon=False, loc='lower left', fontsize=6.4)
axm[1].set_ylabel('Change in $M_{\\mathrm{AT}}$ (GtC)'); axm[2].set_ylabel('Change in $T_{\\mathrm{AT}}$ ($10^{-3}$ $^\\circ$C)')
axm[1].plot([], [], color=ORANGE, label='$\\alpha$ fixed'); axm[1].plot([], [], color=BLUE, label='$\\alpha$ responds')
axm[1].set_ylim(top=axm[1].get_ylim()[1] * 1.28); axm[1].legend(frameon=False, loc='upper left', bbox_to_anchor=(0.1, 1.0), fontsize=6.4)
for a in axm:
    a.axhline(0, color='0.7', lw=0.6); a.set_xlabel('Year')
for a, l in zip(axm, '(a) (b) (c)'.split()):
    panel_label(a, l)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_v6_mechanism.pdf')); plt.close(fig)
