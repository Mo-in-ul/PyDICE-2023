"""Figure 1 and Table 1 of the letter, built from the saved results in ../blockV/results and ../stage2/results."""
import os, sys, json, numpy as np
from scipy.interpolate import CubicSpline
HERE = os.path.dirname(os.path.abspath(__file__)); PB = os.path.join(HERE, '..'); sys.path.insert(0, PB)
import matplotlib; matplotlib.use('Agg')
import house_style as hs, matplotlib.pyplot as plt, matplotlib.colors as mcolors, matplotlib.patches as mp
L = lambda p: json.load(open(os.path.join(PB, p)))
COL = ['#2166ac', '#d6604d', '#1b7837', '#762a83', '0.45']
main = sorted([o for o in L('stage2/results/s2_sigma_main.json') if o['sigma'] == 0.01], key=lambda o: o['eta'])
nop = sorted([o for o in L('stage2/results/s3_noprec.json') if o['eta'] <= 1.75], key=lambda o: o['eta'])
fix = sorted([o for o in L('blockV/results/v14_grid.json') if o['variant'] == 'fixed' and o['tlo'] == 0.01 and o['eta'] <= 1.75], key=lambda o: o['eta'])
s8 = L('stage2/results/s8_checks.json'); B0 = s8['B_x0.95']
W = 390 / 72.27  # LaTeX \textwidth of the manuscript, so font sizes are the printed sizes
fig, axs = plt.subplots(1, 2, figsize=(W, W * 0.74), gridspec_kw={'width_ratios': [1, 1.25]})
ax = axs[0]
ax.plot([o['eta'] for o in main], [o['scc_2025'] for o in main], color=COL[0], lw=1.4, label='As coded')
p = np.array(B0['phi']); k = (p >= 0.5) & (p <= 1.6)
ax.plot(p[k], np.array(B0['scc_fp'])[k], color=COL[3], lw=1.0, ls='-.', label=r'Policy fixed at $\varphi$ = 0.95 optimum')
ax.plot([o['eta'] for o in fix], [o['scc_2025'] for o in fix], color=COL[2], lw=1.0, ls='--', label=r'Precaution fixed at $\varphi$ = 0.95')
ax.plot([o['eta'] for o in nop], [o['scc_2025'] for o in nop], color=COL[4], lw=1.0, ls=':', label='No precautionary term')
ax.axvline(0.95, color='0.7', lw=0.5)
_e = np.array([o['eta'] for o in main]); _y = np.array([o['scc_2025'] for o in main]); _cs = CubicSpline(_e, _y); _f = np.linspace(0.5, 1.75, 20001); _i = np.argmin(_cs(_f))
ax.plot(0.95, float(_cs(0.95)), marker='o', ms=4, mfc='white', mec=COL[0], mew=1.0, zorder=6)
ax.plot(_f[_i], float(_cs(_f[_i])), marker='o', ms=4, color=COL[0], zorder=6)
ax.annotate('default', (0.95, float(_cs(0.95))), xytext=(1.13, 76), ha='center', va='center', fontsize=7, color='0.3', arrowprops=dict(arrowstyle='-', color='0.5', lw=0.5))
ax.annotate('minimum', (_f[_i], float(_cs(_f[_i]))), xytext=(1.53, 48), ha='center', va='center', fontsize=7, color='0.3', arrowprops=dict(arrowstyle='-', color='0.5', lw=0.5))
ax.set_xlim(0.5, 1.75); ax.set_ylim(0, 105); ax.set_xlabel(r'Elasticity of marginal utility $\varphi$'); ax.set_ylabel(r'SCC 2025 (2019 \$/tCO$_2$)')
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=7, ncol=1, handlelength=2.2); ax.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6]); ax.text(0.03, 0.04, '(a)', transform=ax.transAxes)
# (b) rho-phi map
rp = L('stage2/results/s7_rhophi.json'); R095 = {o['rho']: o['scc_2025'] for o in L('stage2/results/s7_rhophi_095.json')}
RP = {(o['rho'], o['eta']): o for o in rp}; R = sorted({o['rho'] for o in rp}); E = sorted({o['eta'] for o in rp})
ax = axs[1]
from scipy.interpolate import PchipInterpolator
ys = [100 * r for r in R]
pm = []
for r in R:
    y = np.array([RP[(r, e)]['scc_2025'] for e in E]); cs = CubicSpline(E, y); f = np.linspace(E[0], E[-1], 20001); pm.append(f[np.argmin(cs(f))])
XMIN, XMAX = 0.5, 1.85
yfill = np.linspace(ys[0], ys[-1], 400)                     # tested rho range only, no extrapolation
bfill = PchipInterpolator(ys, pm)(yfill)
ax.fill_betweenx(yfill, XMIN, bfill, color='#deebf7', lw=0, zorder=0)
ax.fill_betweenx(yfill, bfill, XMAX, color='#f9cdb6', lw=0, zorder=0)
# exact analytical boundary where the long-run utility discount rate is zero: phi0(rho) = sqrt(2 rho_e / ((1 + rho_e) sigma^2 Hbar)), rho_e = exp(rho + beta pi) - 1
rdec = yfill / 100.0; phi0 = np.sqrt(2 * (1 - np.exp(-(rdec + 0.5 * 0.05))) / (0.01 ** 2 * 235))
matplotlib.rcParams['hatch.linewidth'] = 0.4
ax.fill_betweenx(yfill, np.minimum(phi0, XMAX), XMAX, facecolor='none', edgecolor='0.55', hatch='////', lw=0, zorder=1)
ax.plot(phi0, yfill, color='0.4', lw=0.8, ls='--', zorder=3, label=r'$\varphi_0(\rho)$: zero long-run discount rate')
ax.plot(PchipInterpolator(ys, pm)(yfill), yfill, color='k', lw=2.0, zorder=5, label=r'SCC-minimizing $\varphi$ (computed)')
ax.plot(pm, ys, ls='none', marker='o', ms=3.5, color='k', zorder=6, clip_on=False)
pts = [('DICE-23', 0.1, 0.95), ('Stern', 0.1, 1.0), ('Drupp md', 0.5, 1.0), ('Drupp mn', 1.1, 1.35), ('NPP2', 0.1, 1.25), ('NPP2.5', 0.4, 1.4), ('NPP3', 0.8, 1.53), ('DICE-16R', 1.5, 1.45)]
lab_xy = {'DICE-23': (0.78, 0.30), 'Stern': (0.98, 0.42), 'Drupp md': (0.84, 0.66), 'Drupp mn': (1.08, 1.22), 'NPP2': (1.15, 0.31),
          'NPP2.5': (1.60, 0.30), 'NPP3': (1.66, 0.62), 'DICE-16R': (1.12, 1.42)}
for k_, (lab, r_, e_) in enumerate(pts):
    ax.plot(e_, r_, marker='o', ms=4.5, mfc='white', mec='k', mew=0.9, ls='none', zorder=7, clip_on=False, label='Published calibrations' if k_ == 0 else None)
    tx, ty = lab_xy[lab]
    ax.annotate(lab, (e_, r_), xytext=(tx, ty), textcoords='data', fontsize=7, zorder=8, ha='center', va='center', bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.85),
                arrowprops=dict(arrowstyle='-', color='0.45', lw=0.5, shrinkA=1, shrinkB=3))
ax.text(0.70, 0.95, 'SCC\ndecreases', color='#2166ac', fontsize=7.5, linespacing=1.1, ha='center', va='center')
_lab_inc = (1.612, 1.12)  # centred between the solid curve and the dashed line; rotated parallel to them after layout
ax.set_xlim(XMIN, XMAX); ax.set_ylim(ys[0], ys[-1]); ax.set_xlabel(r'Elasticity of marginal utility $\varphi$'); ax.set_ylabel(r'Pure rate of time preference $\rho$ (%)')
ax.legend(frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(0.45, -0.17), ncol=1); ax.text(0.03, 0.04, '(b)', transform=ax.transAxes)
fig.tight_layout()
_ang = axs[1].transData.transform_angles(np.array([np.degrees(np.arctan2(0.5, 0.1))]), np.array([_lab_inc]))[0]
axs[1].text(*_lab_inc, 'SCC increases', color='#b2182b', fontsize=7.5, ha='center', va='center', rotation=_ang, rotation_mode='anchor', zorder=9)
fig.savefig(os.path.join(HERE, 'fig_letter_main.pdf')); plt.close(fig)

# table
sp = {o['eta']: o for o in L('stage2/results/s7_split.json')}
C0 = s8['C_x0.95']; ch = {(o['phi_u'], o['phi_p']): o['scc_2025'] for o in L('stage2/results/s7_channels.json') if o['tlo'] is None}
ro = {'u': ch[(1.3, 0.95)] - ch[(0.95, 0.95)], 'p': ch[(0.95, 1.3)] - ch[(0.95, 0.95)], 'j': ch[(1.3, 1.3)] - ch[(0.95, 0.95)]}; ro['i'] = ro['j'] - ro['u'] - ro['p']
cons = {o['eta']: o for o in L('stage2/results/s5_constrained.json') if o['case'] == 'dice'}
mainD = {o['eta']: o['scc_2025'] for o in main}
hz = {o['eta']: o['scc_2025'] for o in L('stage2/results/s2_sigma_horizon.json') if o['sigma'] == 0.01 and o['cap'] == 47}
nb = {o['eta']: o['scc_2025'] for o in L('blockV/results/v14_grid.json') if o['variant'] == 'dice' and o['tlo'] is None}
f = lambda v: f'{v:+.1f}'
lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
         r'\multicolumn{5}{l}{\emph{A. $d\,\mathrm{SCC}/d\varphi$ split into weights at fixed policy and re-optimization (no floor)}} \\',
         r'$\varphi$ & Total & Weights & Weights, Prop.~1 & Re-optimization \\', r'\midrule']
for e in (0.95, 1.1, 1.3, 1.5):
    o = sp[e]; lines.append(f"{e:.2f} & {f(o['total'])} & {f(o['direct'])} & {f(o['direct_analytic'])} & {f(o['indirect'])} \\\\")
lines += [r'\midrule', r'\multicolumn{5}{l}{\emph{B. Change in SCC when $\varphi_{\mathrm u}$, $\varphi_{\mathrm p}$ or both move from 0.95 to 1.3 (no floor)}} \\',
          r' & Utility only & Precaution only & Both & Interaction \\', r'\midrule',
          f"Policy fixed at $\\varphi = 0.95$ optimum & {f(C0['utility_effect'])} & {f(C0['prec_effect'])} & {f(C0['joint'])} & {f(C0['interaction'])} \\\\",
          f"Re-optimized & {f(ro['u'])} & {f(ro['p'])} & {f(ro['j'])} & {f(ro['i'])} \\\\",
          r'\midrule', r'\multicolumn{5}{l}{\emph{C. Reliability of the 2025 SCC as coded}} \\',
          r'$\varphi$ & SCC, $T \ge 0.5$ & SCC, no floor & Floor term & Horizon to 2620 (\%) \\', r'\midrule']
for e in (1.3, 1.5, 1.75):
    lines.append(f"{e:.2f} & {mainD[e]:.1f} & {nb[e]:.1f} & {f(cons[e]['floor_sum'])} & {100*(hz[e]/mainD[e]-1):+.1f} \\\\")
lines += [r'\bottomrule', r'\end{tabular}']
open(os.path.join(HERE, 'tab_letter.tex'), 'w').write('\n'.join(lines) + '\n')
print('letter assets written')
