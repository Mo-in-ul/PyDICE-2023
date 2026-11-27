"""Figures for the journal manuscript (p2_manuscript/cce_v2). Reads results/*.json only; no model runs.
Writes p2_manuscript/cce_v2/figures/fig1.pdf ... fig5.pdf (and .png previews).
  fig1  timeline of the decision problem (schematic)
  fig2  SCC in 2025 against removal capacity, with chords and Jensen gaps (p4_mech.json)
  figC1_indicator, figC2_timing  Appendix C: peak-warming indicator parity plot (C1) and change in peak by access year (C2) (p3_decomp2.json, p3b_numbers.json)
  fig3  control rate before and after learning by planning strategy, 2% calibration (p6_paths.json)
  fig4  regret of planning strategies against the stochastic optimum (p5_regret.json)
  fig5  premium against the average consumption discount rate (p5_tests.json, p4_mech.json, p4_finish.json, p5_crate.json)"""
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'p2_manuscript', 'cce_v2', 'figures'); os.makedirs(OUT, exist_ok=True)
R = lambda n: json.load(open(os.path.join(HERE, 'results', n))) if os.path.exists(os.path.join(HERE, 'results', n)) else {}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d3'
COL = {'stochastic': '#2a78d6', 'expected': '#eb6834', 'optimist': '#1baf7a', 'pessimist': '#c98500', 'perfect': '#e87ba4'}
CAPS, NOLIM = '#2a78d6', '#eb6834'
W = 5.3                                        # figure width in inches; the journal text width is 5.0 in
plt.rcParams.update({'font.size': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.labelsize': 8, 'axes.titlesize': 8.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK, 'legend.frameon': False,
                     'pdf.fonttype': 42})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf')); fig.savefig(os.path.join(OUT, name + '.png'), dpi=200); plt.close(fig)


# ---------- fig1: timeline (two panels: learning in 2050 and in 2070) ----------
SUCC, FAIL = '#3d3d3a', '#8a8984'                  # states in neutral inks: plan colours stay reserved for Figs 3-5
fig, axs = plt.subplots(2, 1, figsize=(W, 3.6), sharex=True)
for ax, tr, ttl in ((axs[0], 2050, '(a) Outcome learned in 2050'), (axs[1], 2070, '(b) Outcome learned in 2070')):
    ax.set_xlim(2016, 2124); ax.set_ylim(-1.25, 1.35); ax.set_yticks([])
    for sp in ('left', 'top', 'right'): ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_visible(ax is axs[1])
    ax.tick_params(axis='x', length=3 if ax is axs[1] else 0)
    ax.axvline(2070, color=GRID, lw=0.9, ls=(0, (3, 2)), zorder=0)
    ax.plot([2025, tr], [0, 0], color=INK, lw=2.4, solid_capstyle='butt', zorder=3)
    ax.text((2025 + tr) / 2, 0.13, 'same decisions\nin both states' if tr == 2050 else 'same decisions in both states', ha='center', va='bottom', fontsize=7.5, linespacing=1.15)
    ax.plot([tr, tr + 8, 2121], [0, 0.72, 0.72], color=SUCC, lw=1.8, zorder=2)
    ax.plot([tr, tr + 8, 2121], [0, -0.72, -0.72], color=FAIL, lw=1.8, ls=(0, (4, 2)), zorder=2)
    ax.plot([tr], [0], 'o', ms=6.5, mfc='white', mec=INK, mew=1.4, zorder=4)
    ax.text(tr - 1.2, -0.12, 'outcome\nlearned', ha='right', va='top', fontsize=7, color=INK2)
    ax.text(2121, 0.86, r'success, probability $p$: capacity $K$ from 2070', ha='right', va='bottom', fontsize=7)
    ax.text(2121, -0.58, r'failure, probability $1-p$: no removal', ha='right', va='bottom', fontsize=7, color=INK2)
    ax.set_title(ttl, loc='left', fontsize=8.5)
axs[0].text(2071, -1.2, 'removal starts in 2070 if it succeeds', fontsize=7, color=INK2, va='bottom')
axs[1].set_xticks([2025, 2050, 2070, 2100])
fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.09, hspace=0.45); save(fig, 'fig1')

# ---------- fig2: SCC against capacity ----------
M = R('p4_mech.json')
fig, axs = plt.subplots(1, 2, figsize=(W, 2.75))
for ax, cal, ttl in ((axs[0], 'pub', '(a) Published calibration'), (axs[1], 'd2', '(b) 2 percent calibration')):
    gaps = {}
    for v, col, mk, lab in (('A', CAPS, 'o', 'Hard storage caps'), ('D', NOLIM, 's', 'No storage limit')):
        c = M[f'curve:{v}_{cal}']; ks = sorted(c, key=float); x = [float(k) for k in ks]; y = [c[k]['scc2025'] for k in ks]
        ax.plot(x, y, '-' + mk, color=col, lw=1.6, ms=3.5, label=lab, zorder=3)
        ax.plot([0, 1], [y[0], y[-1]], ':', color=col, lw=1.1)
        mid = c['0.5']['scc2025']; chord = 0.5 * (y[0] + y[-1]); gaps[v] = chord - mid
    ax.text(0.98, 0.97, 'Jensen gap at half capacity', transform=ax.transAxes, ha='right', va='top', fontsize=7, color=INK)
    ax.text(0.98, 0.89, f"caps: \\${gaps['A']:.2f}", transform=ax.transAxes, ha='right', va='top', fontsize=7.5, color=CAPS, fontweight='bold')
    ax.text(0.98, 0.81, f"no limit: \\${gaps['D']:.2f}", transform=ax.transAxes, ha='right', va='top', fontsize=7.5, color=NOLIM, fontweight='bold')
    ax.set_xlabel('Removal capacity (share of central)'); ax.set_title(ttl, loc='left')
    ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_xlim(-0.03, 1.03)
axs[0].set_ylabel('SCC in 2025 (\\$ per tCO$_2$)')
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=2, fontsize=7.5)
fig.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, 'fig2')

# ---------- fig3: control-rate paths ----------
P = R('p6_paths.json')
if all(k in P for k in ('d2:2050', 'd2:2070')):
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.3), sharey=True)
    for ax, item, ttl in ((axs[0], 'd2:2050', '(a) Outcome learned in 2050'), (axs[1], 'd2:2070', '(b) Outcome learned in 2070')):
        o = P[item]; yrs = np.array(o['years']) - 5; sel = yrs <= 2100; ty = 2020 + 5 * o['k']     # paths start in 2020; the ceiling array starts in 2025
        cy = np.array(o['years']); cs = cy <= 2100
        ax.axvspan(2020, 2045, color='#f0efec', zorder=0)
        ax.text(2032.5, 0.97, 'all plans\ncoincide while\nthe ceiling\nbinds', fontsize=6.5, color=INK2, ha='center', va='top')
        ax.plot(cy[cs], np.minimum(np.array(o['ceiling'])[cs], 1.0), color=INK2, lw=0.8, label='Ceiling on the control rate', zorder=1)
        ax.plot(yrs[sel], np.array(o['optimist']['MU'])[sel], color=COL['optimist'], lw=1.0, ls=(0, (5, 2)), label='Optimist', zorder=2)
        ax.plot(yrs[sel], np.array(o['pessimist']['MU'])[sel], color=COL['pessimist'], lw=1.0, ls=(0, (5, 1.5, 1, 1.5)), label='Pessimist', zorder=2)
        pre = (yrs < ty) & sel
        ax.plot(yrs[pre], np.array(o['expected_S']['MU'])[pre], color=COL['expected'], lw=2.4, label='Expected capacity, before learning', zorder=3)
        ax.plot(yrs[pre], np.array(o['stochastic_S']['MU'])[pre], color=COL['stochastic'], lw=2.4, label='Stochastic optimum, before learning', zorder=4)
        for k_, lab in (('stochastic_S', 'Stochastic optimum, each state after learning'), ('stochastic_F', None)):
            post = (yrs >= ty - 5) & sel
            ax.plot(yrs[post], np.array(o[k_]['MU'])[post], ':', color=COL['stochastic'], lw=1.4, label=lab, zorder=3)
        ax.axvline(ty, color=INK2, lw=0.7); ax.text(ty + 1, 0.03, 'learned', fontsize=6.5, color=INK2)
        ax.set_title(ttl, loc='left'); ax.set_xlabel('Year'); ax.set_xlim(2020, 2100); ax.set_ylim(0, 1.04)
        ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
    axs[0].set_ylabel('Emissions control rate $\\mu$')
    h, l = axs[1].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=2, fontsize=7, handlelength=2.8, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0.2, 1, 1)); save(fig, 'fig3')

# ---------- fig4: regret ----------
G = R('p5_regret.json')
fig, axs = plt.subplots(2, 2, figsize=(W, 4.6), gridspec_kw={'height_ratios': [1.2, 1]})
cases = (('2050', 'learned\n2050'), ('2070', 'learned\n2070'), ('2050:noceil', '2050,\nno ceiling'))
strat = (('expected', 'Expected capacity'), ('pessimist', 'Pessimist'), ('optimist', 'Optimist'))
w = 0.26
for c, (cal, nm) in enumerate((('pub', 'Published'), ('d2', '2 percent'))):
    ax = axs[0, c]
    for i, (cs, cl) in enumerate(cases):
        r = G[f'{cal}:{cs}']
        for j, (sk, sl) in enumerate(strat):
            vals = [r[p][sk]['regret'] for p in ('0.25', '0.5', '0.75')]
            x = i + (j - 1) * w
            ax.bar(x, vals[1], w * 0.9, color=COL[sk], label=sl if (i == 0 and c == 0) else None, zorder=2)
            ax.plot([x, x], [min(vals), max(vals)], color=INK, lw=0.8, zorder=3)
    ax.set_xticks(range(len(cases))); ax.set_xticklabels([q[1] for q in cases])
    ax.set_title(f"({'ab'[c]}) {nm}: three planning rules", loc='left'); ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax = axs[1, c]; top = 0
    for i, (cs, cl) in enumerate(cases):
        r = G[f'{cal}:{cs}']; vals = [r[p]['expected']['regret'] for p in ('0.25', '0.5', '0.75')]; top = max(top, max(vals))
        ax.bar(i, vals[1], 0.5, color=COL['expected'], zorder=2); ax.plot([i, i], [min(vals), max(vals)], color=INK, lw=0.8, zorder=3)
        ax.text(i, max(vals), f"{vals[1]:.3f}" if vals[1] > 0.05 else f"{vals[1]:.4f}", fontsize=7, ha='center', va='bottom', color=INK)
    ax.set_ylim(0, top * 1.22)
    ax.set_xticks(range(len(cases))); ax.set_xticklabels([q[1] for q in cases])
    ax.set_title(f"({'cd'[c]}) {nm}: expected capacity only", loc='left'); ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
for r_ in (0, 1):
    axs[r_, 0].set_ylabel('Regret (trillion 2019 \\$)')
h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=3, fontsize=7.5)
fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, 'fig4')

# ---------- fig5: premium against the consumption discount rate ----------
T = R('p5_tests.json'); CR = R('p5_crate.json'); F = R('p4_finish.json')
gap = lambda c: 100 * (0.5 * (c['0.0'] + c['1.0']) - c['0.5']) / c['0.5']
tcurve = lambda key, v: {s: T[key][f'{v}|{s}']['scc2025'] for s in ('0.0', '0.5', '1.0')}
mcurve = lambda key: {s: M[key][s]['scc2025'] for s in ('0.0', '0.5', '1.0')}
E95, E001 = 'Elasticity 0.95', 'Elasticity 0.001 (alternative runs)'
series = {E95: [], E001: []}
singles = []                                   # single calibrations with other elasticities: not a series
for ck in CR:
    try:
        if ck.startswith('disc:'):
            _, rho, eta = ck.split(':'); a, d = tcurve(ck, 'A'), tcurve(ck, 'D'); key = E95 if abs(float(eta) - 0.95) < 1e-9 else (E001 if float(eta) < 0.5 else None)
        elif ck == 'pub':
            a, d, key = mcurve('curve:A_pub'), mcurve('curve:D_pub'), E95
        elif ck in ('d2', 'd3'):
            a, d, key = mcurve(f'curve:A_{ck}'), mcurve(f'curve:D_{ck}'), E001
        elif f'disc:A_{ck}' in M:
            a = {s: M[f'disc:A_{ck}'][s]['scc2025'] for s in ('0.0', '0.5', '1.0')}; d = {s: M[f'disc:D_{ck}'][s]['scc2025'] for s in ('0.0', '0.5', '1.0')}; key = E001
        else:
            continue
    except KeyError:
        continue
    if key is None:
        singles.append((100 * CR[ck]['r_2200'], gap(a), gap(d), ck)); continue
    series[key].append((100 * CR[ck]['r_2200'], gap(a), gap(d), ck))
for rate in ('0.025', '0.035'):
    o = F.get(f'disc:{rate}')
    if o:
        a = {s: o[f'A|{float(s)}']['scc2025'] for s in ('0.0', '0.5', '1.0')}; d = {s: o[f'D|{float(s)}']['scc2025'] for s in ('0.0', '0.5', '1.0')}
        series[E001].append((100 * float(rate), gap(a), gap(d), rate))
fig, axs = plt.subplots(1, 2, figsize=(W, 2.9))
SMK = {'disc:0.001:1.1': ('^', r'$\rho$ 0.1%, $\eta$ 1.1'), 'disc:0.001:1.5': ('D', r'$\rho$ 0.1%, $\eta$ 1.5')}
for ax, j, ttl in ((axs[0], 1, '(a) Hard storage caps'), (axs[1], 2, '(b) No storage limit')):
    for (lab, pts), col, mk in zip(series.items(), (CAPS, NOLIM), ('o', 's')):
        pts = sorted(q for q in pts if q[0] >= 1.7)
        ax.plot([q[0] for q in pts], [q[j] for q in pts], '-' + mk, color=col, lw=1.4, ms=3.5, label=lab)
        for q in pts:
            if q[3] in ('pub', 'd2'):
                ax.plot(q[0], q[j], 'o', ms=9, mfc='none', mec=INK, mew=1.0, zorder=4)
    for q in singles:
        mk, lab = SMK.get(q[3], ('x', q[3]))
        ax.plot(q[0], q[j], mk, ms=5, color=INK, mfc=INK, ls='none', zorder=5, label=lab)
    ymax = ax.get_ylim()[1]
    pub = [q for q in series[E95] if q[3] == 'pub'][0]; d2 = [q for q in series[E001] if q[3] == 'd2'][0]
    ax.annotate('Published', (pub[0], pub[j]), xytext=(pub[0] + 0.45, pub[j] + 0.17 * ymax), fontsize=7.5,
                arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6, shrinkA=1, shrinkB=5))
    ax.annotate('2% run', (d2[0], d2[j]), xytext=(1.4, 0.06 * ymax), fontsize=7.5, ha='left', va='bottom',
                arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6, shrinkA=1, shrinkB=5))
    ax.axhline(0, color=INK2, lw=0.5); ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_xlabel('Consumption discount rate (%/yr)'); ax.set_title(ttl, loc='left'); ax.set_xlim(1.3, 5.9)
axs[0].set_ylabel('Jensen gap (% of SCC)')
h, l = axs[0].get_legend_handles_labels(); h.append(plt.Line2D([], [], marker='o', ms=9, mfc='none', mec=INK, ls='none')); l.append('Benchmark calibrations')
fig.legend(h, l, loc='lower center', ncol=3, fontsize=7)
fig.tight_layout(rect=(0, 0.15, 1, 1)); save(fig, 'fig5')
print('figures written to', OUT)


# ---------- Appendix C figures: peak indicator (C1) and timing (C2) ----------
DC, NUMS = R('p3_decomp2.json'), R('p3b_numbers.json')
oos = NUMS['oos']; kA, kR = oos['kA_base'], oos['kR_base']
rws = [(k, v) for k, v in DC.items() if not k.endswith('none')]
pred = np.array([(kA * v['D'] - kR * v['Rn']) / 1000.0 + v['P'] for _, v in rws]); dp = np.array([v['dpeak'] for _, v in rws])
isfit = np.array([k.startswith('base_') for k, _ in rws])
lo = np.floor(min(pred.min(), dp.min()) * 20) / 20 - 0.02; hi = np.ceil(max(pred.max(), dp.max()) * 20) / 20 + 0.02
fig, ax = plt.subplots(figsize=(W * 0.75, W * 0.75))
ax.plot([lo, hi], [lo, hi], color='#c8c7c2', lw=0.7, zorder=1)
ax.axhline(0, color=GRID, lw=0.5, zorder=0); ax.axvline(0, color=GRID, lw=0.5, zorder=0)
ax.scatter(pred[isfit], dp[isfit], s=20, marker='o', facecolor='white', edgecolor=CAPS, linewidth=1.0, zorder=3, label='Fitting runs (39)')
ax.scatter(pred[~isfit], dp[~isfit], s=30, marker='s', facecolor=NOLIM, edgecolor=NOLIM, linewidth=0.8, zorder=3, label='Test runs (28)')
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
ax.set_xlabel('Indicator $k_A D - k_R R_n + P$ (\\textdegree C)'.replace('\\textdegree ', '$^\\circ$')); ax.set_ylabel('Change in peak warming ($^\\circ$C)')
ax.legend(loc='upper left', fontsize=7.5)
fig.tight_layout(); save(fig, 'figC1_indicator')

VARS = (('base', 'Central case', CAPS, '-', 'o'), ('stor05', 'Storage halved', NOLIM, '--', 's'), ('cost2', 'Removal costs doubled', '#1baf7a', ':', '^'))
fig, axs = plt.subplots(1, 2, figsize=(W, 2.7), sharey=True)
for ax, cal, ttl in ((axs[0], 'pub', '(a) Published calibration'), (axs[1], 'd2', '(b) 2 percent calibration')):
    for var, lab, col, ls, mk in VARS:
        ys = sorted(int(k.split('_')[-1]) for k in DC if k.startswith(f'{var}_{cal}_') and not k.endswith('none'))
        ax.plot(ys, [DC[f'{var}_{cal}_{y}']['dpeak'] for y in ys], ls=ls, marker=mk, ms=3.5, color=col, lw=1.4, label=lab)
    ax.axhline(0, color=INK2, lw=0.6); ax.set_xlabel('First year removal is available'); ax.set_title(ttl, loc='left')
    ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
axs[0].set_ylabel('Change in peak warming ($^\\circ$C)')
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, fontsize=7.5)
fig.tight_layout(rect=(0, 0.08, 1, 1)); save(fig, 'figC2_timing')
