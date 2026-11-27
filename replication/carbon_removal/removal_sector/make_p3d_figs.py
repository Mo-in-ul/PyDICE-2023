"""Figures for paper 2 (record and manuscript). Validated categorical palette (blue, orange, aqua, yellow; reference
instance of the dataviz method) for the record; BW=1 writes grayscale versions (*_bw.pdf) for the manuscript, where
identity is carried by line style and marker as well as shade. One y-axis per panel, legends below the plots,
selective direct labels, recessive grid. Revised 5 Nov 2026 after a review of the manuscript figures."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); A = os.path.join(HERE, 'p2_report', 'assets')
sys.path.insert(0, os.path.join(HERE, '..', 'paper_build'))
import matplotlib; matplotlib.use('Agg'); import house_style as hs, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
CCE = os.environ.get('CCE') == '1'                     # Climate Change Economics: 5-inch column, stacked panels, grayscale
BW = os.environ.get('BW') == '1' or CCE                # grayscale versions for the journal manuscript
W = 127 / 25.4                                          # ws-cce text width (30 pc = 5 in)
if BW:
    C1, C2, C3, C4 = '#000000', '#8c8c8c', '#4a4a4a', '#bdbdbd'
_sv = plt.Figure.savefig
def _save(self, fn, *a, **k):
    return _sv(self, fn.replace('.pdf', '_cce.pdf') if CCE else (fn.replace('.pdf', '_bw.pdf') if BW else fn), *a, **k)
plt.Figure.savefig = _save
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d3'
plt.rcParams.update({'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.4, 'axes.axisbelow': True, 'axes.labelsize': 8, 'xtick.labelsize': 7.5,
                     'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'lines.linewidth': 1.4})
TS = 8.5                                                # panel title size
LS = 7.2                                                # annotation size
J = lambda f: json.load(open(os.path.join(HERE, 'results', f)))
FX, MC, UT, DC, WF, ST = J('p3_fixes.json'), J('p3_mech.json'), J('p3_util.json'), J('p3_decomp2.json'), J('p3_welfare.json'), J('p3_stoch.json')
R3, RD, NUMS = J('p3_runs.json'), J('p3_reduced.json'), J('p3b_numbers.json')
YR = 2020 + 5 * np.arange(81)
CALp = {'pub': 'Published', 'd3': '3% discount', 'd2': '2% discount'}


def title(ax, s):
    ax.set_title(s, fontsize=TS, loc='left')


def legend_below(fig, handles, ncol, y=-0.02):
    if CCE:
        ncol = min(ncol, 2 if len(handles) in (2, 4) else 3) if len(handles) > 3 else ncol
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, fontsize=7.5,
               handlelength=2.6, columnspacing=1.6)


def two(h_double, h_each=0.56, **kw):
    """Two panels: side by side at journal double width, or stacked at the 5-inch CCE width."""
    if CCE:
        return plt.subplots(2, 1, figsize=(W, W * h_each * 2))
    return plt.subplots(1, 2, figsize=(hs.DOUBLE, hs.DOUBLE * h_double), **kw)


# ---------- Figure 1: the story in temperature paths ----------
fig, axs = two(0.40)
m = (YR >= 2020) & (YR <= 2200)
T0 = np.array(FX['none_T']['pub']['T']); T70 = np.array(FX['late']['pub_2070']['T']); T110 = np.array(FX['late']['pub_2110']['T'])
D0 = np.array(FX['none_T']['d2']['T']); DS = np.array(ST['U1_d2_2070']['det_S']['T']); DO = np.array(ST['U1_d2_2070']['optimist']['F']['T'])
panels = [(axs[0], [(T0, INK2, '--', 'no removal'), (T70, C1, '-', 'removal from 2070'), (T110, C2, '-', 'removal from 2110')],
           '(a) Published calibration', (0, 2)),
          (axs[1], [(D0, INK2, '--', 'never expected'), (DS, C1, '-', 'expected, arrives 2070'), (DO, C2, '-', 'expected, fails 2070')],
           '(b) 2% discount: removal expected from 2070', (0, 2))]
for ax, series, ttl, (ia, ib) in panels:
    hnd = []; done = []
    for T, col, ls, lab in series:
        ax.plot(YR[m], T[m], color=col, ls=ls, lw=1.6 if ls == '-' else 1.1)
        i = int(np.argmax(T)); ax.plot(YR[i], T[i], 'o', ms=4.5, color=col, mec='white', mew=0.8, zorder=4)
        if not any(abs(T[i] - q) < 0.005 for q in done):
            off = (0, 6) if T[i] >= max(s[0].max() for s in series) - 1e-9 else (0, -11)
            ax.annotate(f'{T[i]:.2f}', (YR[i], T[i]), xytext=off, textcoords='offset points', ha='center', fontsize=LS, color=INK)
            done.append(T[i])
        hnd.append(Line2D([], [], color=col, ls=ls, lw=1.6 if ls == '-' else 1.1, marker='o', ms=4, mec='white', label=lab))
    # bracket: rise of the peak against the reference (dashed) path
    Ta, Tb = series[ia][0], series[ib][0]; pa, pb = Ta.max(), Tb.max(); xb = YR[int(np.argmax(Tb))] + 22
    ax.plot([xb, xb], [pa, pb], color=INK, lw=0.8); ax.plot([xb - 3, xb + 3], [pa, pa], color=INK, lw=0.8); ax.plot([xb - 3, xb + 3], [pb, pb], color=INK, lw=0.8)
    ax.plot([YR[int(np.argmax(Ta))], xb - 3], [pa, pa], color=INK2, lw=0.5, ls=':')
    ax.annotate(f'{pb - pa:+.3f}°C', (xb, 0.5 * (pa + pb)), xytext=(4, 0), textcoords='offset points', va='center', fontsize=LS, color=INK)
    ax.set_xlim(2020, 2200); ax.set_xlabel('Year'); ax.set_ylabel('Warming (°C)'); title(ax, ttl)
    ax.legend(handles=hnd, loc='lower right' if ttl.startswith('(a)') else 'center right', bbox_to_anchor=(1.0, 0.03) if ttl.startswith('(a)') else (1.0, 0.42),
              ncol=1, frameon=True, framealpha=0.9, edgecolor='none', fontsize=7.2, handlelength=2.4)
fig.tight_layout(); fig.savefig(os.path.join(A, 'fig_p3e_story.pdf'), bbox_inches='tight'); plt.close(fig)

# ---------- Figure 3: welfare accounting, access 2110 (published) and 2090 (2%) ----------
fig, axs = two(0.36, 0.52)
POS, NEG, TOT = (C3, C2, C1)
for ax, (c, y) in zip(axs, (('pub', 2110), ('d2', 2090))):
    r = WF['summary'][f'{c}_{y}']
    items = [('Damages\nbefore\npeak', -r['dDam_pre_T']), ('Damages\nafter\npeak', -r['dDam_post_T']), ('Abatement\ncost', -r['dAbate_T']),
             ('Removal\ncost', -r['remcost_T'])]
    other = r['gain_T'] - sum(v for _, v in items); items.append(('Other', other))
    cum = 0.0; G = r['gain_T']
    for i, (lab, v) in enumerate(items):
        ax.bar(i, v, bottom=cum, width=0.62, color=POS if v >= 0 else NEG, edgecolor='white', linewidth=0.8,
               hatch=None if (v >= 0 or not BW) else '////')
        top = max(cum, cum + v)
        ax.text(i, top + 0.025 * G, f'{v:+.1f}', ha='center', va='bottom', fontsize=LS, color=INK)
        ax.plot([i + 0.31, i + 0.69], [cum + v, cum + v], color=INK2, lw=0.6)          # connector to the next bar
        cum += v
    ax.bar(len(items), G, width=0.62, color=TOT, edgecolor='white', linewidth=0.8)
    ax.text(len(items), G * 1.025, f'{G:.1f}', ha='center', va='bottom', fontsize=LS, color=INK)
    ax.set_xticks(range(len(items) + 1)); ax.set_xticklabels([l for l, _ in items] + ['Welfare\ngain'], fontsize=7)
    ax.axhline(0, color=INK2, lw=0.6); ax.set_ylabel('Trillion US\\$'); ax.grid(axis='x', visible=False)
    title(ax, f"({'a' if c == 'pub' else 'b'}) {CALp[c]}, access {y}: peak {r['dpeak']:+.3f}°C")
    lo = min(0, min(np.cumsum([v for _, v in items]).min(), -r['dDam_pre_T']) * 1.6)
    ax.set_ylim(lo, G * (1.32 if CCE else 1.2))
legend_below(fig, [Patch(color=POS, label='raises welfare'), Patch(facecolor=NEG, hatch='////' if BW else None, edgecolor='white', label='lowers welfare'),
                   Patch(color=TOT, label='net welfare gain')], 3, y=0.0)
fig.tight_layout(); fig.savefig(os.path.join(A, 'fig_p3e_welfare.pdf'), bbox_inches='tight'); plt.close(fig)

# ---------- Figure 2: peak indicator ----------
oos = NUMS['oos']; kA, kR = oos['kA_base'], oos['kR_base']               # coefficients fitted on the central runs only
rws = [(k, v) for k, v in DC.items() if not k.endswith('none')]
D_ = np.array([v['D'] for _, v in rws]); Rn = np.array([v['Rn'] for _, v in rws]); P_ = np.array([v['P'] for _, v in rws])
dp = np.array([v['dpeak'] for _, v in rws]); pred = (kA * D_ - kR * Rn) / 1000.0 + P_      # k per 1000 GtCO2
fig, axs = two(0.42, 0.62, gridspec_kw={'width_ratios': [1, 1.55]})
ax = axs[0]
mk = {'base': ('o', C1, 'central (fitting sample)'), 'cost2': ('s', C2, 'removal costs ×2 (test)'), 'stor05': ('^', C3, 'storage ×0.5 (test)')}
for var, (mm, col, lab) in mk.items():
    sel = np.array([k.startswith(var + '_') for k, _ in rws])
    fill = 'white' if var == 'base' else col
    ax.scatter(pred[sel], dp[sel], s=18, marker=mm, facecolor=fill, edgecolor=col, linewidth=0.9, zorder=3)
lim = [-0.3, 0.2]; ax.plot(lim, lim, color=INK2, lw=0.6, ls=':'); ax.text(0.12, 0.155, '1:1', fontsize=LS, color=INK2)
ax.axhline(0, color=INK2, lw=0.5); ax.axvline(0, color=INK2, lw=0.5)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal'); ax.set_xticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2]); ax.set_yticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2])
ax.set_xlabel('Indicator $k_A D - k_R R_n + P$ (°C)'); ax.set_ylabel('Change in peak warming (°C)')
n_fit = oos['Central runs (fitting sample)']; n_test = oos['Doubled costs, halved storage (test sample)']
ax.text(-0.29, 0.17, f"signs right: {n_fit['fit_sign']}/{n_fit['n']} fitting,\n{n_test['fit_sign']}/{n_test['n']} test", fontsize=LS, color=INK, va='top')
title(ax, '(a) Indicator against the model')
ax = axs[1]
for var, (mm, col, lab) in mk.items():
    for c in ('pub', 'd2'):
        ys = sorted(int(k.split('_')[-1]) for k in DC if k.startswith(f'{var}_{c}_') and not k.endswith('none'))
        vals = [DC[f'{var}_{c}_{y}']['dpeak'] for y in ys]
        ax.plot(ys, vals, marker=mm, ms=4, color=col, mfc='white' if var == 'base' else col, lw=1.3, ls='-' if c == 'pub' else '--')
ax.axhline(0, color=INK2, lw=0.6); ax.set_xlabel('First year removal is available'); ax.set_ylabel('Change in peak warming (°C)')
title(ax, '(b) Change in peak by access year')
hnd = [Line2D([], [], color=col, marker=mm, mfc='white' if v == 'base' else col, ls='none', ms=5, label=lab) for v, (mm, col, lab) in mk.items()]
hnd += [Line2D([], [], color=INK2, ls='-', label='published calibration'), Line2D([], [], color=INK2, ls='--', label='2% discount')]
legend_below(fig, hnd, 5, y=0.0)
fig.tight_layout(); fig.savefig(os.path.join(A, 'fig_p3e_indicator.pdf'), bbox_inches='tight'); plt.close(fig)

# ---------- Figure 4: the SCC against removal capacity and the Jensen gap ----------
fig, axs = two(0.46, 0.62)
ss = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
LBL = {('pub', 'A'): (0.58, 61.9), ('pub', 'D'): (0.62, 57.9), ('d2', 'A'): (0.60, 262.0), ('d2', 'D'): (0.04, 192.0)}
for ax, c, tag in zip(axs, ('pub', 'd2'), ('a', 'b')):
    ys_all = []
    for v, col, mm, lab, dx in (('A', C1, 'o', 'hard storage caps', -0.035), ('D', C2, 's', 'no storage limit', 0.035)):
        y = np.array([MC[f'{v}_{c}']['curve'][str(s)]['scc2025'] for s in ss]); ys_all.append(y)
        ax.plot(ss, y, marker=mm, ms=4.5, color=col, lw=1.6, mfc=col if v == 'A' else 'white', label=lab, zorder=3)
        ax.plot([0, 1], [y[0], y[-1]], color=col, lw=0.8, ls=':')
        mid = 0.5 * (y[0] + y[-1]); g = mid - y[2]; xb = 0.5 + dx
        ax.annotate('', xy=(xb, y[2]), xytext=(xb, mid), arrowprops=dict(arrowstyle='<->', color=INK, lw=0.9, shrinkA=0, shrinkB=0))
        ax.plot([0.5, xb], [y[2], y[2]], color=INK2, lw=0.5); ax.plot([0.5, xb], [mid, mid], color=INK2, lw=0.5)
        L = LBL[(c, v)]
        ax.annotate(f'Jensen gap \\${g:.2f}/t', (xb, 0.5 * (mid + y[2])), xytext=L, textcoords='data', ha='left', va='center', fontsize=LS + 0.3,
                    color=INK, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.5, shrinkA=1, shrinkB=2))
    allv = np.concatenate(ys_all); pad = 0.08 * (allv.max() - allv.min())
    ax.set_ylim(allv.min() - pad, allv.max() + pad); ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(['0', '0.25', '0.5\n(expected)', '0.75', '1'])
    ax.set_xlabel('Removal capacity (share of central)'); ax.set_ylabel('SCC in 2025 (\\$/tCO$_2$)')
    title(ax, f'({tag}) {CALp[c]}')
legend_below(fig, [Line2D([], [], color=C1, marker='o', ms=5, lw=1.6, label='hard storage caps'),
                   Line2D([], [], color=C2, marker='s', mfc='white', ms=5, lw=1.6, label='no storage limit'),
                   Line2D([], [], color=INK2, ls=':', lw=0.8, label='chord: probability-weighted SCC')], 3, y=0.0)
fig.tight_layout(); fig.savefig(os.path.join(A, 'fig_p3e_mech.pdf'), bbox_inches='tight'); plt.close(fig)

# ---------- Record figure: premium by variant (the manuscript reports it in a table) ----------
fig, ax = plt.subplots(figsize=(hs.DOUBLE * 0.62, hs.DOUBLE * 0.36))
NUM = NUMS['mech']
vars_ = [('A', 'hard caps'), ('B', 'soft limits'), ('C', 'caps ×4'), ('D', 'no storage limit'), ('E', '+ weak cost convexity'),
         ('H', '+ no temperature floor'), ('F', 'steeper damages > 2°C'), ('J', 'steeper damages > 1.5°C')]
yy = np.arange(len(vars_))[::-1]
for j, (c, col) in enumerate((('pub', C1), ('d2', C2))):
    vals = [NUM.get(f'{v}_{c}', {}).get('prem_pct', np.nan) for v, _ in vars_]
    ax.barh(yy + (0.2 if j == 0 else -0.2), vals, height=0.36, color=col, edgecolor='white', linewidth=0.6, label=CALp[c])
    for y_, v in zip(yy, vals):
        ax.text(v + 0.2, y_ + (0.2 if j == 0 else -0.2), f'{v:.1f}', fontsize=LS, va='center', color=INK2)
ax.set_yticks(yy); ax.set_yticklabels([l for _, l in vars_], fontsize=7.5); ax.set_xlabel('Premium over the expected-capacity SCC (%), p = 0.5')
ax.grid(axis='y', visible=False); ax.legend(frameon=False, loc='lower right'); ax.set_xlim(0, 17)
fig.tight_layout(); fig.savefig(os.path.join(A, 'fig_p3e_variants.pdf'), bbox_inches='tight'); plt.close(fig)

# ---------- Figure 5: the two channels ----------
fig, axs = two(0.40, 0.56)
ax = axs[0]
s4 = [0.25, 0.5, 0.75, 1.0]
ax.plot([0, 1.0], [0, 1.0], color=INK2, lw=0.5, ls=':'); ax.text(0.05, 0.16, 'proportional use', fontsize=LS, color=INK2, ha='left', rotation=0)
for v, col, mm, lab, xy in (('A', C1, 'o', 'hard caps', (0.66, 0.44)), ('D', C2, 's', 'no storage limit', (0.30, 0.60))):
    use = [UT[f'{v}_pub|{s}']['R_2200'] / UT[f'{v}_pub|1.0']['cap_2200'] for s in s4]
    ax.plot(s4, use, marker=mm, ms=4.5, color=col, mfc=col if v == 'A' else 'white', lw=1.6)
    ax.text(*xy, lab, fontsize=LS, color=INK, ha='left')
ax.annotate('storage saturation', xy=(0.88, 0.56), xytext=(0.55, 0.80), fontsize=LS, color=INK,
            arrowprops=dict(arrowstyle='->', color=INK2, lw=0.6))
ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.08); ax.set_xlabel('Removal capacity (share of central)'); ax.set_ylabel('Removal used to 2200\n(share of use at full capacity)')
title(ax, '(a) Storage, published calibration')
ax = axs[1]
LB = {('D', 'd2'): (0.08, 650), ('D', 'pub'): (0.99, 495), ('A', 'd2'): (0.70, 290)}
for v, c, col, mm, lab in (('D', 'd2', C2, 's', '2%, no storage limit'), ('A', 'd2', C3, '^', '2%, hard caps'), ('D', 'pub', C1, 'o', 'published, no storage limit')):
    ab = np.array([sum(MC[f'{v}_{c}']['curve'][str(s)]['ABT'][:17]) * 5 for s in ss]); y = ab[0] - ab
    ax.plot(ss, y, marker=mm, ms=4.5, color=col, mfc='white' if mm == 's' else col, lw=1.6)
    ax.plot([0, 1], [0, y[-1]], color=col, lw=0.7, ls=':')                     # proportional response
    ax.text(*LB[(v, c)], lab, fontsize=LS, color=INK, ha='right' if (v, c) == ('D', 'pub') else 'left')
ax.text(0.03, 900, 'dotted: response proportional\nto capacity', fontsize=LS, color=INK2, va='top')
ax.set_xlim(0, 1.05); ax.set_xlabel('Removal capacity (share of central)'); ax.set_ylabel('Abatement given up to 2100 (GtCO$_2$)')
title(ax, '(b) Displacement of abatement')
fig.tight_layout(); fig.savefig(os.path.join(A, 'fig_p3e_channels.pdf'), bbox_inches='tight'); plt.close(fig)

# ---------- Figure: hedging under uncertain removal (learning in 2070, reduced-space solutions) ----------
# (a) abatement before learning as a weight between the plans; (b, c) value of hedging in trillions, one panel per
# calibration because the scales differ by two orders of magnitude.
if CCE:
    fig = plt.figure(figsize=(W, W * 1.12)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    axs = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
else:
    fig, axs = plt.subplots(1, 3, figsize=(hs.DOUBLE, hs.DOUBLE * 0.37), gridspec_kw={'width_ratios': [1, 1, 1]})
PS = (0.1, 0.25, 0.5, 0.75, 0.9)
VAL = {}
for c, col, mm, ls in (('pub', C1, 'o', '-'), ('d3', C3, '^', '-.'), ('d2', C2, 's', '--')):
    k = f'U1_{c}_2070'; v = ST[k]; lam = R3[f'{c}_none']['lamC2025'] / 5
    aO = 5 * float(np.sum(np.array(v['optimist']['S']['ABT'])[:10])); aP = 5 * float(np.sum(np.array(v['pessimist']['S']['ABT'])[:10]))
    wts, vo, vp, pp = [], [], [], []
    for p in PS:
        cands = [r for tg, r in RD.items() if tg.startswith(f'{k}|{p}|')]
        if not cands:
            continue
        best = max(cands, key=lambda r: r['EU'])
        eo = p * v['optimist']['S']['U'] + (1 - p) * v['optimist']['F']['U']; ep = p * v['pessimist']['S']['U'] + (1 - p) * v['pessimist']['F']['U']
        pp.append(p); wts.append((best['abt_to_res'] - aP) / (aO - aP)); vo.append((best['EU'] - eo) / lam); vp.append((best['EU'] - ep) / lam)
    VAL[c] = (pp, vo, vp)
    axs[0].plot(pp, wts, marker=mm, ms=4.5, color=col, mfc='white' if c == 'd2' else col, ls=ls, lw=1.5)
axs[0].plot([0, 1], [0, 1], color=INK2, lw=0.6, ls=':'); axs[0].text(0.72, 0.60, 'weight = p', fontsize=LS, color=INK2)
axs[0].set_xlim(0, 1); axs[0].set_ylim(0, 1); axs[0].set_xlabel('Probability removal succeeds, p')
axs[0].set_ylabel('Weight on the optimist plan\n(optimist = 1, pessimist = 0)')
title(axs[0], '(a) Abatement before learning')
for ax, c, col, tag in ((axs[1], 'pub', C1, 'b'), (axs[2], 'd2', C2, 'c')):
    pp, vo, vp = VAL[c]
    mk_ = 'o' if c == 'pub' else 's'
    ax.plot(pp, vo, marker=mk_, ms=4.5, color=col, ls='-', lw=1.6)
    ax.plot(pp, vp, marker=mk_, ms=4.5, color=col, mfc='white', ls='--', lw=1.4)
    ax.annotate('vs optimist', (pp[0], vo[0]), xytext=(6, 0), textcoords='offset points', fontsize=LS, color=INK, va='center')
    ax.annotate('vs pessimist', (pp[-1], vp[-1]), xytext=(-6, 0), textcoords='offset points', fontsize=LS, color=INK, va='center', ha='right')
    ax.set_xlim(0, 1); ax.set_ylim(0, max(max(vo), max(vp)) * 1.12); ax.set_xlabel('Probability removal succeeds, p')
    ax.set_ylabel('Value of hedging (trillion US\\$)')
    title(ax, f'({tag}) {CALp[c]}')
legend_below(fig, [Line2D([], [], color=C1, marker='o', ls='-', lw=1.5, ms=5, label='published calibration'),
                   Line2D([], [], color=C3, marker='^', ls='-.', lw=1.5, ms=5, label='3% discount'),
                   Line2D([], [], color=C2, marker='s', mfc='white', ls='--', lw=1.5, ms=5, label='2% discount'),
                   Line2D([], [], color=INK2, marker='o', ls='-', lw=1.4, ms=5, label='(b, c) vs optimist: solid, filled'),
                   Line2D([], [], color=INK2, marker='o', mfc='white', ls='--', lw=1.4, ms=5, label='(b, c) vs pessimist: dashed, open')], 3 if not CCE else 2, y=0.0)
fig.tight_layout(); fig.savefig(os.path.join(A, 'fig_p3e_hedge.pdf'), bbox_inches='tight'); plt.close(fig)
for c in VAL:
    print(c, [f'{a:.3f}/{b:.3f}' for a, b in zip(VAL[c][1], VAL[c][2])])

# ---------- Figure 6 (appendix): damage functions used in the curvature tests ----------
a2 = 0.003467; T = np.linspace(0, 4, 400)
specs = [('DICE-2023, $a_2T^2$', a2 * T ** 2, 2 * a2 * T, INK2, (0, (5, 2))),
         ('steeper above 2°C', a2 * T ** 2 + 2.25 * a2 * np.maximum(T - 2, 0) ** 3, 2 * a2 * T + 6.75 * a2 * np.maximum(T - 2, 0) ** 2, C1, '-'),
         ('steeper above 1.5°C', a2 * T ** 2 + 2.25 * a2 * np.maximum(T - 1.5, 0) ** 3, 2 * a2 * T + 6.75 * a2 * np.maximum(T - 1.5, 0) ** 2, C2, (0, (6, 2, 1.2, 2))),
         ('cubic, level-matched at 2.5°C', a2 / 2.5 * T ** 3, 3 * a2 / 2.5 * T ** 2, C3, (0, (1.2, 1.6)))]
fig, axs = two(0.38, 0.52)
for lab, dm, md, col, ls in specs:
    axs[0].plot(T, 100 * dm, color=col, ls=ls, lw=1.5); axs[1].plot(T, 100 * md, color=col, ls=ls, lw=1.5, label=lab)
for ax in axs:
    ax.axvspan(1.8, 2.6, color=GRID, alpha=0.55, lw=0)
    for tc in (1.5, 2.0):
        ax.axvline(tc, color=INK2, lw=0.5, ls=':')
    ax.set_xlabel('Warming (°C)'); ax.set_xlim(0, 4)
    yt = ax.get_ylim()[1]
    ax.text(2.2, yt * 0.97, 'peak warming\nin the runs', ha='center', va='top', fontsize=LS - 0.4, color=INK2)
    ax.text(1.5, yt * 0.55, '1.5°C', ha='right', fontsize=LS - 0.4, color=INK2, rotation=90, va='center')
    ax.text(2.0, yt * 0.55, '2°C', ha='right', fontsize=LS - 0.4, color=INK2, rotation=90, va='center')
axs[0].set_ylabel('Damages (% of output)'); axs[1].set_ylabel('Marginal damages (% of output per °C)')
title(axs[0], '(a) Damage level'); title(axs[1], '(b) Marginal damages')
h, l = axs[1].get_legend_handles_labels(); legend_below(fig, h, 4, y=0.0)
fig.tight_layout(); fig.savefig(os.path.join(A, 'fig_p3e_damage.pdf'), bbox_inches='tight'); plt.close(fig)
print('figures written', 'BW' if BW else 'colour')
