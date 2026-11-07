"""Table fragments for the v3 additions: FaIR benchmark, eta sweep, mechanism channels, Monte Carlo v2.
Writes tex/tab_fair_benchmark.tex, tex/tab_eta_sweep.tex, tex/tab_channels.tex, tex/tab_mc2_dist.tex,
tex/tab_mc2.tex and results/mc2_summary.json (if the Monte Carlo results exist)."""
import os, glob, json
import numpy as np
from common import HERE, RES, load_json, save_json

TEX = os.path.join(HERE, 'tex')


def w(name, s):
    open(os.path.join(TEX, name), 'w').write(s)


def fair():
    d = load_json('fair_benchmark.json')
    rows = [('dice', 'DFAIR as in DICE-2023 (5-year steps, exact $\\alpha$)', '32.4', '0.019', '4.165', '0'),
            ('dfair_closure', 'DICE parameters, FaIR closure, 1-year steps', '32.4', '0.019', '4.165', '0'),
            ('fair_rU', '\\quad and FaIR v2 uptake sensitivity', '32.4', '0.031', '4.165', '0'),
            ('fair_noA', '\\quad and FaIR v2 $\\sigma_0$, $\\sigma_T$', '29.0', '0.031', '4.0', '0'),
            ('fair_ar6', 'FaIR v2, AR6 calibration', '29.0', '0.031', '4.0', '0.0030')]
    # sigma_C and sigma_A per GtC (FaIR v2: 0.00846 and 0.000819 per GtCO2)
    L = []
    for k, lab, r0, rU, rT, rA in rows:
        v = d['variants'][k]
        sh = ' & '.join(f"{100 * s:.1f}" for s in v['share'])
        L.append(f"{lab} & {r0} & {rU} & {rT} & {rA} & {v['scc'][0]:.2f} & {sh} \\\\")
    s = ("\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n & \\multicolumn{4}{c}{IRF target parameters} & SCC & \\multicolumn{4}{c}{Feedback share (\\%), pulse in} \\\\\n"
         "\\cmidrule(lr){2-5}\\cmidrule(lr){7-10}\nCarbon cycle & $\\sigma_0$ & $\\sigma_C$ & $\\sigma_T$ & $\\sigma_A$ & 2025 & 2025 & 2050 & 2100 & 2150 \\\\\n\\midrule\n"
         + '\n'.join(L) + "\n\\bottomrule\n\\end{tabular}\n")
    w('tab_fair_benchmark.tex', s)


def eta():
    d = load_json('eta_sweep.json')
    by = {(r['precaution'], r['elasmu']): r for r in d}
    etas = sorted({r['elasmu'] for r in d})
    L = []
    for e in etas:
        a, b = by[('dice', e)], by[('fixed', e)]
        L.append(f"{e:.2f} & {100 * a['rprecaut_longrun']:.2f} & {a['scc_2025']:.0f} & {a['peak_T']:.2f} & {100 * a['share_2025']:.1f} & "
                 f"{b['scc_2025']:.0f} & {b['peak_T']:.2f} & {100 * b['share_2025']:.1f} \\\\")
    s = ("\\begin{tabular}{rrrrrrrr}\n\\toprule\n & Precautionary & \\multicolumn{3}{c}{As in DICE-2023} & \\multicolumn{3}{c}{Precautionary term at $\\varphi = 0.95$} \\\\\n"
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n$\\varphi$ & rate (\\%/yr) & SCC 2025 & Peak $T$ & Share (\\%) & SCC 2025 & Peak $T$ & Share (\\%) \\\\\n\\midrule\n"
         + '\n'.join(L) + "\n\\bottomrule\n\\end{tabular}\n")
    w('tab_eta_sweep.tex', s)


def channels():
    d = load_json('formula_check.json')['rows']
    L = []
    for r in d:
        if r['year'] in (2030, 2050, 2075, 2100, 2125, 2150, 2175, 2200):
            sU = r['dalpha_uptake'] / r['dalpha']
            loc = r['local_inflow'] + r['local_stock']
            L.append(f"{r['year']} & {1e3 * r['dalpha']:.2f} & {100 * sU:.0f} & {r['Phi']:.2f} & {r['R_fast']:.1f} & {1e3 * loc:+.2f} & "
                     f"{r['dir_MAT']:.3f} & {r['fb_MAT']:+.3f} \\\\")
    s = ("\\begin{tabular}{rrrrrrrr}\n\\toprule\n & $d\\alpha$ & Uptake & $\\Phi$ & Fast & Local effect & \\multicolumn{2}{c}{$d\\mat$ (GtC)} \\\\\n"
         "\\cmidrule(lr){7-8}\nYear & ($10^{-3}$) & channel (\\%) & (GtC/yr) & boxes (GtC) & ($10^{-3}$ GtC) & direct & feedback \\\\\n\\midrule\n"
         + '\n'.join(L) + "\n\\bottomrule\n\\end{tabular}\n")
    w('tab_channels.tex', s)


def mc2():
    files = sorted(glob.glob(os.path.join(RES, 'mc2_ipopt', '*.json')))   # primary: IPOPT + multiplier SCC
    if not files:
        return
    R = {}
    for f in files:
        for o in json.load(open(f)):
            R[o['draw']] = o
    S = {}                                                                   # cross-check: SLSQP
    for f in sorted(glob.glob(os.path.join(RES, 'mc2', '*.json'))):
        for o in json.load(open(f)):
            S[o['draw']] = o
    import mc_v2 as V
    raw_status = {str(s_): sum(R[k].get('status') == s_ for k in R) for s_ in (0, 1, -1, -2, -3, -4)}
    # primary: IPOPT where it converged (status 0/1); SLSQP where IPOPT did not converge and SLSQP did.
    # Flagged IPOPT draws (IFT total vs multiplier > 0.1%) are kept: a late state bound (TATM >= 0.01 or the
    # carbon floor) is active, and the share is the fixed-policy decomposition (Proposition 1).
    conv = {k: R[k] for k in R if R[k].get('status') in (0, 1)}
    fallback = {k: dict(S[k], src='slsqp') for k in R if k not in conv and k in S and S[k].get('success')}
    n_flag_conv = sum(bool(conv[k].get('flag')) for k in conv)
    gap = np.array([abs(conv[k]['ift_vs_mult_pct_2025']) for k in conv])
    R = {**conv, **{k: v for k, v in fallback.items()}}
    for k in conv:
        R[k]['success'] = True
    both = [k for k in conv if k in S and S[k].get('success')]
    ok = [R[k] for k in sorted(R)]
    P = np.array([[o['params'][k] for k in V.KEYS] for o in ok])
    dscc = np.array([100 * (S[k]['scc_2025'] / R[k]['scc_2025'] - 1) for k in both])
    dsh = np.array([100 * (S[k]['share_2025'] - R[k]['share_2025']) for k in both])
    out = {'n': len(R), 'n_success': len(ok), 'n_ipopt': len(conv), 'n_slsqp_fallback': len(fallback), 'n_flag_bound_active': n_flag_conv,
           'gap_2025_pct_max': float(gap.max()), 'gap_2025_pct_p99': float(np.percentile(gap, 99)),
           'ipopt_status': raw_status,
           'slsqp_vs_ipopt': {'n': len(both), 'scc_absdiff_pct_p50': float(np.median(np.abs(dscc))), 'scc_absdiff_pct_p99': float(np.percentile(np.abs(dscc), 99)),
                              'n_scc_over_0p1pct': int(np.sum(np.abs(dscc) > 0.1)), 'share_absdiff_pp_max': float(np.max(np.abs(dsh))),
                              'n_share_over_0p1pp': int(np.sum(np.abs(dsh) > 0.1))},
 'dice_percentile': {k: float(V.CDF[k](V.DICE[k])) for k in V.KEYS}}
    for y in (2025, 2050, 2100):
        sh = np.array([100 * o[f'share_{y}'] for o in ok]); scc = np.array([o[f'scc_{y}'] for o in ok])
        X = np.c_[np.ones(len(sh)), (P - P.mean(0)) / P.std(0)]
        b = np.linalg.lstsq(X, sh, rcond=None)[0]
        bl = np.linalg.lstsq(X, np.log(scc), rcond=None)[0]
        # first-order sensitivity (correlation ratio) by deciles of each parameter
        S1 = {}
        for j, k in enumerate(V.KEYS):
            q = np.quantile(P[:, j], np.linspace(0, 1, 11)); g = np.clip(np.searchsorted(q, P[:, j], side='right') - 1, 0, 9)
            S1[k] = float(sum(np.sum(g == c) * (sh[g == c].mean() - sh.mean()) ** 2 for c in range(10) if np.any(g == c)) / (len(sh) * sh.var()))
        out[str(y)] = {'coef_pp_per_sd': dict(zip(V.KEYS, b[1:].tolist())), 'R2': float(1 - np.var(sh - X @ b) / np.var(sh)),
                       'S1': S1, 'p5': float(np.percentile(sh, 5)), 'p50': float(np.median(sh)), 'p95': float(np.percentile(sh, 95)),
                       'share_negative': float(np.mean(sh < 0)), 'dice_share_percentile': None,
                       'scc_p5': float(np.percentile(scc, 5)), 'scc_p50': float(np.median(scc)), 'scc_p95': float(np.percentile(scc, 95)),
                       'logscc_coef_per_sd': dict(zip(V.KEYS, bl[1:].tolist())),
                       'fb_p5': float(np.percentile(np.array([o[f'feedback_{y}'] for o in ok]), 5)),
                       'fb_p50': float(np.median([o[f'feedback_{y}'] for o in ok])),
                       'fb_p95': float(np.percentile(np.array([o[f'feedback_{y}'] for o in ok]), 95))}
    sh25 = np.array([100 * o['share_2025'] for o in ok])
    out['2025']['dice_share_percentile'] = float(np.mean(sh25 < 13.04))
    save_json(out, 'mc2_summary.json')
    lab = {'rho': '$\\rho$', 'elasmu': '$\\varphi$', 'ecs': 'ECS', 'psi2': '$\\psi_2$', 'a3': 'Damage exponent',
           'irC': '$\\sigma_C$', 'irT': '$\\sigma_T$', 'eland0': '$E_{\\mathrm{land}}(0)$'}
    L = []
    for k in V.KEYS:
        L.append(f"{lab[k]} & " + ' & '.join(f"{out[str(y)]['coef_pp_per_sd'][k]:+.2f} & {out[str(y)]['S1'][k]:.2f}" for y in (2025, 2100)) + " \\\\")
    L.append('\\midrule')
    L.append("$R^2$ (linear) & \\multicolumn{2}{c}{" + f"{out['2025']['R2']:.2f}" + "} & \\multicolumn{2}{c}{" + f"{out['2100']['R2']:.2f}" + "} \\\\")
    L.append("Median share (\\%) & \\multicolumn{2}{c}{" + f"{out['2025']['p50']:.1f}" + "} & \\multicolumn{2}{c}{" + f"{out['2100']['p50']:.1f}" + "} \\\\")
    L.append("5th--95th percentile & \\multicolumn{2}{c}{" + f"{out['2025']['p5']:.1f} to {out['2025']['p95']:.1f}" + "} & \\multicolumn{2}{c}{" + f"{out['2100']['p5']:.1f} to {out['2100']['p95']:.1f}" + "} \\\\")
    L.append("Draws with negative share (\\%) & \\multicolumn{2}{c}{" + f"{100 * out['2025']['share_negative']:.1f}" + "} & \\multicolumn{2}{c}{" + f"{100 * out['2100']['share_negative']:.1f}" + "} \\\\")
    s = ("\\begin{tabular}{lrrrr}\n\\toprule\n & \\multicolumn{2}{c}{Pulse in 2025} & \\multicolumn{2}{c}{Pulse in 2100} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
         "Parameter & Coef. (pp/s.d.) & $S_1$ & Coef. (pp/s.d.) & $S_1$ \\\\\n\\midrule\n" + '\n'.join(L) + "\n\\bottomrule\n\\end{tabular}\n")
    w('tab_mc2.tex', s)
    D = [('$\\rho$', 'Pure rate of time preference', 'Lognormal, median 0.5\\%, mean 1.1\\%; $[0.01\\%, 8\\%]$', 'Expert survey \\citep{Drupp2018}', '0.1\\%'),
         ('$\\varphi$', 'Elasticity of marginal utility', 'Lognormal, median 1.0, mean 1.35; $[0.3, 1.5]$', 'Expert survey \\citep{Drupp2018}', '0.95'),
         ('ECS', 'Equilibrium climate sensitivity', 'Lognormal, median 3.0\\,\\textdegree C, 90\\%: 1.9--4.7', 'AR6 very likely 2--5 \\citep{Forster2021}', '3.0'),
         ('$\\psi_2$', 'Damage coefficient (at exponent 2)', 'Lognormal, median at DICE, 90\\%: $\\times 0.5$ to $\\times 2$', 'Assumption', '0.003467'),
         ('$a_3$', 'Damage exponent', 'Uniform $[1.5, 3.0]$, damages at 3\\,\\textdegree C held', 'Assumption', '2'),
         ('$\\sigma_C$', 'IRF sensitivity to uptake', 'Uniform $[0.010, 0.030]$ yr/GtC', 'About $\\pm$50\\%', '0.019'),
         ('$\\sigma_T$', 'IRF sensitivity to temperature', 'Uniform $[2.5, 6.0]$ yr/K', 'About $\\pm$45\\%', '4.165'),
         ('$E_{\\mathrm{land}}(0)$', 'Land-use CO$_2$, 2020', 'Normal, mean 5.9, s.d. 2.6 GtCO$_2$/yr; $[0.5, 12]$', 'GCB 2023 \\citep{Friedlingstein2023}', '5.9')]
    L = [f"{a} & {b} & {c} & {d_} & {e} & {100 * out['dice_percentile'][k]:.0f} \\\\" for (a, b, c, d_, e), k in zip(D, V.KEYS)]
    s = ("\\begin{tabular}{llp{0.30\\textwidth}lrr}\n\\toprule\n & Parameter & Distribution & Source & DICE & DICE pct. \\\\\n\\midrule\n"
         + '\n'.join(L) + "\n\\bottomrule\n\\end{tabular}\n")
    w('tab_mc2_dist.tex', s)
    print(json.dumps({y: {k: out[y][k] for k in ('p5', 'p50', 'p95', 'R2', 'share_negative', 'scc_p50')} for y in ('2025', '2100')}, indent=1))
    print('S1 2025', {k: round(v, 3) for k, v in out['2025']['S1'].items()})
    print('coef 2025', {k: round(v, 2) for k, v in out['2025']['coef_pp_per_sd'].items()})
    print('n', out['n'], out['n_success'], 'DICE share pct', out['2025']['dice_share_percentile'])


def ecs():
    d = load_json('ecs_mechanism.json')
    lab = {'ecs': 'ECS (\\textdegree C)', 'psi2': '$\\psi_2$'}
 
    L = []; seen = set()
    for r in d:
        v = f"{r['value']:.1f}" if r['param'] == 'ecs' else f"{r['value']:.4f}"
        a, b = r['fixed_policy'], r['reopt']
        name = '' if r['param'] in seen else lab[r['param']]; seen.add(r['param'])
        L.append(f"{name} & {v} & {a['scc_2025']:.0f} & {100 * a['share_2025']:.1f} & {b['scc_2025']:.0f} & {100 * b['share_2025']:.1f} & {b['peak_T']:.2f} & {b['first_net_negative_year']} \\\\")
        if r is d[4]:
            L.append('\\midrule')
    s = ("\\begin{tabular}{lrrrrrrr}\n\\toprule\n & & \\multicolumn{2}{c}{Policy fixed at DICE-2023 optimum} & \\multicolumn{4}{c}{Policy re-optimized} \\\\\n"
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-8}\nParameter & Value & SCC 2025 & Share (\\%) & SCC 2025 & Share (\\%) & Peak $T$ & Net-negative from \\\\\n\\midrule\n"
         + '\n'.join(L) + "\n\\bottomrule\n\\end{tabular}\n")
    w('tab_ecs_mechanism.tex', s)


def fair_opt():
    d = load_json('fair_opt.json'); b = load_json('fair_benchmark.json')['variants']['dice']
    lab = {'dfair_closure': 'DICE parameters, FaIR closure', 'fair_ar6': 'FaIR v2, AR6 calibration'}
    L = [f"DFAIR as in DICE-2023 & {b['scc'][0]:.2f} & {100 * b['share'][0]:.1f} & {b['scc'][0]:.2f} & {100 * b['share'][0]:.1f} & 2.60 \\\\"]
    for k in ('dfair_closure', 'fair_ar6'):
        v = d[k]
        L.append(f"{lab[k]} & {v['fixed_policy']['scc']['2025']:.2f} & {100 * v['fixed_policy']['share']['2025']:.1f} & "
                 f"{v['reopt']['scc']['2025']:.2f} & {100 * v['reopt']['share']['2025']:.1f} & {v['reopt']['peak_T']:.2f} \\\\")
    s = ("\\begin{tabular}{lrrrrr}\n\\toprule\n & \\multicolumn{2}{c}{DICE-2023 policy} & \\multicolumn{3}{c}{Re-optimized policy} \\\\\n"
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\nCarbon cycle & SCC 2025 & Share (\\%) & SCC 2025 & Share (\\%) & Peak $T$ \\\\\n\\midrule\n"
         + '\n'.join(L) + "\n\\bottomrule\n\\end{tabular}\n")
    w('tab_fair_opt.tex', s)


if __name__ == '__main__':
    fair(); eta(); channels(); ecs(); fair_opt(); mc2()
