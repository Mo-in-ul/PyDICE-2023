"""Write numbers.tex (macros) and LaTeX table fragments from results/*.json.
Run after run_validation.py, run_faithful.py and scc_gams_controls.py."""
import os, glob
import numpy as np
from common import load_json, save_json, RES, HERE

OUT = os.path.join(HERE, 'tex'); os.makedirs(OUT, exist_ok=True)
NAMES = {1: '1\\% discounting', 2: '2\\% discounting', 3: '3\\% discounting', 4: '4\\% discounting',
         5: '5\\% discounting', 6: '1.5\\textdegree{}C limit', 7: '2\\textdegree{}C limit', 8: 'Paris',
         9: 'Cost-benefit optimal', 10: 'Baseline'}
ROMAN = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten'}
macros = []
def mac(name, val):
    macros.append(f'\\newcommand{{\\{name}}}{{{val}}}')

def exit_label(js):
    if js.get('meta', {}).get('infeasible') or js.get('infeasible'):
        return 'best effort'
    if 'attempts' in js:
        st = js['final_status']; n = len(js['attempts'])
        return ('0' if st == 0 else str(st)) + (f' ({n} passes)' if n > 1 else '')
    m = js['meta']['message']
    return '0' if js['meta']['success'] else ('8' if 'directional' in m else m[:12])

def validation_table(prefix, fname, caption_key):
    rows = []
    for sc in range(1, 11):
        f = os.path.join(RES, f'{prefix}_scen{sc}.json')
        if not os.path.exists(f):
            continue
        js = load_json(os.path.basename(f))
        gap, d1, d2 = js['welfare_gap_pct'], js['max_dT_to_2100'], js['max_dT_to_2215']
        sci = lambda v: '0' if abs(v) < 5e-13 else f"{v:.0e}".replace('e-0', 'e-').replace('e+0', 'e+')
        rows.append(f"{NAMES[sc]} & {sci(gap)} & {sci(d1)} & {sci(d2)} & {js['peak_T_pydice']:.3f} & {js['peak_T_gams']:.3f} & {exit_label(js)} \\\\")
        R = ROMAN[sc]
        mac(f'{caption_key}Gap{R}', f'{gap:+.4f}')
        mac(f'{caption_key}dTa{R}', f'{d1:.4f}')
        mac(f'{caption_key}dTb{R}', f'{d2:.4f}')
    body = '\n'.join(rows)
    tex = ("\\begin{tabular}{lrrrrrl}\n\\toprule\n"
           "Scenario & Welfare gap (\\%) & $\\max|\\Delta T|$ to 2100 & $\\max|\\Delta T|$ to 2215 & Peak $T$ PyDICE & Peak $T$ GAMS & Exit mode \\\\\n"
           "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(OUT, fname), 'w').write(tex)

def scc_table():
    js = load_json('scc_gams_controls.json')
    rows = []
    for sc in ['4', '5', '8', '9', '10', '3', '2', '1', '7', '6']:
        o = js[sc]; yrs = o['years']
        cells = []
        for y in (2025, 2050, 2100):
            j = yrs.index(y)
            cells += [f"{o['gams'][j]:.1f}", f"{o['resp'][j]:.1f}", f"{o['fixed'][j]:.1f}", f"{100*o['share'][j]:.1f}"]
        rows.append(NAMES[int(sc)] + ' & ' + ' & '.join(cells) + ' \\\\')
        if sc in ('5', '9'):
            pass
        R = ROMAN[int(sc)]
        for y in (2025, 2050, 2100):
            j = yrs.index(y); Y = {2025: 'A', 2050: 'B', 2100: 'C'}[y]
            mac(f'sccGams{R}{Y}', f"{o['gams'][j]:.1f}")
            mac(f'sccResp{R}{Y}', f"{o['resp'][j]:.1f}")
            mac(f'sccFixed{R}{Y}', f"{o['fixed'][j]:.1f}")
            mac(f'sccShare{R}{Y}', f"{100*o['share'][j]:.1f}")
            mac(f'sccRespGap{R}{Y}', f"{o['resp_vs_gams_pct'][j]:+.2f}")
    clean = ['4', '5', '8', '9', '10']
    sh = [100 * js[s]['share'][0] for s in clean]
    mac('sccShareMinA', f'{min(sh):.0f}'); mac('sccShareMaxA', f'{max(sh):.0f}')
    rg = [abs(js[s]['resp_vs_gams_pct'][js[s]['years'].index(y)]) for s in clean for y in (2025, 2050, 2100)]
    mac('sccRespMaxAbsGap', f'{max(rg):.2f}')
    tex = ("\\begin{tabular}{l" + "rrrr" * 3 + "}\n\\toprule\n"
           " & \\multicolumn{4}{c}{2025} & \\multicolumn{4}{c}{2050} & \\multicolumn{4}{c}{2100} \\\\\n"
           "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}\n"
           "Scenario" + " & GAMS & Resp. & Fixed & Share" * 3 + " \\\\\n\\midrule\n"
           + '\n'.join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(OUT, 'tab_scc.tex'), 'w').write(tex)


def _get(name):
    p = os.path.join(RES, name)
    return load_json(name) if os.path.exists(p) else None


def scc_checks_table(year):
    """All SCC methods side by side at one year (GAMS controls unless noted)."""
    fd, ift, mult, pipe = _get('scc_gams_controls.json'), _get('check3_ift.json'), _get('check5_multiplier_scc.json'), _get('check1_pipeline.json')
    if not all((fd, ift, mult)):
        return
    rows = []
    for sc in ['4', '5', '8', '9', '10', '1', '2', '3', '7', '6']:
        f = fd[sc]; j = f['years'].index(year)
        k = ift[sc]['years'].index(year)
        mkey = '6d' if sc == '6' else sc
        m = mult.get(mkey, {})
        def mv(key):
            return f"{m[key][m['years'].index(year)]:.2f}" if key in m else '--'
        pv = f"{pipe[sc]['pydice'][pipe[sc]['years'].index(year)]:.2f}" if pipe and sc in pipe else '--'
        rows.append(f"{NAMES[int(sc)]}{' (disposal)' if sc == '6' else ''} & {f['gams'][j]:.2f} & {mv('scc_gams_formula')} & {mv('scc_exact')} & "
                    f"{ift[sc]['total'][k]:.2f} & {f['resp'][j]:.2f} & {pv} & {ift[sc]['direct'][k]:.2f} & {f['fixed'][j]:.2f} \\\\")
        if sc == '10':
            rows.append('\\midrule')
    tex = ("\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
           " & & \\multicolumn{2}{c}{Multiplier (IPOPT)} & \\multicolumn{3}{c}{Responsive $\\alpha$} & \\multicolumn{2}{c}{Fixed $\\alpha$} \\\\\n"
           "\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}\n"
           "Scenario & GAMS & GAMS formula & exact & IFT & pulse & own optimum & IFT & pulse \\\\\n\\midrule\n"
           + '\n'.join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(OUT, f'tab_scc_checks_{year}.tex'), 'w').write(tex)
    # summary macros
    clean = ['4', '5', '8', '9', '10']
    Y = {2025: 'A', 2050: 'B', 2100: 'C'}[year]
    dev_ift = max(abs(ift[s]['total'][ift[s]['years'].index(year)] / fd[s]['resp'][fd[s]['years'].index(year)] - 1) for s in fd)
    dev_dir = max(abs(ift[s]['direct'][ift[s]['years'].index(year)] / fd[s]['fixed'][fd[s]['years'].index(year)] - 1) for s in fd)
    mac(f'chkIftMaxDev{Y}', f'{dev_ift:.0e}'); mac(f'chkIftDirectMaxDev{Y}', f'{dev_dir:.0e}')
    mk = [s for s in mult if 'scc_gams_formula' in mult[s] and s not in ('6', '6d')]
    if mk:
        dev_m = max(abs(mult[s]['formula_vs_gams_pct'][mult[s]['years'].index(year)]) for s in mk)
        mac(f'chkMultMaxDevPct{Y}', f'{dev_m:.4f}')


def pulse_table():
    d = _get('check2_pulse_sweep.json')
    if not d:
        return
    H = d['9']['h']
    rows = []
    for lab, key in (('One-sided $s(h)$', 'relerr_raw'), ('$2s(h/2)-s(h)$', 'relerr_richardson'),
                     ('$(4s(h/2)-s(h))/3$', 'relerr_richardson_central_formula')):
        cells = [f"{max(x[key][j] for s in d for m in ('resp', 'fixed') for x in d[s][m]):.1e}" for j in range(len(H))]
        rows.append(lab + ' & ' + ' & '.join(cells) + ' \\\\')
    tex = ("\\begin{tabular}{l" + 'r' * len(H) + "}\n\\toprule\n"
           "$h$ (GtCO$_2$/yr) & " + ' & '.join(f'{h:.0e}' for h in H) + " \\\\\n\\midrule\n"
           + '\n'.join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(OUT, 'tab_pulse_sweep.tex'), 'w').write(tex)
    j = H.index(0.1)
    mac('chkPulseErrBest', f"{max(x['relerr_richardson'][j] for s in d for m in ('resp', 'fixed') for x in d[s][m]):.0e}")


def envelope_table():
    d = _get('check4_envelope.json'); fd = _get('scc_gams_controls.json')
    if not d:
        return
    rows = []
    for sc, o in d.items():
        for j, y in enumerate(o['years']):
            fc = fd[sc]['resp'][fd[sc]['years'].index(y)] if fd else o['fixed_control'][j]
            rows.append(f"{NAMES[int(sc)]} & {y} & {o['gams'][j]:.3f} & {o['multiplier'][j]:.3f} & {o['reopt_richardson'][j]:.3f} & {fc:.3f} \\\\")
    tex = ("\\begin{tabular}{lrrrrr}\n\\toprule\n"
           "Scenario & Year & GAMS & Multiplier & Re-optimized & Fixed controls \\\\\n\\midrule\n"
           + '\n'.join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(OUT, 'tab_envelope.tex'), 'w').write(tex)


def tlo_table():
    d = _get('check6_tlo_effect.json')
    if not d:
        return
    rows = []
    for sc, o in d.items():
        rows.append(f"{NAMES[int(sc)]} & " + ' & '.join(f"{a:.2f} & {b:.2f}" for a, b in zip(o['with_bound'], o['without_bound'])) + ' \\\\')
    tex = ("\\begin{tabular}{lrrrrrr}\n\\toprule\n"
           " & \\multicolumn{2}{c}{2025} & \\multicolumn{2}{c}{2050} & \\multicolumn{2}{c}{2100} \\\\\n"
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
           "Scenario & $T\\ge 0.5$ & $T\\ge 0.01$ & $T\\ge 0.5$ & $T\\ge 0.01$ & $T\\ge 0.5$ & $T\\ge 0.01$ \\\\\n\\midrule\n"
           + '\n'.join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(OUT, 'tab_tlo_effect.tex'), 'w').write(tex)



STEP_LABELS = {'submitted model': 'Submitted model', 'land': '+ control rate on land CO$_2$',
               'ccatot': '+ lagged cumulative emissions', 'fghg': '+ lagged other-GHG forcing',
               'savings': '+ savings bounds', 'matfloor': '+ carbon floor 10 GtC',
               'tlo': '+ temperature floor 0.5\\,\\textdegree C', 'carry': '+ carried-over bounds',
               'miu1': '+ base MIU(2020)'}


def decomposition_table():
    d = {}
    for f in sorted(glob.glob(os.path.join(RES, 'decomp_*.json'))):
        d.update(load_json(os.path.basename(f)))
    if not d:
        return
    order = [s for s in ['9', '4', '5', '8', '10', '7', '2', '3', '1'] if s in d]
    rows = []
    nsteps = max(len(d[s]['cumulative']) for s in order)
    for k in range(nsteps):
        lab = None; cells = []
        for s in order:
            c = d[s]['cumulative']
            if k < len(c) and 'error' not in c[k]:
                lab = c[k]['added']
                cells.append(f"{c[k]['dT2215']:.4f} / {abs(c[k]['sccA_gap_pct']):.2f}" + ('$^*$' if c[k]['status'] not in (0, 1) else ''))
            else:
                cells.append('--')
        rows.append(STEP_LABELS.get(lab, lab or '') + ' & ' + ' & '.join(cells) + ' \\\\')
    tex = ("\\begin{tabular}{l" + 'r' * len(order) + "}\n\\toprule\n"
           " & " + ' & '.join(NAMES[int(s)] for s in order) + " \\\\\n\\midrule\n" + '\n'.join(rows) +
           "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(OUT, 'tab_decomposition.tex'), 'w').write(tex)
    # leave-one-out where available: max over scenarios of dT2215 with each correction removed
    loo = [s for s in order if d[s]['leave_one_out']]
    if loo:
        rows = []
        for j, step in enumerate(STEPS_ORDER):
            vals = [d[s]['leave_one_out'][j] for s in loo if j < len(d[s]['leave_one_out'])]
            rows.append(STEP_LABELS[step].replace('+ ', '$-$ ') + ' & ' + ' & '.join(f"{v['dT2215']:.4f} / {abs(v['sccA_gap_pct']):.2f}" for v in vals) + ' \\\\')
        tex = ("\\begin{tabular}{l" + 'r' * len(loo) + "}\n\\toprule\n & " + ' & '.join(NAMES[int(s)] for s in loo) +
               " \\\\\n\\midrule\n" + '\n'.join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
        open(os.path.join(OUT, 'tab_decomposition_loo.tex'), 'w').write(tex)


STEPS_ORDER = ['land', 'ccatot', 'fghg', 'savings', 'matfloor', 'tlo', 'carry', 'miu1']


def policy_table():
    d = _get('policy_fixed_alpha.json')
    if not d:
        return
    o9, o4 = d.get('9'), d.get('4')
    def row(lab, key, fmt):
        return lab + ' & ' + ' & '.join(fmt(o) for o in (o9, o4) if o) + ' \\\\'
    rows = [row('Carbon price 2030 (\\%)', 'p30', lambda o: f"{o['price_gap_2030_pct']:+.1f}"),
            row('Carbon price 2050 (\\%)', 'p50', lambda o: f"{o['price_gap_2050_pct']:+.1f}"),
            row('Peak warming (\\textdegree C)', 'dT', lambda o: f"{o['dT_peak']:+.3f}"),
            row('Welfare, consumption equivalent (\\%)', 'ce', lambda o: f"{o['feedback_ignored']['ce_pct']:+.4f}")]
    tex = ("\\begin{tabular}{lrr}\n\\toprule\n & Cost-benefit optimal & 4\\% discounting \\\\\n\\midrule\n" + '\n'.join(rows) +
           "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(OUT, 'tab_policy_fixed_alpha.tex'), 'w').write(tex)
    mac('polPriceGapA', f"{abs(o9['price_gap_2030_pct']):.1f}"); mac('polDTpeak', f"{o9['dT_peak']:.3f}")
    mac('polCE', f"{abs(o9['feedback_ignored']['ce_pct']):.4f}")


def onepointfive_table():
    d = _get('onepointfive.json')
    if not d:
        return
    runs = [o for o in d['runs'] if 'error' not in o and o['status'] in (0, 1)]
    rows = []
    for cap in sorted({o['cap'] for o in runs}):
        a = next((o for o in runs if o['cap'] == cap and o['disposal']), None)
        b = next((o for o in runs if o['cap'] == cap and not o['disposal']), None)
        fmt = lambda o: f"{o['U']:,.1f} & {o['cpc2025']:.2f} & {o['scc2025']:,.0f}" if o else '-- & -- & --'
        rows.append(f"{cap:.4g} & {fmt(a)} & {fmt(b)} \\\\")
    tex = ("\\begin{tabular}{lrrrrrr}\n\\toprule\n & \\multicolumn{3}{c}{Capital can be scrapped} & \\multicolumn{3}{c}{Equality accumulation} \\\\\n"
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nCap (\\textdegree C) & Welfare & CPC 2025 & SCC 2025 & Welfare & CPC 2025 & SCC 2025 \\\\\n\\midrule\n"
           + '\n'.join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(OUT, 'tab_1p5c_cap.tex'), 'w').write(tex)


def mc_summary():
    from fig_mc import load
    runs = load()
    if not runs:
        return None
    import numpy as np
    ok = [o for o in runs if o['success']]
    S = {}
    for y in (2025, 2050, 2100):
        sh = np.array([100 * o[f'share_{y}'] for o in ok]); sc = np.array([o[f'scc_{y}'] for o in ok])
        S[y] = {'share_p5': float(np.percentile(sh, 5)), 'share_p50': float(np.median(sh)), 'share_p95': float(np.percentile(sh, 95)),
                'scc_p5': float(np.percentile(sc, 5)), 'scc_p50': float(np.median(sc)), 'scc_p95': float(np.percentile(sc, 95))}
    S['n'] = len(runs); S['n_success'] = len(ok)
    save_json(S, 'mc_summary.json')
    mac('mcN', str(len(runs))); mac('mcShareMedA', f"{S[2025]['share_p50']:.1f}")
    mac('mcShareLoA', f"{S[2025]['share_p5']:.1f}"); mac('mcShareHiA', f"{S[2025]['share_p95']:.1f}")
    return S


def timing_table():
    t = _get('timing.json')
    if not t:
        return
    ms = lambda v: (f"{1000 * v:.2g} ms" if v < 0.01 else f"{1000 * v:.0f} ms") if v < 1 else f"{v:.1f} s"
    rows = [('Forward model, one evaluation (Numba)', ms(t['forward_numba_s'][0])),
            ('Saturation path $\\alpha$ for given controls (root finding)', ms(t['solve_alpha_s'][0])),
            ('Optimization, SLSQP, cold start', f"{t['slsqp_optimize_s']:.1f} s ({t['slsqp_iterations']} iterations)"),
            ('Optimization, IPOPT with exact derivatives, cold / warm start', f"{t['ipopt_solve_only_cold_s']:.1f} s / {t['ipopt_solve_only_warm_s']:.1f} s"),
            ('SCC by pulse, all 80 periods, $\\alpha$ held fixed', ms(t['scc_fd_fixed_all_s'][0])),
            ('SCC by pulse, all 80 periods, $\\alpha$ re-solved', ms(t['scc_fd_responsive_all_s'][0])),
            ('SCC by the implicit function theorem, all periods', ms(t['ift_all_periods_s'][0]) + f" (+{t['ift_compile_s']:.1f} s compilation, once)"),
            ('One Monte Carlo draw (SLSQP + decomposition), median', ms(t['mc_draw_s'][0])),
            ('One step of the Gymnasium environment', f"{t['gym_ms_per_step']:.2f} ms")]
    tex = ("\\begin{tabular}{lr}\n\\toprule\nComputation & Wall time \\\\\n\\midrule\n" +
           '\n'.join(f"{a} & {b} \\\\" for a, b in rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(OUT, 'tab_timing.tex'), 'w').write(tex)


def mc_table():
    r = _get('mc_regression.json'); s = _get('mc_summary.json')
    if not r or not s:
        return
    lab = {'rho': 'Pure time preference $\\rho$', 'psi2': 'Damage coefficient $\\psi_2$', 'irT': 'IRF temperature sensitivity $\\sigma_T$',
           'irC': 'IRF carbon sensitivity $\\sigma_C$', 'eland0': 'Land-use emissions in 2020'}
    rows = []
    for k in ('rho', 'psi2', 'irT', 'irC', 'eland0'):
        rows.append(lab[k] + ' & ' + ' & '.join(f"{r[y]['coef_pp_per_sd'][k]:+.2f}" for y in ('2025', '2050', '2100')) + ' \\\\')
    rows.append('\\midrule')
    rows.append('$R^2$ & ' + ' & '.join(f"{r[y]['R2']:.2f}" for y in ('2025', '2050', '2100')) + ' \\\\')
    rows.append('Median share (\\%) & ' + ' & '.join(f"{r[y]['p50']:.1f}" for y in ('2025', '2050', '2100')) + ' \\\\')
    rows.append('5th--95th percentile (\\%) & ' + ' & '.join(f"{r[y]['p5']:.1f} to {r[y]['p95']:.1f}" for y in ('2025', '2050', '2100')) + ' \\\\')
    rows.append('Draws with a negative share (\\%) & ' + ' & '.join(f"{100 * r[y]['share_negative']:.0f}" for y in ('2025', '2050', '2100')) + ' \\\\')
    tex = ("\\begin{tabular}{lrrr}\n\\toprule\n & 2025 & 2050 & 2100 \\\\\n\\midrule\n" + '\n'.join(rows) +
           "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(OUT, 'tab_mc.tex'), 'w').write(tex)


def scc_clean_table():
    js = _get('scc_gams_controls.json')
    if not js:
        return
    rows = []
    for sc in ['9', '4', '5', '8', '10']:
        o = js[sc]; cells = []
        for y in (2025, 2050, 2100):
            j = o['years'].index(y)
            cells += [f"{o['gams'][j]:.1f}", f"{o['resp'][j]:.1f}", f"{o['fixed'][j]:.1f}", f"{100 * o['share'][j]:.1f}"]
        rows.append(NAMES[int(sc)] + ' & ' + ' & '.join(cells) + ' \\\\')
    tex = ("\\begin{tabular}{l" + "rrrr" * 3 + "}\n\\toprule\n"
           " & \\multicolumn{4}{c}{2025} & \\multicolumn{4}{c}{2050} & \\multicolumn{4}{c}{2100} \\\\\n"
           "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}\n"
           "Scenario" + " & GAMS & Resp. & Fixed & Share" * 3 + " \\\\\n\\midrule\n" + '\n'.join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(OUT, 'tab_scc_clean.tex'), 'w').write(tex)

if __name__ == '__main__':
    validation_table('val', 'tab_validation_default.tex', 'val')
    validation_table('faith', 'tab_validation_faithful.tex', 'fa')
    if os.path.exists(os.path.join(RES, 'scc_gams_controls.json')):
        scc_table()
    for y in (2025, 2100):
        scc_checks_table(y)
    pulse_table()
    envelope_table()
    tlo_table()
    decomposition_table()
    policy_table()
    onepointfive_table()
    mc_summary()
    timing_table()
    mc_table()
    scc_clean_table()
    open(os.path.join(OUT, 'numbers.tex'), 'w').write('% Generated by paper_build/make_tables.py. Do not edit by hand.\n' + '\n'.join(macros) + '\n')
    print(len(macros), 'macros written to', OUT)
