"""Richardson (pulse) SCC vs Lagrange-multiplier SCC vs the published GAMS SCC, and why they differ.

Three numbers per scenario and year:
  FD     : pulse SCC with alpha re-solved, Richardson 2 f(h/2) - f(h), h = 0.1, on the GAMS controls
  exact  : multiplier SCC from IPOPT, -1000 eco2eq.m / cc.m            (check5_multiplier_scc.json)
  GAMS   : the published value, -1000 eco2eq.m / (1e-5 + cc.m)
Their differences split into three parts:
  (a) denominator term: GAMS adds 1e-5 to cc.m, so GAMS = exact * cc.m / (cc.m + 1e-5);
      the gap is about 1e-5 / cc.m and grows as discounting shrinks cc.m
  (b) binding state bounds: where a temperature bound binds, the multiplier includes its shadow
      value and the fixed-control pulse does not (FD vs exact)
  (c) numerics: FD vs exact where no bound binds (pulse size and rounding), and IPOPT vs GAMS
      convergence (exact with the 1e-5 term vs GAMS)
Writes results/compare_fd_multiplier.json and tex/tab_fd_vs_multiplier.tex.
"""
import os
import numpy as np
from common import dm, save_json, load_json, scenario_params, gams_controls, HERE
from scc_gams_controls import scc_rich

YEARS = (2025, 2050, 2100, 2150, 2200)
NAMES = {1: '1%', 2: '2%', 3: '3%', 4: '4%', 5: '5%', 7: '2 C', 8: 'Paris', 9: 'Optimal', 10: 'Baseline', 6: '1.5 C'}


def fd_values(sc):
    _, p = scenario_params(sc)
    if sc == 1:
        p['k0'] = 409.0
    MIU, S, g = gams_controls(sc, p)
    a, _ = dm.solve_alpha(p, MIU, S)
    base = dm.diceTrajectory(p, MIU, S, a)
    W0, cpc = dm._welfare_sum(p, base[1])
    return {y: scc_rich(p, MIU, S, a, W0, cpc, (y - 2020) // 5 + 1, 0.1, True)[0] for y in YEARS}


def table(out):
    TEXN = {'9': 'Optimal', '2': '2\\% discounting'}
    def pct(v):
        return '0' if abs(v) < 5e-7 else (f"{v:+.4f}" if abs(v) >= 1e-3 else f"{v:+.0e}".replace('e-0', 'e-'))
    lines = []
    for sc in ('9', '2'):
        for r in out[sc]:
            lines.append(f"{TEXN[sc]} & {r['year']} & {r['gams']:.2f} & {r['exact']:.2f} & {r['fd']:.2f} & "
                         f"{pct(r['a_denominator_pct'])} & {pct(r['b_fd_vs_exact_pct'])} \\\\")
        lines.append('\\midrule')
    lines = lines[:-1]
    tex = ("\\begin{tabular}{llrrrrr}\n\\toprule\nRun & Year & GAMS & Multiplier & Richardson & Denominator term (\\%) & Richardson vs multiplier (\\%) \\\\\n\\midrule\n"
           + '\n'.join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(HERE, 'tex', 'tab_fd_vs_multiplier.tex'), 'w').write(tex)


if __name__ == '__main__':
    import sys
    if '--table-only' in sys.argv:
        table(load_json('compare_fd_multiplier.json')); sys.exit()
    mult = load_json('check5_multiplier_scc.json')
    out = {}
    for sc in (9, 4, 5, 8, 10, 3, 2, 1, 7):
        m = mult[str(sc)]
        fd = fd_values(sc)
        g = dm.load_gams(sc)
        rows = []
        for y in YEARS:
            i = (y - 2020) // 5
            exact = m['scc_exact_full'][i]; form = m['scc_gams_formula_full'][i]; G = float(g['SCC'][i])
            ccm = 1e-5 * form / (exact - form) if exact != form else float('inf')   # cc.m implied by the two
            rows.append({'year': y, 'gams': G, 'fd': fd[y], 'exact': exact, 'formula': form, 'ccm': ccm,
                         'fd_vs_gams_pct': 100 * (fd[y] / G - 1),
                         'a_denominator_pct': 100 * (exact / form - 1),
                         'b_fd_vs_exact_pct': 100 * (fd[y] / exact - 1),
                         'c_formula_vs_gams_pct': 100 * (form / G - 1)})
        out[str(sc)] = rows
        print(NAMES[sc])
        for r in rows:
            print(f"  {r['year']}: GAMS {r['gams']:10.3f}  FD {r['fd']:10.3f} ({r['fd_vs_gams_pct']:+.4f}%) = "
                  f"denominator {r['a_denominator_pct']:+.4f}% + FD-vs-exact {r['b_fd_vs_exact_pct']:+.4f}% + IPOPT-vs-GAMS {r['c_formula_vs_gams_pct']:+.5f}%   cc.m {r['ccm']:.3g}")
    save_json(out, 'compare_fd_multiplier.json')
    table(out)
