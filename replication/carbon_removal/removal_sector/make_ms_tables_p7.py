"""Tables of the 12 Nov 2026 revision runs (tab_disc, tab_ceiling, tab_gradual). Reads results/p5_regret.json, p5_crate.json, p7_runs.json, p7_tree.json."""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__)); T = os.path.join(HERE, 'p2_manuscript', 'cce_v2', 'tables')
R = lambda n: json.load(open(os.path.join(HERE, 'results', n)))
G, C, P, TR = R('p5_regret.json'), R('p5_crate.json'), R('p7_runs.json'), R('p7_tree.json')
def g(item):
    o = G[item]['0.5']; s, e = o['stochastic'], o['expected']; V = (o['optimist']['US'] - o['pessimist']['UF']) / o['lam']
    return dict(prem=100 * (s['scc'] - e['scc']) / e['scc'], abs_=s['abt'], abe=e['abt'], reg=e['regret'], pct=100 * e['regret'] / (0.5 * V))
def p(k):
    v = P[k]; return dict(prem=v['premium_pct'], abs_=v['stochastic']['abt'], abe=v['plans']['expected']['abt'], reg=v['regret_expected'], pct=v['regret_pct_pV'])
pc = lambda a: '$<$0.01' if a['pct'] < 0.005 else f"{a['pct']:.2f}"
def row(lab, a, b):
    return f"{lab} & {a['prem']:.1f} & {a['reg']:.4f} & {pc(a)} & {b['prem']:.1f} & {b['reg']:.3f} & {pc(b)} \\\\"
# discounting
rate = lambda k: 100 * C[k]['r_2200']
rows = [row('Published calibration (3.45\\%)', g('pub:2050'), g('pub:2070')),
        row(f"$\\rho$ 0.5\\%, $\\eta$ 0.95 ({rate('disc:0.005:0.95'):.1f}\\%)", p('disc:0.005:0.95@1@2050'), p('disc:0.005:0.95@1@2070')),
        row(f"$\\rho$ 0.1\\%, $\\eta$ 1.1 ({rate('disc:0.001:1.1'):.1f}\\%)", p('disc:0.001:1.1@1@2050'), p('disc:0.001:1.1@1@2070')),
        row(f"$\\rho$ 2\\%, $\\eta$ 0.001 ({rate('disc:0.02:0.001'):.1f}\\%)", p('d2@1@2050'), p('d2@1@2070'))]
open(os.path.join(T, 'tab_disc.tex'), 'w').write('\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n'
 ' & \\multicolumn{3}{c}{Learned 2050} & \\multicolumn{3}{c}{Learned 2070} \\\\\n\\cline{2-4}\\cline{5-7}\n'
 'Calibration (average consumption discount rate) & Prem.\\ \\% & Regret T\\$ & \\% of $pV$ & Prem.\\ \\% & Regret T\\$ & \\% of $pV$ \\\\\n\\colrule\n'
 + '\n'.join(rows) + '\n\\botrule\n\\end{tabular}\n')
# ceiling
def crow(lab, a):
    return f"{lab} & {a['prem']:.1f} & {a['abs_']:.0f} & {a['abe']:.0f} & {a['reg']:.4f} & {pc(a)} \\\\"
rows = ['\\multicolumn{6}{@{}l}{\\rule{0pt}{11pt}\\textit{2 percent calibration, learned in 2050}} \\\\',
        crow('\\quad DICE-2023 ceiling', p('d2@1@2050')), crow('\\quad Growth after 2030 $\\times$1.5', p('d2@1.5@2050')),
        crow('\\quad $\\times$2', p('d2@2@2050')), crow('\\quad $\\times$3', p('d2@3@2050')), crow('\\quad No ceiling', g('d2:2050:noceil')),
        '\\multicolumn{6}{@{}l}{\\rule{0pt}{11pt}\\textit{2 percent calibration, learned in 2070}} \\\\',
        crow('\\quad DICE-2023 ceiling', p('d2@1@2070')), crow('\\quad $\\times$1.5', p('d2@1.5@2070')),
        crow('\\quad $\\times$2', p('d2@2@2070')), crow('\\quad $\\times$3', p('d2@3@2070')),
        '\\multicolumn{6}{@{}l}{\\rule{0pt}{11pt}\\textit{Lower discount rates, learned in 2050, no ceiling}} \\\\',
        crow('\\quad $\\rho$ 0.5\\%, $\\eta$ 0.95', p('disc:0.005:0.95@1@2050@noceil')), crow('\\quad $\\rho$ 0.1\\%, $\\eta$ 1.1', p('disc:0.001:1.1@1@2050@noceil'))]
open(os.path.join(T, 'tab_ceiling.tex'), 'w').write('\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n'
 ' & Prem.\\ \\% & \\multicolumn{2}{c}{Abatement (Gt)} & \\multicolumn{2}{c}{Regret} \\\\\n\\cline{3-4}\\cline{5-6}\n'
 'Near-term ceiling & & Stoch. & Exp. & T\\$ & \\% of $pV$ \\\\\n\\colrule\n' + '\n'.join(rows) + '\n\\botrule\n\\end{tabular}\n')
# gradual learning
def trow(lab, k):
    v = TR[k]; return f"{lab} & {v['premium_pct']:.1f} & {v['regret_2070only_expected']:.4f} & {v['regret_noupdate']:.4f} & {v['regret_updating']:.4f} & {v['value_of_signal']:.4f} \\\\"
rows = [trow('Published calibration, signal 0.75 / 0.25', 'pub@0.75@0.25'), trow('2 percent calibration, signal 0.75 / 0.25', 'd2@0.75@0.25'),
        trow('2 percent calibration, signal 0.625 / 0.375', 'd2@0.625@0.375')]
open(os.path.join(T, 'tab_gradual.tex'), 'w').write('\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n'
 ' & Prem.\\ \\% & \\multicolumn{3}{c}{Regret of the expected-capacity plan (T\\$)} & Value of signal \\\\\n\\cline{3-5}\n'
 'Case & & Learned 2070 only & Ignores signal & Uses signal & (T\\$) \\\\\n\\colrule\n' + '\n'.join(rows) + '\n\\botrule\n\\end{tabular}\n')
for n in ('tab_disc', 'tab_ceiling', 'tab_gradual'): print(open(os.path.join(T, n + '.tex')).read())
