"""Tables for the journal manuscript (p2_manuscript/cce_v2/tables). Reads results/*.json and p2_report/assets; no model runs."""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__)); T = os.path.join(HERE, 'p2_manuscript', 'cce_v2', 'tables')
R = lambda n: json.load(open(os.path.join(HERE, 'results', n)))
G = R('p5_regret.json')
rows = []
for cal, head in (('pub', 'Published calibration'), ('d2', '2 percent calibration')):
    rows.append(f"\\multicolumn{{11}}{{@{{}}l}}{{\\rule{{0pt}}{{11pt}}\\textit{{{head}}}}} \\\\")
    for case, learned, ceil in (('2050', '2050', 'on'), ('2070', '2070', 'on'), ('2050:noceil', '2050', 'off')):
        item = f'{cal}:{case}'
        o = G[item]['0.5']; s, e = o['stochastic'], o['expected']
        rg = [G[item][p]['expected']['regret'] for p in ('0.25', '0.5', '0.75')]
        V = (o['optimist']['US'] - o['pessimist']['UF']) / o['lam']; sh = 100 * rg[1] / (0.5 * V)
        print(item, 'value of removal', round(V, 1), 'regret/EV %', round(sh, 3), 'regret/(regret+PI) %', round(100 * rg[1] / (rg[1] + o['perfect']['gain']), 1))
        rows.append(f"\\quad {learned} & {ceil} & {s['scc']:.2f} & {e['scc']:.2f} & {100 * (s['scc'] - e['scc']) / e['scc']:.1f} & {s['abt']:.1f} & {e['abt']:.1f} & "
                    f"{s['peakF']:.3f} & {e['peakF']:.3f} & {rg[1]:.4f} & {sh:.2f} \\\\")
with open(os.path.join(T, 'tab_main.tex'), 'w') as f:
    f.write('\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{@{}llrrrrrrrrr@{}}\n\\toprule\n'
            ' & & \\multicolumn{3}{c}{SCC 2025 (\\$/t)} & \\multicolumn{2}{c}{Abatement (Gt)} & \\multicolumn{2}{c}{Peak if fails (\\textdegree C)} & \\multicolumn{2}{c}{Regret} \\\\\n'
            '\\cline{3-5}\\cline{6-7}\\cline{8-9}\\cline{10-11}\n'
            'Learned & Ceiling & Stoch. & Exp. & \\% & Stoch. & Exp. & Stoch. & Exp. & T\\$ & \\% of $pV$ \\\\\n\\colrule\n')
    f.write('\n'.join(rows) + '\n\\botrule\n\\end{tabular}\n')
# appendix tables converted from the research record
A = os.path.join(HERE, 'p2_report', 'assets')
for src, dst in (('tab_p4_held.tex', 'tab_held.tex'), ('tab_p4_lottery.tex', 'tab_lottery.tex')):
    s = open(os.path.join(A, src)).read().replace('\\midrule', '\\colrule').replace('\\bottomrule', '\\botrule')
    if dst == 'tab_held.tex':                    # group the rows by calibration, blank repeated labels
        out, prev = [], None
        for line in s.split('\n'):
            parts = line.split(' & ')
            if len(parts) == 6 and parts[0] in ('Published', '3\\%', '2\\%'):
                if prev is not None and parts[0] != prev:
                    out.append('\\noalign{\\vskip 3pt}')
                lab = parts[0] if parts[0] != prev else ''
                prev = parts[0]; parts[0] = lab
                line = ' & '.join(parts)
            out.append(line)
        s = '\n'.join(out)
    open(os.path.join(T, dst), 'w').write(s)
print(open(os.path.join(T, 'tab_main.tex')).read())
