"""Table of the exact premium decomposition (results/p3_jensen.json) for the record and the manuscript."""
import json
r = json.load(open('results/p3_jensen.json'))
LAB = {'A': 'Hard caps', 'D': 'No storage limit', 'K': 'Non-CO$_2$ fixed, caps', 'L': 'Non-CO$_2$ fixed, no storage limit',
       'J': r'Steeper damages above 1.5\,\textdegree C', 'N': 'No storage limit, no near-term ceiling'}
CAL = {'pub': 'Published', 'd3': r'3\%', 'd2': r'2\%'}; order = {'pub': 0, 'd3': 1, 'd2': 2}; vo = 'ADJKLN'
rows = sorted(((k.split('_'), o) for k, o in r.items()), key=lambda z: (order[z[0][1]], vo.index(z[0][0]), z[0][2]))
lines = []
for (v, c, p), o in rows:
    lines.append(f"{CAL[c]} & {LAB[v]} & {p} & {o['premium']:+.3f} & {o['jensen']:+.3f} & {o['pathS']:+.3f} & {o['pathF']:+.3f} & "
                 f"{o['pathS'] + o['pathF']:+.3f} & {o['resid']:.0e} \\\\")
tab = ("\\begin{tabular}{llrrrrrrr}\n\\toprule\nCalibration & Variant & $p$ & Premium & Jensen gap & Path, success & Path, failure & Net path & Residual \\\\\n\\midrule\n"
       + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
for f in ('p2_report/assets/tab_p3_jensen.tex', 'p2_manuscript/tables/tab_jensen.tex'):
    open(f, 'w').write(tab)
