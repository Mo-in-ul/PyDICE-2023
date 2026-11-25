"""Compare NEOS/GAMS listing files with the PyDICE-2023 results used in the eta letter.

Put the NEOS listing for each phi in this folder as dice2023_eta<phi>.lst (e.g. dice2023_eta1p30.lst),
then run: python compare_neos.py. Writes comparison.md next to this script.
PyDICE values: ../stage2/results/s2_sigma_main.json (sigma = 1%, 235-year cap,
0.5 C temperature bound, horizon 2420), i.e. the 'as coded' curve of Fig. 1a.
"""
import os, re, json
HERE = os.path.dirname(os.path.abspath(__file__))
PB = os.path.join(HERE, '..')
PHIS = (0.95, 1.30, 1.50, 1.75)
NUM = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'

def blocks(txt):
    """Split a GAMS listing into display blocks keyed by the name after '----  <line> PARAMETER/VARIABLE'."""
    out = {}
    parts = re.split(r'\n----\s+\d+\s+', txt)
    for p in parts[1:]:
        for m in re.finditer(r'(?:PARAMETER|VARIABLE|MODEL)\s+([A-Za-z0-9_.]+)\s*=\s*(' + NUM + ')', p):
            out[m.group(1).lower()] = float(m.group(2))
        first = p.split('\n')[0].split()
        name = (first[1] if first and first[0] in ('PARAMETER', 'VARIABLE') and len(first) > 1 else '').lower()
        if name and name not in out:
            body = '\n'.join(p.split('\n')[1:])
            out[name] = {int(k): float(v) for k, v in re.findall(r'(?<![\d.])(\d+)\s+(' + NUM + ')', body)}
    return out

py = {round(o['eta'], 2): o for o in json.load(open(os.path.join(PB, 'stage2', 'results', 's2_sigma_main.json')))
      if o['sigma'] == 0.01 and o['tlo'] == 0.5 and o['N'] == 81}
rows = ['| phi | status (model/solve) | SCC 2025 GAMS | PyDICE | diff % | SCC 2050 GAMS | PyDICE | diff % | peak T GAMS | PyDICE | objective GAMS | PyDICE | max SCC path diff % (2025-2100) |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for phi in PHIS:
    f = os.path.join(HERE, 'dice2023_eta%s.lst' % ('%.2f' % phi).replace('.', 'p'))
    p = py[round(phi, 2)]
    if not os.path.exists(f):
        rows.append('| %.2f | listing missing | | %.4f | | | %.4f | | | %.4f | | %.6f | |' % (phi, p['scc_2025'], p['scc_2050'], p['peakT'], p['U']))
        continue
    b = blocks(open(f, encoding='latin-1').read())
    g = lambda k: b.get(k, float('nan'))
    pct = lambda a, c: 100 * (a - c) / c
    path = b.get('scc', {})
    dpath = max((abs(pct(path[t], p['scc_path'][t - 1])) for t in range(2, 18) if t in path), default=float('nan'))
    rows.append('| %.2f | %s/%s | %.4f | %.4f | %+.4f | %.4f | %.4f | %+.4f | %.4f | %.4f | %.6f | %.6f | %.4f |' % (
        phi, g('co2.modelstat'), g('co2.solvestat'), g('scc2025'), p['scc_2025'], pct(g('scc2025'), p['scc_2025']),
        g('scc2050'), p['scc_2050'], pct(g('scc2050'), p['scc_2050']), g('peakt'), p['peakT'], g('utility.l'), p['U'], dpath))
open(os.path.join(HERE, 'comparison.md'), 'w').write('# GAMS (NEOS) vs PyDICE-2023, eta letter\n\n' + '\n'.join(rows) + '\n')
print('\n'.join(rows))
