"""Numbers reported in Section 6.3 for the correlated-preference Monte Carlo.

c6_pref_correlation.py was run as `python c6_pref_correlation.py 0.5 200` and `... -0.5 200` (the re-pairing is
computed over the first 200 draws) and stopped after 111 and 121 draws; the paper uses the first 111 draws of each.
This script reports, for those draws, the achieved rank correlation, the 5th/50th/95th percentiles of the 2025
feedback share, and the same percentiles for draws 0-110 of the main run with independent pairing.
"""
import glob
import json
import os
import numpy as np
from scipy.stats import spearmanr
from common import RES

K = 111
for r in ('+0.5', '-0.5'):
    d = json.load(open(os.path.join(RES, f'c6_pref_corr_{r}.json')))
    ok = [x for x in d['draws'][:K] if x.get('success')]
    sh = 100 * np.array([x['share_2025'] for x in ok])
    rs = spearmanr([x['rho'] for x in ok], [x['elasmu'] for x in ok])[0]
    print(f'r = {r}: draws {len(ok)}, rank correlation {rs:+.2f}, share p5/p50/p95 =', np.percentile(sh, [5, 50, 95]).round(1))

main = {}
for f in sorted(glob.glob(os.path.join(RES, 'mc2_ipopt_v4', 'ip_*.json'))):
    x = json.load(open(f))
    for o in (x.values() if isinstance(x, dict) else x):
        if isinstance(o, dict) and 'share_2025' in o:
            main[int(o.get('k', o.get('index', len(main))))] = o
sh = 100 * np.array([main[k]['share_2025'] for k in sorted(main)[:K]])
print('independent pairing, same draws: share p5/p50/p95 =', np.percentile(sh, [5, 50, 95]).round(1))
