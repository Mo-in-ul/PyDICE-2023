"""Re-solve of the 'removal from 2090' robustness case (Table 11, published calibration), whose earlier joint solve in
p3_mech.py ended at IPOPT status 1. Same procedure as p7_run.run (benchmark plans, joint solve accepted only with status 0
or 1 and expected welfare above every benchmark, otherwise reduced space). 26 Nov 2026. Results: results/p8_audit.json.
Usage: python p8_p90.py pub|d2"""
import os, sys, json
os.environ.setdefault('SLOT', 'p8')
import numpy as np
import p7_run as P7, removal3_stoch as ST
from p4_common import capcal, RES
OUT = os.path.join(RES, 'p8_audit.json'); P90 = dict(startN=2090.0, startG=2090.0, startM=2090.0)
cal = sys.argv[1]; item = f'P90:{cal}'
d = json.load(open(OUT)) if os.path.exists(OUT) else {}
if item in d: print('done', d[item]); sys.exit()
P7.set_factor(1.0); sc, mk = P7.mkses(cal); k = P7.iy(2050); p = 0.5
calS, calF, calE = capcal(1.0, P90), {}, capcal(p, P90)
xS, sS, _ = P7.det(sc, calS, 'NGM', mk()); xF, sF, _ = P7.det(sc, calF, '', mk()); xE, sE, _ = P7.det(sc, calE, 'NGM', mk())
lam = sF['lamC2025'] / 5; plans = {}
for lab, xc in (('optimist', xS), ('pessimist', xF), ('expected', xE)):
    a = P7.fixed(sc, mk, calS, 'NGM', xc, k); b = P7.fixed(sc, mk, calF, '', xc, k); plans[lab] = p * a['U'] + (1 - p) * b['U']
m = ST.TwoState(sc, calS, 'NGM', calF, '', p, 2050, session=mk()); x, info, scc, dt = m.solve(x0=m.start(xS, xF, common_from='S'))
s2 = m.summary(x, info, scc)
o = {'joint_status': s2['status'], 'EU': s2['EU'], 'scc': s2['scc2025'], 'sccE': sE['scc2025'], 'plans_EU': plans,
     'beats_all_plans': bool(s2['EU'] >= max(plans.values()) - 1e-9), 'premium_pct': 100 * (s2['scc2025'] - sE['scc2025']) / sE['scc2025'],
     'regret_expected': (s2['EU'] - plans['expected']) / lam}
d[item] = o; open(OUT, 'w').write(json.dumps(d, indent=1)); print(item, o)
