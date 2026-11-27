"""Reduced-space cross-check of the joint solutions that ended at IPOPT status 1 in p7_runs.json (26 Nov 2026).
Same setup as p7_run.run; results in results/p8_audit.json under 'xcheck:ITEM'. Usage: python p8_crosscheck.py ITEM"""
import os, sys, json
os.environ.setdefault('SLOT', 'p8')
import p7_run as P7
from p4_common import capcal, RES
OUT = os.path.join(RES, 'p8_audit.json'); item = sys.argv[1]; key = 'xcheck:' + item
d = json.load(open(OUT)) if os.path.exists(OUT) else {}
if key in d: print('done', d[key]); sys.exit()
case, f, t, *rest = item.split('@'); f = float(f); t = int(t); noceil = 'noceil' in rest
P7.set_factor(f); sc, mk = P7.mkses(case); k = P7.iy(t); base = dict(ramp_set=False) if noceil else {}
calS, calF = capcal(1.0, base), dict(base)
_, sF, _ = P7.det(sc, calF, '', mk()); lam = sF['lamC2025'] / 5
st = P7.reduced('x' + item, 0.5, sc, mk, calS, calF, k, lam)
j = json.load(open(os.path.join(RES, 'p7_runs.json')))[item]
st['joint'] = j['stochastic']; st['premium_pct'] = 100 * (st['scc'] - j['sccE']) / j['sccE']; st['premium_joint'] = j['premium_pct']
st['dEU_reduced_minus_joint_T$'] = (st['EU'] - j['stochastic']['EU']) / lam
d[key] = st; open(OUT, 'w').write(json.dumps(d, indent=1)); print(key, st)
