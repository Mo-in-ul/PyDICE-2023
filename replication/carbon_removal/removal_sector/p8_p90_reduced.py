"""Reduced-space confirmation of the removal-from-2090 case (published calibration, learning 2050). Results: results/p8_audit.json key P90:pub:reduced."""
import os, sys, json
os.environ.setdefault('SLOT', 'p8')
import p7_run as P7
from p4_common import capcal, RES
OUT = os.path.join(RES, 'p8_audit.json'); P90 = dict(startN=2090.0, startG=2090.0, startM=2090.0)
d = json.load(open(OUT)); key = 'P90:pub:reduced'
if key in d: print('done', d[key]); sys.exit()
P7.set_factor(1.0); sc, mk = P7.mkses('pub'); k = P7.iy(2050)
_, sF, _ = P7.det(sc, {}, '', mk()); lam = sF['lamC2025'] / 5
st = P7.reduced('P90pub', 0.5, sc, mk, capcal(1.0, P90), {}, k, lam)
st['premium_pct'] = 100 * (st['scc'] - d['P90:pub']['sccE']) / d['P90:pub']['sccE']
d[key] = st; open(OUT, 'w').write(json.dumps(d, indent=1)); print(key, st)
