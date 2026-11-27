"""Finite-difference check of the 2025 SCC for the reduced-space cases (learning in 2070; learning in 2050 without the
near-term ceiling), 26 Nov 2026. Results: results/p8_scc_fd.json (keys red:...).
Step 1 re-solves the reduced-space problem exactly as p7_run.reduced (checkpointed, resumable; state in results/p7_statep8.json)
and stores the pre-learning decisions. Step 2 perturbs 2025 emissions and consumption by +-h with the pre-learning decisions
held at that optimum and each state's later decisions re-optimised. By the envelope theorem the derivative of expected welfare
is the same as with the pre-learning decisions re-optimised as well, so this checks the SCC that the paper reports for these cases.
Usage: python p8_scc_fd_reduced.py d2@1@2070 | d2@1@2050@noceil"""
import os, sys, json, inspect, time
os.environ.setdefault('SLOT', 'p8')
import numpy as np
import p7_run as P7
from p8_scc_fd import DetP, rd, wr, HS, I25
from p4_common import capcal

src = inspect.getsource(P7.reduced).replace("return {'EU'", "return xf, {'EU'").replace('def reduced(', 'def reduced_x(')
exec(src, P7.__dict__); reduced_x = P7.reduced_x
ZF = os.path.join(P7.RES, 'p8_reduced_x.json')


def main(item):
    key = 'red:' + item
    if key in rd(): print('done', key); return
    case, f, t, *rest = item.split('@'); f = float(f); t = int(t); noceil = 'noceil' in rest; p = 0.5
    P7.set_factor(f); sc, mk = P7.mkses(case); k = P7.iy(t); base = dict(ramp_set=False) if noceil else {}
    calS, calF = capcal(1.0, base), dict(base)
    zs = json.load(open(ZF)) if os.path.exists(ZF) else {}
    if item not in zs:
        _, sF, _ = P7.det(sc, calF, '', mk()); lam = sF['lamC2025'] / 5
        xf, st = reduced_x(item, p, sc, mk, calS, calF, k, lam)
        zs[item] = {'xf': np.asarray(xf).tolist(), 'st': st}; open(ZF, 'w').write(json.dumps(zs))
        print(item, 'reduced solve done', st, flush=True)
    xf = np.array(zs[item]['xf']); st = zs[item]['st']
    prs = {}
    for lab, cal, al in (('S', calS, 'NGM'), ('F', calF, '')):
        pr = DetP(sc, cal=cal, allow=al, session=mk()); pr.fixed = (xf, k); pr.fixed_blocks = range(P7.NCTRL); N = pr.N
        pr.setpert(np.zeros(N + 1), np.zeros(N + 1)); x, info, r, d, _ = pr.solve(x0=xf); prs[lab] = (pr, x, d)
    scc_rep = p * float(prs['S'][2]['scc'][I25]) + (1 - p) * float(prs['F'][2]['scc'][I25])
    out = {'scc_reported': scc_rep, 'scc_in_results': st['scc'], 'status': st['status'], 'pg': st['pg'], 'fd': {}}
    for h in HS:
        res = {}
        for v in ('E', 'C'):
            W = []
            for sgn in (1, -1):
                e = np.zeros(N + 1); e[I25] = sgn * h; EU = 0.0; sts = []
                for lab, w in (('S', p), ('F', 1 - p)):
                    pr, x0, _ = prs[lab]; pr.setpert(e if v == 'E' else 0 * e, e if v == 'C' else 0 * e)
                    x, info, *_ = pr.solve(x0=x0); EU += w * (-pr.objective(x)); sts.append(int(info['status']))
                W.append((EU, sts))
            res['d' + v] = (W[0][0] - W[1][0]) / (2 * h); res['st' + v] = [W[0][1], W[1][1]]
        res['scc_fd'] = -1000.0 * res['dE'] / res['dC']; res['rel_err'] = res['scc_fd'] / scc_rep - 1; out['fd'][str(h)] = res
        print(key, h, f"reported {scc_rep:.6f} (results file {st['scc']:.6f}) fd {res['scc_fd']:.6f} rel {res['rel_err']:.2e} st {res['stE']}{res['stC']}", flush=True)
    wr(key, out)


if __name__ == '__main__':
    for it in sys.argv[1:]: main(it)
