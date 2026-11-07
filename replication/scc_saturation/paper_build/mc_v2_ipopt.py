"""Monte Carlo v2 solved with IPOPT (primary results). Same draws as mc_v2.py.
Per draw: IPOPT optimum, multiplier SCC, IFT split; flag if IPOPT status is not 0/1 or the IFT total
differs from the multiplier SCC by more than 0.1% (mc_ift.evaluate).
The SLSQP results (results/mc2/) are kept as a cross-check (mc_v2_polish.py re-solves 20 of them with IPOPT;
slsqp_disagree.py examines the draws where the two differ).
Usage: python mc_v2_ipopt.py <first> <last> -> results/mc2_ipopt/ip_<first>_<last>.json"""
import os, sys, json
from common import save_json, RES
import mc_v2 as V
import mc_ift as M

if __name__ == '__main__':
    a, z = int(sys.argv[1]), int(sys.argv[2])
    X = V.design()
    from ipopt_model import Problem
    xg = Problem(9).start_from_gams()
    D = os.environ.get('MC_DIR', 'mc2_ipopt')
    os.makedirs(os.path.join(RES, D), exist_ok=True)
    name = f'{D}/ip_{a}_{z}.json'
    out = json.load(open(os.path.join(RES, name))) if os.path.exists(os.path.join(RES, name)) else []
    done = {o['draw'] for o in out}
    for k in range(a, z):
        if k in done:
            continue
        try:
            o, _ = M.evaluate(X[k], xg)
            o['success'] = bool(o['status'] in (0, 1) and not o['flag'])
        except Exception as e:
            o = {'params': dict(zip(V.KEYS, map(float, X[k]))), 'error': repr(e), 'success': False}
        o['draw'] = k; out.append(o)
        if k % 10 == 0 or k == z - 1:
            save_json(out, name)
        print(k, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in o.items() if kk in ('status', 'success', 'scc_2025', 'share_2025', 'elapsed_s')}, flush=True)
    save_json(out, name)
