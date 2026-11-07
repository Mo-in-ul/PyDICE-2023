"""Check 4: envelope check by brute-force re-optimization.

The perturbation SCC (scc_gams_controls.py, compute_SCC in the model) holds the controls
MIU and S fixed while the pulse is applied. The envelope theorem says this gives the
right first-order welfare change at an optimum where no state constraint binds. Here the
welfare change is computed without that assumption: add a pulse +h and -h to the
emissions equation in period t, re-solve the full model with IPOPT (warm start from the
unperturbed optimum, tol 1e-12), and take the central difference
    SCC_reopt(t) = -1000 * [U*(+h) - U*(-h)] / (2h) / cc.m(t).
This is also a direct test of the multiplier formula in ipopt_model.py.
Expected: in the optimal run, SCC_reopt = multiplier SCC = fixed-control SCC = GAMS.
In runs with a binding state bound (1% and 2%: TATM >= 0.5; 2 C: TATM <= 2), SCC_reopt =
multiplier SCC = GAMS, while the fixed-control SCC differs.
Writes results/check4_envelope.json.
"""
import sys
import numpy as np
from common import save_json, load_json
from ipopt_model import solve

H = (0.1, 0.05)
YEARS = (2025, 2050)


def run(sc):
    base = solve(sc, tol=1e-12)
    pr = base['problem']; N = pr.N
    o = {'scenario': sc, 'base_status': base['status'], 'U0': base['U'], 'years': list(YEARS), 'h': list(H),
         'multiplier': [], 'reopt': [], 'reopt_richardson': [], 'fixed_control': [], 'gams': [], 'statuses': []}
    fd = load_json('scc_gams_controls.json')[str(sc)]
    for y in YEARS:
        i = (y - 2020) // 5 + 1
        ccm = base['ccm_analytic'][i]
        vals = []
        for h in H:
            Us = []
            for s in (+1, -1):
                e = np.zeros(N + 1); e[i] = s * h
                r = solve(sc, tol=1e-12, x0=base['x'], eps_E=e)
                Us.append(r['U']); o['statuses'].append(r['status'])
            vals.append(-1000.0 * (Us[0] - Us[1]) / (2 * h) / ccm)
        o['reopt'].append(vals)
        o['reopt_richardson'].append((4 * vals[1] - vals[0]) / 3)   # central difference: error O(h^2)
        o['multiplier'].append(float(base['scc'][i]))
        j = fd['years'].index(y)
        o['fixed_control'].append(fd['resp'][j]); o['gams'].append(fd['gams'][j])
    return o


if __name__ == '__main__':
    scs = [int(s) for s in sys.argv[1:]] or [9, 2, 7, 1]
    import os
    from common import RES
    allr = load_json('check4_envelope.json') if os.path.exists(os.path.join(RES, 'check4_envelope.json')) else {}
    for sc in scs:
        o = run(sc); allr[str(sc)] = o
        save_json(allr, 'check4_envelope.json')
        print(f"S{sc} statuses {set(o['statuses'])}")
        for j, y in enumerate(o['years']):
            print(f"   {y}: GAMS {o['gams'][j]:9.3f}  multiplier {o['multiplier'][j]:9.3f}  re-optimized {o['reopt_richardson'][j]:9.3f} "
                  f"(h={H[0]}: {o['reopt'][j][0]:.3f}, h={H[1]}: {o['reopt'][j][1]:.3f})  fixed controls {o['fixed_control'][j]:9.3f}")
        sys.stdout.flush()
