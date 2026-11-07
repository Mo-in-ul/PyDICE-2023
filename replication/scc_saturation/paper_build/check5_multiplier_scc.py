"""Check 5: the GAMS SCC definition computed in Python (multiplier SCC from IPOPT).

For each scenario, solve the model with IPOPT under the GAMS session settings and compute
SCC(t) = -1000 * eco2eq.m(t) / cc.m(t) from the Lagrange multipliers (ipopt_model.py).
Two versions are reported:
  scc_exact : the multiplier ratio itself
  scc_gams  : the formula GAMS actually prints, -1000*eco2eq.m/(0.00001 + cc.m)
              (def-*.gms, line "scc(t) = ..."); the 1e-5 biases the reported SCC down by
              about 1e-5/cc.m, which grows as discounting makes cc.m small
The 1.5 C scenario is solved twice: with equality capital accumulation (infeasible, as in
the main code) and with a free-disposal control D >= 0, which is what the GAMS =L= allows.

Writes results/check5_multiplier_scc.json.
Usage: python check5_multiplier_scc.py [scenarios...]   (default: all ten, plus 6 with disposal)
"""
import sys, time
import numpy as np
from common import dm, save_json, load_json, RES
from ipopt_model import solve
import os

YEARS = (2025, 2030, 2050, 2100, 2150, 2200)


def one(sc, disposal=False):
    t0 = time.time()
    # The free-disposal 1.5 C problem starts at a point with variables on their bounds (S = 0,
    # K near its floor); IPOPT's warm-start settings keep it from pushing that point away.
    if disposal:
        r = solve(sc, disposal=True, warm='gams', warm_start=True, tol=1e-11, max_iter=5000)
    else:
        r = solve(sc, disposal=False, warm='gams')
    g = dm.load_gams(sc)
    pr = r['problem']; N = pr.N
    ccm = -r['dVdC']; eco = -r['dVdE']
    scc_gams = np.zeros(N + 1); scc_gams[1:] = -1000 * eco[1:] / (1e-5 + ccm[1:])
    T = r['traj']['TATM'][1:]
    o = {'scenario': sc, 'disposal': disposal, 'status': r['status'], 'status_msg': r['status_msg'],
         'U': r['U'], 'U_gamsC': float(dm._welfare(pr.p, g['C'][:N])), 'irf_max': r['irf_max'],
         'max_dT_to_2215': float(np.max(np.abs(T[:40] - g['TATM'][:40]))),
         'peak_T': float(T[:40].max()), 'min_T': float(T.min()),
         'periods_at_Tup': int(np.sum(T > pr.b['Tup'] - 1e-7)),
         'gams_periods_at_Tup': int(np.sum(g['TATM'][:N] > pr.b['Tup'] - 1e-7)),
         'ccm_vs_analytic_max_relerr': float(np.max(np.abs(ccm[1:] / r['ccm_analytic'][1:] - 1))),
         'years': list(YEARS), 'gams': [], 'scc_exact': [], 'scc_gams_formula': [],
         'exact_vs_gams_pct': [], 'formula_vs_gams_pct': [], 'elapsed_s': time.time() - t0,
         'scc_exact_full': r['scc'][1:].tolist(), 'scc_gams_formula_full': scc_gams[1:].tolist()}
    if disposal:
        o['K'] = r['traj']['K'][1:].tolist(); o['K_gams'] = g['K'][:N].tolist()
        o['D'] = r['x'][3 * N:].tolist(); o['x'] = r['x'].tolist()
    for y in YEARS:
        i = (y - 2020) // 5 + 1
        gg = float(g['SCC'][i - 1])
        o['gams'].append(gg); o['scc_exact'].append(float(r['scc'][i]))
        o['scc_gams_formula'].append(float(scc_gams[i]))
        o['exact_vs_gams_pct'].append(100 * (r['scc'][i] / gg - 1))
        o['formula_vs_gams_pct'].append(100 * (scc_gams[i] / gg - 1))
    return o


if __name__ == '__main__':
    jobs = [(int(s.rstrip('d')), s.endswith('d')) for s in sys.argv[1:]] or \
           [(s, False) for s in range(1, 11)] + [(6, True)]
    path = os.path.join(RES, 'check5_multiplier_scc.json')
    allr = load_json('check5_multiplier_scc.json') if os.path.exists(path) else {}
    for sc, disp in jobs:
        try:
            o = one(sc, disp)
        except Exception as e:   # record failures instead of stopping the batch
            o = {'scenario': sc, 'disposal': disp, 'error': repr(e)}
        key = f'{sc}{"d" if disp else ""}'
        allr[key] = o
        save_json(allr, 'check5_multiplier_scc.json')
        if 'error' in o:
            print(key, 'ERROR', o['error']); continue
        print(f"S{key:3s} status {o['status']} U {o['U']:.6f} (GAMS C: {o['U_gamsC']:.6f}) dT {o['max_dT_to_2215']:.2e} "
              f"peak {o['peak_T']:.3f} irf {o['irf_max']:.1e} {o['elapsed_s']:.0f}s")
        for j, y in enumerate(o['years']):
            print(f"   {y}: GAMS {o['gams'][j]:10.3f}  exact {o['scc_exact'][j]:10.3f} ({o['exact_vs_gams_pct'][j]:+.4f}%)"
                  f"  GAMS formula {o['scc_gams_formula'][j]:10.3f} ({o['formula_vs_gams_pct'][j]:+.5f}%)")
        sys.stdout.flush()
