"""SCC on the GAMS control paths: GAMS shadow-price SCC vs perturbation SCC with alpha
re-solved (responsive) and alpha held at its baseline path (fixed).
Using the GAMS controls isolates the SCC definition from optimizer differences.
Pulse h = 0.1 GtCO2/yr, Richardson for a one-sided difference: SCC = 2 f(h/2) - f(h)
(see check2_pulse_sweep.py for the choice of h).
Usage: python scc_gams_controls.py [scenarios...]   -> results/scc_gams_controls.json"""
import sys
import numpy as np
from common import dm, save_json, scenario_params, gams_controls, idx, YEARS

def welfare_sum(p, C):
    return dm._welfare_sum(p, C)

def scc_point(p, MIU, S, a_base, W0, cpc, i, h, responsive):
    N = p['num_periods']
    pulse = np.zeros(N + 1); pulse[i] = h
    if responsive:
        a, _ = dm.solve_alpha(p, MIU, S, pulse=pulse, alpha_start=a_base, first_period=i)
    else:
        a = a_base
    alt = dm.diceTrajectory(p, MIU, S, a, pulse)
    dW = welfare_sum(p, alt[1])[0] - W0
    lam = cpc[i - 1] ** (-p['elasmu']) * p['RR'][i]
    return -(dW / h) / lam

def scc_rich(p, MIU, S, a_base, W0, cpc, i, h, responsive):
    s1 = scc_point(p, MIU, S, a_base, W0, cpc, i, h, responsive)
    s2 = scc_point(p, MIU, S, a_base, W0, cpc, i, h / 2, responsive)
    return 2 * s2 - s1, s1, s2

def run(sc, years=(2025, 2030, 2050, 2100, 2150), h=0.1):
    m, p = scenario_params(sc)
    if sc == 1:
        p['k0'] = 409.0   # the published GAMS 1% run keeps K(1) from the 2% run (see run_faithful.py)
    MIU, S, g = gams_controls(sc, p)
    N = p['num_periods']
    a_base, res = dm.solve_alpha(p, MIU, S)
    base = dm.diceTrajectory(p, MIU, S, a_base)
    W0, cpc = welfare_sum(p, base[1])
    T = base[4][1:]
    n40 = 40
    out = {'scenario': sc, 'name': dm.SCENARIO_NAMES[sc], 'irf_residual': res,
           'max_dT_to_2215': float(np.max(np.abs(T[:n40] - g['TATM'][:n40]))),
           'max_dalpha_to_2215': float(np.max(np.abs(a_base[1:n40 + 1] - g['ALPHA'][:n40]))),
           'max_dK_pct_to_2100': float(np.max(np.abs(base[5][1:18] / g['K'][:17] - 1)) * 100),
           'years': list(years), 'gams': [], 'resp': [], 'fixed': [], 'resp_h': [], 'resp_h2': [],
           'fixed_h': [], 'fixed_h2': []}
    for y in years:
        i = idx(y) + 1   # 1-based period
        r, r1, r2 = scc_rich(p, MIU, S, a_base, W0, cpc, i, h, True)
        f, f1, f2 = scc_rich(p, MIU, S, a_base, W0, cpc, i, h, False)
        out['gams'].append(float(g['SCC'][i - 1]))
        out['resp'].append(r); out['fixed'].append(f)
        out['resp_h'].append(r1); out['resp_h2'].append(r2)
        out['fixed_h'].append(f1); out['fixed_h2'].append(f2)
    out['share'] = [(r - f) / r for r, f in zip(out['resp'], out['fixed'])]
    out['resp_vs_gams_pct'] = [100 * (r / gg - 1) for r, gg in zip(out['resp'], out['gams'])]
    out['fixed_vs_gams_pct'] = [100 * (f / gg - 1) for f, gg in zip(out['fixed'], out['gams'])]
    return out

if __name__ == '__main__':
    scs = [int(s) for s in sys.argv[1:]] or list(range(1, 11))
    allres = {}
    for sc in scs:
        o = run(sc)
        allres[sc] = o
        print(f"S{sc:2d} {o['name']:<22} dT={o['max_dT_to_2215']:.2e} dalpha={o['max_dalpha_to_2215']:.2e} "
              f"dK%={o['max_dK_pct_to_2100']:.2e} irf={o['irf_residual']:.1e}")
        for j, y in enumerate(o['years']):
            print(f"    {y}: GAMS {o['gams'][j]:9.3f}  resp {o['resp'][j]:9.3f} ({o['resp_vs_gams_pct'][j]:+.3f}%)  "
                  f"fixed {o['fixed'][j]:9.3f} ({o['fixed_vs_gams_pct'][j]:+.2f}%)  share {100*o['share'][j]:.2f}%")
        sys.stdout.flush()
    save_json(allres, 'scc_gams_controls' + ('' if len(scs) == 10 else '_' + '_'.join(map(str, scs))) + '.json')
