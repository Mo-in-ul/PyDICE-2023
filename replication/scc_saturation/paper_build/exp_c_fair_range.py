"""Experiment C: range of the saturation-feedback share across published FaIR v2 carbon-cycle calibrations.

Carbon-cycle substitution as in fair_benchmark.py: DICE-2023 economy, optimal policy, forcing and two-box
temperature model held fixed; only the CO2 cycle (FaIR v2 closure, 1-year steps) changes. Parameter sets:
  * the eleven CMIP6 Earth-system-model calibrations of Leach et al. (2021, GMD 14, 3007, Table 3),
  * the FaIRv2.0.0 default (Leach et al. 2021: r0 33.9, ru 0.0188, rT 2.67, ra 0),
  * the FaIR v2.2 AR6 default (as in fair_benchmark.py).
Leach et al. give ru and ra per GtC (FaIRv2.0.0 emissions in GtC); they are divided by 3.667 for FaIR 2.x,
which works in GtCO2. For each set: feedback share of the SCC for pulses every 5 years 2025-2200 and the first
pulse year with a negative feedback. Writes results/exp_c_fair_range.json."""
import numpy as np
from common import save_json
import fair_benchmark as FB

C2 = FB.C2
LEACH = {   # model: (r0, ru per GtC, rT per K, ra per GtC), Leach et al. (2021) Table 3
    'ACCESS-ESM1-5': (36.7, 0.035, 3.04, -0.00066), 'BCC-CSM2-MR': (25.6, 0.00598, 5.2, 0.00439),
    'CESM2': (40.7, 0.0107, 1.28, 0.00421), 'CNRM-ESM2-1': (38.1, 0.000581, 2.47, 0.00978),
    'CanESM5': (35.7, -0.00596, -0.104, 0.0181), 'GFDL-ESM4': (34.3, 0.0219, 4.86, -0.00424),
    'IPSL-CM6A-LR': (32.2, 0.0166, 1.07, 0.0123), 'MIROC-ES2L': (33.4, 0.0131, 3.46, 0.00399),
    'MPI-ESM1-2-LR': (33.3, 0.031, 1.5, -0.00257), 'NorESM2-LM': (40.7, 0.00947, 1.56, 0.00489),
    'UKESM1-0-LL': (37.9, 0.0201, 2.67, 0.00181), 'FaIRv2.0.0 default': (33.9, 0.0188, 2.67, 0.0)}
YEARS = list(range(2025, 2201, 5))

if __name__ == '__main__':
    p, q, MIUj, Sj, aj, w, lamC, E, T, Fnon = FB.setup()
    sets = {k: dict(r0=v[0], rU=v[1] / C2, rT=v[2], rA=v[3] / C2) for k, v in LEACH.items()}
    sets['FaIR v2.2 AR6 default'] = FB.AR6
    out = {'years': YEARS, 'sets': {}}
    for name, par in sets.items():
        m = FB.Annual(p, Fnon, **par)
        sh, scc = [], []
        for y in YEARS:
            i = (y - 2020) // 5 + 1
            dT_dir, dT_fb, T0, a0 = m.split(E, i)
            d, f = FB.scc_parts(w, lamC, i, dT_dir, dT_fb)
            sh.append(f / (d + f)); scc.append(d + f)
        neg = [y for y, s in zip(YEARS, sh) if s < 0]
        out['sets'][name] = {'params_per_GtCO2': par, 'share': sh, 'scc': scc, 'share_2025': sh[0], 'scc_2025': scc[0],
                             'first_negative_pulse_year': neg[0] if neg else None, 'peak_T_2020_2215': float(np.max(T0[1:41]))}
        print(f"{name:24s} share2025 {100 * sh[0]:6.2f}%  SCC2025 {scc[0]:7.2f}  first negative {neg[0] if neg else None}", flush=True)
        save_json(out, 'exp_c_fair_range.json')
    v = np.array([o['share_2025'] for k, o in out['sets'].items() if k in LEACH and k != 'FaIRv2.0.0 default'])
    out['cmip6_summary'] = {'n': int(len(v)), 'min': float(v.min()), 'median': float(np.median(v)), 'max': float(v.max())}
    save_json(out, 'exp_c_fair_range.json'); print(out['cmip6_summary'])
