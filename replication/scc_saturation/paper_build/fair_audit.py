"""Audit of the carbon-cycle substitution (Experiment C), requested in review of v5.

For each FaIR v2 parameter set (Leach et al. 2021 Table 3 CMIP6 calibrations, FaIRv2.0.0 default, FaIR v2.2 AR6):
  * units: rU and rA are given per GtC by Leach et al. and divided by 3.667 for FaIR 2.x, which works in GtCO2;
    the boxes, airborne stock and cumulative emissions are carried in GtCO2 (DICE GtC stocks times 3.667);
  * alpha in 2020 implied by the common DICE-2023 initial state, against DICE-2023's own alpha;
  * the response of alpha to a 2025 pulse: average d alpha over 2025-2100 and 2100-2200, the fraction of years with
    d alpha > 0, and the contribution of each channel (uptake rU dG_U, temperature rT dT, airborne rA dG_A) to
    d iIRF averaged over 2025-2100;
  * initialization: the 2025 share when r0 is re-anchored so that each set reproduces DICE-2023's 2020 alpha from
    the common DICE initial state (the other parameters unchanged).
Writes results/fair_audit.json."""
import numpy as np
from common import save_json
import fair_benchmark as FB
from fair_benchmark import calculate_alpha
from exp_c_fair_range import LEACH

C2 = FB.C2


def run_full(p, Fnon, par, Eper):
    r0, rU, rT, rA = par['r0'], par['rU'], par['rT'], par['rA']
    boxes = np.array([p['res00'], p['res10'], p['res20'], p['res30']]) * C2
    airb = boxes.sum(); cum = p['CumEmiss0'] * C2
    B1, B2 = p['tbox10'], p['tbox20']; T = B1 + B2
    rec = {'a': [], 'GU': [], 'GA': [], 'T': []}
    for i in range(2, FB.N + 1):
        for yr in range(5):
            a = float(calculate_alpha(np.array([airb]), np.array([cum]), FB.G0, FB.G1, r0, rA, rT, rU, np.array([T]), 100.0)[0])
            rec['a'].append(a); rec['GU'].append(cum - airb); rec['GA'].append(airb); rec['T'].append(T)
            boxes = FB.step(Eper[i], boxes, a); airb = float(boxes.sum()); cum += Eper[i]
            MAT = p['mateq'] + airb / C2
            F = p['fco22x'] * np.log(MAT / p['mateq']) / np.log(2) + Fnon[i]
            B1 = B1 * np.exp(-1 / p['d1']) + p['teq1'] * F * (1 - np.exp(-1 / p['d1']))
            B2 = B2 * np.exp(-1 / p['d2']) + p['teq2'] * F * (1 - np.exp(-1 / p['d2']))
            T = B1 + B2
    return {k: np.array(v) for k, v in rec.items()}


if __name__ == '__main__':
    p, q, MIUj, Sj, aj, w, lamC, E, T, Fnon = FB.setup()
    sets = {k: dict(r0=v[0], rU=v[1] / C2, rT=v[2], rA=v[3] / C2) for k, v in LEACH.items()}
    sets['FaIR v2.2 AR6 default'] = dict(FB.AR6)
    yrs = 2020 + np.arange(len(run_full(p, Fnon, sets['FaIR v2.2 AR6 default'], E)['a']))   # yearly from 2020
    i25 = (2025 - 2020) // 5 + 1; h = 1e-3
    a_dice = float(np.asarray(aj)[1])
    out = {'alpha_2020_dice': a_dice, 'sets': {}}
    for name, par in sets.items():
        b = run_full(p, Fnon, par, E)
        Ep = E.copy(); Ep[i25] += h; Em = E.copy(); Em[i25] -= h
        rp, rm = run_full(p, Fnon, par, Ep), run_full(p, Fnon, par, Em)
        da = (rp['a'] - rm['a']) / (2 * h)
        dGU = (rp['GU'] - rm['GU']) / (2 * h); dGA = (rp['GA'] - rm['GA']) / (2 * h); dT = (rp['T'] - rm['T']) / (2 * h)
        m1 = (yrs >= 2025) & (yrs <= 2100); m2 = (yrs > 2100) & (yrs <= 2200)
        ch = {'uptake': float(np.mean(par['rU'] * dGU[m1])), 'temperature': float(np.mean(par['rT'] * dT[m1])), 'airborne': float(np.mean(par['rA'] * dGA[m1]))}
        # r0 re-anchored to DICE-2023's 2020 alpha from the same initial state
        airb0 = b['GA'][0]; GU0 = b['GU'][0]; T0 = b['T'][0]
        iirf_target = float(np.log(a_dice / FB.G0) * FB.G1)     # alpha = g0 exp(iIRF/g1)
        r0a = iirf_target - par['rU'] * GU0 - par['rT'] * T0 - par['rA'] * airb0
        pa = dict(par, r0=r0a)
        m = FB.Annual(p, Fnon, **pa)
        dT_dir, dT_fb, _, _ = m.split(E, i25)
        d, f = FB.scc_parts(w, lamC, i25, dT_dir, dT_fb)
        out['sets'][name] = {'params_per_GtCO2': par, 'alpha_2020': float(b['a'][0]),
                             'dalpha_mean_2025_2100': float(np.mean(da[m1])), 'dalpha_mean_2100_2200': float(np.mean(da[m2])),
                             'frac_years_dalpha_pos_2025_2100': float(np.mean(da[m1] > 0)), 'diirf_channels_2025_2100': ch,
                             'r0_reanchored': r0a, 'share_2025_reanchored': f / (d + f), 'scc_2025_reanchored': d + f}
        o = out['sets'][name]
        print(f"{name:24s} a2020 {o['alpha_2020']:.3f} (DICE {a_dice:.3f})  da>0 {100 * o['frac_years_dalpha_pos_2025_2100']:5.1f}%  "
              f"da 25-100 {o['dalpha_mean_2025_2100']:+.2e}  ch U {ch['uptake']:+.3e} T {ch['temperature']:+.3e} A {ch['airborne']:+.3e}  "
              f"r0 {par['r0']:.1f}->{r0a:.1f}  share re-anch {100 * o['share_2025_reanchored']:.2f}", flush=True)
    save_json(out, 'fair_audit.json')
