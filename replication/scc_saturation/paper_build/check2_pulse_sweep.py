"""Check 2: numerical robustness of the perturbation SCC.

On the GAMS control paths (clean scenarios 4, 5, 8, 9, 10; years 2025, 2050, 2100):
  (a) pulse-size sweep, h = 1e-6 ... 1 GtCO2/yr, for the responsive and the fixed SCC.
      s(h)  = one-sided difference quotient (error O(h))
      R2(h) = 2 s(h/2) - s(h), Richardson for a one-sided difference (error O(h^2))
      R3(h) = (4 s(h/2) - s(h)) / 3, the formula that was in compute_SCC; it is the Richardson
              step for a central difference and leaves 1/3 of the O(h) error in place
      Errors are measured against the exact derivative from check3_ift.py, so the slopes
      of log error vs log h should be about 1 for s and R3 and 2 for R2 until rounding takes over.
  (b) alpha-solver tolerance: the responsive SCC (h = 0.1, 2 f(h/2) - f(h)) with the Newton tolerance in
      solve_alpha set to 1e-8, 1e-10 and 1e-13 (the model default).
Writes results/check2_pulse_sweep.json.
"""
import sys, functools
import numpy as np
from common import dm, save_json, load_json, scenario_params, gams_controls
import scipy.optimize as so

HS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
YEARS = (2025, 2050, 2100)
SCEN = (4, 5, 8, 9, 10)


def setup(sc):
    _, p = scenario_params(sc)
    MIU, S, g = gams_controls(sc, p)
    a, _ = dm.solve_alpha(p, MIU, S)
    base = dm.diceTrajectory(p, MIU, S, a)
    W0, cpc = dm._welfare_sum(p, base[1])
    return p, MIU, S, a, W0, cpc


def quotient(p, MIU, S, a, W0, cpc, i, h, responsive):
    N = p['num_periods']
    pulse = np.zeros(N + 1); pulse[i] = h
    al = dm.solve_alpha(p, MIU, S, pulse=pulse, alpha_start=a, first_period=i)[0] if responsive else a
    alt = dm.diceTrajectory(p, MIU, S, al, pulse)
    lam = cpc[i - 1] ** (-p['elasmu']) * p['RR'][i]
    return -((dm._welfare_sum(p, alt[1])[0] - W0) / h) / lam


def slope(hs, err):
    hs, err = np.log10(hs), np.log10(np.maximum(err, 1e-300))
    return float(np.polyfit(hs, err, 1)[0])


def run(sc, ift):
    p, MIU, S, a, W0, cpc = setup(sc)
    o = {'scenario': sc, 'h': HS, 'years': list(YEARS)}
    for mode, resp in (('resp', True), ('fixed', False)):
        o[mode] = []
        for y in YEARS:
            i = (y - 2020) // 5 + 1
            ref = ift['total' if resp else 'direct'][ift['years'].index(y)]
            s = {h: quotient(p, MIU, S, a, W0, cpc, i, h, resp) for h in sorted(set(HS + [h / 2 for h in HS]))}
            raw = [s[h] for h in HS]
            rich = [2 * s[h / 2] - s[h] for h in HS]
            rich3 = [(4 * s[h / 2] - s[h]) / 3 for h in HS]
            e_raw = [abs(v / ref - 1) for v in raw]; e_rich = [abs(v / ref - 1) for v in rich]
            e_rich3 = [abs(v / ref - 1) for v in rich3]
            # slopes over the range where truncation dominates (h >= 1e-3)
            k = [j for j, h in enumerate(HS) if h >= 1e-2]
            o[mode].append({'year': y, 'exact': ref, 'raw': raw, 'richardson': rich,
                            'relerr_raw': e_raw, 'relerr_richardson': e_rich,
                            'richardson_central_formula': rich3, 'relerr_richardson_central_formula': e_rich3,
                            'slope_richardson_central_formula': slope([HS[j] for j in k], [e_rich3[j] for j in k]),
                            'slope_raw': slope([HS[j] for j in k], [e_raw[j] for j in k]),
                            'slope_richardson': slope([HS[j] for j in k], [e_rich[j] for j in k])})
    # (b) solver tolerance
    o['tolerance'] = {}
    orig = so.newton
    for tol in (1e-8, 1e-10, 1e-13):
        dm.newton = functools.partial(orig, tol=tol) if tol != 1e-13 else orig
        try:
            vals = []
            for y in YEARS:
                i = (y - 2020) // 5 + 1
                r1 = quotient(p, MIU, S, a, W0, cpc, i, 0.1, True)
                r2 = quotient(p, MIU, S, a, W0, cpc, i, 0.05, True)
                vals.append(2 * r2 - r1)
            o['tolerance'][f'{tol:.0e}'] = vals
        finally:
            dm.newton = orig
    return o


if __name__ == '__main__':
    scs = [int(s) for s in sys.argv[1:]] or list(SCEN)
    ift = load_json('check3_ift.json')
    allr = {}
    for sc in scs:
        o = run(sc, ift[str(sc)]); allr[str(sc)] = o
        save_json(allr, 'check2_pulse_sweep.json')
        for mode in ('resp', 'fixed'):
            for d in o[mode]:
                j = HS.index(1e-3)
                print(f"S{sc} {mode:5s} {d['year']}: exact {d['exact']:.4f}  relerr at h=1e-3: raw {d['relerr_raw'][j]:.1e} "
                      f"R2 {d['relerr_richardson'][j]:.1e} R3 {d['relerr_richardson_central_formula'][j]:.1e} | slopes (h>=1e-2) "
                      f"raw {d['slope_raw']:.2f} R2 {d['slope_richardson']:.2f} R3 {d['slope_richardson_central_formula']:.2f}")
        print('   tolerance:', {k: [round(v, 5) for v in vs] for k, vs in o['tolerance'].items()})
        sys.stdout.flush()
