"""Mechanism and calibration tests asked for by the review of 11 Nov 2026 (P0). Results: results/p5_tests.json.

Each item computes the deterministic SCC at capacity 0, 0.5 and 1 (removal from 2070), hard caps (A) and no storage
limit (D), and hence the Jensen gap at p = 0.5, which earlier work showed equals the premium closely.
  disc:RHO:ETA      published GAMS session (k0 = 295, no price cap) with the utility discount factors rebuilt at a
                    constant rate RHO (dice2023_corrected.apply_disc_prstp) and elasticity ETA. Isolates discounting
                    from the rest of the alternative-calibration package.
  hz:CAL:N          horizon of N five-year periods (61 = 2320, 81 = 2420, 121 = 2620), session built as in
                    common.gams_session but with N periods.
  dam:CAL:KIND      low-temperature damages below 1.0 C: lin (marginal damage held at its 1.0 C value) or vanish (smooth;
                    marginal damage reaches zero at 0.5 C instead of 0 C).
  two:ITEM:V        two-state premium (p = 0.5, learned 2050) for one of the items above and storage variant V.
Usage: python p5_tests.py disc:0.02:0.95 hz:d2:121 dam:d2:lin two:disc:0.02:0.95:A"""
import sys, copy
import numpy as np
import removal3_stoch as ST
import dice2023_corrected as dm
import common
from p4_common import *

OUT = 'p5_tests.json'


def session_N(sc, N):
    """common.gams_session for scenario sc with N periods (same flags)."""
    orig = common.scenario_params

    def sp(scenario, _N=N):
        m = dm.Dice2023Model(_N, scenario); p = m.params
        if scenario in (1, 2, 3, 4, 5):
            p['k0'] = {1: 420, 2: 409, 3: 370, 4: 326, 5: 290}[scenario]
            dm.apply_disc_prstp(p, prstp_value=0.01 * scenario)
        return m, p
    common.scenario_params = sp
    try:
        return common.gams_session(sc)
    finally:
        common.scenario_params = orig


def spec(item):
    """(scenario, session factory, extra calibration) for an item."""
    f = item.split(':')
    if f[0] == 'disc':
        rho, eta = float(f[1]), float(f[2])
        def mk():
            p, b = gams_session(9); dm.apply_disc_prstp(p, prstp_value=rho, elasmu_value=eta); return p, b
        return 9, mk, {}
    if f[0] == 'hz':
        return SC[f[1]], (lambda: session_N(SC[f[1]], int(f[2]))), {}
    if f[0] == 'dam':
        extra = {'lin': dict(dam_lin_t1=1.0), 'vanish': dict(dam_van_t1=1.0)}[f[2]]
        return SC[f[1]], (lambda: None), extra
    raise ValueError(item)


if __name__ == '__main__':
    for item in sys.argv[1:]:
        out = load(OUT).get(item, {})
        if item.startswith('two:'):
            if 'premium' in out:
                continue
            inner, v = item[4:].rsplit(':', 1); sc, mk, extra = spec(inner); base = dict(VAR[v], **extra)
            calS, calF = capcal(1.0, base), dict(base)
            xS, sS, _ = det(sc, calS, 'NGM', mk()); xF, sF, _ = det(sc, calF, '', mk()); _, sE, _ = det(sc, capcal(0.5, base), 'NGM', mk())
            m = ST.TwoState(sc, calS, 'NGM', calF, '', 0.5, 2050, session=mk())
            x, info, scc, dt = m.solve(x0=m.start(xS, xF, common_from='S'))
            out = {'status': int(info['status']), 'scc2s': float(scc[2]), 'sccE': sE['scc2025'], 'premium': float(scc[2]) - sE['scc2025'],
                   'jensen': 0.5 * (sS['scc2025'] + sF['scc2025']) - sE['scc2025']}
            save(OUT, item, out)
            print(f"{item}: st{out['status']} premium {out['premium']:+.4f} ({100 * out['premium'] / sE['scc2025']:.2f}%) Jensen {out['jensen']:+.4f} {dt:.0f}s", flush=True)
            continue
        sc, mk, extra = spec(item)
        variants = ('A', 'D') + (('H',) if item.startswith('dam') and item.endswith('vanish') else ())
        for v in variants:
            for s in (0.0, 0.5, 1.0):
                key = f'{v}|{s}'
                if key in out:
                    continue
                base = dict(VAR[v], **extra)
                cal, allow = (dict(base), '') if s == 0.0 else (capcal(s, base), 'NGM')
                x, r, pr = det(sc, cal, allow, mk())
                out[key] = rec(r, pr); out[key]['N'] = pr.N; save(OUT, item, out)
                print(f"{item} {key}: st{r['status']} SCC {r['scc2025']:.4f} peak {r['peakT']:.4f} minT {min(r['paths']['T']):.3f}", flush=True)
            c = {s: out[f'{v}|{s}']['scc2025'] for s in (0.0, 0.5, 1.0)}
            g = 0.5 * (c[0.0] + c[1.0]) - c[0.5]
            print(f"{item} {v}: Jensen gap {g:+.4f} ({100 * g / c[0.5]:.2f}%)", flush=True)
    print('done')
