"""Consumption discount rate implied by each calibration (P0, discounting test). Results: results/p5_crate.json.

For the no-removal optimum of each calibration, the average annual consumption discount rate between 2025 and year Y
is (lambda_C(2025) / lambda_C(Y))^(1/(Y - 2025)) - 1, with lambda_C the marginal welfare of consumption per period
(removal3 duals, dVdC). This is the rate at which the SCC discounts future damages, and it combines the utility
discount rate with the elasticity times consumption growth."""
import sys
import numpy as np
import removal3 as R3
import dice2023_corrected as dm
from p4_common import *

OUT = 'p5_crate.json'


def mk(item):
    f = item.split(':')
    if f[0] == 'disc':
        def m():
            p, b = gams_session(9); dm.apply_disc_prstp(p, prstp_value=float(f[1]), elasmu_value=float(f[2])); return 9, (p, b)
        return m
    return lambda: (SC[item], None)


if __name__ == '__main__':
    for item in sys.argv[1:]:
        if item in load(OUT):
            continue
        sc, sess = mk(item)()
        pr = R3.Problem(sc, cal={}, allow='', session=sess)
        x, info, r, d, _ = pr.solve()
        lam = -np.asarray(d['dVdC'])[1:] / 5.0
        C = np.asarray(r['C'])[1:]; L = np.asarray(pr.qd['L'])[1:]; c = C / L
        o = {'status': int(info['status']), 'scc2025': float(d['scc'][2]), 'elasmu': float(pr.p['elasmu'])}
        for Y in (2100, 2200, 2300):
            j = iy(Y)
            o[f'r_{Y}'] = float((lam[1] / lam[j]) ** (1 / (Y - 2025)) - 1)
            o[f'g_{Y}'] = float((c[j] / c[1]) ** (1 / (Y - 2025)) - 1)
        save(OUT, item, o)
        print(item, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in o.items()}, flush=True)
    print('done')
