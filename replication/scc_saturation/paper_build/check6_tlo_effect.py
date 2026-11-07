"""Check 6: how much of the published SCC comes from the TATM >= 0.5 stability bound.

Solve the 1%, 2% and 3% runs (where the bound binds late in the horizon) with IPOPT under
the GAMS session settings, once with the bound (as published) and once with TATM >= 0.01,
and compare the multiplier SCCs. Writes results/check6_tlo_effect.json.
"""
import numpy as np
from common import dm, save_json
from ipopt_model import solve

YEARS = (2025, 2050, 2100)
if __name__ == '__main__':
    out = {}
    for sc in (1, 2, 3):
        g = dm.load_gams(sc)
        r_on = solve(sc, flags=('tlo', 'miulo', 'cpcap', 'k0'))
        r_off = solve(sc, flags=('miulo', 'cpcap', 'k0'))
        T = r_on['traj']['TATM'][1:]
        o = {'scenario': sc, 'status': [r_on['status'], r_off['status']], 'years': list(YEARS),
             'gams': [], 'with_bound': [], 'without_bound': [], 'bound_share_pct': [],
             'years_at_bound': [int(2020 + 5 * k) for k in np.where(T <= 0.5 + 1e-7)[0]]}
        for y in YEARS:
            i = (y - 2020) // 5 + 1
            a, b = float(r_on['scc'][i]), float(r_off['scc'][i])
            o['gams'].append(float(g['SCC'][i - 1])); o['with_bound'].append(a); o['without_bound'].append(b)
            o['bound_share_pct'].append(100 * (a - b) / a)
        out[str(sc)] = o
        yb = o['years_at_bound']
        print(f"S{sc} status {o['status']} bound binds in {len(yb)} periods {yb[:1]}..{yb[-1:]}: " + '  '.join(
            f"{y}: GAMS {gg:.2f} with {a:.2f} without {b:.2f} ({s:+.1f}%)" for y, gg, a, b, s in
            zip(YEARS, o['gams'], o['with_bound'], o['without_bound'], o['bound_share_pct'])))
    save_json(out, 'check6_tlo_effect.json')
