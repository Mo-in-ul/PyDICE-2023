"""Experiment B, order check: the sequential split of the change in the 2025 feedback share with the savings path
replaced before the abatement path,
    physical  = share(v, x0)              - share(v0, x0)
    savings   = share(v, mu0, S*(v))      - share(v, x0)
    abatement = share(v, x*(v))           - share(v, mu0, S*(v))
compared with the order of exp_b_physical_policy.py (abatement first). Writes results/exp_b_order.json."""
import numpy as np
from common import save_json, load_json
import mc_v2 as V, mc_fast as F
from exp_b_physical_policy import at_controls, controls

if __name__ == '__main__':
    from ipopt_model import Problem
    xg = Problem(9).start_from_gams()
    row0 = np.array([V.DICE[k] for k in V.KEYS])
    o0, x0 = F.evaluate_ipopt(row0, xg)
    p0, _ = V.draw_params(row0); MIU0, S0, a0 = controls(x0, p0)
    base = at_controls(row0, MIU0, S0, a0)
    B = {(r['param'], round(r['value'], 6)): r for r in load_json('exp_b_physical_policy.json')['rows']}
    out = []
    grids = (('ecs', np.round(np.arange(2.0, 5.01, 0.25), 2)), ('psi2', (0.0017, 0.0026, 0.003467, 0.0052, 0.0069)))
    for key, vals in grids:
        for v in vals:
            row = row0.copy(); row[V.KEYS.index(key)] = v
            o, x = F.evaluate_ipopt(row, x0)
            p, _ = V.draw_params(row); MIUv, Sv, av = controls(x, p)
            phys = at_controls(row, MIU0, S0, a0)
            sav = at_controls(row, MIU0, Sv, av)
            full = at_controls(row, MIUv, Sv, av)
            rev = {'physical': phys['share_2025'] - base['share_2025'], 'savings': sav['share_2025'] - phys['share_2025'],
                   'abatement': full['share_2025'] - sav['share_2025'], 'total': full['share_2025'] - base['share_2025']}
            fwd = B[(key, round(float(v), 6))]['split']
            r = {'param': key, 'value': float(v), 'success': o['success'], 'reverse': rev, 'forward': fwd,
                 'abatement_diff_pp': 100 * (rev['abatement'] - fwd['abatement']), 'savings_diff_pp': 100 * (rev['savings'] - fwd['savings'])}
            out.append(r)
            print(key, v, o['success'], {k: round(100 * w, 3) for k, w in rev.items()}, 'fwd', {k: round(100 * w, 3) for k, w in fwd.items()}, flush=True)
            save_json(out, 'exp_b_order.json')
    print('ORDER_DONE')
