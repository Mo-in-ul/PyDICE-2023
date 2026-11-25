"""v8: sensitivity of the saturation-feedback share to abatement costs (referee comment 7).

The policy channel of Section 6.1 runs through abatement: when abatement is cheaper the planner cuts emissions faster,
the net flux into the sinks shrinks and the share falls. This script varies (i) the backstop price, which scales the
whole abatement-cost curve, and (ii) the cost exponent theta2, keeping the cost of full abatement (mu = 1) unchanged.
With the policy held at the DICE-2023 optimum abatement costs barely move the share (they enter only through
consumption and the welfare weights), so the change comes from re-optimizing. Same machinery as
exp_b_physical_policy.py: IPOPT optimum, decomposition at the optimum, alpha re-solved for every control path.
Writes results/c4_abatement_cost.json.
"""
import numpy as np
from common import save_json
import mc_v2 as V, mc_fast as F
import mc_ift as M
import exp_b_physical_policy as B

_orig = V.draw_params
SET = {'pback': 1.0, 'theta2': None}


def draw_params_cost(row):
    p, b = _orig(row)
    th0 = p['expcost2']
    th = SET['theta2'] or th0
    # cost1tot = pback * sigmatot / theta2 / 1000; scale the backstop and change theta2 consistently
    p['cost1tot'] = p['cost1tot'] * SET['pback'] * th0 / th
    p['PBACKTIME'] = p['PBACKTIME'] * SET['pback']
    p['expcost2'] = th
    return p, b


V.draw_params = draw_params_cost
M.draw_params = draw_params_cost
B.V.draw_params = draw_params_cost

if __name__ == '__main__':
    from ipopt_model import Problem
    xg = Problem(9).start_from_gams()
    row0 = np.array([V.DICE[k] for k in V.KEYS])
    o0, x0 = F.evaluate_ipopt(row0, xg)
    p0, _ = V.draw_params(row0); MIU0, S0, a0 = B.controls(x0, p0)
    base = B.at_controls(row0, MIU0, S0, a0)
    out = {'baseline': base, 'rows': []}
    print('baseline', round(100 * base['share_2025'], 2), flush=True)
    cases = [('pback', v, None) for v in (0.5, 0.75, 1.5, 2.0)] + [('theta2', 1.0, v) for v in (2.2, 3.0)]
    for key, pb, th in cases:
        SET['pback'] = pb; SET['theta2'] = th
        o, x = F.evaluate_ipopt(row0, x0)
        p, _ = V.draw_params(row0); MIUv, Sv, av = B.controls(x, p)
        fixed = B.at_controls(row0, MIU0, S0, a0)
        reopt = B.at_controls(row0, MIUv, Sv, av)
        cp2030 = float(p['PBACKTIME'][3] * MIUv[3] ** (p['expcost2'] - 1))
        r = {'case': key, 'pback_scale': pb, 'theta2': th or float(p0['expcost2']), 'success': o['success'],
             'fixed': fixed, 'reopt': reopt, 'cprice_2030': cp2030}
        out['rows'].append(r)
        print(key, pb, th, o['success'], 'fixed', round(100 * fixed['share_2025'], 2), 'reopt', round(100 * reopt['share_2025'], 2),
              'SCC', round(reopt['scc_2025'], 2), 'feedback', round(reopt['feedback_2025'], 2), 'cumE', round(reopt['cumE_2100']),
              'neg', reopt['feedback_negative_from'], flush=True)
        save_json(out, 'c4_abatement_cost.json')
    SET['pback'] = 1.0; SET['theta2'] = None
