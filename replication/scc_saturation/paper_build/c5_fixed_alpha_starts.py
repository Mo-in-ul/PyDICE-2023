"""v8: the planner who ignores the saturation feedback, started from several alpha paths (referee comment 8).

The fixed point of policy_fixed_alpha.py (choose MIU, S with alpha given; re-solve alpha on the chosen path;
repeat) is computed from five starting alpha paths: the optimal path, the path on the published baseline
(no-policy) controls, a constant path at the 2020 value, and the optimal path scaled by 0.8 and 1.2. If the
fixed point is unique on this evidence, all starts end at the same policy. Also reported: cumulative CO2
emissions 2020-2100 of the optimum and of the fixed point.
Writes results/c5_fixed_alpha_starts.json.
"""
import numpy as np
from common import dm, save_json, gams_controls, scenario_params
from ipopt_model import solve

N = 81


def fixed_point(sc, alpha0, x0, p):
    alpha = alpha0.copy(); x = x0.copy(); it = 0; d = np.inf
    for it in range(1, 81):
        r = solve(sc, tol=1e-11, fix_alpha=alpha, x0=x)
        x = r['x']
        MIU = np.concatenate([[0.0], x[:N]]); S = np.concatenate([[0.0], x[N:2 * N]])
        new, _ = dm.solve_alpha(p, MIU, S, alpha_start=alpha)
        d = float(np.max(np.abs(new[2:] - alpha[2:])))
        alpha = new
        if d < 1e-9:
            break
    price = p['PBACKTIME'][1:] * MIU[1:] ** (p['expcost2'] - 1)
    E = r['traj']['ECO2'][1:]
    return {'iterations': it, 'last_change': d, 'price2030': float(price[2]), 'price2050': float(price[6]),
            'cumE_2100_GtCO2': float(5 * E[:17].sum()), 'alpha': alpha, 'MIU': MIU}


if __name__ == '__main__':
    out = {}
    for sc in (9, 4):
        opt = solve(sc, tol=1e-11)
        p = opt['problem'].p
        a_opt = np.concatenate([[0.0], opt['x'][2 * N:3 * N]])
        MIUb, Sb, _ = gams_controls(10, scenario_params(10)[1])
        a_base, _ = dm.solve_alpha(p, MIUb, Sb)
        starts = {'optimal': a_opt, 'baseline_controls': a_base, 'constant_2020': np.full(N + 1, p['a0']),
                  'optimal_x0.8': 0.8 * a_opt, 'optimal_x1.2': 1.2 * a_opt}
        for k in starts:
            starts[k][1] = p['a0']; starts[k][0] = 0.0
            starts[k][2:] = np.clip(starts[k][2:], 0.1, 100)
        price_opt = p['PBACKTIME'][1:] * np.concatenate([[0], opt['x'][:N]])[1:] ** (p['expcost2'] - 1)
        res = {'optimal': {'price2030': float(price_opt[2]), 'price2050': float(price_opt[6]),
                           'cumE_2100_GtCO2': float(5 * opt['traj']['ECO2'][1:][:17].sum())}, 'starts': {}}
        fps = {}
        for name, a0 in starts.items():
            fp = fixed_point(sc, a0, opt['x'].copy(), p)
            fps[name] = fp
            res['starts'][name] = {k: v for k, v in fp.items() if k not in ('alpha', 'MIU')}
            print(sc, name, res['starts'][name], flush=True)
        ref = fps['optimal']
        res['max_alpha_diff_across_starts'] = float(max(np.max(np.abs(f['alpha'][2:] - ref['alpha'][2:])) for f in fps.values()))
        res['max_miu_diff_across_starts'] = float(max(np.max(np.abs(f['MIU'][2:] - ref['MIU'][2:])) for f in fps.values()))
        res['cumE_gap_GtCO2'] = ref['cumE_2100_GtCO2'] - res['optimal']['cumE_2100_GtCO2']
        out[str(sc)] = res
        print(sc, 'max diffs', res['max_alpha_diff_across_starts'], res['max_miu_diff_across_starts'], 'cumE gap', res['cumE_gap_GtCO2'], flush=True)
        save_json(out, 'c5_fixed_alpha_starts.json')
