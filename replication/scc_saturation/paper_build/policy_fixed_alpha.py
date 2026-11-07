"""What it costs to ignore the saturation feedback when choosing policy.

A planner who computes the SCC with alpha held fixed is a planner who treats the alpha path
as given when choosing controls. The resulting policy is a fixed point:
  1. given an alpha path, choose MIU and S to maximize welfare (IPOPT, alpha fixed, no IRF
     condition: ipopt_model.Problem(fix_alpha=...))
  2. re-solve alpha from the IRF condition on the chosen controls (solve_alpha)
  3. repeat until alpha changes by < 1e-9
At the fixed point the planner's marginal valuation of emissions is the fixed-alpha SCC of the
realized path, and the alpha path is consistent with the policy. Compared with the optimum:
carbon price, control rate, temperature, and welfare (% and consumption equivalent: the uniform
% change in optimal consumption that gives the same welfare).
Scenarios: cost-benefit optimal (9) and 4% (4).
Writes results/policy_fixed_alpha.json.
"""
import numpy as np
from scipy.optimize import brentq
from common import dm, save_json
from ipopt_model import solve

N = 81


def ce(p, C_opt, W):
    return 100 * brentq(lambda z: dm._welfare(p, C_opt * (1 + z)) - W, -0.5, 0.5, xtol=1e-14)


def run(sc):
    opt = solve(sc, tol=1e-11)
    p = opt['problem'].p
    x = opt['x'].copy()
    alpha = np.concatenate([[0.0], x[2 * N:3 * N]])
    hist = []
    for it in range(60):
        r = solve(sc, tol=1e-11, fix_alpha=alpha, x0=x)
        x = r['x']
        MIU = np.concatenate([[0.0], x[:N]]); S = np.concatenate([[0.0], x[N:2 * N]])
        new, _ = dm.solve_alpha(p, MIU, S, alpha_start=alpha)
        d = float(np.max(np.abs(new[2:] - alpha[2:]))); hist.append(d)
        alpha = new
        if d < 1e-9:
            break
    # realized path with the consistent alpha
    traj = dm.diceTrajectory(p, MIU, S, alpha)
    W_fp = traj[0]
    yrs = 2020 + 5 * np.arange(N)
    def summ(lab, MIUv, T, W):
        price = p['PBACKTIME'][1:] * MIUv[1:] ** (p['expcost2'] - 1)
        return {'welfare': float(W), 'welfare_change_pct': float(100 * (W - opt['U']) / abs(opt['U'])),
                'ce_pct': float(ce(p, opt['traj']['C'][1:], W)),
                'peak_T': float(T[:40].max()), 'peak_year': int(yrs[np.argmax(T[:40])]), 'T2100': float(T[16]),
                'price2030': float(price[2]), 'price2050': float(price[6]), 'price2100': float(price[16]),
                'MIU2030': float(MIUv[3]), 'MIU2050': float(MIUv[7])}
    MIU_o = np.concatenate([[0.0], opt['x'][:N]])
    res = {'scenario': sc, 'iterations': len(hist), 'last_change': hist[-1],
           'optimal': summ('optimal', MIU_o, opt['traj']['TATM'][1:], opt['U']),
           'feedback_ignored': summ('fixed', MIU, traj[4][1:], W_fp)}
    res['dT_peak'] = res['feedback_ignored']['peak_T'] - res['optimal']['peak_T']
    res['price_gap_2030_pct'] = 100 * (res['feedback_ignored']['price2030'] / res['optimal']['price2030'] - 1)
    res['price_gap_2050_pct'] = 100 * (res['feedback_ignored']['price2050'] / res['optimal']['price2050'] - 1)
    return res


if __name__ == '__main__':
    out = {}
    for sc in (9, 4):
        out[str(sc)] = run(sc)
        print(out[str(sc)], flush=True)
        save_json(out, 'policy_fixed_alpha.json')
