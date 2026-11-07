"""Sweep the elasticity of marginal utility (elasmu, eta) with everything else at DICE-2023.

DICE-2023 discounts with RR(t) = RR1(t) (1 + rprecaut(t))^(-5(t-1)), rprecaut(t) = -0.5 varpcc(t) eta^2,
varpcc(t) = siggc1^2 * 5 min(t-1, 47): a precautionary term for uncertain growth that grows with the
horizon (to -0.0118 eta^2 per year from 2255). Two runs per eta: as coded in DICE-2023, and with the
precautionary term held at its DICE value (eta = 0.95), so that eta acts only through utility curvature.
Writes results/eta_sweep.json.
"""
import sys
import numpy as np
import mc_v2 as V, mc_fast as F
from ipopt_model import Problem
from common import save_json

ETAS = (0.5, 0.75, 0.95, 1.25, 1.5, 1.75, 2.0)


def dp_fixed(r):
    p, b = V.draw_params(r)
    t = np.arange(1, 82)
    p['rprecaut'][1:] = -0.5 * p['varpcc'][1:] * 0.95 ** 2
    p['RR'][1:] = p['RR1'][1:] * (1 + p['rprecaut'][1:]) ** (-p['tstep'] * (t - 1))
    return p, b


if __name__ == '__main__':
    xg = Problem(9).start_from_gams()
    out = []
    for prec in ('dice', 'fixed'):
        F.M.draw_params = V.draw_params if prec == 'dice' else dp_fixed
        for el in ETAS:
            row = np.array([V.DICE[k] for k in V.KEYS]); row[V.KEYS.index('elasmu')] = el
            o, _ = F.evaluate(row, xg)
            r = {'precaution': prec, 'elasmu': el, 'success': o['success'], 'scc_2025': o['scc_2025'],
                 'share_2025': o['share_2025'], 'share_2100': o['share_2100'], 'peak_T': o['peak_T'],
                 'first_net_negative_year': o['first_net_negative_year'],
                 'rprecaut_longrun': -0.5 * 0.01 ** 2 * 5 * 47 * el ** 2}
            out.append(r); print(r, flush=True)
            save_json(out, 'eta_sweep.json')
