"""Experiments for the letter on the elasticity of marginal utility (eta) in DICE-2023.

All runs: cost-benefit optimum (DICE-2023 'opt' settings, lower temperature bound relaxed to 0.01),
solved with IPOPT (ipopt_model.solve), multiplier SCC. Every parameter except eta at its DICE value.
Variants
  dice   : discounting as in the DICE-2023 program, RR(t) = RR1(t) (1 + rprecaut(t))^(-5(t-1)),
           rprecaut(t) = -0.5 varpcc(t) eta^2, varpcc(t) = siggc1^2 * 5 * min(t-1, 47)
  fixed  : rprecaut held at its value for eta = 0.95, so eta acts only through utility curvature
  noprec : rprecaut = 0 (no precautionary term)
Experiments
  grid    : eta = 0.50, 0.55, ..., 2.00 for dice and fixed (N = 81); noprec on a coarser grid
  nofloor : dice discounting without the lower temperature bound, eta >= 1.45 (the 0.01 C bound binds from eta = 1.6)
  horizon : N = 61, 81, 101 periods (horizon 2320, 2420, 2520) for eta in {0.5, 0.95, 1.25, 1.5, 1.75, 2.0}, dice
  paths   : consumption discount rate r(t) = (lambda_C(t)/lambda_C(t+1))^(1/5) - 1 along each optimum,
            from the analytic marginal utility of consumption (GAMS cc.m)
Writes results/eta_letter_grid.json and results/eta_letter_horizon.json (incrementally).
Usage: python eta_letter.py grid | horizon
"""
import sys, copy
import numpy as np
import ipopt_model as IM
from common import dm, save_json, load_json, RES
import os

ETA_GRID = np.round(np.arange(0.5, 2.0001, 0.05), 2)
ETA_COARSE = (0.5, 0.75, 0.95, 1.25, 1.5, 1.75, 2.0)


def params(N, eta, variant):
    m = dm.Dice2023Model(N, 9)
    p = m.params
    p['rartp'] = np.exp(p['prstp'] + p['betaclim'] * p['pi']) - 1
    p['elasmu'] = 1.0 + 1e-4 if abs(eta - 1.0) < 1e-12 else eta   # DICE utility (c^(1-eta)-1)/(1-eta) is singular at 1
    t = np.arange(1, N + 1)
    e_prec = {'dice': eta, 'dice_nofloor': eta, 'fixed': 0.95, 'noprec': 0.0}[variant]
    p['rprecaut'][1:] = -0.5 * p['varpcc'][1:] * e_prec ** 2
    p['RR1'][1:] = 1 / ((1 + p['rartp']) ** (p['tstep'] * (t - 1)))
    p['RR'][1:] = p['RR1'][1:] * (1 + p['rprecaut'][1:]) ** (-p['tstep'] * (t - 1))
    miu_up = p['miuup'].copy(); miu_lo = np.zeros(N + 1)
    miu_lo[1] = miu_up[1] = p['miu1']
    # dice_nofloor: no lower temperature bound (damages a2 T^2 are symmetric, so T < 0 is never optimal)
    b = {'miu_lo': miu_lo, 'miu_up': miu_up, 'Tlo': -10.0 if variant == 'dice_nofloor' else 0.01, 'Tup': 20.0, 'free_miu1': False}
    return p, b


def run(N, eta, variant, x0, max_iter=3000, tol=1e-10):
    p, b = params(N, eta, variant)
    IM.gams_session = lambda sc, flags=None, _pb=(p, b): (copy.deepcopy(_pb[0]), copy.deepcopy(_pb[1]))
    r = IM.solve(9, x0=x0, tol=tol, max_iter=max_iter)
    lam = r['ccm_analytic'][1:]
    rate = (lam[:-1] / lam[1:]) ** (1 / 5) - 1
    T = r['traj']['TATM'][1:]; E = r['traj']['ECO2'][1:]
    yrs = 2020 + 5 * np.arange(N)
    o = {'N': N, 'eta': float(eta), 'variant': variant, 'status': r['status'], 'U': r['U'],
         'scc_2025': float(r['scc'][2]), 'scc_2050': float(r['scc'][7]), 'scc_2100': float(r['scc'][17]),
         'peak_T': float(T[:40].max()), 'min_T': float(T.min()), 'T_2100': float(T[16]),
         'first_net_negative_year': int(yrs[E < 0][0]) if np.any(E < 0) else None,
         'cum_E_2100': float(5 * E[1:17].sum()), 'miu_2030': float(r['x'][2]), 'miu_2050': float(r['x'][6]),
         'rate_path': rate.tolist(), 'rprecaut': p['rprecaut'][1:].tolist(),
         'cpc': (1000 * r['traj']['C'][1:] / p['L'][1:]).tolist(), 'irf_max': r['irf_max']}
    return o, r['x']


def save(out, what):
    save_json(out, f'eta_letter_{what}.json')


if __name__ == '__main__':
    what = sys.argv[1]
    path = os.path.join(RES, f'eta_letter_{what}.json')
    out = load_json(f'eta_letter_{what}.json') if os.path.exists(path) else []
    done = {(o['N'], o['eta'], o['variant']) for o in out}
    jobs = []
    if what == 'grid':
        jobs = [(81, e, 'dice') for e in ETA_GRID] + [(81, e, 'fixed') for e in ETA_GRID] + [(81, e, 'noprec') for e in ETA_COARSE]
    elif what == 'nofloor':
        jobs = [(81, e, 'dice_nofloor') for e in ETA_GRID if e >= 1.45] + [(N, e, 'dice_nofloor') for N in (61, 101) for e in (1.5, 1.75, 2.0)]
    elif what == 'horizon':
        jobs = [(N, e, 'dice') for N in (61, 101) for e in ETA_COARSE] + [(N, e, 'fixed') for N in (61, 101) for e in ETA_COARSE]
    x_prev = {}
    for N, e, v in jobs:
        if (N, float(e), v) in done:
            continue
        x0 = x_prev.get((N, v))
        if x0 is None:
            if N == 81:
                p, b = params(N, e, v)
                IM.gams_session = lambda sc, flags=None, _pb=(p, b): (copy.deepcopy(_pb[0]), copy.deepcopy(_pb[1]))
                x0 = IM.Problem(9).start_from_gams()
            else:
                p, b = params(N, e, v)
                IM.gams_session = lambda sc, flags=None, _pb=(p, b): (copy.deepcopy(_pb[0]), copy.deepcopy(_pb[1]))
                x0 = IM.Problem(9).start_default()
        o, x = run(N, e, v, x0)
        x_prev[(N, v)] = x
        out.append(o); save(out, what)
        print(N, e, v, o['status'], round(o['scc_2025'], 2), round(o['peak_T'], 3), flush=True)
