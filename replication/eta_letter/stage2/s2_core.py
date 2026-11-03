"""Stage 2 core: eta-letter model with explicit growth-uncertainty (sigma), precautionary cap and
effective time-preference (rho_e) settings. Imports the baseline modules without modifying them.

rprecaut(t) = -0.5 * sigma^2 * 5 * min(t-1, cap) * eta^2   (DICE-2023: sigma = 0.01, cap = 47 periods)
RR(t)       = (1+rho_e)^(-5(t-1)) * (1+rprecaut(t))^(-5(t-1)),  rho_e = exp(rho + beta*pi) - 1 (DICE: 0.0263)
In the DICE-2023 code rho and beta enter only through rho_e (checked: rartp is used only in RR1 and in
optlrsav, which is unused when the savings rate is fixed at 0.28 from 2205).
Default lower temperature bound: 0.5 C, as in the GAMS program (Stage 2 baseline)."""
import os, sys, copy
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'blockV'))
import numpy as np
import jax.numpy as jnp
import ipopt_model as IM
import eta_letter as EL
import dice_jax as dj
import bv_core as B
import json

SIG0, CAP0 = 0.01, 47


def problem(N, eta, sigma=SIG0, cap=CAP0, rho_e=None, tlo=0.5):
    p, b = EL.params(N, eta, 'dice')
    t = np.arange(1, N + 1)
    if rho_e is not None:
        p['rartp'] = rho_e
    p['varpcc'][1:] = sigma ** 2 * 5 * np.minimum(t - 1, cap)
    p['rprecaut'][1:] = -0.5 * p['varpcc'][1:] * p['elasmu'] ** 2
    p['RR1'][1:] = 1 / ((1 + p['rartp']) ** (p['tstep'] * (t - 1)))
    p['RR'][1:] = p['RR1'][1:] * (1 + p['rprecaut'][1:]) ** (-p['tstep'] * (t - 1))
    b['Tlo'] = -10.0 if tlo is None else float(tlo)
    IM.gams_session = lambda sc, flags=None, _pb=(p, b): (copy.deepcopy(_pb[0]), copy.deepcopy(_pb[1]))
    return IM.Problem(9)


def solve(pr, x0, max_iter=int(os.environ.get('S2_MAXIT', 20000))):
    x, info, n, dt = B.solve(pr, x0, max_iter=max_iter)
    d = B.diagnostics(pr, x, info)
    r, _ = pr.run(jnp.asarray(x), pr.z, pr.z)
    lam = np.asarray(dj.marginal_utility_C(pr.q, r))          # periods 1..N, includes RR
    N = pr.N
    o = {'N': N, 'status': d['status'], 'iters': n, 'kkt': d['kkt_stationarity_inf'], 'irf_max': d['irf_max'],
         'U': d['U'], 'scc_2025': d['scc'][2], 'scc_2050': d['scc'][7], 'scc_2100': d['scc'][17],
         'peakT': d['peakT'], 'minT': d['minT'], 'Tlo': d['Tlo'],
         'Tlo_binds': bool(d['minT'] - d['Tlo'] < 1e-6), 'alpha_at_lo': d['alpha_at_lo_hi'][0],
         'cumE_2100': float(5 * np.sum(d['E'][1:17])), 'scc_path': d['scc'][1:].tolist(),
         'lamC': lam.tolist()}
    return o, x


def R_h(lam, h):
    """Consumption discount rate from 2020 over h years (h multiple of 5), from lambda_C."""
    k = h // 5
    return (lam[0] / lam[k]) ** (1.0 / h) - 1


def save(obj, name):
    json.dump(obj, open(os.path.join(HERE, 'results', name), 'w'))


def load(name):
    p = os.path.join(HERE, 'results', name)
    return json.load(open(p)) if os.path.exists(p) else None
