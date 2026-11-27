"""Carbon-cycle sensitivity (C1). Results: results/p4_carbon.json (resumable).

DICE-2023 uses a simplified FaIR carbon cycle with the FaIR v1 feedback parameters: the 100-year integrated impulse
response target is iIRF = r0 + rU G_U + rT T with r0 = 32.4 yr, rU = 0.019 yr/GtC, rT = 4.165 yr/K. This script replaces
them with the FaIR v2 (AR6) values r0 = 29 yr, rU = 0.00846 yr/GtCO2 (0.0310 yr/GtC), rT = 4.0 yr/K and adds the
airborne term rA G_A with rA = 0.000819 yr/GtCO2 (0.0030 yr/GtC), keeping the DICE closure (exact alpha on five-year
steps), the four boxes, the forcing and the two-box temperature model. The initial alpha is re-solved so that the
first-period condition holds. This is a parameter swap inside the DICE structure, not a run of FaIR itself.
The whole process uses the AR6 values (dice_jax.irf_residual is replaced before any model function is traced).
Usage: python p4_carbon.py pub d2"""
import sys
import numpy as np, jax.numpy as jnp
from scipy.optimize import brentq
import removal3 as R3          # sets the import path for dice_jax
import dice_jax as dj
from p4_common import *

OUT = 'p4_carbon.json'
AR6 = dict(irf0=29.0, irC=0.00846 * 3.667, irT=4.0); IRA = 0.000819 * 3.667


def irf_residual_ar6(q, res, alpha):
    lhs = (q['irf0'] + q['irC'] * (res['CCATOT'][1:] - (res['MAT'][1:] - q['mateq'])) + q['irT'] * res['TATM'][1:]
           + IRA * (res['MAT'][1:] - q['mateq']))
    return dj.irf_rhs(q, alpha[1:]) - lhs
dj.irf_residual = irf_residual_ar6


def sess_ar6(sc):
    p, b = gams_session(sc); p.update(AR6)
    lhs = p['irf0'] + p['irC'] * (p['CumEmiss0'] - (p['mat0'] - p['mateq'])) + p['irT'] * p['tatm0'] + IRA * (p['mat0'] - p['mateq'])
    q = {k: float(p[k]) for k in ('emshare0', 'emshare1', 'emshare2', 'emshare3', 'tau0', 'tau1', 'tau2', 'tau3')}
    p['a0'] = float(brentq(lambda a: float(dj.irf_rhs(q, a)) - lhs, 0.01, 50.0))
    return p, b


if __name__ == '__main__':
    A = lambda y: dict(startN=float(y), startG=float(y), startM=float(y))
    for c in sys.argv[1:]:
        sc = SC[c]; out = load(OUT).get(c, {})
        runs = {'none': ({}, ''), 'central': ({}, 'NGM'), 'a2070': (A(2070), 'NGM'), 'a2090': (A(2090), 'NGM'), 'a2110': (A(2110), 'NGM'),
                'a2070_half': (capcal(0.5), 'NGM'), 'a2070_quarter': (capcal(0.25), 'NGM'), 'a2070_3q': (capcal(0.75), 'NGM'),
                'D_a2070': (capcal(1.0, VAR['D']), 'NGM'), 'D_a2070_half': (capcal(0.5, VAR['D']), 'NGM')}
        for name, (cal, allow) in runs.items():
            if name in out:
                continue
            x, s, pr = det(sc, cal, allow, sess_ar6(sc))
            out[name] = dict(rec(s, pr), a0=float(pr.p['a0'])); save(OUT, c, out)
            print(f"{c} {name}: st{s['status']} SCC {s['scc2025']:.3f} peak {s['peakT']:.4f} in {s['peakT_year']} T2100 {s['T2100']:.3f} a0 {pr.p['a0']:.4f}", flush=True)
    print('done')
