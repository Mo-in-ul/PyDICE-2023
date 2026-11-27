"""Shared helpers for the p4 runs (paper 2, revision analysis of 2026-11-11). Nothing here changes removal3.py.

Welfare differences are converted to trillion 2019 dollars of 2025 consumption with lam = lamC2025 / 5 of the no-removal
run of the same calibration, as in replicate.py."""
import os, sys, json
import numpy as np
import removal3 as R3, removal3_stoch as ST
from common import gams_session

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, 'results')
YR = R3.YR[:81]
iy = lambda y: int(np.where(YR == y)[0][0])
SC = {'pub': 9, 'd1': 1, 'd2': 2, 'd3': 3, 'd4': 4, 'd5': 5}
A70 = dict(startN=2070.0, startG=2070.0, startM=2070.0)
CAP = dict(RN_inf=5.0, RG_inf=17.0, RM_inf=4.0)
VAR = {'A': {}, 'D': dict(capN=None, capG=None),          # A hard storage caps (central), D no storage limit
       'H': dict(capN=None, capG=None, tlo_set=-3.0)}     # H no storage limit and no lower bound on temperature


def load(name):
    f = os.path.join(RES, name)
    return json.load(open(f)) if os.path.exists(f) else {}


def save(name, key, val):
    """Read-modify-write under an exclusive lock with an atomic replace, so parallel processes can share one file."""
    import fcntl
    f = os.path.join(RES, name)
    with open(f + '.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        r = json.load(open(f)) if os.path.exists(f) else {}
        r[key] = val
        json.dump(r, open(f + '.tmp', 'w')); os.replace(f + '.tmp', f)
        fcntl.flock(lk, fcntl.LOCK_UN)


def session(sc, abcost=1.0, elasmu=None, rr_kappa=None):
    """GAMS session for scenario sc with optional changes: abatement cost multiplier, elasticity of marginal utility,
    and discount factors RR raised to the power rr_kappa (scales the utility discount rate in every period)."""
    p, b = gams_session(sc)
    if abcost != 1.0:
        p['cost1tot'] = np.asarray(p['cost1tot']) * abcost
    if elasmu is not None:
        p['elasmu'] = float(elasmu)
    if rr_kappa is not None:
        p['RR'] = np.asarray(p['RR'], float) ** rr_kappa
    return p, b


def capcal(s, base=None, start=A70, cm=1.0):
    """Calibration with removal capacity scaled by s (all three families)."""
    return dict(start, **(base or {}), **{k: s * cm * v for k, v in CAP.items()})


def dice_ceiling(pr):
    """Published DICE-2023 ceiling on the control rate (1.10 from 2120, 1.05 from 2200, 1.00 from 2300), with the
    session's carbon-price cap applied as in removal3.mu_ceiling but without the cap at one."""
    up = pr.b['miu_up'].copy(); p = pr.p
    if pr.sc in R3.CPCAP and pr.cpcap:
        up[1:] = np.minimum(up[1:], (R3.CPCAP[pr.sc] / p['PBACKTIME'][1:]) ** (1 / (p['expcost2'] - 1)))
    return up


def det(sc, cal, allow, sess=None, up=None, fix=None, x0=None, tol=1e-10):
    """Deterministic solve. up: 'dice' for the published ceiling. fix: full decision vector whose control-rate block is
    imposed in every period (savings, removal and the carbon-cycle variable stay free)."""
    pr = R3.Problem(sc, cal=cal, allow=allow, session=sess)
    if up == 'dice':
        pr.up = dice_ceiling(pr)
    if fix is not None:
        pr.fixed = (np.asarray(fix), pr.N); pr.fixed_blocks = [0]
    x, info, r, d, dt = pr.solve(x0=x0 if x0 is not None else fix, tol=tol)
    s = pr.summary(x, info, r, d)
    return x, s, pr


def rec(s, pr):
    """Compact record of a run, with the diagnostics on the control-rate bounds."""
    P = s['paths']; MU = np.array(P['MU']); up = np.asarray(pr.up[1:pr.N + 1])
    YRn = R3.YR[:len(MU)]
    cap = np.where(MU >= np.minimum(up, 1.0) - 1e-6)[0]; cap = cap[(YRn[cap] >= 2070)]
    near = [int(y) for y, m, u in zip(YR, MU, up) if 2025 <= y < 2070 and m >= u - 1e-6]
    i22 = iy(2200) + 1
    return {'status': s['status'], 'U': s['U'], 'scc2025': s['scc2025'], 'peakT': s['peakT'], 'peakT_year': s['peakT_year'],
            'T2100': s['T2100'], 'lamC2025': s['lamC2025'], 'cumABT_2100': s['cumABT_2100'], 'cumABT_2200': s['cumABT_2200'],
            'cumR_2200': s['cumN_2200'] + s['cumG_2200'] + s['cumM_2200'], 'cumR_2100': s['cumN_2100'] + s['cumG_2100'] + s['cumM_2100'],
            'first_at_max': int(YRn[cap[0]]) if len(cap) else None, 'n_at_max_2200': int(np.sum(YRn[cap] <= 2200)),
            'last_on_ramp': max(near) if near else None, 'n_on_ramp': len(near), 'mu_max': float(MU.max()),
            'MU': MU.tolist(), 'T': P['T'], 'ABT': P['ABT'], 'E': P['E']}


def jensen(curve, p=0.5):
    """Jensen gap of a two-point lottery over capacity {0, 1} with success probability p, from a curve keyed by scale."""
    return p * curve['1.0']['scc2025'] + (1 - p) * curve['0.0']['scc2025'] - curve[str(p)]['scc2025']
