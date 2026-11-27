"""Units check: value of removal as an exact consumption equivalent (uniform percentage change in consumption in every
period, balanced-growth style) against the first-order conversion at the marginal welfare of 2025 consumption."""
import json, numpy as np, jax.numpy as jnp
from scipy.optimize import brentq
import removal3 as R3
SC = {'pub': 9, 'd3': 3, 'd2': 2}; out = {}
for c, sc in SC.items():
    res = {}
    for lab, allow in (('none', ''), ('NGM', 'NGM')):
        pr = R3.Problem(sc, cal={}, allow=allow); x, info, r, d, _ = pr.solve(); s = pr.summary(x, info, r, d)
        qd = pr.qd; C = np.asarray(r['C'])[1:]; L = np.asarray(qd['L'])[1:]; RR = np.asarray(qd['RR'])[1:]; el = float(qd['elasmu'])
        res[lab] = dict(U=s['U'], lam=s['lamC2025'] / 5, C=C, L=L, RR=RR, el=el, ts=float(qd['tstep']), s1=float(qd['scale1']), s2=float(qd['scale2']))
    b, a = res['none'], res['NGM']
    def W(rr, x):
        cpc = 1000 * rr['C'] * (1 + x) / rr['L']
        return rr['ts'] * rr['s1'] * np.sum(rr['L'] * rr['RR'] * (((cpc ** (1 - rr['el'])) - 1) / (1 - rr['el']) - 1)) + rr['s2']
    chk = (W(b, 0) - b['U'], W(a, 0) - a['U'])
    x = brentq(lambda x: W(b, x) - a['U'], -0.5, 0.5)
    pvC = np.sum(b['C'] * 5 * np.r_[0, 0, 0] .sum() + 0) if False else None
    lin = (a['U'] - b['U']) / b['lam']
    # first-order value of a uniform 1% consumption change, at the same measuring rod
    one = (W(b, 0.01) - W(b, 0)) / b['lam']
    out[c] = dict(gain_T_linear=lin, cev_pct=100 * x, check=chk, value_of_1pct_T=one, implied_pct=lin / one)
    print(c, f"linear {lin:.2f} T$  CEV {100 * x:.4f}%  1% of consumption = {one:.1f} T$  -> linear implies {lin / one:.4f}%  check {chk}")
json.dump(out, open('results/p3_cev.json', 'w'), indent=1)
