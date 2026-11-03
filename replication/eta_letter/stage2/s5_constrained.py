"""Exact SCC accounting when the temperature floor binds.

IPOPT minimizes f = -W with Lagrangian f + lam_R' R + lam_T' T. With alpha interior, stationarity in alpha
lets the envelope derivative be written as
  dVf/de = d f^fp/de + lam_T' dT/de,
where both total derivatives hold MIU and S at the optimum and move alpha through the IRF equalities (IFT).
So SCC(t) = sum_s consumption-loss contribution(s) + sum_k floor contribution(k), with
  consumption contribution(s) =  1000 lambda_C(s) dC(s)/dE(t) / dVf/dC(t)
  floor contribution(k)       = -1000 lam_T(k) dT(k)/dE(t)   / dVf/dC(t)
and no residual. Usage: python s5_constrained.py"""
import numpy as np, jax, jax.numpy as jnp, s2_core as S, dice_jax as dj, bv_core as B

def accounting(pr, x, info, t=2):
    N = pr.N; q = pr.q; x = jnp.asarray(x); afix = x[2 * N]
    build = lambda af: jnp.concatenate([x[:2 * N], jnp.array([afix]), af])
    def run(af, e):
        r, al = pr.run(build(af), pr.z.at[t].set(e), pr.z); return r, al
    C = lambda af, e: run(af, e)[0]['C'][1:]
    T = lambda af, e: run(af, e)[0]['TATM'][2:]
    R = lambda af, e: dj.irf_residual(q, *run(af, e))[1:]
    af0 = x[2 * N + 1:3 * N]
    da = -jnp.linalg.solve(jax.jacfwd(R, 0)(af0, 0.0), jax.jacfwd(R, 1)(af0, 0.0))
    dC = np.asarray(jax.jacfwd(C, 1)(af0, 0.0) + jax.jacfwd(C, 0)(af0, 0.0) @ da)
    dT = np.asarray(jax.jacfwd(T, 1)(af0, 0.0) + jax.jacfwd(T, 0)(af0, 0.0) @ da)
    lam = np.asarray(dj.marginal_utility_C(q, run(af0, 0.0)[0]))
    scc, dVdE, dVdC = pr.multiplier_scc(np.asarray(x), info['mult_g'])
    lamT = np.asarray(info['mult_g'])[N:N + N - 1]
    cons = 1000.0 * lam * dC / dVdC[t]                      # periods 1..N
    floor = np.zeros(N); floor[1:] = -1000.0 * lamT * dT / dVdC[t]   # periods 2..N
    return cons, floor, float(scc[t])

YR = 2020 + 5 * np.arange(81)
BANDS = {'2025-2100': (YR >= 2025) & (YR <= 2100), '2105-2255': (YR > 2100) & (YR <= 2255), '2260-2420': YR > 2255}
cal = {o['eta']: o['rho_e'] for o in S.load('s2_rho_h80.json')}
JOBS = [('dice', e, {}) for e in (1.1, 1.3, 1.5, 1.75)] + [('dice_nobound', 1.75, {'tlo': None})] + \
       [('recal_h80', e, {'rho_e': cal[e]}) for e in (1.3, 1.5)] + [('sigma_1.25', e, {'sigma': 0.0125}) for e in (0.95, 1.3)]
out = S.load('s5_constrained.json') or []; done = {(o['case'], o['eta']) for o in out}
for case, e, kw in JOBS:
    if (case, e) in done: continue
    pr = S.problem(81, e, **kw); x, info, n, _ = B.solve(pr, pr.start_from_gams(), max_iter=20000)
    if np.any(np.asarray(info['mult_x_L'])[2 * pr.N + 1:3 * pr.N] > 1e-8):
        print('warning: alpha at a bound; accounting assumes interior alpha', flush=True)
    cons, floor, m = accounting(pr, x, info)
    tot = cons.sum() + floor.sum()
    o = {'case': case, 'eta': e, 'status': int(info['status']), 'scc_mult': m, 'cons_sum': float(cons.sum()), 'floor_sum': float(floor.sum()),
         'resid_pct': 100 * (m - tot) / m, 'cons_bands': {k: float(cons[v].sum()) for k, v in BANDS.items()},
         'floor_bands': {k: float(floor[v].sum()) for k, v in BANDS.items()},
         'floor_years': [int(YR[i]) for i in np.where(np.abs(floor) > 1e-3)[0]]}
    out.append(o); S.save(out, 's5_constrained.json')
    print(case, e, o['status'], round(m, 3), 'cons', round(o['cons_sum'], 2), 'floor', round(o['floor_sum'], 2), 'resid %.1e%%' % o['resid_pct'],
          o['floor_years'][:1], o['floor_years'][-1:], flush=True)
