"""SCC(2025) split by the period in which consumption is lost, at the optimum.

With MIU and S held at their optimal values, a unit change in the emissions equation in period t
changes consumption in every later period s (damages lower output; part of the loss is carried by
capital, since S is fixed). alpha is re-solved from the IRF equalities (implicit function theorem).
  dV/dE(t) |policy fixed = sum_s lambda_C(s) dC(s)/dE(t),  lambda_C(s) = dU/dC(s) (closed form)
  contribution_s = -1000 lambda_C(s) dC(s)/dE(t) / [dV/dC(t)]
By the envelope theorem the sum equals the multiplier SCC when no state bound binds; the residual
measures the shadow value of binding bounds (here the 0.5 C temperature bound).
Usage: python s3_decomp.py"""
import numpy as np, jax, jax.numpy as jnp, s2_core as S, dice_jax as dj
import bv_core as B

def decompose(pr, x, info, t=2):
    N = pr.N; q = pr.q
    x = jnp.asarray(x); afix = x[2 * N]
    def build(af):
        return jnp.concatenate([x[:2 * N], jnp.array([afix]), af])
    def run(af, e):
        eE = pr.z.at[t].set(e)
        r, al = pr.run(build(af), eE, pr.z)
        return r, al
    def C(af, e):
        r, _ = run(af, e); return r['C'][1:]
    def R(af, e):
        r, al = run(af, e); return dj.irf_residual(q, r, al)[1:]
    af0 = x[2 * N + 1:3 * N]
    dR_da = jax.jacfwd(R, 0)(af0, 0.0); dR_de = jax.jacfwd(R, 1)(af0, 0.0)
    da_de = -jnp.linalg.solve(dR_da, dR_de)
    dC_de = jax.jacfwd(C, 1)(af0, 0.0) + jax.jacfwd(C, 0)(af0, 0.0) @ da_de
    r, _ = run(af0, 0.0)
    lam = np.asarray(dj.marginal_utility_C(q, r))
    scc, dVdE, dVdC = pr.multiplier_scc(np.asarray(x), info['mult_g'])
    contrib = 1000.0 * lam * np.asarray(dC_de) / dVdC[t]   # dVdC is d(-U)/dC
    return contrib, float(scc[t]), float(np.max(np.abs(np.asarray(R(af0, 0.0)))))

YR = 2020 + 5 * np.arange(81)
BANDS = {'2025-2100': (YR >= 2025) & (YR <= 2100), '2105-2255': (YR > 2100) & (YR <= 2255), '2260-2420': YR > 2255}
ETAS = (0.5, 0.75, 0.95, 1.1, 1.3, 1.5, 1.75)
cal = {o['eta']: o['rho_e'] for o in S.load('s2_rho_h80.json')}
CASES = {
    'dice':          lambda e: dict(),
    'dice_nobound':  lambda e: dict(tlo=None),
    'fixed_prec':    lambda e: dict(sigma=0.01 * 0.95 / e),     # rprecaut held at its phi = 0.95 value
    'no_prec':       lambda e: dict(sigma=0.0),
    'recal_h80':     lambda e: dict(rho_e=cal[e]),
    'sigma_0.75':    lambda e: dict(sigma=0.0075),
    'sigma_1.25':    lambda e: dict(sigma=0.0125),
}
out = S.load('s3_decomp.json') or []
done = {(o['case'], o['eta']) for o in out}
for case, kw in CASES.items():
    xprev = None
    for e in ETAS:
        pr = S.problem(81, e, **kw(e))
        x0 = pr.start_from_gams() if xprev is None else xprev
        x, info, n, _ = B.solve(pr, x0, max_iter=20000)
        xprev = x
        if (case, e) in done: continue
        c, m, irf = decompose(pr, x, info)
        d = B.diagnostics(pr, x, info)
        o = {'case': case, 'eta': e, 'status': int(info['status']), 'scc_mult': m, 'scc_sum': float(c.sum()),
             'resid': m - float(c.sum()), 'resid_pct': 100 * (m - float(c.sum())) / m, 'minT': d['minT'],
             'bands': {k: float(c[v].sum()) for k, v in BANDS.items()}, 'contrib': c.tolist()}
        out.append(o); S.save(out, 's3_decomp.json')
        print(case, e, o['status'], round(m, 3), round(o['scc_sum'], 3), '%.2e%%' % o['resid_pct'],
              {k: round(v, 2) for k, v in o['bands'].items()}, flush=True)
