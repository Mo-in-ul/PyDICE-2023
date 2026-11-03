"""Split dSCC/dphi into a fixed-policy (weights) part and a re-optimization part. No temperature floor.

At policy x (MIU, S fixed; alpha re-solved), the consumption-loss SCC under preference phi' is
  SCC_fp(phi', x) = -1000 * sum_s omega_s(phi') dC(s)/dE(t),  omega_s = lambda_s(phi') / lambda_t(phi'),
since eps_C(t) changes only C(t). dC/dE(t) depends on x but not on phi'. Then, with D = 0.025,
  direct   = [SCC_fp(phi+D, x(phi)) - SCC_fp(phi-D, x(phi))] / (2D)      (weights only)
  total    = [SCC(phi+D) - SCC(phi-D)] / (2D)                              (re-optimized multiplier SCC)
  indirect = total - direct                                                (policy and path response)
The direct part is also computed from the analytical derivative of omega (Eq. weight condition) as a check."""
import numpy as np, jax, jax.numpy as jnp, s2_core as S, dice_jax as dj, bv_core as B

def dCdE(pr, x, t=2):
    N = pr.N; q = pr.q; x = jnp.asarray(x); afix = x[2 * N]
    build = lambda af: jnp.concatenate([x[:2 * N], jnp.array([afix]), af])
    run = lambda af, e: pr.run(build(af), pr.z.at[t].set(e), pr.z)
    C = lambda af, e: run(af, e)[0]['C'][1:]
    R = lambda af, e: dj.irf_residual(q, *run(af, e))[1:]
    af0 = x[2 * N + 1:3 * N]
    da = -jnp.linalg.solve(jax.jacfwd(R, 0)(af0, 0.0), jax.jacfwd(R, 1)(af0, 0.0))
    dC = np.asarray(jax.jacfwd(C, 1)(af0, 0.0) + jax.jacfwd(C, 0)(af0, 0.0) @ da)
    cpc = 1000.0 * np.asarray(run(af0, 0.0)[0]['C'][1:]) / np.asarray(q['L'][1:])
    return dC, cpc

def omega_dlog(phi, cpc, t=2, sig=0.01, cap=235):
    """Analytical d ln omega_s / d phi at fixed consumption (exact form)."""
    N = len(cpc); h = 5.0 * np.arange(N); m = np.minimum(h, cap); a = 0.5 * phi ** 2 * sig ** 2
    g = h * phi * sig ** 2 * m / (1 - a * m)
    return -np.log(cpc / cpc[t - 1]) + (g - g[t - 1])

D = 0.025
out = S.load('s7_split.json') or []; done = {o['eta'] for o in out}
for e in (0.95, 1.1, 1.3, 1.5, 1.75):
    if e in done: continue
    res = {}
    x = None
    for ee in (e - D, e, e + D):
        pr = S.problem(81, round(ee, 4), tlo=None)
        x, info, n, _ = B.solve(pr, pr.start_from_gams() if x is None else x, max_iter=20000)
        scc, _, _ = pr.multiplier_scc(x, info['mult_g'])
        res[ee] = (float(scc[2]), x, pr, int(info['status']))
    pr0, x0 = res[e][2], res[e][1]
    dC, cpc = dCdE(pr0, x0)
    # consumption-loss SCC at fixed policy: -1000 sum_s omega_s dC_s (omega relative to period t = 2)
    def sfp(phi):
        q = S.problem(81, phi, tlo=None).q
        lam = cpc ** (-q['elasmu']) * np.asarray(q['RR'][1:])
        return float(-1000.0 * np.sum(lam / lam[1] * dC))
    base = sfp(e); direct = (sfp(e + D) - sfp(e - D)) / (2 * D)
    w = (lambda q: cpc ** (-q['elasmu']) * np.asarray(q['RR'][1:]))(S.problem(81, e, tlo=None).q); w = w / w[1]
    direct_an = float(-1000.0 * np.sum(w * omega_dlog(e, cpc) * dC))
    total = (res[e + D][0] - res[e - D][0]) / (2 * D)
    o = {'eta': e, 'scc_mult': res[e][0], 'scc_fp_at_x': base, 'direct': direct, 'direct_analytic': direct_an, 'total': total,
         'indirect': total - direct, 'status': [r[3] for r in res.values()]}
    out.append(o); S.save(out, 's7_split.json')
    print('split', e, o['status'], 'SCC %.3f fp %.3f' % (res[e][0], base), 'total %.2f direct %.2f (analytic %.2f) indirect %.2f' % (total, direct, direct_an, total - direct), flush=True)
