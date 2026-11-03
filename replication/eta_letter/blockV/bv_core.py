"""Block V core: isolated solve wrapper for the eta-letter experiments.

Imports eta_letter.params and ipopt_model.Problem without modifying them. Differences from
eta_letter.run are explicit arguments, never silent:
  tlo      : lower bound on TATM (0.5 = GAMS TATM.lo, 0.01 = letter 'dice' runs, None = no bound)
  eps_E/C  : additive perturbations of the emissions / consumption equations (1-based, per year)
  x0       : starting point;  warm : IPOPT warm-start options (for re-optimization after a pulse)
Returns the solution plus KKT and constraint diagnostics.
"""
import os, sys, copy, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax, jax.numpy as jnp
import cyipopt
import ipopt_model as IM
import eta_letter as EL
import dice_jax as dj

YEARS = lambda N: 2020 + 5 * np.arange(N)


def problem(N, eta, variant='dice', tlo=0.01):
    """variant in {'dice','fixed','noprec'}; tlo overrides the lower temperature bound."""
    p, b = EL.params(N, eta, variant)
    b['Tlo'] = -10.0 if tlo is None else float(tlo)
    IM.gams_session = lambda sc, flags=None, _pb=(p, b): (copy.deepcopy(_pb[0]), copy.deepcopy(_pb[1]))
    return IM.Problem(9)


def solve(pr, x0, eps_E=None, eps_C=None, tol=1e-10, max_iter=5000, warm=None, print_level=0):
    z = pr.z
    eE = z if eps_E is None else jnp.asarray(eps_E)
    eC = z if eps_C is None else jnp.asarray(eps_C)
    if eps_E is not None or eps_C is not None:
        pr._f = jax.jit(lambda x: pr.f_g(x, eE, eC))
        pr._grad = jax.jit(jax.grad(lambda x: pr.f_g(x, eE, eC)))
        pr._g = jax.jit(lambda x: pr.g_g(x, eE, eC))
        pr._jac = jax.jit(jax.jacfwd(lambda x: pr.g_g(x, eE, eC)))
        pr._hess = jax.jit(jax.hessian(lambda x, lam, of: of * pr.f_g(x, eE, eC) + lam @ pr.g_g(x, eE, eC)))
    xl, xu, gl, gu = pr.bounds()
    x0 = np.clip(x0, xl, xu)
    it = {'n': 0}
    pr.intermediate = lambda *a: it.__setitem__('n', a[1]) or True
    nlp = cyipopt.Problem(n=pr.nv, m=len(gl), problem_obj=pr, lb=xl, ub=xu, cl=gl, cu=gu)
    U0 = abs(pr.objective(x0))
    opts = [('tol', tol), ('acceptable_tol', tol * 100), ('print_level', print_level), ('max_iter', max_iter),
            ('obj_scaling_factor', 1.0 / max(U0, 1.0)), ('mu_strategy', 'adaptive'), ('bound_relax_factor', 0.0),
            ('honor_original_bounds', 'yes'), ('linear_solver', 'mumps')]
    if warm is not None:
        opts += [('warm_start_init_point', 'yes'), ('mu_init', 1e-9), ('bound_push', 1e-12), ('bound_frac', 1e-12),
                 ('warm_start_bound_push', 1e-12), ('warm_start_bound_frac', 1e-12), ('warm_start_mult_bound_push', 1e-12)]
    for k, v in opts:
        nlp.add_option(k, v)
    t0 = time.time()
    if warm is not None:
        x, info = nlp.solve(x0, lagrange=warm['mult_g'], zl=warm['mult_x_L'], zu=warm['mult_x_U'])
    else:
        x, info = nlp.solve(x0)
    del pr.intermediate
    return x, info, it['n'], time.time() - t0


def diagnostics(pr, x, info):
    """Multiplier SCC, KKT residual, active bounds, state-bound shadow prices."""
    N = pr.N
    scc, dVdE, dVdC = pr.multiplier_scc(x, info['mult_g'])
    xj = jnp.asarray(x)
    g = np.asarray(pr._grad(xj)); J = np.asarray(pr._jac(xj)); lam = np.asarray(info['mult_g'])
    xl, xu, gl, gu = pr.bounds()
    free = xu > xl                                # IPOPT removes fixed variables (make_parameter)
    stat = (g + J.T @ lam - info['mult_x_L'] + info['mult_x_U'])[free]
    r, _ = pr.run(xj, pr.z, pr.z)
    T = np.asarray(r['TATM'])[1:]; MAT = np.asarray(r['MAT'])[1:]; E = np.asarray(r['ECO2'])[1:]
    lamT = lam[N:N + N - 1]                       # multipliers of TATM(2..N) rows
    tol_b = 1e-7
    act = lambda lo, hi, v: (int(np.sum((v - lo < tol_b) & (hi > lo))), int(np.sum((hi - v < tol_b) & (hi > lo))))
    miu, s, al = x[:N], x[N:2 * N], x[2 * N:3 * N]
    Tbind = np.where(np.abs(T[1:] - pr.b['Tlo']) < 1e-6)[0] + 2   # 1-based periods
    return {
        'scc': scc, 'dVdE': dVdE, 'dVdC': dVdC,
        'kkt_stationarity_inf': float(np.max(np.abs(stat)) / max(1.0, np.max(np.abs(g[free])))),
        'iters_note': 'see solve()',
        'irf_max': float(np.max(np.abs(np.asarray(pr._g(xj))[:N]))),
        'minT': float(T.min()), 'argminT_year': int(YEARS(N)[T.argmin()]),
        'peakT': float(T[:40].max()), 'minMAT': float(MAT.min()),
        'mat_floor_active': bool(MAT.min() <= 10.0 + 1e-9),
        'Tlo': float(pr.b['Tlo']), 'Tlo_binding_years': [int(YEARS(N)[k - 1]) for k in Tbind],
        'Tlo_mult_sum': float(np.sum(np.abs(lamT))), 'Tlo_mult_max': float(np.max(np.abs(lamT))),
        'miu_at_lo_hi': act(xl[:N], xu[:N], miu), 's_at_lo_hi': act(xl[N:2 * N], xu[N:2 * N], s),
        'alpha_at_lo_hi': act(xl[2 * N:3 * N], xu[2 * N:3 * N], al),
        'U': -float(info['obj_val']), 'status': int(info['status']),
        'E': E, 'T': T,
    }
