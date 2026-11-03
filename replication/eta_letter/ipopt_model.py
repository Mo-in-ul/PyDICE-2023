"""Solve PyDICE-2023 with IPOPT (exact JAX derivatives) and compute the multiplier SCC.

Why: GAMS defines the SCC as a ratio of Lagrange multipliers,
    SCC(t) = -1000 * eco2eq.m(t) / cc.m(t),
i.e. the marginal value of one more unit of emissions in the emissions equation over the
marginal value of one more unit of consumption in the consumption equation, both measured
at the optimum, with all constraints (IRF condition, temperature bounds, ...) active.
SLSQP does not return multipliers; IPOPT does.

How: in the reduced space used here (decision variables MIU, S, alpha and optionally the
capital disposal D), the two GAMS equations are not separate constraints. Their multipliers
are recovered with the envelope theorem of nonlinear programming. Add perturbations eps_E(t)
to the emissions equation and eps_C(t) to the consumption equation (dice_jax.trajectory).
At a KKT point of   min f(x; eps)  s.t.  g(x; eps) in [gl, gu],  x in [xl, xu],
with IPOPT's Lagrangian L = f + lambda' g, the value function V(eps) = f(x*(eps); eps) has
    dV/deps = df/deps + lambda' dg/deps.
With f = -UTILITY, eco2eq.m(t) = -dV/deps_E(t) and cc.m(t) = -dV/deps_C(t), so
    SCC(t) = -1000 * [dV/deps_E(t)] / [dV/deps_C(t)].
This includes the shadow prices of any binding state bounds (TATM >= 0.5, a temperature
cap), which a fixed-control perturbation cannot capture. check4_envelope.py verifies the
formula against brute-force re-optimization.

Usage (module): res = solve(scenario, disposal=False, warm='gams'); res['scc'] (1-based)
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
import cyipopt
from common import dm, gams_session, GAMS_ALPHA
import dice_jax as dj


class Problem:
    def __init__(self, scenario, disposal=False, flags=('tlo', 'miulo', 'cpcap', 'k0', 'miu1'),
                 switches=None, old_savings=False, fix_alpha=None):
        """switches: dice_jax model switches (old_land, old_ccatot, old_fghg, mat_floor);
        old_savings: the submitted model's savings rule (S >= 0.2, fixed at optlrsav from period 38)."""
        self.sc = scenario
        self.p, self.b = gams_session(scenario, flags)
        if old_savings:
            p = self.p; N = p['num_periods']
            p['sLBounds'] = np.full(N + 1, 0.2); p['sUBounds'] = np.full(N + 1, 1.0)
            p['sLBounds'][38:] = p['optlrsav']; p['sUBounds'][38:] = p['optlrsav']
        self.q = dj.pack_params(self.p)
        self.q['free_miu1'] = self.b['free_miu1']
        self.q.update(switches or {})
        # fix_alpha (1-based array): alpha held at this path and the IRF condition dropped, i.e. a
        # planner who treats alpha as given (policy_fixed_alpha.py)
        self.fix_alpha = None if fix_alpha is None else np.asarray(fix_alpha, dtype=np.float64)
        irf_on = self.fix_alpha is None
        self.N = N = self.p['num_periods']
        self.disposal = disposal
        self.nv = 3 * N + (N if disposal else 0)
        q = self.q

        def unpack(x):
            z = jnp.zeros(1)
            MIU = jnp.concatenate([z, x[0:N]])
            S = jnp.concatenate([z, x[N:2 * N]])
            al = jnp.concatenate([z, x[2 * N:3 * N]])
            D = jnp.concatenate([z, x[3 * N:4 * N]]) if disposal else None
            return MIU, S, al, D
        self.unpack = unpack

        def run(x, eE, eC):
            MIU, S, al, D = unpack(x)
            return dj.trajectory(q, MIU, S, al, eps_E=eE, eps_C=eC, D=D, clamp=False), al

        def f(x, eE, eC):
            r, _ = run(x, eE, eC)
            return -r['UTILITY']

        def g(x, eE, eC):
            r, al = run(x, eE, eC)
            parts = ([dj.irf_residual(q, r, al)] if irf_on else []) + [r['TATM'][2:]]
            if disposal:
                parts += [r['K'][2:], r['C'][1:]]
            return jnp.concatenate(parts)
        self.run = run
        z = jnp.zeros(N + 1)
        self.z = z
        self._f = jax.jit(lambda x: f(x, z, z))
        self._grad = jax.jit(jax.grad(lambda x: f(x, z, z)))
        self._g = jax.jit(lambda x: g(x, z, z))
        self._jac = jax.jit(jax.jacfwd(lambda x: g(x, z, z)))
        self._hess = jax.jit(jax.hessian(lambda x, lam, of: of * f(x, z, z) + lam @ g(x, z, z)))
        # derivatives with respect to the perturbations, at eps = 0
        self._dfdE = jax.jit(jax.grad(lambda e, x: f(x, e, z)))
        self._dfdC = jax.jit(jax.grad(lambda e, x: f(x, z, e)))
        self._dgdE = jax.jit(jax.jacfwd(lambda e, x: g(x, e, z)))
        self._dgdC = jax.jit(jax.jacfwd(lambda e, x: g(x, z, e)))
        self.f_g = f; self.g_g = g
        self.ng = (N if irf_on else 0) + (N - 1) + ((N - 1) + N if disposal else 0)
        self.hrow, self.hcol = np.tril_indices(self.nv)

    # bounds
    def bounds(self):
        N, p, b = self.N, self.p, self.b
        xl = np.concatenate([b['miu_lo'][1:], p['sLBounds'][1:], np.full(N, GAMS_ALPHA[0])])
        xu = np.concatenate([b['miu_up'][1:], p['sUBounds'][1:], np.full(N, GAMS_ALPHA[1])])
        xl[2 * N] = xu[2 * N] = p['a0']                    # alpha(1) is fixed by the initial state
        if self.fix_alpha is not None:
            xl[2 * N:3 * N] = xu[2 * N:3 * N] = self.fix_alpha[1:]; xl[2 * N] = xu[2 * N] = p['a0']
        if self.disposal:
            xl = np.concatenate([xl, np.zeros(N)]); xu = np.concatenate([xu, np.full(N, 1e4)])
            xl[3 * N] = xu[3 * N] = 0.0                    # no disposal in period 1 (K(1) fixed)
        nirf = N if self.fix_alpha is None else 0
        gl = np.concatenate([np.zeros(nirf), np.full(N - 1, b['Tlo'])])
        gu = np.concatenate([np.zeros(nirf), np.full(N - 1, b['Tup'])])
        if self.disposal:                                  # GAMS K.LO = 1, C.LO = 2
            gl = np.concatenate([gl, np.full(N - 1, 1.0), np.full(N, 2.0)])
            gu = np.concatenate([gu, np.full(N - 1, 1e8), np.full(N, 1e8)])
        return xl, xu, gl, gu

    # cyipopt callbacks
    def objective(self, x): return float(self._f(jnp.asarray(x)))
    def gradient(self, x): return np.asarray(self._grad(jnp.asarray(x)))
    def constraints(self, x): return np.asarray(self._g(jnp.asarray(x)))
    def jacobian(self, x): return np.asarray(self._jac(jnp.asarray(x))).ravel()
    def hessianstructure(self): return self.hrow, self.hcol
    def hessian(self, x, lam, of):
        H = np.asarray(self._hess(jnp.asarray(x), jnp.asarray(lam), of))
        return H[self.hrow, self.hcol]

    def start_from_gams(self):
        N, p = self.N, self.p
        gm = dm.load_gams(self.sc)
        x = np.concatenate([gm['MIU'][:N], gm['S'][:N], gm['ALPHA'][:N]])
        if self.disposal:
            K, I = gm['K'][:N], gm['I'][:N]
            D = np.zeros(N)
            D[1:] = np.maximum((1 - p['dk']) ** p['tstep'] * K[:-1] + p['tstep'] * I[:-1] - K[1:], 0.0)
            x = np.concatenate([x, D])
        xl, xu, _, _ = self.bounds()
        return np.clip(x, xl, xu)

    def start_default(self):
        prob = dm.DiceFunc(self.N, self.p)
        x = np.concatenate([prob.MIU[1:], prob.S[1:], prob.Alpha[1:]])
        if self.disposal:
            x = np.concatenate([x, np.zeros(self.N)])
        xl, xu, _, _ = self.bounds()
        return np.clip(x, xl, xu)

    def multiplier_scc(self, x, lam):
        """SCC(t), t = 1..N (1-based, index 0 unused), from the envelope theorem (see module doc)."""
        xj = jnp.asarray(x); lj = jnp.asarray(lam); z = self.z
        dVdE = np.asarray(self._dfdE(z, xj)) + np.asarray(lj @ self._dgdE(z, xj))
        dVdC = np.asarray(self._dfdC(z, xj)) + np.asarray(lj @ self._dgdC(z, xj))
        scc = np.zeros(self.N + 1)
        scc[1:] = -1000.0 * dVdE[1:] / dVdC[1:]
        return scc, dVdE, dVdC


def solve(scenario, disposal=False, warm='gams', tol=1e-10, print_level=0, flags=None,
          x0=None, eps_E=None, max_iter=3000, warm_start=False, switches=None, old_savings=False, tup=None,
          fix_alpha=None):
    """Solve one scenario. eps_E (1-based array) adds a fixed emissions perturbation, used by
    check4_envelope.py for brute-force re-optimization."""
    kw = {} if flags is None else {'flags': flags}
    pr = Problem(scenario, disposal=disposal, switches=switches, old_savings=old_savings, fix_alpha=fix_alpha, **kw)
    if tup is not None:      # override the temperature cap (onepointfive.py)
        pr.b['Tup'] = tup
    if eps_E is not None:   # rebuild the callbacks with the perturbation baked in
        eE = jnp.asarray(eps_E); z = pr.z
        pr._f = jax.jit(lambda x: pr.f_g(x, eE, z))
        pr._grad = jax.jit(jax.grad(lambda x: pr.f_g(x, eE, z)))
        pr._g = jax.jit(lambda x: pr.g_g(x, eE, z))
        pr._jac = jax.jit(jax.jacfwd(lambda x: pr.g_g(x, eE, z)))
        pr._hess = jax.jit(jax.hessian(lambda x, lam, of: of * pr.f_g(x, eE, z) + lam @ pr.g_g(x, eE, z)))
    xl, xu, gl, gu = pr.bounds()
    if x0 is None:
        x0 = pr.start_from_gams() if warm == 'gams' else pr.start_default()
    nlp = cyipopt.Problem(n=pr.nv, m=len(gl), problem_obj=pr, lb=xl, ub=xu, cl=gl, cu=gu)
    U0 = abs(pr.objective(x0))
    for k, v in (('tol', tol), ('acceptable_tol', tol * 100), ('print_level', print_level),
                 ('max_iter', max_iter), ('obj_scaling_factor', 1.0 / max(U0, 1.0)),
                 ('mu_strategy', 'adaptive'), ('bound_relax_factor', 0.0),
                 ('honor_original_bounds', 'yes'), ('linear_solver', 'mumps')):
        nlp.add_option(k, v)
    if warm_start:   # keep IPOPT from pushing a good starting point away from its bounds
        for k, v in (('warm_start_init_point', 'yes'), ('mu_init', 1e-9), ('bound_push', 1e-12),
                     ('bound_frac', 1e-12), ('warm_start_bound_push', 1e-12),
                     ('warm_start_bound_frac', 1e-12), ('warm_start_mult_bound_push', 1e-12)):
            nlp.add_option(k, v)
    t0 = time.time()
    x, info = nlp.solve(x0)
    out = {'problem': pr, 'x': x, 'info': info, 'status': int(info['status']),
           'status_msg': info['status_msg'].decode() if isinstance(info['status_msg'], bytes) else str(info['status_msg']),
           'U': -float(info['obj_val']), 'elapsed_s': time.time() - t0}
    scc, dVdE, dVdC = pr.multiplier_scc(x, info['mult_g'])
    out.update({'scc': scc, 'dVdE': dVdE, 'dVdC': dVdC})
    r, _ = pr.run(jnp.asarray(x), pr.z, pr.z)
    out['traj'] = {k: np.asarray(v) for k, v in r.items() if k != 'UTILITY'}
    out['irf_max'] = float(np.max(np.abs(np.asarray(pr._g(jnp.asarray(x)))[:pr.N]))) if pr.fix_alpha is None else None
    # analytic marginal utility as a cross-check of the cc.m multiplier
    out['ccm_analytic'] = np.concatenate([[0.0], np.asarray(dj.marginal_utility_C(pr.q, r))])
    return out
