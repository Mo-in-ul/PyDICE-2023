"""Two-state uncertainty about carbon removal (paper 2, model v3).

The planner chooses one policy for the periods before a resolution date t_r, then learns whether removal succeeds
(state S, probability p) or fails (state F, probability 1-p) and chooses a separate policy in each state afterwards.
Welfare is the expected value p W_S + (1-p) W_F. Each state is a full DICE-2023 trajectory (removal3.trajectory) with its
own removal options; decisions in periods before t_r are shared, which is the non-anticipativity condition.

Decision vector: [common (nb blocks x k periods), tail_S (nb x (N-k)), tail_F (nb x (N-k))], k = index of t_r.
Variable bounds in common periods are the intersection of the two states' bounds, so a technology that may fail
cannot be used before the uncertainty is resolved.

The SCC in a common period is the expected marginal welfare loss of one more tonne emitted in that period (in both
states, since the period is shared), divided by the expected marginal welfare of consumption in that period."""
import time
import numpy as np, jax, jax.numpy as jnp, cyipopt
import removal3 as R3

YR = R3.YR


class TwoState:
    def __init__(self, scenario, calS, allowS, calF, allowF, p, t_res, session=None, gamma=None):
        self.prS = R3.Problem(scenario, cal=calS, allow=allowS, session=session)
        self.prF = R3.Problem(scenario, cal=calF, allow=allowF, session=session)
        assert self.prS.nb == self.prF.nb and (self.prS.capN, self.prS.capG, self.prS.capM) == (self.prF.capN, self.prF.capG, self.prF.capM)
        self.p = float(p); self.t_res = t_res
        N = self.N = self.prS.N; nb = self.nb = self.prS.nb
        k = self.k = int(np.where(YR[:N] == t_res)[0][0])
        self.nc = nb * k; self.nt = nb * (N - k); self.nv = self.nc + 2 * self.nt
        FS, FF = self.prS.F, self.prF.F
        qdS, qdF, rpS, rpF = self.prS.qd, self.prF.qd, self.prS.rp, self.prF.rp
        z = jnp.zeros(N + 1)

        def assemble(x):
            c = x[:self.nc].reshape(nb, k); tS = x[self.nc:self.nc + self.nt].reshape(nb, N - k)
            tF = x[self.nc + self.nt:].reshape(nb, N - k)
            return jnp.concatenate([c, tS], axis=1).ravel(), jnp.concatenate([c, tF], axis=1).ravel()
        self.assemble = assemble
        pp = self.p

        self.gamma = gamma

        def per_period_u(r, qd):
            cpc = 1000.0 * r['C'][1:] / qd['L'][1:]; el = qd['elasmu']
            return ((cpc ** (1 - el)) - 1) / (1 - el) - 1, qd['L'][1:] * qd['RR'][1:]

        def f(x, eE, eC):
            xS, xF = assemble(x)
            if gamma is None:
                return pp * FS['f_raw'](xS, qdS, rpS, eE, eC, z, z) + (1 - pp) * FF['f_raw'](xF, qdF, rpF, eE, eC, z, z)
            # Epstein-Zin-type risk aversion at the learning date (v0.8): the two states' post-learning welfare is mapped to
            # constant-consumption equivalents c_s, combined as [p c_S^(1-g) + (1-p) c_F^(1-g)]^(1/(1-g)), and mapped back.
            # With g = elasmu this is expected utility.
            rS, _ = FS['run'](xS, qdS, rpS, eE, eC, z, z); rF, _ = FF['run'](xF, qdF, rpF, eE, eC, z, z)
            uS, w = per_period_u(rS, qdS); uF, _ = per_period_u(rF, qdF); el = qdS['elasmu']
            kk = k                                             # first post-learning period in the [1:] arrays (index j = year YR[j])
            Wc = jnp.sum((uS * w)[:kk]); W = jnp.sum(w[kk:])
            cS = ((1 - el) * (jnp.sum((uS * w)[kk:]) / W + 1) + 1) ** (1 / (1 - el))
            cF = ((1 - el) * (jnp.sum((uF * w)[kk:]) / W + 1) + 1) ** (1 / (1 - el))
            cCE = (pp * cS ** (1 - gamma) + (1 - pp) * cF ** (1 - gamma)) ** (1 / (1 - gamma))
            uCE = (cCE ** (1 - el) - 1) / (1 - el) - 1
            return -(qdS['tstep'] * qdS['scale1'] * (Wc + W * uCE) + qdS['scale2'])

        def g(x, eE, eC):
            xS, xF = assemble(x)
            return jnp.concatenate([FS['g_raw'](xS, qdS, rpS, eE, eC, z, z), FF['g_raw'](xF, qdF, rpF, eE, eC, z, z)])
        self._f = jax.jit(lambda x: f(x, z, z)); self._grad = jax.jit(jax.grad(lambda x: f(x, z, z)))
        self._g = jax.jit(lambda x: g(x, z, z)); self._jac = jax.jit(jax.jacfwd(lambda x: g(x, z, z)))
        self._hess = jax.jit(jax.hessian(lambda x, lam, of: of * f(x, z, z) + lam @ g(x, z, z)))
        self._dfE = jax.jit(jax.grad(lambda e, x: f(x, e, z))); self._dfC = jax.jit(jax.grad(lambda e, x: f(x, z, e)))
        self._dgE = jax.jit(jax.jacfwd(lambda e, x: g(x, e, z))); self._dgC = jax.jit(jax.jacfwd(lambda e, x: g(x, z, e)))
        self.hrow, self.hcol = np.tril_indices(self.nv)
        self.z = z

    def split(self, v):
        """Split a full-length per-state array (nb*N) into common and tail parts in this solver's layout."""
        N, nb, k = self.N, self.nb, self.k
        m = v.reshape(nb, N)
        return m[:, :k].ravel(), m[:, k:].ravel()

    def bounds(self):
        lS, uS, glS, guS = self.prS.bounds(); lF, uF, glF, guF = self.prF.bounds()
        cS_l, tS_l = self.split(lS); cF_l, tF_l = self.split(lF)
        cS_u, tS_u = self.split(uS); cF_u, tF_u = self.split(uF)
        xl = np.concatenate([np.maximum(cS_l, cF_l), tS_l, tF_l]); xu = np.concatenate([np.minimum(cS_u, cF_u), tS_u, tF_u])
        xu = np.maximum(xu, xl)
        return xl, xu, np.concatenate([glS, glF]), np.concatenate([guS, guF])

    def warm(self):
        """Solve each state as a deterministic problem and use the solutions as a starting point (common periods
        taken from the more likely state)."""
        xS, iS, _, _, _ = self.prS.solve(); xF, iF, _, _, _ = self.prF.solve()
        return self.start(xS, xF, common_from='S' if self.p >= 0.5 else 'F')

    def start(self, xS=None, xF=None, common_from='S'):
        xS = self.prS.start(xS); xF = self.prF.start(xF)
        cS, tS = self.split(xS); cF, tF = self.split(xF)
        if common_from == 'F':
            cS = cF
        x0 = np.concatenate([cS, tS, tF]); xl, xu, _, _ = self.bounds()
        return np.clip(x0, xl, xu)

    def objective(self, x): return float(self._f(jnp.asarray(x)))
    def gradient(self, x): return np.asarray(self._grad(jnp.asarray(x)))
    def constraints(self, x): return np.asarray(self._g(jnp.asarray(x)))
    def jacobian(self, x): return np.asarray(self._jac(jnp.asarray(x))).ravel()
    def hessianstructure(self): return self.hrow, self.hcol
    def hessian(self, x, lam, of):
        return np.asarray(self._hess(jnp.asarray(x), jnp.asarray(lam), of))[self.hrow, self.hcol]

    def solve(self, x0=None, tol=1e-10, max_iter=5000, **opts):
        x0 = self.warm() if x0 is None else x0
        xl, xu, gl, gu = self.bounds(); x0 = np.clip(x0, xl, xu)
        nlp = cyipopt.Problem(n=self.nv, m=len(gl), problem_obj=self, lb=xl, ub=xu, cl=gl, cu=gu)
        U0 = abs(self.objective(x0))
        for kk, v in [('tol', tol), ('acceptable_tol', tol * 100), ('print_level', 0), ('max_iter', max_iter),
                      ('obj_scaling_factor', 1.0 / max(U0, 1.0)), ('mu_strategy', 'adaptive'), ('bound_relax_factor', 0.0),
                      ('honor_original_bounds', 'yes'), ('linear_solver', 'mumps')] + list(opts.items()):
            nlp.add_option(kk, v)
        t0 = time.time(); x, info = nlp.solve(x0)
        lam = jnp.asarray(info['mult_g']); xj = jnp.asarray(x); z = self.z
        dVE = np.asarray(self._dfE(z, xj)) + np.asarray(lam @ self._dgE(z, xj))
        dVC = np.asarray(self._dfC(z, xj)) + np.asarray(lam @ self._dgC(z, xj))
        scc = -1000.0 * dVE / dVC                    # meaningful in common periods (both states perturbed together)
        return x, info, scc, time.time() - t0

    def summary(self, x, info, scc):
        xS, xF = [np.asarray(v) for v in self.assemble(jnp.asarray(x))]
        out = {'p': self.p, 't_res': self.t_res, 'status': int(info['status']), 'EU': -float(self.objective(x)),
               'scc2025': float(scc[2]), 'scc2030': float(scc[3])}
        for lab, pr, xx in (('S', self.prS, xS), ('F', self.prF, xF)):
            r, _ = pr.run(xx)
            d = {'scc': np.zeros(self.N + 1), 'vN': np.zeros(self.N + 1), 'vG': np.zeros(self.N + 1), 'dVdC': -np.ones(self.N + 1)}
            s = pr.summary(xx, info, r, d)
            out[lab] = {k: s[k] for k in ('U', 'peakT', 'peakT_year', 'T2100', 'T2200', 'miu2030', 'miu2050', 'cumN', 'cumG', 'cumM',
                                          'cumABT_2100', 'cumNCO2_2100', 'cumE_2020_2100')}
            P = s['paths']; YRn = YR[:self.N]
            out[lab]['MU'] = P['MU']; out[lab]['T'] = P['T']; out[lab]['ABT'] = P['ABT']
            out[lab]['cumABT_to_res'] = float(5 * np.array(P['ABT'])[:self.k].sum())
            out[lab]['miu2040'] = float(P['MU'][int(np.where(YRn == 2040)[0][0])]); out[lab]['miu2045'] = float(P['MU'][int(np.where(YRn == 2045)[0][0])])
        out['E_peakT'] = self.p * out['S']['peakT'] + (1 - self.p) * out['F']['peakT']
        out['lamC2025'] = float(self.p * 0 + 1)                                   # placeholder; welfare converted outside
        return out
