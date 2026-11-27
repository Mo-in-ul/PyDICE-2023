"""n-state uncertainty about carbon removal (paper 2, revision analysis). Generalises removal3_stoch.TwoState.

The planner chooses one policy for the periods before the learning date t_r, then learns which of n states holds
(probabilities w_i) and chooses a separate policy in each state. Welfare is sum_i w_i W_i. Each state is a full
removal3 trajectory with its own removal options; decisions before t_r are shared.

Decision vector: [common (nb blocks x k periods), tail_1 (nb x (N-k)), ..., tail_n]. Bounds in common periods are the
intersection of the states' bounds. The SCC in a common period is the expected marginal welfare loss of a tonne emitted
in that period in every state, divided by the expected marginal welfare of consumption in that period.
With n = 2 this is removal3_stoch.TwoState under expected utility (checked in p4_nstate.py)."""
import time
import numpy as np, jax, jax.numpy as jnp, cyipopt
import removal3 as R3

YR = R3.YR


class NState:
    def __init__(self, scenario, states, t_res, session=None):
        """states: list of (cal, allow, probability)."""
        self.prs = [R3.Problem(scenario, cal=c, allow=a, session=session) for c, a, _ in states]
        self.w = np.array([float(s[2]) for s in states]); assert abs(self.w.sum() - 1.0) < 1e-12
        self.n = n = len(states); self.t_res = t_res
        N = self.N = self.prs[0].N; nb = self.nb = self.prs[0].nb
        assert all(pr.nb == nb for pr in self.prs)
        k = self.k = int(np.where(YR[:N] == t_res)[0][0])
        self.nc = nb * k; self.nt = nb * (N - k); self.nv = self.nc + n * self.nt
        Fs = [pr.F for pr in self.prs]; qds = [pr.qd for pr in self.prs]; rps = [pr.rp for pr in self.prs]
        z = jnp.zeros(N + 1); w = self.w; nc, nt = self.nc, self.nt

        def assemble(x):
            c = x[:nc].reshape(nb, k)
            return [jnp.concatenate([c, x[nc + i * nt:nc + (i + 1) * nt].reshape(nb, N - k)], axis=1).ravel() for i in range(n)]
        self.assemble = assemble

        def f(x, eE, eC):
            xs = assemble(x)
            return sum(w[i] * Fs[i]['f_raw'](xs[i], qds[i], rps[i], eE, eC, z, z) for i in range(n))

        def g(x, eE, eC):
            xs = assemble(x)
            return jnp.concatenate([Fs[i]['g_raw'](xs[i], qds[i], rps[i], eE, eC, z, z) for i in range(n)])
        self._f = jax.jit(lambda x: f(x, z, z)); self._grad = jax.jit(jax.grad(lambda x: f(x, z, z)))
        self._g = jax.jit(lambda x: g(x, z, z)); self._jac = jax.jit(jax.jacfwd(lambda x: g(x, z, z)))
        self._hess = jax.jit(jax.hessian(lambda x, lam, of: of * f(x, z, z) + lam @ g(x, z, z)))
        self._dfE = jax.jit(jax.grad(lambda e, x: f(x, e, z))); self._dfC = jax.jit(jax.grad(lambda e, x: f(x, z, e)))
        self._dgE = jax.jit(jax.jacfwd(lambda e, x: g(x, e, z))); self._dgC = jax.jit(jax.jacfwd(lambda e, x: g(x, z, e)))
        self.hrow, self.hcol = np.tril_indices(self.nv)
        self.z = z

    def split(self, v):
        m = np.asarray(v).reshape(self.nb, self.N)
        return m[:, :self.k].ravel(), m[:, self.k:].ravel()

    def bounds(self):
        bs = [pr.bounds() for pr in self.prs]
        sp_l = [self.split(b[0]) for b in bs]; sp_u = [self.split(b[1]) for b in bs]
        cl = np.max([s[0] for s in sp_l], axis=0); cu = np.min([s[0] for s in sp_u], axis=0)
        xl = np.concatenate([cl] + [s[1] for s in sp_l]); xu = np.concatenate([cu] + [s[1] for s in sp_u])
        xu = np.maximum(xu, xl)
        return xl, xu, np.concatenate([b[2] for b in bs]), np.concatenate([b[3] for b in bs])

    def start(self, xs, common_from=0):
        """xs: one full decision vector per state (e.g. the deterministic optima); common periods from state common_from."""
        sp = [self.split(pr.start(x)) for pr, x in zip(self.prs, xs)]
        x0 = np.concatenate([sp[common_from][0]] + [s[1] for s in sp]); xl, xu, _, _ = self.bounds()
        return np.clip(x0, xl, xu)

    def objective(self, x): return float(self._f(jnp.asarray(x)))
    def gradient(self, x): return np.asarray(self._grad(jnp.asarray(x)))
    def constraints(self, x): return np.asarray(self._g(jnp.asarray(x)))
    def jacobian(self, x): return np.asarray(self._jac(jnp.asarray(x))).ravel()
    def hessianstructure(self): return self.hrow, self.hcol
    def hessian(self, x, lam, of):
        return np.asarray(self._hess(jnp.asarray(x), jnp.asarray(lam), of))[self.hrow, self.hcol]

    def solve(self, x0, tol=1e-10, max_iter=5000):
        xl, xu, gl, gu = self.bounds(); x0 = np.clip(x0, xl, xu)
        nlp = cyipopt.Problem(n=self.nv, m=len(gl), problem_obj=self, lb=xl, ub=xu, cl=gl, cu=gu)
        U0 = abs(self.objective(x0))
        for kk, v in [('tol', tol), ('acceptable_tol', tol * 100), ('print_level', 0), ('max_iter', max_iter),
                      ('obj_scaling_factor', 1.0 / max(U0, 1.0)), ('mu_strategy', 'adaptive'), ('bound_relax_factor', 0.0),
                      ('honor_original_bounds', 'yes'), ('linear_solver', 'mumps')]:
            nlp.add_option(kk, v)
        t0 = time.time(); x, info = nlp.solve(x0)
        lam = jnp.asarray(info['mult_g']); xj = jnp.asarray(x); z = self.z
        dVE = np.asarray(self._dfE(z, xj)) + np.asarray(lam @ self._dgE(z, xj))
        dVC = np.asarray(self._dfC(z, xj)) + np.asarray(lam @ self._dgC(z, xj))
        with np.errstate(all='ignore'):
            scc = -1000.0 * dVE / dVC                              # meaningful in common periods
        return x, info, scc, time.time() - t0

    def state_paths(self, x):
        """Control rate, abatement, temperature and welfare of each state at the solution."""
        out = []
        for pr, xx in zip(self.prs, [np.asarray(v) for v in self.assemble(jnp.asarray(x))]):
            r, _ = pr.run(xx); N = self.N
            MU = np.array(pr.unpack(jnp.asarray(xx))[0])[1:]; MU[0] = pr.p['miu1']
            YG = np.asarray(r['YGROSS'])[1:]; T = np.asarray(r['TATM'])[1:]
            ABT = (np.asarray(pr.q['sigma'][1:]) * YG + np.asarray(pr.q['eland'][1:])) * MU
            out.append({'U': float(r['UTILITY']), 'peakT': float(T.max()), 'peakT_year': int(YR[int(np.argmax(T))]),
                        'cumABT_to_res': float(5 * ABT[:self.k].sum()), 'MU': MU.tolist(), 'T': T.tolist(), 'x': xx})
        return out
