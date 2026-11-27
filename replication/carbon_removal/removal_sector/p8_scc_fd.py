"""Finite-difference check of the 2025 SCC (26 Nov 2026, pre-submission audit). Results: results/p8_scc_fd.json.
The reported SCC is -1000 dW/dE / dW/dC, both derivatives of the Lagrangian with respect to an exogenous addition to
2025 emissions (eE) and to 2025 consumption (eC), evaluated at the optimum (removal3.Problem.duals, removal3_stoch.TwoState.solve).
Here both derivatives are computed instead by re-solving the full problem with eE = +-h and eC = +-h in the 2025 period
(central differences, all decisions re-optimised, IPOPT tol 1e-10, warm start from the unperturbed optimum).
In the two-state problems the perturbation is common to both states and the pre-learning decisions stay common, so the
nonanticipativity constraint is kept. Units as in the model: eE in 5 GtCO2 per unit (a five-year flow), eC in 5 trillion $.
Usage: python p8_scc_fd.py ITEM [ITEM ...]   ITEM = det:CAL:CAP | two:CAL:T   (CAL pub or d2; CAP 0 or 0.5 capacity scale)"""
import os, sys, json, time
import numpy as np, jax, jax.numpy as jnp
import removal3 as R3, removal3_stoch as ST
from p4_common import SC, capcal, RES

OUT = os.path.join(RES, 'p8_scc_fd.json'); I25 = 2; HS = (1e-2, 1e-3, 1e-4)


def rd(): return json.load(open(OUT)) if os.path.exists(OUT) else {}
def wr(k, v):
    d = rd(); d[k] = v; open(OUT, 'w').write(json.dumps(d, indent=1))


class DetP(R3.Problem):
    """Deterministic problem with an exogenous perturbation vector (eE, eC) built into objective and constraints."""
    def setpert(self, eE, eC):
        F = self.F; qd, rp = self.qd, self.rp; z = jnp.zeros(self.N + 1); self._e = (jnp.asarray(eE), jnp.asarray(eC))
        if not hasattr(self, '_pf'):
            self._pf = jax.jit(lambda x, e, c: F['f_raw'](x, qd, rp, e, c, z, z))
            self._pg = jax.jit(jax.grad(lambda x, e, c: F['f_raw'](x, qd, rp, e, c, z, z)))
            self._pc = jax.jit(lambda x, e, c: F['g_raw'](x, qd, rp, e, c, z, z))
            self._pj = jax.jit(jax.jacfwd(lambda x, e, c: F['g_raw'](x, qd, rp, e, c, z, z)))
            self._ph = jax.jit(jax.hessian(lambda x, lam, of, e, c: of * F['f_raw'](x, qd, rp, e, c, z, z) + lam @ F['g_raw'](x, qd, rp, e, c, z, z)))
    def objective(self, x): return float(self._pf(jnp.asarray(x), *self._e))
    def gradient(self, x): return np.asarray(self._pg(jnp.asarray(x), *self._e))
    def constraints(self, x): return np.asarray(self._pc(jnp.asarray(x), *self._e))
    def jacobian(self, x): return np.asarray(self._pj(jnp.asarray(x), *self._e)).ravel()
    def hessian(self, x, lam, of): return np.asarray(self._ph(jnp.asarray(x), jnp.asarray(lam), of, *self._e))[self.hrow, self.hcol]


class TwoP(ST.TwoState):
    def setpert(self, eE, eC):
        self._e = (jnp.asarray(eE), jnp.asarray(eC))
        if not hasattr(self, '_pf'):
            FS, FF = self.prS.F, self.prF.F; qdS, qdF, rpS, rpF = self.prS.qd, self.prF.qd, self.prS.rp, self.prF.rp
            z = self.z; pp = self.p; A = self.assemble
            def f(x, e, c):
                xS, xF = A(x); return pp * FS['f_raw'](xS, qdS, rpS, e, c, z, z) + (1 - pp) * FF['f_raw'](xF, qdF, rpF, e, c, z, z)
            def g(x, e, c):
                xS, xF = A(x); return jnp.concatenate([FS['g_raw'](xS, qdS, rpS, e, c, z, z), FF['g_raw'](xF, qdF, rpF, e, c, z, z)])
            self._pf = jax.jit(f); self._pg = jax.jit(jax.grad(f)); self._pc = jax.jit(g); self._pj = jax.jit(jax.jacfwd(g))
            self._ph = jax.jit(jax.hessian(lambda x, lam, of, e, c: of * f(x, e, c) + lam @ g(x, e, c)))
    objective = DetP.objective; gradient = DetP.gradient; constraints = DetP.constraints; jacobian = DetP.jacobian; hessian = DetP.hessian


def run(item):
    kind, cal, a = item.split(':'); sc = SC[cal]
    if kind == 'det':
        s = float(a); pr = DetP(sc, cal=capcal(s) if s > 0 else {}, allow='NGM' if s > 0 else '')
        N = pr.N; pr.setpert(np.zeros(N + 1), np.zeros(N + 1))
        x0, info, r, d, _ = pr.solve(); scc_rep = float(d['scc'][I25]); W0 = -pr.objective(x0); st0 = int(info['status'])
        solve = lambda: pr.solve(x0=x0)[:2]
    else:
        t = int(a); m = TwoP(sc, capcal(1.0), 'NGM', {}, '', 0.5, t)
        N = m.N; m.setpert(np.zeros(N + 1), np.zeros(N + 1)); pr = m
        x0, info, scc, _ = m.solve(); scc_rep = float(scc[I25]); W0 = -m.objective(x0); st0 = int(info['status'])
        solve = lambda: m.solve(x0=x0)[:2]
    out = {'scc_reported': scc_rep, 'status0': st0, 'W0': W0, 'fd': {}}
    for h in HS:
        res = {}
        for lab in ('E', 'C'):
            Wp = []
            for sgn in (1, -1):
                e = np.zeros(N + 1); e[I25] = sgn * h
                pr.setpert(e if lab == 'E' else np.zeros(N + 1), e if lab == 'C' else np.zeros(N + 1))
                x, inf = solve(); Wp.append((-pr.objective(x), int(inf['status'])))
            res['d' + lab] = (Wp[0][0] - Wp[1][0]) / (2 * h); res['st' + lab] = [Wp[0][1], Wp[1][1]]
        res['scc_fd'] = -1000.0 * res['dE'] / res['dC']; res['rel_err'] = res['scc_fd'] / scc_rep - 1
        out['fd'][str(h)] = res
        print(item, h, f"reported {scc_rep:.6f} fd {res['scc_fd']:.6f} rel {res['rel_err']:.2e} st {res['stE']}{res['stC']}", flush=True)
    pr.setpert(np.zeros(N + 1), np.zeros(N + 1)); wr(item, out)


if __name__ == '__main__':
    for it in sys.argv[1:]:
        if it in rd(): print('done', it); continue
        t0 = time.time(); run(it); print(it, 'time', round(time.time() - t0, 1), flush=True)
