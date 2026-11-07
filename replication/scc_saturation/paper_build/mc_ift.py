"""Monte Carlo of the SCC and its saturation-feedback share.

Design: the submitted paper's Latin hypercube (1000 draws, seed 42) over
  psi2 (damage coefficient a2base) [0.002, 0.010], rho (prstp) [0.001, 0.050],
  irC [0.010, 0.030], irT [2.5, 6.0], eland0 [3, 9] GtCO2/yr.
Each draw is the cost-benefit optimal run (GAMS opt settings) solved with IPOPT. For each
draw, at 2025, 2050 and 2100:
  * the multiplier SCC (GAMS definition, ipopt_model.py),
  * the implicit-function-theorem decomposition on the draw's optimal controls (check3_ift.py):
    direct (alpha fixed) + feedback = total; share = feedback / total.
Where no state bound binds, total equals the multiplier SCC (check 4); the gap between them
is recorded as a check, and draws where it exceeds 0.1% are flagged.
The JAX functions take the parameters as arguments, so they compile once per process.
Usage: python mc_ift.py <first> <last> [--every K | --draws k1,k2,...]  (draw indices, half-open)
       -> results/mc/mc_<first>_<last>.json
"""
import os, sys, time, copy
import numpy as np
import jax
import jax.numpy as jnp
import cyipopt
from scipy.optimize import brentq
from scipy.stats import qmc
from common import dm, save_json, gams_session, GAMS_ALPHA, RES
import dice_jax as dj

BOUNDS = {'psi2': (0.002, 0.010), 'rho': (0.001, 0.050), 'irC': (0.010, 0.030), 'irT': (2.5, 6.0), 'eland0': (3.0, 9.0)}
YEARS = (2025, 2050, 2100)
N = 81


def design(n=1000, seed=42):
    keys = list(BOUNDS)
    lo = np.array([BOUNDS[k][0] for k in keys]); hi = np.array([BOUNDS[k][1] for k in keys])
    return keys, qmc.scale(qmc.LatinHypercube(d=len(keys), seed=seed).random(n=n), lo, hi)


def draw_params(row):
    """Cost-benefit optimal params (GAMS opt settings) with one draw applied."""
    p, b = gams_session(9)
    psi2, rho, irC, irT, eland0 = row
    p['a2base'] = psi2; p['prstp'] = rho; p['irC'] = irC; p['irT'] = irT; p['eland0'] = eland0
    t = np.arange(1, N + 1)
    p['eland'][1:] = eland0 * (1 - p['deland']) ** (t - 1)
    p['rartp'] = np.exp(p['prstp'] + p['betaclim'] * p['pi']) - 1
    p['RR1'][1:] = 1 / ((1 + p['rartp']) ** (p['tstep'] * (t - 1)))
    p['RR'][1:] = p['RR1'][1:] * (1 + p['rprecaut'][1:]) ** (-p['tstep'] * (t - 1))
    lhs = p['irf0'] + p['irC'] * (p['CumEmiss0'] - (p['mat0'] - p['mateq'])) + p['irT'] * p['tatm0']
    p['a0'] = brentq(lambda a: dm.irf_rhs(p, a) - lhs, 0.05, 50.0, xtol=1e-14)
    return p, b


# ---- JAX functions with the parameters as a pytree argument (compiled once) ----
def qdyn(p):
    q = dj.pack_params(p); q.pop('N'); return q


def full(qd):
    q = dict(qd); q['N'] = N; return q


def unpack(x):
    z = jnp.zeros(1)
    return (jnp.concatenate([z, x[0:N]]), jnp.concatenate([z, x[N:2 * N]]), jnp.concatenate([z, x[2 * N:3 * N]]))


def f_(x, qd, eE, eC):
    MIU, S, al = unpack(x)
    return -dj.trajectory(full(qd), MIU, S, al, eps_E=eE, eps_C=eC, clamp=False)['UTILITY']


def g_(x, qd, eE, eC):
    q = full(qd); MIU, S, al = unpack(x)
    r = dj.trajectory(q, MIU, S, al, eps_E=eE, eps_C=eC, clamp=False)
    return jnp.concatenate([dj.irf_residual(q, r, al), r['TATM'][2:]])


Z = jnp.zeros(N + 1)
F = jax.jit(lambda x, qd: f_(x, qd, Z, Z))
G = jax.jit(lambda x, qd: g_(x, qd, Z, Z))
GRAD = jax.jit(jax.grad(lambda x, qd: f_(x, qd, Z, Z)))
JAC = jax.jit(jax.jacfwd(lambda x, qd: g_(x, qd, Z, Z)))
HESS = jax.jit(jax.hessian(lambda x, qd, lam, of: of * f_(x, qd, Z, Z) + lam @ g_(x, qd, Z, Z)))
DFDE = jax.jit(jax.grad(lambda e, x, qd: f_(x, qd, e, Z)))
DFDC = jax.jit(jax.grad(lambda e, x, qd: f_(x, qd, Z, e)))
DGDE = jax.jit(jax.jacfwd(lambda e, x, qd: g_(x, qd, e, Z)))
DGDC = jax.jit(jax.jacfwd(lambda e, x, qd: g_(x, qd, Z, e)))


def R_(al1, e, MIU, S, qd):
    q = full(qd)
    al = jnp.concatenate([jnp.zeros(1), al1]); ee = jnp.concatenate([jnp.zeros(1), e])
    return dj.irf_residual(q, dj.trajectory(q, MIU, S, al, eps_E=ee), al)


def U_(al1, e, MIU, S, qd):
    al = jnp.concatenate([jnp.zeros(1), al1]); ee = jnp.concatenate([jnp.zeros(1), e])
    return dj.trajectory(full(qd), MIU, S, al, eps_E=ee)['UTILITY']


JA = jax.jit(jax.jacfwd(R_, argnums=0)); JE = jax.jit(jax.jacfwd(R_, argnums=1))
GUA = jax.jit(jax.grad(U_, argnums=0)); GUE = jax.jit(jax.grad(U_, argnums=1))
HROW, HCOL = np.tril_indices(3 * N)


class NLP:
    def __init__(self, qd): self.qd = qd
    def objective(self, x): return float(F(jnp.asarray(x), self.qd))
    def gradient(self, x): return np.asarray(GRAD(jnp.asarray(x), self.qd))
    def constraints(self, x): return np.asarray(G(jnp.asarray(x), self.qd))
    def jacobian(self, x): return np.asarray(JAC(jnp.asarray(x), self.qd)).ravel()
    def hessianstructure(self): return HROW, HCOL
    def hessian(self, x, lam, of): return np.asarray(HESS(jnp.asarray(x), self.qd, jnp.asarray(lam), of))[HROW, HCOL]


def solve_draw(p, b, x0):
    qd = qdyn(p)
    xl = np.concatenate([b['miu_lo'][1:], p['sLBounds'][1:], np.full(N, GAMS_ALPHA[0])])
    xu = np.concatenate([b['miu_up'][1:], p['sUBounds'][1:], np.full(N, GAMS_ALPHA[1])])
    xl[2 * N] = xu[2 * N] = p['a0']
    gl = np.concatenate([np.zeros(N), np.full(N - 1, b['Tlo'])]); gu = np.concatenate([np.zeros(N), np.full(N - 1, b['Tup'])])
    nlp = cyipopt.Problem(n=3 * N, m=len(gl), problem_obj=NLP(qd), lb=xl, ub=xu, cl=gl, cu=gu)
    x0 = np.clip(x0, xl, xu)
    for k, v in (('tol', 1e-9), ('acceptable_tol', 1e-7), ('print_level', 0), ('max_iter', 2000),
                 ('obj_scaling_factor', 1.0 / max(abs(float(F(jnp.asarray(x0), qd))), 1.0)),
                 ('mu_strategy', 'adaptive'), ('bound_relax_factor', 0.0), ('honor_original_bounds', 'yes')):
        nlp.add_option(k, v)
    x, info = nlp.solve(x0)
    return qd, x, info


def evaluate(row, x0):
    t0 = time.time()
    p, b = draw_params(row)
    qd, x, info = solve_draw(p, b, x0)
    xj = jnp.asarray(x); lam = jnp.asarray(info['mult_g'])
    dVdE = np.asarray(DFDE(Z, xj, qd)) + np.asarray(lam @ DGDE(Z, xj, qd))
    dVdC = np.asarray(DFDC(Z, xj, qd)) + np.asarray(lam @ DGDC(Z, xj, qd))
    mult = -1000 * dVdE / dVdC
    MIU, S, al = unpack(xj)
    MIU = MIU.at[1].set(p['miu1'])
    a1 = al[1:].at[0].set(p['a0']); e0 = jnp.zeros(N)
    Ja = np.asarray(JA(a1, e0, MIU, S, qd)); Je = np.asarray(JE(a1, e0, MIU, S, qd))
    gUa = np.asarray(GUA(a1, e0, MIU, S, qd)); gUE = np.asarray(GUE(a1, e0, MIU, S, qd))
    dadE = -np.linalg.solve(Ja, Je)
    traj = dj.trajectory(full(qd), MIU, S, jnp.concatenate([jnp.zeros(1), a1]))
    lamC = np.asarray(dj.marginal_utility_C(full(qd), traj))
    direct = -1000 * gUE / lamC; feedback = -1000 * (gUa @ dadE) / lamC
    T = np.asarray(traj['TATM'][1:])
    o = {'params': dict(zip(BOUNDS, map(float, row))), 'status': int(info['status']), 'U': -float(info['obj_val']),
         'peak_T': float(T[:40].max()), 'min_T': float(T.min()), 'Tlo_binds': bool(T[1:].min() < 0.5 + 1e-6), 'T_at_bound': bool(T[1:].min() <= b['Tlo'] + 1e-6),
         'alpha_min': float(np.min(np.asarray(a1))), 'alpha_at_floor': bool(np.min(np.asarray(a1)) <= p['AlphaLowerBound'] + 1e-6),
         'irf_max': float(np.max(np.abs(np.asarray(G(xj, qd))[:N]))), 'elapsed_s': time.time() - t0}
    for y in YEARS:
        i = (y - 2020) // 5
        tot = direct[i] + feedback[i]
        o[f'scc_{y}'] = float(mult[i + 1]); o[f'direct_{y}'] = float(direct[i]); o[f'feedback_{y}'] = float(feedback[i])
        o[f'share_{y}'] = float(feedback[i] / tot); o[f'ift_vs_mult_pct_{y}'] = float(100 * (tot / mult[i + 1] - 1))
    o['flag'] = bool(max(abs(o[f'ift_vs_mult_pct_{y}']) for y in YEARS) > 0.1 or o['status'] not in (0, 1))
    return o, x


if __name__ == '__main__':
    a, z = int(sys.argv[1]), int(sys.argv[2])
    step = int(sys.argv[sys.argv.index('--every') + 1]) if '--every' in sys.argv else 1
    # --draws k1,k2,...: an explicit list (e.g. draws where the temperature floor binds)
    draws = [int(v) for v in sys.argv[sys.argv.index('--draws') + 1].split(',')] if '--draws' in sys.argv else None
    keys, X = design()
    os.makedirs(os.path.join(RES, 'mc'), exist_ok=True)
    pr0, _ = gams_session(9)
    from ipopt_model import Problem
    x_gams = Problem(9).start_from_gams()
    out = []
    for k in (draws if draws is not None else range(a, z, step)):
        try:
            o, _ = evaluate(X[k], x_gams)
        except Exception as e:
            o = {'params': dict(zip(BOUNDS, map(float, X[k]))), 'error': repr(e)}
        o['draw'] = k; out.append(o)
        if k % 10 == 0 or k == z - 1:
            save_json(out, f'mc/mc_{a}_{z}.json')
        print(k, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in o.items() if kk in ('status', 'scc_2025', 'share_2025', 'ift_vs_mult_pct_2025', 'flag', 'elapsed_s')}, flush=True)
    save_json(out, f'mc/mc_{a}_{z}.json')
