"""Re-optimize DICE-2023 with the FaIR v2 carbon cycle, then split the SCC at the new optimum.

fair_benchmark.py held the DICE-2023 optimal policy fixed and changed only the carbon cycle. Here
the planner re-optimizes with the new carbon cycle. The model is DICE-2023 (economy, abatement,
non-CO2 forcing, two-box temperature, damages, welfare) with the carbon cycle replaced:
within each 5-year period the emissions rate E(t) (GtCO2/yr) is applied over five 1-year steps
of the FaIR v2 gas cycle (fair.gas_cycle.calculate_alpha and step_concentration, same formulas),
    iIRF = r0 + rU (cum - airb) + rT T + rA airb,   alpha = g0 exp(min(iIRF, 100) / g1),
with temperature updated each year by the DICE two-box model and the DICE forcing. alpha is an
explicit function of the state, so the optimization has no implicit constraint.
SCC(t) = -1000 dU/dE(t) / lambda_C(t) by automatic differentiation at the optimum (no state bound
binds; checked). Direct part: the same derivative with the annual alpha path held at its optimal
values. Feedback share = 1 - direct / total.
Variants: dfair_closure (DICE-2023 IRF parameters, FaIR closure) and fair_ar6 (FaIR v2 AR6 defaults).
Writes results/fair_opt.json.
"""
import time
import numpy as np
import jax, jax.numpy as jnp
import cyipopt
from common import dm, save_json, gams_session, gams_controls
import dice_jax as dj

jax.config.update("jax_enable_x64", True)
N = 81
C2 = 3.667
PF = jnp.array([0.2173, 0.224, 0.2824, 0.2763])
LT = jnp.array([1e9, 394.4, 36.54, 4.304])
G0, G1 = 0.010178, 11.412622
AR6 = dict(r0=29.0, rU=0.00846, rT=4.0, rA=0.000819)


def trajectory(q, cp, MIU, S, eps_E=None, alpha_fix=None):
    """DICE-2023 with the FaIR v2 carbon cycle on annual substeps. cp: dict r0, rU, rT, rA (per GtCO2).
    alpha_fix: optional array (5*(N-1),) of annual alpha values (frozen saturation)."""
    z = jnp.zeros(N + 1)
    eps_E = z if eps_E is None else eps_E
    MIU = MIU.at[1].set(q['miu1'])
    ts = q['tstep']
    K1 = q['k0']; T1 = q['tatm0']
    DAM1 = q['a1'] * T1 + q['a2base'] * T1 ** q['a3']
    YG1 = q['eco2Param'][1] * K1 ** q['gama']
    AB1 = YG1 * q['cost1tot'][1] * MIU[1] ** q['expcost2']
    Y1 = YG1 * (1 - DAM1) - AB1
    I1 = S[1] * Y1; C1 = Y1 - I1
    boxes0 = jnp.array([q['res00'], q['res10'], q['res20'], q['res30']]) * C2
    cum0 = q['CumEmiss0'] * C2
    afix = jnp.zeros(5 * (N - 1)) if alpha_fix is None else alpha_fix
    use_fix = alpha_fix is not None

    def year(carry, inp):
        boxes, cum, B1, B2, T = carry
        E, Fnon, k = inp
        airb = boxes.sum()
        iirf = cp['r0'] + cp['rU'] * (cum - airb) + cp['rT'] * T + cp['rA'] * airb
        a = G0 * jnp.exp(jnp.minimum(iirf, 100.0) / G1)
        a = afix[k] if use_fix else a
        rate = 1.0 / (a * LT); dec = jnp.exp(-rate)
        boxes = PF * E / rate * (1 - dec) + boxes * dec
        cum = cum + E
        MAT = q['mateq'] + boxes.sum() / C2
        F = q['fco22x'] * jnp.log(MAT / q['mateq']) / jnp.log(2.0) + Fnon
        B1 = B1 * jnp.exp(-1 / q['d1']) + q['teq1'] * F * (1 - jnp.exp(-1 / q['d1']))
        B2 = B2 * jnp.exp(-1 / q['d2']) + q['teq2'] * F * (1 - jnp.exp(-1 / q['d2']))
        return (boxes, cum, B1, B2, B1 + B2), (a, MAT)

    def period(carry, i):
        K, I, FG, Mprev, boxes, cum, B1, B2, T = carry
        K = (1 - q['dk']) ** ts * K + ts * I
        YG = q['eco2Param'][i] * K ** q['gama']
        E = (q['sigma'][i] * YG + q['eland'][i]) * (1 - MIU[i]) + eps_E[i]
        FG = q['Fcoef2'] * FG + q['Fcoef1'] * q['CO2E_GHGabateB'][i - 1] * (1 - Mprev)
        Fnon = q['F_Misc'][i] + FG
        ks = 5 * (i - 2) + jnp.arange(5)
        (boxes, cum, B1, B2, T), (al, MATy) = jax.lax.scan(year, (boxes, cum, B1, B2, T),
                                                          (jnp.full(5, E), jnp.full(5, Fnon), ks))
        DAM = q['a1'] * T + q['a2base'] * T ** q['a3']
        AB = YG * q['cost1tot'][i] * MIU[i] ** q['expcost2']
        Y = YG * (1 - DAM) - AB
        I = S[i] * Y; C = Y - I
        return (K, I, FG, MIU[i], boxes, cum, B1, B2, T), (C, T, E, MATy[-1], al)

    carry0 = (K1, I1, q['F_GHGabate2020'], MIU[1], boxes0, cum0, q['tbox10'], q['tbox20'], T1)
    _, (C, T, E, MAT, al) = jax.lax.scan(period, carry0, jnp.arange(2, N + 1))
    C = jnp.concatenate([jnp.atleast_1d(C1), C]); T = jnp.concatenate([jnp.atleast_1d(T1), T])
    cpc = 1000.0 * C / q['L'][1:]
    el = q['elasmu']
    U = ((cpc ** (1 - el)) - 1) / (1 - el) - 1
    W = ts * q['scale1'] * jnp.sum(U * q['L'][1:] * q['RR'][1:]) + q['scale2']
    return {'UTILITY': W, 'C': C, 'TATM': T, 'ECO2': jnp.concatenate([jnp.zeros(1), E]), 'MAT': MAT, 'alpha': al.ravel()}


class NLP:
    def __init__(self, q, cp, lo, hi):
        self.q, self.cp = q, cp
        def unpack(x):
            MIU = jnp.concatenate([jnp.zeros(2), x[:N - 1]])
            S = jnp.concatenate([jnp.zeros(1), x[N - 1:]])
            return MIU, S
        self.unpack = unpack
        f = lambda x: -trajectory(q, cp, *unpack(x))['UTILITY']
        self._f = jax.jit(f); self._g = jax.jit(jax.grad(f)); self._h = jax.jit(jax.hessian(f))
        n = 2 * N - 1
        self.hr, self.hc = np.tril_indices(n)
    def objective(self, x): return float(self._f(jnp.asarray(x)))
    def gradient(self, x): return np.asarray(self._g(jnp.asarray(x)))
    def constraints(self, x): return np.zeros(0)
    def jacobian(self, x): return np.zeros(0)
    def hessianstructure(self): return self.hr, self.hc
    def hessian(self, x, lam, of): return of * np.asarray(self._h(jnp.asarray(x)))[self.hr, self.hc]


def solve(cp, p, b, x0):
    q = dj.pack_params(p)
    prob = NLP(q, cp, None, None)
    xl = np.concatenate([b['miu_lo'][2:], p['sLBounds'][1:]]); xu = np.concatenate([b['miu_up'][2:], p['sUBounds'][1:]])
    nlp = cyipopt.Problem(n=2 * N - 1, m=0, problem_obj=prob, lb=xl, ub=xu, cl=[], cu=[])
    U0 = abs(prob.objective(np.clip(x0, xl, xu)))
    for k, v in (('tol', 1e-10), ('print_level', 0), ('max_iter', 3000), ('obj_scaling_factor', 1.0 / max(U0, 1.0)),
                 ('mu_strategy', 'adaptive'), ('bound_relax_factor', 0.0), ('honor_original_bounds', 'yes')):
        nlp.add_option(k, v)
    x, info = nlp.solve(np.clip(x0, xl, xu))
    return q, prob, x, info


def split(q, cp, MIU, S):
    base = trajectory(q, cp, MIU, S)
    af = base['alpha']
    gE = np.asarray(jax.grad(lambda e: trajectory(q, cp, MIU, S, eps_E=e)['UTILITY'])(jnp.zeros(N + 1)))
    gD = np.asarray(jax.grad(lambda e: trajectory(q, cp, MIU, S, eps_E=e, alpha_fix=af)['UTILITY'])(jnp.zeros(N + 1)))
    cpc = 1000.0 * np.asarray(base['C']) / np.asarray(q['L'][1:])
    lam = q['tstep'] * q['scale1'] * 1000.0 * cpc ** (-q['elasmu']) * np.asarray(q['RR'][1:])
    tot = -1000 * gE[1:] / lam; dirc = -1000 * gD[1:] / lam
    return tot, dirc, base


if __name__ == '__main__':
    p, b = gams_session(9)
    b['Tlo'] = 0.01
    MIUg, Sg, g = gams_controls(9, p)
    x_g = np.concatenate([MIUg[2:], Sg[1:]])
    variants = {'dfair_closure': dict(r0=p['irf0'], rU=p['irC'] / C2, rT=p['irT'], rA=0.0), 'fair_ar6': AR6}
    out = {}
    for name, cp in variants.items():
        t0 = time.time()
        q, prob, x, info = solve(cp, p, b, x_g)
        MIU, S = prob.unpack(jnp.asarray(x))
        tot, dirc, base = split(q, cp, MIU, S)
        T = np.asarray(base['TATM']); E = np.asarray(base['ECO2'][1:])   # periods 2..N
        yrs = 2025 + 5 * np.arange(N - 1)
        # fixed-policy split for comparison (DICE-2023 GAMS controls)
        totg, dircg, baseg = split(q, cp, jnp.asarray(MIUg), jnp.asarray(Sg))
        o = {'status': int(info['status']), 'elapsed_s': time.time() - t0, 'U': -float(info['obj_val']),
             'reopt': {'scc': {y: float(tot[(y - 2020) // 5]) for y in (2025, 2050, 2100, 2150)},
                       'share': {y: float(1 - dirc[(y - 2020) // 5] / tot[(y - 2020) // 5]) for y in (2025, 2050, 2100, 2150)},
                       'peak_T': float(T[:40].max()), 'first_net_negative_year': int(yrs[E < 0][0]) if np.any(E < 0) else None,
                       'miu_2030': float(np.asarray(MIU)[3]), 'cprice_2030': float(p['PBACKTIME'][3] * np.asarray(MIU)[3] ** (p['expcost2'] - 1))},
             'fixed_policy': {'scc': {y: float(totg[(y - 2020) // 5]) for y in (2025, 2050, 2100, 2150)},
                              'share': {y: float(1 - dircg[(y - 2020) // 5] / totg[(y - 2020) // 5]) for y in (2025, 2050, 2100, 2150)},
                              'peak_T': float(np.asarray(baseg['TATM'])[:40].max())}}
        out[name] = o
        print(name, o, flush=True)
        save_json(out, 'fair_opt.json')
