"""Robustness checks requested in review of v5.

Part 1, temperature bound at high climate sensitivity. Experiment B's re-optimized rows, solved with IPOPT under
three treatments of the lower bound on T_AT: 0.01 C (as in the paper), 0.5 C (the GAMS stability bound), and no
bound (-50 C). The decomposition is computed on the unclamped model (clamp=False), with alpha taken from the
optimum, so it is exact for each treatment; the IFT total is compared with the multiplier SCC.
Part 2, carbon-cycle sensitivities. sigma_C (irC) and sigma_T (irT) varied one at a time over the Monte Carlo
ranges, DICE-2023 values elsewhere: share and feedback with policy fixed at the DICE-2023 optimum and re-optimized,
with the same physical / policy split as Experiment B.
Writes results/exp_e_robust_p<parts>.json."""
import sys
import numpy as np
import jax, jax.numpy as jnp
from common import save_json, dm
import mc_v2 as V, mc_ift as M
import dice_jax as dj

N = M.N
_orig = V.draw_params
TLO = {'v': 0.01}


def patched(row):
    p, b = _orig(row); b['Tlo'] = TLO['v']; return p, b


V.draw_params = patched; M.draw_params = patched


def R_(al1, e, MIU, S, qd):
    q = M.full(qd); al = jnp.concatenate([jnp.zeros(1), al1]); ee = jnp.concatenate([jnp.zeros(1), e])
    return dj.irf_residual(q, dj.trajectory(q, MIU, S, al, eps_E=ee, clamp=False), al)


def U_(al1, e, MIU, S, qd):
    al = jnp.concatenate([jnp.zeros(1), al1]); ee = jnp.concatenate([jnp.zeros(1), e])
    return dj.trajectory(M.full(qd), MIU, S, al, eps_E=ee, clamp=False)['UTILITY']


JA = jax.jit(jax.jacfwd(R_, argnums=0)); JE = jax.jit(jax.jacfwd(R_, argnums=1))
RR = jax.jit(R_)
GUA = jax.jit(jax.grad(U_, argnums=0)); GUE = jax.jit(jax.grad(U_, argnums=1))


def decomp(p, MIU, S, a1):
    """Exact split on the unclamped model. a1: alpha for periods 1..N (Newton-refined to R = 0)."""
    qd = M.qdyn(p); MIUj, Sj = jnp.asarray(MIU), jnp.asarray(S); e0 = jnp.zeros(N)
    a1 = jnp.asarray(a1).at[0].set(p['a0'])
    for _ in range(20):
        r = np.asarray(RR(a1, e0, MIUj, Sj, qd))
        if np.max(np.abs(r)) < 1e-11:
            break
        J = np.asarray(JA(a1, e0, MIUj, Sj, qd)); a1 = a1 - jnp.asarray(np.linalg.solve(J, r))
    Ja = np.asarray(JA(a1, e0, MIUj, Sj, qd)); Je = np.asarray(JE(a1, e0, MIUj, Sj, qd))
    gUa = np.asarray(GUA(a1, e0, MIUj, Sj, qd)); gUE = np.asarray(GUE(a1, e0, MIUj, Sj, qd))
    dadE = -np.linalg.solve(Ja, Je)
    tr = dj.trajectory(M.full(qd), MIUj, Sj, jnp.concatenate([jnp.zeros(1), a1]), clamp=False)
    lamC = np.asarray(dj.marginal_utility_C(M.full(qd), tr))
    d = -1000 * gUE / lamC; f = -1000 * (gUa @ dadE) / lamC
    T = np.asarray(tr['TATM'][2:]); E = np.asarray(tr['ECO2'][1:])
    return {'scc_2025': float(d[1] + f[1]), 'feedback_2025': float(f[1]), 'share_2025': float(f[1] / (d[1] + f[1])),
            'min_T': float(T.min()), 'cumE_2100': float(5 * E[:17].sum()), 'irf_resid': float(np.max(np.abs(np.asarray(RR(a1, e0, MIUj, Sj, qd))))),
            'alpha_min': float(np.min(np.asarray(a1)))}


def ctrl(x, p):
    x = np.asarray(x, float)
    return np.r_[0, x[:N]].copy(), np.r_[0, x[N:2 * N]].copy(), x[2 * N:].copy()


def solve(row, x0):
    p, b = V.draw_params(row)
    qd, x, info = M.solve_draw(p, b, x0)
    x = np.asarray(x, float); x[2 * N] = p['a0']
    xj = jnp.asarray(x); lam = jnp.asarray(info['mult_g'])
    dVdE = np.asarray(M.DFDE(M.Z, xj, qd)) + np.asarray(lam @ M.DGDE(M.Z, xj, qd))
    dVdC = np.asarray(M.DFDC(M.Z, xj, qd)) + np.asarray(lam @ M.DGDC(M.Z, xj, qd))
    MIU, S, a1 = ctrl(x, p); MIU[1] = p['miu1']
    o = decomp(p, MIU, S, a1)
    o['status'] = int(info['status']); o['scc_mult_2025'] = float(-1000 * dVdE[2] / dVdC[2])
    o['ift_vs_mult_pct'] = 100 * (o['scc_2025'] / o['scc_mult_2025'] - 1)
    return o, x


if __name__ == '__main__':
    from ipopt_model import Problem
    xg = Problem(9).start_from_gams()
    row0 = np.array([V.DICE[k] for k in V.KEYS])
    out = {'part1': [], 'part2': []}
    parts = sys.argv[1:] or ['1', '2']
    if '1' in parts:
        for tlo in (0.01, 0.5, -50.0):
            TLO['v'] = tlo
            b0, x0 = solve(row0, xg)
            out['part1'].append({'Tlo': tlo, 'ecs': 3.0, **b0}); print('P1', tlo, 3.0, {k: round(v, 4) for k, v in b0.items()}, flush=True)
            for ecs in (2.0, 2.5, 3.5, 4.0, 4.25, 4.5, 4.75, 5.0):
                row = row0.copy(); row[V.KEYS.index('ecs')] = ecs
                o, _ = solve(row, x0)
                out['part1'].append({'Tlo': tlo, 'ecs': ecs, **o}); print('P1', tlo, ecs, {k: round(v, 4) for k, v in o.items()}, flush=True)
                save_json(out, 'exp_e_robust_p' + ''.join(parts) + '.json')
    if '2' in parts:
        TLO['v'] = 0.01
        b0, x0 = solve(row0, xg); p0, _ = V.draw_params(row0); MIU0, S0, a0 = ctrl(x0, p0); MIU0[1] = p0['miu1']
        for key, vals in (('irC', (0.010, 0.0145, 0.019, 0.0245, 0.030)), ('irT', (2.5, 3.3, 4.165, 5.0, 6.0))):
            for v in vals:
                row = row0.copy(); row[V.KEYS.index(key)] = v
                p, _ = V.draw_params(row)
                fixed = decomp(p, MIU0, S0, a0)          # alpha re-solved by Newton for the new parameters
                re, _ = solve(row, x0)
                r = {'param': key, 'value': v, 'fixed': fixed, 'reopt': re,
                     'physical_pp': 100 * (fixed['share_2025'] - b0['share_2025']), 'policy_pp': 100 * (re['share_2025'] - fixed['share_2025']),
                     'total_pp': 100 * (re['share_2025'] - b0['share_2025'])}
                out['part2'].append(r)
                print('P2', key, v, 'fixed', round(100 * fixed['share_2025'], 2), round(fixed['feedback_2025'], 2), 'reopt', round(100 * re['share_2025'], 2), round(re['feedback_2025'], 2), 'phys', round(r['physical_pp'], 2), 'pol', round(r['policy_pp'], 2), flush=True)
                save_json(out, 'exp_e_robust_p' + ''.join(parts) + '.json')
    print('E_DONE')
