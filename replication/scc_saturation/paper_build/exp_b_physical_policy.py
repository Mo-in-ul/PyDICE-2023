"""Experiment B: physical response versus policy response of the saturation-feedback share.

For a parameter value v (climate sensitivity or damage coefficient) and the baseline optimum x0 (DICE-2023
calibration, solved by PyDICE), the change in the 2025 feedback share splits exactly into
    physical  = share(v, x0)                 - share(v0, x0)       policy held at the baseline optimum
    abatement = share(v, mu*(v), S0)          - share(v, x0)        planner re-optimizes the control rate only
    savings   = share(v, x*(v))              - share(v, mu*(v), S0) planner also re-optimizes the savings rate
so that share(v, x*(v)) - share(v0, x0) = physical + abatement + savings. alpha is re-solved from the IRF
condition for every control path. Also reported: cumulative emissions 2020-2100, first net-negative year,
peak warming, and the first year the feedback (the SCC component for a pulse in that year) turns negative.
Each optimum is solved with IPOPT on the unclamped model (mc_fast.evaluate_ipopt), so the 0.01 C lower bound on
temperature is a constraint. Writes results/exp_b_physical_policy.json."""
import numpy as np
import jax.numpy as jnp
from common import dm, save_json
import mc_v2 as V, mc_fast as F
import mc_ift as M
import dice_jax as dj

N = 81
YRS = 2020 + 5 * np.arange(N)


def at_controls(row, MIU, S, alpha_start=None):
    """Feedback share and path diagnostics for parameters `row` and fixed controls (1-based MIU, S)."""
    p, b = V.draw_params(row)
    MIU = MIU.copy(); MIU[1] = p['miu1']
    alpha, _ = dm.solve_alpha(p, MIU, S, alpha_start=alpha_start)
    qd = M.qdyn(p); MIUj, Sj = jnp.asarray(MIU), jnp.asarray(S)
    a1 = jnp.asarray(alpha[1:]); e0 = jnp.zeros(N)
    Ja = np.asarray(M.JA(a1, e0, MIUj, Sj, qd)); Je = np.asarray(M.JE(a1, e0, MIUj, Sj, qd))
    gUa = np.asarray(M.GUA(a1, e0, MIUj, Sj, qd)); gUE = np.asarray(M.GUE(a1, e0, MIUj, Sj, qd))
    dadE = -np.linalg.solve(Ja, Je)
    traj = dj.trajectory(M.full(qd), MIUj, Sj, jnp.asarray(alpha))
    lamC = np.asarray(dj.marginal_utility_C(M.full(qd), traj))
    d = -1000 * gUE / lamC; f = -1000 * (gUa @ dadE) / lamC
    E = np.asarray(traj['ECO2'][1:]); T = np.asarray(traj['TATM'][1:])
    neg = np.where(f[:41] < 0)[0]
    return {'scc_2025': float(d[1] + f[1]), 'direct_2025': float(d[1]), 'feedback_2025': float(f[1]),
            'share_2025': float(f[1] / (d[1] + f[1])),
            'cumE_2100': float(5 * E[:17].sum()), 'first_net_negative_year': int(YRS[E < 0][0]) if np.any(E < 0) else None,
            'feedback_negative_from': int(YRS[neg[0]]) if len(neg) else None, 'peak_T': float(T[:40].max()),
            'alpha_2100': float(alpha[17])}


def controls(x, p):
    MIU = np.zeros(N + 1); S = np.zeros(N + 1)
    MIU[1:] = x[:N]; S[1:] = x[N:2 * N]; MIU[1] = p['miu1']
    return MIU, S, np.concatenate([[0], x[2 * N:]])


if __name__ == '__main__':
    from ipopt_model import Problem
    xg = Problem(9).start_from_gams()
    row0 = np.array([V.DICE[k] for k in V.KEYS])
    o0, x0 = F.evaluate_ipopt(row0, xg)
    p0, _ = V.draw_params(row0); MIU0, S0, a0 = controls(x0, p0)
    base = at_controls(row0, MIU0, S0, a0)
    print('baseline', base, flush=True)
    out = {'baseline': base, 'rows': []}
    grids = (('ecs', np.round(np.arange(2.0, 5.01, 0.25), 2)), ('psi2', (0.0017, 0.0026, 0.003467, 0.0052, 0.0069)))
    for key, vals in grids:
        for v in vals:
            row = row0.copy(); row[V.KEYS.index(key)] = v
            o, x = F.evaluate_ipopt(row, x0)
            p, _ = V.draw_params(row); MIUv, Sv, av = controls(x, p)
            phys = at_controls(row, MIU0, S0, a0)
            abat = at_controls(row, MIUv, S0, av)
            full = at_controls(row, MIUv, Sv, av)
            r = {'param': key, 'value': float(v), 'success': o['success'], 'fixed': phys, 'mu_only': abat, 'reopt': full,
                 'split': {'physical': phys['share_2025'] - base['share_2025'], 'abatement': abat['share_2025'] - phys['share_2025'],
                           'savings': full['share_2025'] - abat['share_2025'], 'total': full['share_2025'] - base['share_2025']}}
            out['rows'].append(r)
            print(key, v, o['success'], {k: round(100 * w, 2) for k, w in r['split'].items()}, flush=True)
            save_json(out, 'exp_b_physical_policy.json')
    print('done')
