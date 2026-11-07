"""Why does a higher climate sensitivity (or higher damages) lower the feedback share?
Hypothesis: through the policy response. A more dangerous climate makes the optimal path abate sooner
and reach net-negative emissions earlier, so the period of negative feedback (Proposition 2) comes
sooner and carries more weight. Test: for each ECS (and damage coefficient), compute the share
  (a) at the re-optimized policy, and
  (b) with the policy held at the DICE-2023 GAMS optimum (no re-optimization).
If the share falls with ECS in (a) and not in (b), the policy response is the channel.
Writes results/ecs_mechanism.json."""
import numpy as np
import jax.numpy as jnp
from common import dm, save_json, gams_controls
import mc_v2 as V, mc_fast as F
import mc_ift as M
import dice_jax as dj

N = 81


def split_fixed_policy(row):
    p, b = V.draw_params(row)
    MIU, S, g = gams_controls(9, p)
    alpha, _ = dm.solve_alpha(p, MIU, S)
    qd = M.qdyn(p); MIUj, Sj = jnp.asarray(MIU), jnp.asarray(S)
    a1 = jnp.asarray(alpha[1:]); e0 = jnp.zeros(N)
    Ja = np.asarray(M.JA(a1, e0, MIUj, Sj, qd)); Je = np.asarray(M.JE(a1, e0, MIUj, Sj, qd))
    gUa = np.asarray(M.GUA(a1, e0, MIUj, Sj, qd)); gUE = np.asarray(M.GUE(a1, e0, MIUj, Sj, qd))
    dadE = -np.linalg.solve(Ja, Je)
    traj = dj.trajectory(M.full(qd), MIUj, Sj, jnp.asarray(alpha))
    lamC = np.asarray(dj.marginal_utility_C(M.full(qd), traj))
    d = -1000 * gUE / lamC; f = -1000 * (gUa @ dadE) / lamC
    E = np.asarray(traj['ECO2'][1:]); yrs = 2020 + 5 * np.arange(N)
    return {'scc_2025': float(d[1] + f[1]), 'share_2025': float(f[1] / (d[1] + f[1])),
            'first_net_negative_year': int(yrs[E < 0][0]) if np.any(E < 0) else None,
            'peak_T': float(np.asarray(traj['TATM'][1:41]).max())}


if __name__ == '__main__':
    from ipopt_model import Problem
    xg = Problem(9).start_from_gams()
    out = []
    for key, vals in (('ecs', (2.0, 2.5, 3.0, 4.0, 5.0)), ('psi2', (0.0017, 0.003467, 0.0069))):
        for v in vals:
            row = np.array([V.DICE[k] for k in V.KEYS]); row[V.KEYS.index(key)] = v
            o, _ = F.evaluate(row, xg)
            fx = split_fixed_policy(row)
            r = {'param': key, 'value': v,
                 'reopt': {'scc_2025': o['scc_2025'], 'share_2025': o['share_2025'], 'first_net_negative_year': o['first_net_negative_year'], 'peak_T': o['peak_T'], 'success': o['success']},
                 'fixed_policy': fx}
            out.append(r); print(r, flush=True)
            save_json(out, 'ecs_mechanism.json')
