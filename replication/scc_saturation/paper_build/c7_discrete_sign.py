"""v8: the local sign rule (Proposition 2) in the discrete model (referee comment 5).

Proposition 2 is stated in continuous time: d(dM_AT/dt)/d alpha = Phi / alpha. The model steps five years at a time
(Eq. 6). For each period t of a policy path this script computes, with R_i(t-1) and E(t) given,
    dM(t)/d alpha(t)  from the discrete reservoir equation (Appendix D, Eq. D1),
the discrete net flux into the reservoirs over the period, Phi_d(t) = sum_i [xi_i E(t)/3.667 * 5 - (R_i(t) - R_i(t-1))] / 5,
and the continuous-time rule evaluated at the start of the period, Phi(t-1)/alpha(t). It reports in how many periods the
signs agree, on the published optimal, 4% and 5% paths (the ones with net-negative emissions).
Writes results/c7_discrete_sign.json.
"""
import numpy as np
from common import dm, save_json, scenario_params, gams_controls

N = 81


def reservoirs(p, MIU, S, alpha):
    traj = dm.diceTrajectory(p, MIU, S, alpha)
    import dice_jax as dj, jax.numpy as jnp
    r = dj.trajectory(dj.pack_params(p), jnp.asarray(MIU), jnp.asarray(S), jnp.asarray(alpha))
    E = np.asarray(r['ECO2'])            # 1-based, GtCO2/yr
    xi = np.array([p[f'emshare{i}'] for i in range(4)]); tau = np.array([p[f'tau{i}'] for i in range(4)])
    R = np.zeros((N + 1, 4)); R[1] = [p['res00'], p['res10'], p['res20'], p['res30']]
    for t in range(2, N + 1):
        a = alpha[t]; eta = np.exp(-5 / (a * tau))
        R[t] = xi * tau * a * E[t] / 3.667 * (1 - eta) + R[t - 1] * eta
    MAT = np.asarray(r['MAT'])
    return E, R, xi, tau, MAT


def run(sc):
    _, p = scenario_params(sc)
    MIU, S, _ = gams_controls(sc, p)
    alpha, _ = dm.solve_alpha(p, MIU, S)
    E, R, xi, tau, MAT = reservoirs(p, MIU, S, alpha)
    rows = []
    for t in range(2, N + 1):
        a = alpha[t]; eta = np.exp(-5 / (a * tau)); J = E[t] / 3.667
        dM = np.sum(xi * tau * J * (1 - eta) + (R[t - 1] - xi * tau * a * J) * eta * 5 / (tau * a ** 2))
        phi_d = np.sum(xi * J * 5 - (R[t] - R[t - 1])) / 5
        phi_c = np.sum(R[t - 1] / (a * tau)) / a
        rows.append((2020 + 5 * (t - 1), dM, phi_d, phi_c))
    rows = np.array(rows)
    yrs = rows[:, 0].astype(int)
    agree_d = np.sign(rows[:, 1]) == np.sign(rows[:, 2])
    agree_c = np.sign(rows[:, 1]) == np.sign(rows[:, 3])
    first = lambda m: int(yrs[np.where(m)[0][0]]) if np.any(m) else None
    o = {'scenario': sc, 'periods': int(len(yrs)), 'agree_with_discrete_flux': int(agree_d.sum()),
         'agree_with_continuous_rule': int(agree_c.sum()),
         'disagree_years_discrete': yrs[~agree_d].tolist(), 'disagree_years_continuous': yrs[~agree_c].tolist(),
         'first_negative_dM_dalpha': first(rows[:, 1] < 0), 'first_negative_flux': first(rows[:, 2] < 0),
         'max_MAT_check': float(np.max(np.abs(p['mateq'] + R[1:].sum(1) - MAT[1:])))}
    return o


if __name__ == '__main__':
    out = {}
    for sc in (9, 4, 5):
        out[str(sc)] = run(sc)
        print(out[str(sc)], flush=True)
    save_json(out, 'c7_discrete_sign.json')
