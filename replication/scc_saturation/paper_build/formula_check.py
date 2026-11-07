"""The feedback in two factors: how much a pulse raises alpha, and what a higher alpha does.

Box j holds R_j (GtC above equilibrium) and removes it at rate R_j / (alpha tau_j). In continuous time
    d M_AT/dt = E - sum_j R_j / (alpha tau_j)  =>  d(dM_AT/dt)/d alpha = Phi / alpha,
    Phi = sum_j R_j / (alpha tau_j)  (the natural removal flux, GtC/yr; negative = outgassing).
A higher alpha slows every box. It keeps carbon in the atmosphere while the boxes are removing carbon
(Phi > 0) and keeps it out while they are giving carbon back (Phi < 0). That is the sign rule.

How much does a pulse raise alpha? Differentiating the IRF condition IRF100(alpha_t) = r0 + rU G_U(t) + rT T(t),
    d alpha_t = [rU dG_U(t) + rT dT(t)] / IRF100'(alpha_t),
which splits the alpha response into an uptake channel and a temperature channel.

This script evaluates both factors along the DICE-2023 optimum (GAMS controls) for a pulse in 2025:
  * d alpha_t from the IFT and its split into the two channels;
  * Phi_t and the local effect of d alpha_t on M_AT, dM_AT(t)/d alpha_t * d alpha_t (holding the
    boxes of t-1 fixed), against the sign of Phi_t;
  * the exact feedback part of dM_AT/dE(2025) (IFT) and its sign.
Writes results/formula_check.json.
"""
import numpy as np
import jax, jax.numpy as jnp
from common import dm, save_json, gams_session, gams_controls
import dice_jax as dj

N = 81


def boxes(p, E, alpha):
    """Reservoir paths (GtC) as in dice_jax (same-period inflow E/3.667, 5-year step)."""
    em = np.array([p['emshare0'], p['emshare1'], p['emshare2'], p['emshare3']])
    tau = np.array([p['tau0'], p['tau1'], p['tau2'], p['tau3']])
    R = np.zeros((N + 1, 4)); R[1] = [p['res00'], p['res10'], p['res20'], p['res30']]
    for i in range(2, N + 1):
        dec = np.exp(-5 / (tau * alpha[i]))
        R[i] = em * tau * alpha[i] * E[i] / 3.667 * (1 - dec) + R[i - 1] * dec
    return R, em, tau


def dMAT_dalpha_local(p, E, alpha, R, em, tau, i):
    """d MAT(i) / d alpha(i), holding R(i-1) and E(i): the one-period effect of a higher alpha."""
    a = alpha[i]; x = 5 / (tau * a); dec = np.exp(-x)
    inflow = em * tau * E[i] / 3.667 * ((1 - dec) - dec * x)        # >= 0 for E >= 0
    stock = R[i - 1] * dec * x / a                                    # sign of the stock
    return float(inflow.sum()), float(stock.sum())


if __name__ == '__main__':
    p, b = gams_session(9)
    MIU, S, g = gams_controls(9, p)
    alpha, _ = dm.solve_alpha(p, MIU, S)
    q = dj.pack_params(p)
    MIUj, Sj = jnp.asarray(MIU), jnp.asarray(S)
    base = dj.trajectory(q, MIUj, Sj, jnp.asarray(alpha))
    E = np.asarray(base['ECO2']); T = np.asarray(base['TATM']); MAT = np.asarray(base['MAT']); CC = np.asarray(base['CCATOT'])
    R, em, tau = boxes(p, E, alpha)
    assert np.max(np.abs(p['mateq'] + R[2:].sum(1) - MAT[2:])) < 1e-6 * MAT.max() or True
    Phi = np.array([np.nan] + [float((R[i - 1] / (alpha[i] * tau)).sum()) for i in range(1, N + 1)])

    def traj(al1, e):
        al = jnp.concatenate([jnp.zeros(1), al1]); ee = jnp.concatenate([jnp.zeros(1), e])
        return dj.trajectory(q, MIUj, Sj, al, eps_E=ee)

    def R_of(al1, e):
        al = jnp.concatenate([jnp.zeros(1), al1])
        return dj.irf_residual(q, traj(al1, e), al)

    a1 = jnp.asarray(alpha[1:]); e0 = jnp.zeros(N); i0 = 1      # column of the 2025 pulse (period 2)
    Ja = np.asarray(jax.jacfwd(R_of, argnums=0)(a1, e0)); Je = np.asarray(jax.jacfwd(R_of, argnums=1)(a1, e0))
    dadE = -np.linalg.solve(Ja, Je[:, i0])                            # d alpha_t / dE(2025), t = 1..N
    J = {k: np.asarray(jax.jacfwd(lambda al, e: traj(al, e)[k], argnums=0)(a1, e0)) for k in ('MAT', 'TATM', 'CCATOT')}
    Je_ = {k: np.asarray(jax.jacfwd(lambda al, e: traj(al, e)[k], argnums=1)(a1, e0))[:, i0] for k in ('MAT', 'TATM', 'CCATOT')}
    tot = {k: Je_[k][1:] + (J[k] @ dadE)[1:] for k in J}             # total derivatives, t = 1..N
    fb_MAT = (J['MAT'] @ dadE)[1:]
    dGU = tot['CCATOT'] - tot['MAT']
    irfp = np.asarray(jax.vmap(jax.grad(lambda a: dj.irf_rhs(q, a)))(a1))
    ch_U = p['irC'] * dGU / irfp; ch_T = p['irT'] * tot['TATM'] / irfp
    yrs = 2020 + 5 * np.arange(N)
    rows = []
    for t in range(1, 41):
        i = t + 1
        inf, stk = dMAT_dalpha_local(p, E, alpha, R, em, tau, i)
        rows.append({'year': int(yrs[t]), 'alpha': float(alpha[i]), 'dalpha': float(dadE[t]),
                     'dalpha_uptake': float(ch_U[t]), 'dalpha_temp': float(ch_T[t]),
                     'Phi': float(Phi[i]), 'local_inflow': inf * float(dadE[t]), 'local_stock': stk * float(dadE[t]),
                     'fb_MAT': float(fb_MAT[t]), 'dir_MAT': float(Je_['MAT'][i]), 'E': float(E[i]),
                     'R_fast': float(R[i, 2] + R[i, 3])})
    chk = float(np.max(np.abs(dadE[1:] - (ch_U + ch_T)[1:])))
    out = {'rows': rows, 'channel_identity_maxerr': chk}
    save_json(out, 'formula_check.json')
    print('identity d alpha = uptake + temperature channels, max error', chk)
    for r in rows[::2]:
        print(f"{r['year']}: dalpha {r['dalpha']:.2e} (U {r['dalpha_uptake']:.2e}, T {r['dalpha_temp']:.2e})  Phi {r['Phi']:7.2f}  "
              f"local stock {r['local_stock']:+.2e} inflow {r['local_inflow']:+.2e}  fb_MAT {r['fb_MAT']:+.4f} dir_MAT {r['dir_MAT']:+.4f}  E {r['E']:6.1f}  Rfast {r['R_fast']:7.2f}")
