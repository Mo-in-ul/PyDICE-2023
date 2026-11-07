"""Diagonal of the IFT Jacobian dR/dalpha and its nonsingularity (referee check for Eq. (15) / Section 4.2).

The impulse-response condition of period t is R_t = IRF100(alpha_t) - [s0 + sC Cacc(t) + sT T(t)] = 0, with
Cacc(t) = Ccum(t) - [M(t) - Meq]. Ccum(t) adds the previous period's emissions (GAMS CCATOT), so it does not
depend on alpha_t, while M(t) and T(t) do: R_i(t) uses alpha_t (Eq. reservoir) and T(t) uses the forcing of M(t).
R_t involves alpha_1..alpha_t only, so dR/dalpha is lower triangular and nonsingular iff every diagonal entry is
nonzero. The full diagonal entry is
    D_t = IRF100'(alpha_t) + sC dM(t)/dalpha_t - sT dT(t)/dalpha_t,
    dM(t)/dalpha_t = sum_i [ xi_i tau_i I_t (1 - d_i) + (R_i(t-1) - xi_i tau_i alpha_t I_t) d_i 5 / (tau_i alpha_t^2) ],
    dT(t)/dalpha_t = [teq1 (1 - e^{-5/d1}) + teq2 (1 - e^{-5/d2})] F2x / (ln 2 M(t)) dM(t)/dalpha_t,
with I_t = E(t)/3.667 and d_i = exp(-5/(tau_i alpha_t)). This script evaluates D_t analytically, checks it against
the automatic-differentiation Jacobian, and reports min_t D_t and min_t D_t / IRF100'(alpha_t) on (i) the ten published
GAMS policy paths and (ii) a targeted subset of Monte Carlo draws re-solved with IPOPT (largest sT*ECS/sC, i.e. the
draws where the temperature term works hardest against the others, the smallest sC, and draws at the temperature
bound). Writes results/jacobian_check.json.
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp
from common import dm, save_json, scenario_params, gams_controls
import dice_jax as dj

jax.config.update("jax_enable_x64", True)
N = 81
TS = 5.0


def irf_prime(p, a):
    out = 0.0
    for j in range(4):
        tau = p[f'tau{j}']; x = 100.0 / (a * tau)
        out = out + p[f'emshare{j}'] * tau * ((1 - np.exp(-x)) - x * np.exp(-x))
    return out


def analytic_diag(p, alpha, E, MAT):
    """D_t for t = 2..N from the formula in the docstring (alpha, E, MAT 1-based numpy arrays)."""
    R = np.array([p['res00'], p['res10'], p['res20'], p['res30']], float)
    tau = np.array([p[f'tau{j}'] for j in range(4)]); xi = np.array([p[f'emshare{j}'] for j in range(4)])
    kT = p['teq1'] * (1 - np.exp(-TS / p['d1'])) + p['teq2'] * (1 - np.exp(-TS / p['d2']))
    D, dM, Ip = np.zeros(N + 1), np.zeros(N + 1), np.zeros(N + 1)
    for t in range(2, N + 1):
        a = alpha[t]; I = E[t] / 3.667; d = np.exp(-TS / (tau * a))
        dMt = np.sum(xi * tau * I * (1 - d) + (R - xi * tau * a * I) * d * TS / (tau * a ** 2))
        dTt = kT * p['fco22x'] / (np.log(2.0) * MAT[t]) * dMt
        Ip[t] = irf_prime(p, a); dM[t] = dMt
        D[t] = Ip[t] + p['irC'] * dMt - p['irT'] * dTt
        R = xi * tau * a * I * (1 - d) + R * d
    return D, Ip, dM


def ad_jacobian(p, MIU, S, alpha):
    q = dj.pack_params(p); MIUj, Sj = jnp.asarray(MIU), jnp.asarray(S)
    def R(al1):
        al = jnp.concatenate([jnp.zeros(1), al1])
        return dj.irf_residual(q, dj.trajectory(q, MIUj, Sj, al, clamp=False), al)
    Ja = np.asarray(jax.jacfwd(R)(jnp.asarray(alpha[1:])))
    tr = dj.trajectory(q, MIUj, Sj, jnp.asarray(alpha), clamp=False)
    return Ja, np.asarray(tr['ECO2']), np.asarray(tr['MAT'])


def summarize(p, MIU, S, alpha):
    Ja, E, MAT = ad_jacobian(p, MIU, S, alpha)
    D, Ip, dM = analytic_diag(p, alpha, E, MAT)
    dAD = np.diag(Ja)[1:]                      # periods 2..N
    rel = np.abs(D[2:] - dAD) / np.abs(dAD)
    yrs = 2020 + 5 * np.arange(1, N)
    k = int(np.argmin(D[2:] / Ip[2:]))
    return {'min_D': float(D[2:].min()), 'min_ratio_D_over_IRFprime': float((D[2:] / Ip[2:]).min()),
            'year_min_ratio': int(yrs[k]), 'max_ratio': float((D[2:] / Ip[2:]).max()),
            'D_2025': float(D[2]), 'IRFprime_2025': float(Ip[2]),
            'analytic_vs_AD_max_rel': float(rel.max()), 'upper_triangle_max': float(np.abs(np.triu(Ja, 1)).max()),
            'min_alpha': float(alpha[1:].min()), 'abs_logdet': float(np.sum(np.log(np.abs(np.diag(Ja)))))}


def scenarios():
    out = {}
    for sc in range(1, 11):
        _, p = scenario_params(sc)
        if sc == 1:
            p['k0'] = 409.0
        MIU, S, g = gams_controls(sc, p)
        alpha, _ = dm.solve_alpha(p, MIU, S)
        out[str(sc)] = summarize(p, MIU, S, alpha)
        print(sc, out[str(sc)], flush=True)
    return out


def mc_subset(n_each=10):
    import json
    import mc_v2 as V
    import mc_ift as M
    from ipopt_model import Problem
    recs = json.load(open('results/mc2_ipopt/ip_0_250.json')) + json.load(open('results/mc2_ipopt/ip_250_500.json'))
    X = V.design()
    P = {r['draw']: r['params'] for r in recs}
    stress = sorted(P, key=lambda k: -P[k]['irT'] * P[k]['ecs'] / P[k]['irC'])[:n_each]
    lowC = sorted(P, key=lambda k: P[k]['irC'])[:n_each]
    bound = [r['draw'] for r in recs if r.get('T_at_bound')][:n_each]
    pick = list(dict.fromkeys(stress + lowC + bound))
    xg = Problem(9).start_from_gams()
    out = {}
    for k in pick:
        o, x = M.evaluate(X[k], xg)
        p, b = M.draw_params(X[k])
        MIU = np.zeros(N + 1); S = np.zeros(N + 1)
        MIU[1:] = x[:N]; S[1:] = x[N:2 * N]; MIU[1] = p['miu1']
        alpha, _ = dm.solve_alpha(p, MIU, S, alpha_start=np.concatenate([[0], x[2 * N:]]))
        s = summarize(p, MIU, S, alpha)
        s.update({'draw': k, 'group': [g for g, L in (('stress', stress), ('low_sC', lowC), ('T_bound', bound)) if k in L],
                  'params': P[k], 'share_2025_check': o['share_2025'], 'share_2025_stored': next(r['share_2025'] for r in recs if r['draw'] == k)})
        out[str(k)] = s
        print(k, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in s.items() if kk != 'params'}, flush=True)
    return out


if __name__ == '__main__':
    res = {'scenarios': scenarios()}
    save_json(res, 'jacobian_check.json')
    if 'mc' in sys.argv[1:]:
        res['mc'] = mc_subset()
        v = [o['min_ratio_D_over_IRFprime'] for o in res['mc'].values()]
        res['mc_summary'] = {'n': len(v), 'min_ratio': float(min(v)),
                             'max_share_mismatch': float(max(abs(o['share_2025_check'] - o['share_2025_stored']) for o in res['mc'].values()))}
    sv = [o['min_ratio_D_over_IRFprime'] for o in res['scenarios'].values()]
    res['scenario_summary'] = {'min_ratio': float(min(sv)), 'max_analytic_vs_AD': float(max(o['analytic_vs_AD_max_rel'] for o in res['scenarios'].values()))}
    save_json(res, 'jacobian_check.json')
    print(res.get('scenario_summary'), res.get('mc_summary'))
