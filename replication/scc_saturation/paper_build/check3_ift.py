"""Check 3: SCC decomposition by the implicit function theorem (no pulses, no re-solving).

Hold the controls (MIU, S) at the GAMS paths. alpha(1..N) is defined implicitly by the IRF
condition R(alpha, eps) = 0, where eps(t) is an additive term in the emissions equation.
Differentiating R(alpha(eps), eps) = 0:
    d alpha / d eps = -(dR/dalpha)^(-1) dR/deps          (dR/dalpha is lower triangular)
The marginal welfare effect of emissions in period t then splits exactly into
    dU/deps(t) = dU/deps(t)|alpha fixed  +  dU/dalpha . d alpha/d eps(t)
                 -------- direct --------   ------- saturation feedback -------
and SCC(t) = -1000 * dU/deps(t) / dU/dC(t). All partial derivatives come from automatic
differentiation of dice_jax.trajectory (exact to rounding), so this check shares no
numerics with the finite-difference SCCs in scc_gams_controls.py:
  direct            should equal the fixed-alpha perturbation SCC
  direct + feedback should equal the responsive-alpha perturbation SCC
Writes results/check3_ift.json (per scenario: the three SCCs for every period, the feedback
share, and the comparison with the finite-difference values at the report years).
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp
from common import dm, save_json, load_json, scenario_params, gams_controls
import dice_jax as dj

YEARS = (2025, 2030, 2050, 2100, 2150)


def ift_scc(p, MIU, S):
    alpha, res = dm.solve_alpha(p, MIU, S)
    q = dj.pack_params(p)
    N = p['num_periods']
    MIUj, Sj = jnp.asarray(MIU), jnp.asarray(S)
    z = jnp.zeros(N + 1)

    def R(al1, e):     # al1: alpha(1..N), e: eps(1..N)
        al = jnp.concatenate([jnp.zeros(1), al1]); ee = jnp.concatenate([jnp.zeros(1), e])
        r = dj.trajectory(q, MIUj, Sj, al, eps_E=ee)
        return dj.irf_residual(q, r, al)

    def U(al1, e):
        al = jnp.concatenate([jnp.zeros(1), al1]); ee = jnp.concatenate([jnp.zeros(1), e])
        return dj.trajectory(q, MIUj, Sj, al, eps_E=ee)['UTILITY']

    a1 = jnp.asarray(alpha[1:]); e0 = jnp.zeros(N)
    Ja = np.asarray(jax.jacfwd(R, argnums=0)(a1, e0))
    JE = np.asarray(jax.jacfwd(R, argnums=1)(a1, e0))
    gUa = np.asarray(jax.grad(U, argnums=0)(a1, e0))
    gUE = np.asarray(jax.grad(U, argnums=1)(a1, e0))
    dadE = -np.linalg.solve(Ja, JE)            # N x N, column t = response of alpha to eps(t)
    traj = dj.trajectory(q, MIUj, Sj, jnp.asarray(alpha))
    lam = np.asarray(dj.marginal_utility_C(q, traj))
    direct = -1000.0 * gUE / lam
    feedback = -1000.0 * (gUa @ dadE) / lam
    tri_err = float(np.max(np.abs(np.triu(Ja, 1))))   # should be 0: residual t depends on alpha(1..t)
    return direct, feedback, dadE, tri_err, res


def run(sc):
    _, p = scenario_params(sc)
    if sc == 1:
        p['k0'] = 409.0
    MIU, S, g = gams_controls(sc, p)
    N = p['num_periods']
    d, f, dadE, tri, res = ift_scc(p, MIU, S)
    tot = d + f
    o = {'scenario': sc, 'irf_residual': res, 'upper_triangle_max': tri,
         'years_all': (2020 + 5 * np.arange(N)).tolist(),
         'direct_all': d.tolist(), 'feedback_all': f.tolist(), 'total_all': tot.tolist(),
         'share_all': np.where(np.abs(tot) > 0, f / np.where(tot == 0, 1, tot), 0).tolist(),
         'dalpha_dE_2025': dadE[:, 1].tolist(),
         'years': list(YEARS)}
    fd = load_json('scc_gams_controls.json')[str(sc)]
    for k in ('direct', 'feedback', 'total', 'fd_fixed', 'fd_resp', 'gams', 'dev_direct_vs_fixed_pct', 'dev_total_vs_resp_pct'):
        o[k] = []
    for y in YEARS:
        i = (y - 2020) // 5          # 0-based period index
        j = fd['years'].index(y)
        o['direct'].append(float(d[i])); o['feedback'].append(float(f[i])); o['total'].append(float(tot[i]))
        o['fd_fixed'].append(fd['fixed'][j]); o['fd_resp'].append(fd['resp'][j]); o['gams'].append(fd['gams'][j])
        o['dev_direct_vs_fixed_pct'].append(100 * (d[i] / fd['fixed'][j] - 1))
        o['dev_total_vs_resp_pct'].append(100 * (tot[i] / fd['resp'][j] - 1))
    return o


if __name__ == '__main__':
    scs = [int(s) for s in sys.argv[1:]] or list(range(1, 11))
    allr = {}
    for sc in scs:
        o = run(sc); allr[str(sc)] = o
        print(f"S{sc:2d} triangular {o['upper_triangle_max']:.1e}")
        for j, y in enumerate(o['years']):
            print(f"   {y}: direct {o['direct'][j]:9.3f} vs FD fixed {o['fd_fixed'][j]:9.3f} ({o['dev_direct_vs_fixed_pct'][j]:+.2e}%)  "
                  f"total {o['total'][j]:9.3f} vs FD resp {o['fd_resp'][j]:9.3f} ({o['dev_total_vs_resp_pct'][j]:+.2e}%)  "
                  f"share {100*o['feedback'][j]/o['total'][j]:6.2f}%")
        sys.stdout.flush()
    save_json(allr, 'check3_ift.json')
