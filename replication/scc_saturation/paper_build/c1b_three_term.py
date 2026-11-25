"""v8: three-term split of the multiplier SCC when a temperature bound binds (referee comment 4).

Maximize W subject to the IRF conditions R = 0 and the temperature bounds g <= 0. With alpha eliminated through the
implicit function theorem, the envelope theorem gives

    dV/deps_t = W_eps + W_alpha dalpha/deps_t - eta' (g_eps + g_alpha dalpha/deps_t),

direct + saturation feedback + constraint valuation. The first two terms are those of scc_decomposition.py at the
optimal policy; the third is computed here directly from the bound multipliers and checked against
SCC - direct - feedback. In code the problem is minimize f = -W with Lagrangian f + lambda' g (ipopt_model), so the
same identity reads dV/dE = f_E + f_alpha alpha_E + lambda_T' (gT_E + gT_alpha alpha_E).
Writes results/c1b_three_term.json.
"""
import numpy as np
import jax, jax.numpy as jnp
from common import save_json
from ipopt_model import solve

NAMES = {1: '1% discounting', 2: '2% discounting', 3: '3% discounting', 7: '2 C limit', 9: 'Cost-benefit optimum'}


def three(sc):
    r = solve(sc, disposal=False, warm='gams')
    pr = r['problem']; N = pr.N; x = jnp.asarray(r['x']); lam = np.asarray(r['info']['mult_g']); z = pr.z
    fE = np.asarray(pr._dfdE(z, x)); gE = np.asarray(pr._dgdE(z, x))
    fx = np.asarray(pr._grad(x)); gx = np.asarray(pr._jac(x))
    ia = slice(2 * N, 3 * N)
    Ra, RE = gx[:N, ia], gE[:N]
    aE = -np.linalg.solve(Ra, RE)                      # d alpha / d eps, N x (N+1)
    fa = fx[ia]
    lT = lam[N:]; gTE, gTa = gE[N:], gx[N:, ia]
    dVdC = r['dVdC']
    conv = lambda v: -1000.0 * v[1:] / dVdC[1:]
    direct = conv(fE); feed = conv(fa @ aE); cons = conv(lT @ (gTE + gTa @ aE))
    tot = r['scc'][1:]
    yrs = 2025 + 5 * np.arange(36)                      # 2025..2200
    idx = (yrs - 2020) // 5
    res = np.abs(tot[idx] - direct[idx] - feed[idx] - cons[idx]) / np.abs(tot[idx])
    out_res = {'max_rel_residual_2025_2200': float(res.max()), 'argmax_year': int(yrs[np.argmax(res)])}
    rows = []
    for y in (2025, 2050, 2100):
        i = (y - 2020) // 5
        rows.append({'year': y, 'scc': float(tot[i]), 'direct': float(direct[i]), 'feedback': float(feed[i]),
                     'constraint': float(cons[i]), 'residual': float(tot[i] - direct[i] - feed[i] - cons[i])})
    return {'scenario': sc, 'name': NAMES[sc], 'status': r['status'], 'rows': rows, **out_res}


if __name__ == '__main__':
    out = {}
    for sc in (9, 3, 2, 1, 7):
        o = three(sc); out[str(sc)] = o
        for rw in o['rows']:
            print(f"{o['name']:22s} {rw['year']}: {rw['scc']:9.2f} = {rw['direct']:9.2f} + {rw['feedback']:8.2f} + {rw['constraint']:9.2f}  (res {rw['residual']:.1e})", flush=True)
        print(o['name'], 'max rel residual 2025-2200', o['max_rel_residual_2025_2200'], o['argmax_year'], flush=True)
        save_json(out, 'c1b_three_term.json')
