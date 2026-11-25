"""Shadow-price form of the SCC decomposition (v8, referee comment 1 and 4).

At an IPOPT optimum of the reduced problem (decision variables MIU, S, alpha; constraints: the
impulse-response conditions R = 0 and the temperature bounds), the envelope theorem gives

    dV/dE_t = df/dE_t  +  nu' dR/dE_t  +  eta' dT/dE_t,

with nu the multipliers of the IRF conditions and eta those of the temperature bounds. Divided by the
marginal value of consumption and scaled to $/tCO2 (as in ipopt_model.multiplier_scc), the three terms are

    direct     : welfare effect of the pulse with alpha and the policy held fixed,
    feedback   : the pulse valued through the IRF constraints at their shadow prices,
    constraint : the pulse valued through any binding temperature bound.

Because the first-order condition for alpha is W_alpha + nu' R_alpha = 0, nu' R_E = -W_alpha R_alpha^-1 R_E,
so the feedback term equals the implicit-function feedback of scc_decomposition.py whenever the policy is the
optimum (third check of the decomposition). When a temperature bound binds, the constraint term is the scarcity
value of the bound, and the three terms still add to the multiplier SCC.

Writes results/c1_multiplier_split.json.
"""
import numpy as np
import jax.numpy as jnp
from common import save_json, scenario_params
from ipopt_model import solve
from scc_decomposition import scc_decomposition

NAMES = {1: '1% discounting', 2: '2% discounting', 3: '3% discounting', 4: '4% discounting',
         5: '5% discounting', 7: '2 C limit', 8: 'Paris', 9: 'Cost-benefit optimum', 10: 'Baseline'}
YEARS = (2025, 2050, 2100)


def split(sc):
    r = solve(sc, disposal=False, warm='gams')
    pr = r['problem']; N = pr.N; x = jnp.asarray(r['x']); lam = np.asarray(r['info']['mult_g'])
    z = pr.z
    dfdE = np.asarray(pr._dfdE(z, x)); dgdE = np.asarray(pr._dgdE(z, x))
    dVdC = r['dVdC']
    nirf = N
    t_direct = dfdE
    t_feed = lam[:nirf] @ dgdE[:nirf]
    t_cons = lam[nirf:] @ dgdE[nirf:]
    conv = lambda v: -1000.0 * v[1:] / dVdC[1:]
    d, f, c = conv(t_direct), conv(t_feed), conv(t_cons)
    tot = r['scc'][1:]
    # IFT decomposition at the IPOPT optimum (same controls), for comparison
    MIU, S, al, _ = pr.unpack(x)
    ift = scc_decomposition(pr.p, np.asarray(MIU), np.asarray(S), alpha=np.asarray(al))
    T = r['traj']['TATM'][1:]
    o = {'scenario': sc, 'name': NAMES[sc], 'status': r['status'],
         'periods_T_at_bound': int(np.sum((T[1:] < pr.b['Tlo'] + 1e-6) | (T[1:] > pr.b['Tup'] - 1e-6))),
         'min_T': float(T.min()), 'peak_T': float(T[:40].max()), 'years': list(YEARS), 'rows': []}
    for y in YEARS:
        i = (y - 2020) // 5   # 0-based over periods 1..N
        o['rows'].append({'year': y, 'scc': float(tot[i]), 'direct': float(d[i]), 'feedback': float(f[i]),
                          'constraint': float(c[i]), 'sum_check': float(d[i] + f[i] + c[i] - tot[i]),
                          'ift_direct': float(ift['direct'][i]), 'ift_feedback': float(ift['feedback'][i]),
                          'ift_total': float(ift['total'][i]),
                          'share_feedback_pct': 100 * float(f[i] / tot[i])})
    return o


if __name__ == '__main__':
    import sys
    scs = [int(s) for s in sys.argv[1:]] or [9, 8, 10, 4, 5, 7, 3, 2, 1]
    out = {}
    for sc in scs:
        try:
            o = split(sc)
        except Exception as e:
            print(sc, 'ERROR', repr(e)); continue
        out[str(sc)] = o
        save_json(out, 'c1_multiplier_split.json')
        print(f"S{sc} {o['name']}: status {o['status']} T-bound periods {o['periods_T_at_bound']} minT {o['min_T']:.3f}")
        for rw in o['rows']:
            print(f"  {rw['year']}: SCC {rw['scc']:9.2f} = direct {rw['direct']:9.2f} + feedback {rw['feedback']:8.2f}"
                  f" + constraint {rw['constraint']:8.2f}  (res {rw['sum_check']:.1e});"
                  f"  IFT: {rw['ift_direct']:9.2f} + {rw['ift_feedback']:8.2f} = {rw['ift_total']:9.2f}")
        sys.stdout.flush()
