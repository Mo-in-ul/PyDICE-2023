"""v9: is the welfare-weighted temperature form of the SCC (Eq. 18) exact? (second-round referee comment 1)

Eq. 18 writes SCC(t) = -1000/lambda_C(t) * sum_s w(s) dT(s)/dE(t), with w(s) = dW/dT(s) computed with emissions
held at the optimal path (so the weight does not contain the output -> emissions -> carbon loop).
Three calculations on the DICE-2023 optimum (GAMS controls), DFAIR carbon cycle:
  A  full automatic derivative of welfare (scc_decomposition.py), emissions respond to output
  B  Eq. 18 with the coupled temperature response: dT/dE from the full forward model, in which a pulse lowers
     output and therefore later industrial emissions (fair_benchmark.dice_split)
  C  Eq. 18 with the temperature response computed at frozen emissions (later emissions held at the optimal path),
     which is how the FaIR v2 substitution of Section 5.2 is run
If welfare depends on emissions only through temperature, B equals A exactly (chain rule), and C differs from A by
the output -> emissions loop. Reported to ten significant figures. Writes results/c8_eq18_check.json.
"""
import numpy as np
import jax, jax.numpy as jnp
from common import save_json
import fair_benchmark as FB
import dice_jax as dj
from scc_decomposition import scc_decomposition

N = 81
p, q, MIUj, Sj, aj, w, lamC, E, T, Fnon = FB.setup()
A = scc_decomposition(p, np.asarray(MIUj), np.asarray(Sj), alpha=np.asarray(aj))
Efix = jnp.asarray(E)


def split_frozen(i):
    def T_of(al1, e):
        al = jnp.concatenate([jnp.zeros(1), al1]); ee = jnp.concatenate([jnp.zeros(1), e])
        return dj.trajectory(q, MIUj, Sj, al, eps_E=ee, E_fix=Efix)['TATM']

    def R_of(al1, e):
        al = jnp.concatenate([jnp.zeros(1), al1]); ee = jnp.concatenate([jnp.zeros(1), e])
        return dj.irf_residual(q, dj.trajectory(q, MIUj, Sj, al, eps_E=ee, E_fix=Efix), al)
    a1 = aj[1:]; e0 = jnp.zeros(N)
    dTdE = np.asarray(jax.jacfwd(T_of, argnums=1)(a1, e0))[1:, i - 1]
    dTda = np.asarray(jax.jacfwd(T_of, argnums=0)(a1, e0))[1:]
    Ja = np.asarray(jax.jacfwd(R_of, argnums=0)(a1, e0)); Je = np.asarray(jax.jacfwd(R_of, argnums=1)(a1, e0))
    dadE = -np.linalg.solve(Ja, Je[:, i - 1])
    return dTdE, dTda @ dadE


out = {}
for y in (2025, 2050, 2100):
    i = (y - 2020) // 5 + 1
    k = i - 1
    a = (float(A['direct'][k]), float(A['feedback'][k]))
    b = FB.scc_parts(w, lamC, i, *FB.dice_split(q, MIUj, Sj, aj, w, lamC, i))
    c = FB.scc_parts(w, lamC, i, *split_frozen(i))
    rel = lambda x, z: (x[0] + x[1]) / (z[0] + z[1]) - 1
    out[y] = {'A_full': [a[0] + a[1], a[1], a[1] / (a[0] + a[1])], 'B_eq18_coupled': [b[0] + b[1], b[1], b[1] / (b[0] + b[1])],
              'C_eq18_frozen_emissions': [c[0] + c[1], c[1], c[1] / (c[0] + c[1])],
              'B_vs_A_rel': rel(b, a), 'C_vs_A_rel': rel(c, a),
              'share_C_minus_A_pp': 100 * (c[1] / (c[0] + c[1]) - a[1] / (a[0] + a[1]))}
    print(y, {kk: (["%.10g" % v for v in vv] if isinstance(vv, list) else "%.3e" % vv) for kk, vv in out[y].items()}, flush=True)
save_json(out, 'c8_eq18_check.json')
