"""Exact decomposition of the DICE-2023 social cost of carbon into a direct part and the saturation feedback.

    from scc_decomposition import scc_decomposition
    out = scc_decomposition(p, MIU, S)        # p: parameter dict; MIU, S: 1-based control paths (length 82)
    out['total'][1], out['direct'][1], out['feedback'][1]   # 2025 values, 2019 $/tCO2

For fixed controls (MIU, S), alpha is solved from the impulse-response conditions R(alpha, eps) = 0 and the implicit
function theorem gives d alpha / d eps = -(dR/dalpha)^-1 dR/deps. With W the welfare function and lambda_C the
marginal utility of consumption in each period,
    direct(t)   = -1000 * dW/deps_t |alpha fixed           / lambda_C(t)
    feedback(t) = -1000 * dW/dalpha . d alpha / d eps_t     / lambda_C(t)
total = direct + feedback equals the multiplier SCC when no state bound binds (Proposition 1 of the paper) and the
pulse SCC with alpha re-solved; direct alone equals a pulse SCC with alpha held at its optimized path.
Index 0 of each returned array is period 1 (2020), index 1 is 2025, and so on."""
import numpy as np
import jax.numpy as jnp
from common import dm
import mc_ift as M
import dice_jax as dj

YEARS = 2020 + 5 * np.arange(M.N)


def scc_decomposition(p, MIU, S, alpha=None):
    MIU = np.asarray(MIU, float); S = np.asarray(S, float)
    if alpha is None:
        alpha, _ = dm.solve_alpha(p, MIU, S)
    qd = M.qdyn(p); MIUj, Sj = jnp.asarray(MIU), jnp.asarray(S)
    a1 = jnp.asarray(np.asarray(alpha)[1:]); e0 = jnp.zeros(M.N)
    Ja = np.asarray(M.JA(a1, e0, MIUj, Sj, qd)); Je = np.asarray(M.JE(a1, e0, MIUj, Sj, qd))
    gUa = np.asarray(M.GUA(a1, e0, MIUj, Sj, qd)); gUE = np.asarray(M.GUE(a1, e0, MIUj, Sj, qd))
    dadE = -np.linalg.solve(Ja, Je)
    lamC = np.asarray(dj.marginal_utility_C(M.full(qd), dj.trajectory(M.full(qd), MIUj, Sj, jnp.asarray(alpha))))
    direct = -1000 * gUE / lamC; feedback = -1000 * (gUa @ dadE) / lamC
    total = direct + feedback
    with np.errstate(divide='ignore', invalid='ignore'):
        share = feedback / total
    return {'years': YEARS, 'direct': direct, 'feedback': feedback, 'total': total, 'share': share, 'alpha': np.asarray(alpha)}


if __name__ == '__main__':
    from common import scenario_params, gams_controls
    _, p = scenario_params(9)
    MIU, S, _ = gams_controls(9, p)
    o = scc_decomposition(p, MIU, S)
    for k in (1, 6, 16):
        print(f"{o['years'][k]}: SCC {o['total'][k]:8.2f} = direct {o['direct'][k]:8.2f} + feedback {o['feedback'][k]:6.2f}  (share {100 * o['share'][k]:5.2f}%)")
