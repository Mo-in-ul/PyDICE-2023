"""Does the 0.01 C temperature clamp of the Numba forward model touch the SLSQP solutions of Experiments A and B?
Re-solves the Experiment A configurations and the Experiment B re-optimized rows and reports the lowest
temperature on each SLSQP optimal path, with the clamp (Numba, as SLSQP sees it) and without it (JAX, clamp=False).
This is why Experiments A and B are solved with IPOPT (mc_fast.evaluate_ipopt).
Writes results/clamp_check.json."""
import numpy as np, jax.numpy as jnp
from common import save_json, dm
import exp_a_removal_sign as A
import mc_v2 as V, mc_fast as F, mc_ift as M
import dice_jax as dj

N = 81


def minT(row, x):
    p, _ = V.draw_params(row)
    MIU = np.r_[0, x[:N]]; S = np.r_[0, x[N:2 * N]]; al = np.r_[0, x[2 * N:]]; MIU[1] = p['miu1']
    Tn = dm.diceTrajectory(p, MIU, S, al)[4][2:]
    Tj = np.asarray(dj.trajectory(M.full(M.qdyn(p)), jnp.asarray(MIU), jnp.asarray(S), jnp.asarray(al), clamp=False)['TATM'][2:])
    return float(Tn.min()), float(np.nanmin(Tj)) if not np.all(np.isnan(Tj)) else None, int(np.isnan(Tj).sum())


if __name__ == '__main__':
    from ipopt_model import Problem
    xg = Problem(9).start_from_gams()
    out = []
    for cal, (irC, irT) in (('DICE-2023', (0.019, 4.165)), ('slow saturation', (0.010, 2.5))):
        row = np.array([V.DICE[k] for k in V.KEYS]); row[V.KEYS.index('irC')] = irC; row[V.KEYS.index('irT')] = irT
        x0 = xg
        for cfg in ('published', 'cap_1.00', 'cap_1.05', 'deep_1.20', 'early_2070'):
            A.STATE['cfg'] = cfg
            o, x = F.evaluate(row, x0)
            if cfg == 'published':
                x0 = x
            mn, mj, nn = minT(row, x)
            out.append({'exp': 'A', 'calibration': cal, 'config': cfg, 'minT_clamped': mn, 'minT_unclamped': mj, 'n_nan': nn, 'scc_2025': o['scc_2025']})
            print(out[-1], flush=True)
    A.STATE['cfg'] = 'published'
    row0 = np.array([V.DICE[k] for k in V.KEYS])
    _, x0 = F.evaluate(row0, xg)
    for key, vals in (('ecs', np.round(np.arange(2.0, 5.01, 0.25), 2)), ('psi2', (0.0017, 0.0026, 0.003467, 0.0052, 0.0069))):
        for v in vals:
            row = row0.copy(); row[V.KEYS.index(key)] = v
            o, x = F.evaluate(row, x0)
            mn, mj, nn = minT(row, x)
            out.append({'exp': 'B', 'param': key, 'value': float(v), 'minT_clamped': mn, 'minT_unclamped': mj, 'n_nan': nn, 'scc_2025': o['scc_2025']})
            print(out[-1], flush=True)
    save_json(out, 'clamp_check.json')
    print('CLAMP_DONE')
