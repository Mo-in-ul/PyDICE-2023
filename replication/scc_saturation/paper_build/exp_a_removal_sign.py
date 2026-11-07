"""Experiment A: when does the saturation feedback change sign? Sensitivity to net-negative emissions.

DICE-2023 allows net-negative emissions through a control rate above one (upper bound 1.1 from 2120, 1.05 from
2200). Configurations of that upper bound (all else as the cost-benefit optimum without the 0.5 C floor, as in
the Monte Carlo):
  published   bound as in DICE-2023
  cap_1.00    no control rate above one (no net-negative emissions from abatement)
  cap_1.05    at most 1.05 throughout
  deep_1.20   1.2 wherever DICE-2023 allows more than one
  early_2070  the 1.1 bound already from 2070
each under two DFAIR calibrations: DICE-2023 (irC 0.019, irT 4.165) and a slow-saturation alternative at the
lower end of the Monte Carlo ranges (irC 0.010, irT 2.5). For each solution: first net-negative year; first year
the net flux into the carbon sinks Phi = E/3.667 - dM_AT/dt turns negative; first pulse year with a negative
saturation feedback; 2025 share. Each optimum is solved with IPOPT on the unclamped model (mc_fast.evaluate_ipopt).
Writes results/exp_a_removal_sign.json."""
import numpy as np
from common import save_json
import mc_v2 as V, mc_fast as F

N = 81
YRS = 2020 + 5 * np.arange(N)
_orig = V.draw_params
STATE = {'cfg': 'published'}


def bound(p, b, cfg):
    up = b['miu_up'].copy()
    if cfg == 'cap_1.00':
        up = np.minimum(up, 1.0)
    elif cfg == 'cap_1.05':
        up = np.minimum(up, 1.05)
    elif cfg == 'deep_1.20':
        up = np.where(up > 1.0 + 1e-9, 1.2, up)
    elif cfg == 'early_2070':
        t = np.arange(len(up)); yr = 2015 + 5 * t
        up = np.where((yr >= 2070) & (yr < 2120), np.maximum(up, 1.1), up)
    b['miu_up'] = up; p['miuup'] = up.copy()
    return p, b


def patched(row):
    p, b = _orig(row)
    return bound(p, b, STATE['cfg'])


V.draw_params = patched
F.M.draw_params = patched


def phi_path(row, x):
    """Net flux into the sinks, GtC per year, from the solved path."""
    import jax.numpy as jnp, dice_jax as dj, mc_ift as M
    from common import dm
    p, _ = patched(row)
    MIU = np.zeros(N + 1); S = np.zeros(N + 1); MIU[1:] = x[:N]; S[1:] = x[N:2 * N]; MIU[1] = p['miu1']
    alpha, _ = dm.solve_alpha(p, MIU, S, alpha_start=np.concatenate([[0], x[2 * N:]]))
    tr = dj.trajectory(M.full(M.qdyn(p)), jnp.asarray(MIU), jnp.asarray(S), jnp.asarray(alpha))
    E = np.asarray(tr['ECO2'][1:]); MAT = np.asarray(tr['MAT'][1:])
    return E[:-1] / 3.667 - np.diff(MAT) / 5.0, E


if __name__ == '__main__':
    from ipopt_model import Problem
    xg = Problem(9).start_from_gams()
    out = []
    for cal, (irC, irT) in (('DICE-2023', (0.019, 4.165)), ('slow saturation', (0.010, 2.5))):
        row = np.array([V.DICE[k] for k in V.KEYS]); row[V.KEYS.index('irC')] = irC; row[V.KEYS.index('irT')] = irT
        x0 = xg
        for cfg in ('published', 'cap_1.00', 'cap_1.05', 'deep_1.20', 'early_2070'):
            STATE['cfg'] = cfg
            o, x = F.evaluate_ipopt(row, x0)
            if cfg == 'published':
                x0 = x
            phi, E = phi_path(row, x)
            negphi = YRS[:-1][phi < 0]
            sh = np.array(o['share_path'])
            negsh = YRS[:41][sh < 0]
            r = {'calibration': cal, 'config': cfg, 'success': o['success'], 'share_2025': o['share_2025'], 'scc_2025': o['scc_2025'],
                 'first_net_negative_year': o['first_net_negative_year'], 'first_negative_flux_year': int(negphi[0]) if len(negphi) else None,
                 'first_negative_feedback_pulse_year': int(negsh[0]) if len(negsh) else None, 'peak_T': o['peak_T'],
                 'min_E_GtCO2': float(E.min())}
            out.append(r); save_json(out, 'exp_a_removal_sign.json')
            print(cal, cfg, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ('calibration', 'config')}, flush=True)
    print('done')
