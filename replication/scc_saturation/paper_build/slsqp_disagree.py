"""Diagnostics for the Monte Carlo draws where the SLSQP and IPOPT 2025 SCCs differ by more than 0.1%.
For each draw both solvers are re-run from the same start. Reported: largest IRF residual of each solution; the
lowest temperature on each path with the 0.01 C clamp of the Numba model (what SLSQP sees) and without it (the
model IPOPT solves, where the bound is a constraint); the number of periods in which the clamp is active on the
SLSQP path; the largest difference in the control rate; the IPOPT exit status (status 0: KKT conditions met to
tolerance 1e-9); and the SCC and share from each solver. Writes results/slsqp_disagree.json."""
import json, glob, os
import numpy as np
import jax.numpy as jnp
from common import RES, save_json, dm
import mc_v2 as V, mc_fast as Fs, mc_ift as M
import dice_jax as dj

N = M.N
YRS = 2020 + 5 * np.arange(N)


def temps(p, x):
    MIU = np.r_[0, x[:N]]; S = np.r_[0, x[N:2 * N]]; al = np.r_[0, x[2 * N:]]; MIU[1] = p['miu1']
    Tc = dm.diceTrajectory(p, MIU, S, al)[4][2:]
    Tu = np.asarray(dj.trajectory(M.full(M.qdyn(p)), jnp.asarray(MIU), jnp.asarray(S), jnp.asarray(al), clamp=False)['TATM'][2:])
    return Tc, Tu


if __name__ == '__main__':
    I = {}; S = {}
    for f in glob.glob(os.path.join(RES, 'mc2_ipopt', '*.json')):
        for o in json.load(open(f)): I[o['draw']] = o
    for f in glob.glob(os.path.join(RES, 'mc2', '*.json')):
        for o in json.load(open(f)): S[o['draw']] = o
    both = [k for k in I if I[k].get('status') in (0, 1) and S.get(k, {}).get('success')]
    bad = sorted(k for k in both if abs(S[k]['scc_2025'] / I[k]['scc_2025'] - 1) > 1e-3)
    X = V.design()
    from ipopt_model import Problem
    xg = Problem(9).start_from_gams()
    out = []
    for k in bad:
        row = X[k]; p, b = V.draw_params(row)
        qd, xi, info = M.solve_draw(p, b, xg); xi = np.asarray(xi); xi[2 * N] = p['a0']
        res, xs, irf_s = Fs.solve_slsqp(p, b, xg)
        gi = np.asarray(M.G(jnp.asarray(xi), qd))
        Tci, Tui = temps(p, xi); Tcs, Tus = temps(p, xs)
        below = np.where(~(Tus >= b['Tlo'] - 1e-9))[0]
        r = {'draw': k, 'scc_2025_ipopt': I[k]['scc_2025'], 'scc_2025_slsqp': S[k]['scc_2025'],
             'scc_diff_pct': 100 * (S[k]['scc_2025'] / I[k]['scc_2025'] - 1),
             'share_2025_ipopt': I[k]['share_2025'], 'share_2025_slsqp': S[k]['share_2025'],
             'irf_max_ipopt': float(np.max(np.abs(gi[:N]))), 'irf_max_slsqp': irf_s,
             'ipopt_status': int(info['status']), 'slsqp_success': bool(res.success),
             'minT_ipopt': float(Tui.min()), 'minT_slsqp_clamped': float(Tcs.min()),
             'minT_slsqp_unclamped': float(np.nanmin(Tus)) if np.any(~np.isnan(Tus)) else None,
             'slsqp_first_year_below_bound': int(YRS[1:][below[0]]) if len(below) else None,
             'slsqp_periods_below_bound': int(len(below)),
             'max_abs_miu_diff': float(np.max(np.abs(xi[:N] - xs[:N])))}
        out.append(r); print(json.dumps({q: (round(v, 6) if isinstance(v, float) else v) for q, v in r.items()}), flush=True)
        save_json(out, 'slsqp_disagree.json')
    print('DIAG_DONE')
