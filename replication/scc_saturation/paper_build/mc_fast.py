"""Monte Carlo, production run: SLSQP optimum per draw + exact IFT decomposition.

Same design and parameter mapping as mc_ift.py (1000 LHS draws, seed 42). Each draw is solved
with SLSQP on the Numba model (as run_faithful.py, GAMS opt settings, warm start from the GAMS
optimum), then the SCC is split into direct (alpha fixed) and feedback parts with the implicit
function theorem (JAX functions from mc_ift.py, compiled once). mc_ift.py, which uses IPOPT
and the multiplier SCC, validates a subsample (mc_validate in make_tables / the notebook).
Usage: python mc_fast.py <first> <last> -> results/mc/fast_<first>_<last>.json
       python mc_fast.py --no-floor --draws k1,k2,... -> results/mc/fast_nofloor.json
--no-floor removes the GAMS stability bound TATM >= 0.5 (it binds late in 253 of 1000 draws, where
the fixed-control SCC then overstates the multiplier SCC by 0.1-0.9% in 2025; see mc_ift.py --draws).
"""
import os, sys, time, warnings
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize
from common import dm, save_json, RES
import dice_jax as dj
import mc_ift as M

N = 81
YEARS = (2025, 2050, 2100)


def solve_slsqp(p, b, x0):
    prob = dm.DiceFunc(N, p, TempUpperConstraint=b['Tup'], TempLowerConstraint=b['Tlo'])
    mb = [(p['miu1'], p['miu1'])] + [(b['miu_lo'][t], b['miu_up'][t]) for t in range(2, N + 1)]
    sb = [(p['sLBounds'][i], p['sUBounds'][i]) for i in range(1, N + 1)]
    ab = [(p['AlphaLowerBound'], p['AlphaUpperBound'])] * N
    lo = np.array([m[0] for m in mb + sb + ab]); hi = np.array([m[1] for m in mb + sb + ab])
    x = np.clip(x0, lo, hi); x[2 * N] = p['a0']
    cons = [{'type': 'eq', 'fun': prob.irf_residual}, {'type': 'ineq', 'fun': prob.temp_up}, {'type': 'ineq', 'fun': prob.temp_lo}]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for ftol in (1e-7, 1e-9):
            res = minimize(prob.objective, x, method='SLSQP', bounds=mb + sb + ab, constraints=cons,
                           options={'maxiter': 3000, 'ftol': ftol, 'eps': 1e-6})
            x = res.x
            if res.success:
                break
    return res, x, float(np.max(np.abs(prob.irf_residual(x))))


def evaluate(row, x0, no_floor=False):
    t0 = time.time()
    p, b = M.draw_params(row)
    if no_floor:
        b['Tlo'] = 0.01
    res, x, irf = solve_slsqp(p, b, x0)
    return postprocess(row, p, x, bool(res.success), int(res.status), irf, t0)


def evaluate_ipopt(row, x0):
    """As evaluate, but solved with IPOPT on the unclamped JAX model (mc_ift.solve_draw). The Numba forward model
    used by SLSQP clamps temperature at 0.01 C, which is also the lower bound in these runs, so where the bound is
    active SLSQP sees a floored temperature instead of a constraint. IPOPT imposes the bound as a constraint."""
    t0 = time.time()
    p, b = M.draw_params(row)
    qd, x, info = M.solve_draw(p, b, x0)
    x = np.asarray(x, float); x[2 * N] = p['a0']
    g = np.asarray(M.G(jnp.asarray(x), qd))
    o, x = postprocess(row, p, x, bool(info['status'] in (0, 1)), int(info['status']), float(np.max(np.abs(g[:N]))), t0)
    o['min_T_unclamped'] = float(g[N:].min())
    return o, x


def postprocess(row, p, x, success, status, irf, t0):
    MIU = np.zeros(N + 1); S = np.zeros(N + 1)
    MIU[1:] = x[:N]; S[1:] = x[N:2 * N]; MIU[1] = p['miu1']
    alpha, ares = dm.solve_alpha(p, MIU, S, alpha_start=np.concatenate([[0], x[2 * N:]]))
    qd = M.qdyn(p)
    MIUj, Sj = jnp.asarray(MIU), jnp.asarray(S)
    a1 = jnp.asarray(alpha[1:]); e0 = jnp.zeros(N)
    Ja = np.asarray(M.JA(a1, e0, MIUj, Sj, qd)); Je = np.asarray(M.JE(a1, e0, MIUj, Sj, qd))
    gUa = np.asarray(M.GUA(a1, e0, MIUj, Sj, qd)); gUE = np.asarray(M.GUE(a1, e0, MIUj, Sj, qd))
    dadE = -np.linalg.solve(Ja, Je)
    traj = dj.trajectory(M.full(qd), MIUj, Sj, jnp.asarray(alpha))
    lamC = np.asarray(dj.marginal_utility_C(M.full(qd), traj))
    direct = -1000 * gUE / lamC; feedback = -1000 * (gUa @ dadE) / lamC
    T = np.asarray(traj['TATM'][1:]); E = np.asarray(traj['ECO2'][1:])
    yrs = 2020 + 5 * np.arange(N)
    o = {'params': dict(zip(M.BOUNDS, map(float, row))), 'success': success, 'status': status,
         'U': float(traj['UTILITY']), 'irf_max': irf, 'peak_T': float(T[:40].max()),
         'peak_year': int(yrs[np.argmax(T[:40])]), 'Tlo_binds': bool(T[1:].min() < 0.5 + 1e-6),
         'first_net_negative_year': int(yrs[E < 0][0]) if np.any(E < 0) else None, 'elapsed_s': time.time() - t0,
         'scc_path': (direct + feedback)[:41].tolist(), 'share_path': (feedback / (direct + feedback))[:41].tolist()}
    for y in YEARS:
        i = (y - 2020) // 5
        o[f'scc_{y}'] = float(direct[i] + feedback[i]); o[f'direct_{y}'] = float(direct[i])
        o[f'feedback_{y}'] = float(feedback[i]); o[f'share_{y}'] = float(feedback[i] / (direct[i] + feedback[i]))
    return o, x


if __name__ == '__main__':
    if '--no-floor' in sys.argv:
        draws = [int(v) for v in sys.argv[sys.argv.index('--draws') + 1].split(',')]
        keys, X = M.design()
        from ipopt_model import Problem
        xg = Problem(9).start_from_gams()
        out = []
        for k in draws:
            o, _ = evaluate(X[k], xg, no_floor=True); o['draw'] = k; o['floor_removed'] = True; out.append(o)
            if len(out) % 10 == 0:
                save_json(out, f'mc/{sys.argv[-1]}.json')
            print(k, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in o.items() if kk in ('success', 'scc_2025', 'share_2025', 'Tlo_binds')}, flush=True)
        save_json(out, f'mc/{sys.argv[-1]}.json')
        sys.exit()
    a, z = int(sys.argv[1]), int(sys.argv[2])
    keys, X = M.design()
    os.makedirs(os.path.join(RES, 'mc'), exist_ok=True)
    from ipopt_model import Problem
    xg = Problem(9).start_from_gams()
    path = os.path.join(RES, 'mc', f'fast_{a}_{z}.json')
    out = []
    if os.path.exists(path):      # resume: keep finished draws
        from common import load_json
        out = [o for o in load_json(f'mc/fast_{a}_{z}.json') if 'error' not in o]
    done = {o['draw'] for o in out}
    for k in range(a, z):
        if k in done:
            continue
        try:
            o, _ = evaluate(X[k], xg)
        except Exception as e:
            o = {'params': dict(zip(M.BOUNDS, map(float, X[k]))), 'error': repr(e)}
        o['draw'] = k; out.append(o)
        if k % 10 == 0 or k == z - 1:
            save_json(out, f'mc/fast_{a}_{z}.json')
        print(k, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in o.items()
                  if kk in ('success', 'scc_2025', 'share_2025', 'share_2100', 'elapsed_s', 'error')}, flush=True)
    save_json(out, f'mc/fast_{a}_{z}.json')
