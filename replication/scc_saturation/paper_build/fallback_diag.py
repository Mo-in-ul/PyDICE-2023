"""Diagnostics for the seven Monte Carlo draws solved by SLSQP (IPOPT from the DICE-2023 start did not converge).
For each: feasibility of the SLSQP solution (largest IRF residual; lowest and highest temperature against the
bounds 0.01 and 20 C; lowest and highest alpha against 0.1 and 100; lowest atmospheric carbon against the floor of
the forward model), the number of active control bounds, and the IPOPT re-solve warm-started from the SLSQP
solution: exit status, KKT stationarity residual over the free variables (fixed variables excluded), largest
constraint violation, and multiplier SCC against the reported SLSQP decomposition total.
Also the Monte Carlo summary with the seven draws excluded. Writes results/fallback_diag.json."""
import json, glob, os
import numpy as np
import jax, jax.numpy as jnp
from common import RES, save_json, dm
import mc_v2 as V, mc_fast as Fs, mc_ift as M
import dice_jax as dj
from ipopt_model import GAMS_ALPHA
from fig_mc2 import load

N = M.N

if __name__ == '__main__':
    I = {}; S = {}
    for f in glob.glob(os.path.join(RES, 'mc2_ipopt', '*.json')):
        for o in json.load(open(f)): I[o['draw']] = o
    for f in glob.glob(os.path.join(RES, 'mc2', '*.json')):
        for o in json.load(open(f)): S[o['draw']] = o
    fb = sorted(k for k in I if I[k].get('status') not in (0, 1))
    X = V.design()
    from ipopt_model import Problem
    xg = Problem(9).start_from_gams()
    out = {'draws': []}
    for k in fb:
        p, b = V.draw_params(X[k]); qd = M.qdyn(p)
        res, xs, irf = Fs.solve_slsqp(p, b, xg)
        xs = np.asarray(xs, float); xs[2 * N] = p['a0']
        MIU = np.r_[0, xs[:N]]; MIU[1] = p['miu1']; Sv = np.r_[0, xs[N:2 * N]]; al = np.r_[0, xs[2 * N:]]
        tr = dj.trajectory(M.full(qd), jnp.asarray(MIU), jnp.asarray(Sv), jnp.asarray(al), clamp=False)
        T = np.asarray(tr['TATM'][2:]); MAT = np.asarray(tr['MAT'][1:])
        xl = np.concatenate([b['miu_lo'][1:], p['sLBounds'][1:], np.full(N, GAMS_ALPHA[0])])
        xu = np.concatenate([b['miu_up'][1:], p['sUBounds'][1:], np.full(N, GAMS_ALPHA[1])]); xl[2 * N] = xu[2 * N] = p['a0']
        free = xl < xu - 1e-12
        act = int(np.sum(free[:2 * N] & ((xs[:2 * N] <= xl[:2 * N] + 1e-7) | (xs[:2 * N] >= xu[:2 * N] - 1e-7))))
        q2, xi, info = M.solve_draw(p, b, xs)
        xi = np.asarray(xi, float)
        gf = np.asarray(jax.grad(lambda z: M.F(z, qd))(jnp.asarray(xi)))
        Jg = np.asarray(jax.jacobian(lambda z: M.G(z, qd))(jnp.asarray(xi)))
        kkt = gf + Jg.T @ np.asarray(info['mult_g']) - np.asarray(info['mult_x_L']) + np.asarray(info['mult_x_U'])
        g = np.asarray(M.G(jnp.asarray(xi), qd))
        viol = max(float(np.max(np.abs(g[:N]))), float(max(0.0, b['Tlo'] - g[N:].min())), float(max(0.0, g[N:].max() - b['Tup'])))
        o, _ = M.evaluate(X[k], xs)
        r = {'draw': k, 'slsqp_success': bool(res.success), 'slsqp_irf_max': float(irf), 'minT': float(T.min()), 'maxT': float(T.max()),
             'alpha_min': float(al[1:].min()), 'alpha_max': float(al[1:].max()), 'min_MAT': float(MAT.min()), 'mat_floor': float(p.get('mat_floor', np.nan)) if 'mat_floor' in p else None,
             'n_control_bounds_active': act, 'ipopt_warm_status': int(info['status']),
             'ipopt_kkt_inf_rel': float(np.max(np.abs(kkt[free])) / max(1.0, np.max(np.abs(gf[free])))), 'ipopt_constr_viol': viol,
             'scc_2025_reported': S[k]['scc_2025'], 'scc_2025_ipopt_mult': o['scc_2025'], 'diff_pct': 100 * (S[k]['scc_2025'] / o['scc_2025'] - 1)}
        out['draws'].append(r); print(json.dumps({q: (round(v, 8) if isinstance(v, float) else v) for q, v in r.items()}), flush=True)
    # Monte Carlo without the seven
    rec = {o['draw']: o for o in load()}
    keep = [rec[k] for k in sorted(rec) if k not in fb]
    for y in (2025, 2100):
        s = np.array([100 * o[f'share_{y}'] for o in keep])
        out[f'excl_{y}'] = {'n': len(s), 'p5': float(np.percentile(s, 5)), 'p50': float(np.median(s)), 'p95': float(np.percentile(s, 95)), 'neg': float(np.mean(s < 0))}
    print(out['excl_2025'], out['excl_2100'])
    save_json(out, 'fallback_diag.json')
    print('FBD_DONE')
