"""V2: independent SCC by re-optimization (welfare perturbation) and by a fixed-policy pulse.
Usage: python v2_bruteforce.py [tlo]   (tlo: none | 0.01 | 0.5)"""
import sys, numpy as np, bv_core as B, bv_io as IO
TLO = {'none': None, '0.01': 0.01, '0.5': 0.5}[sys.argv[1] if len(sys.argv) > 1 else 'none']
ETAS = [float(a) for a in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0.5, 0.95, 1.28, 1.5, 1.75, 2.0]
HE, HC = 0.1, 0.006
PERIODS = {2025: 2, 2050: 7}
name = f"v2_bf_tlo{sys.argv[1] if len(sys.argv) > 1 else 'none'}.json"
out = IO.load(name) or []
done = {o['eta'] for o in out}

def fixed_policy(pr, x):
    """Freeze MIU and S at x; alpha stays free and is pinned by the IRF equality; no T bound."""
    xl0, xu0, gl0, gu0 = pr.bounds()
    N = pr.N
    xl, xu = xl0.copy(), xu0.copy()
    xl[:2 * N] = xu[:2 * N] = x[:2 * N]
    gl, gu = gl0.copy(), gu0.copy(); gl[N:] = -10.0; gu[N:] = 20.0
    pr.bounds = lambda: (xl, xu, gl, gu)

def V(pr, x0, warm, which, t, h):
    e = np.zeros(pr.N + 1); e[t] = h
    kw = {'eps_E': e} if which == 'E' else {'eps_C': e}
    x, info, n, dt = B.solve(pr, x0, warm=warm, tol=1e-12, **kw)
    return -float(info['obj_val']), int(info['status']), n

def derivs(mk, x0, warm, t):
    res = {}
    for which, h in (('E', HE), ('C', HC)):
        D = {}; st = []
        for hh in (h, h / 2):
            pr = mk(); vp, s1, _ = V(pr, x0, warm, which, t, +hh)
            pr = mk(); vm, s2, _ = V(pr, x0, warm, which, t, -hh)
            D[hh] = (vp - vm) / (2 * hh); st += [s1, s2]
        res[which] = {'D_h': D[h], 'D_h2': D[h / 2], 'rich': (4 * D[h / 2] - D[h]) / 3, 'status': st}
    scc_h = -1000 * res['E']['D_h'] / res['C']['D_h']
    scc_h2 = -1000 * res['E']['D_h2'] / res['C']['D_h2']
    scc_r = -1000 * res['E']['rich'] / res['C']['rich']
    return {'scc_h': scc_h, 'scc_h2': scc_h2, 'scc_rich': scc_r, 'parts': res}

for eta in ETAS:
    if eta in done: continue
    pr = B.problem(81, eta, 'dice', TLO)
    x, info, n, _ = B.solve(pr, pr.start_from_gams(), tol=1e-12, max_iter=20000)
    d = B.diagnostics(pr, x, info)
    warm = {k: info[k] for k in ('mult_g', 'mult_x_L', 'mult_x_U')}
    rec = {'eta': eta, 'tlo': TLO, 'status': d['status'], 'kkt': d['kkt_stationarity_inf'], 'minT': d['minT'],
           'Tlo_binding_years': d['Tlo_binding_years'], 'U': d['U']}
    for yr, t in PERIODS.items():
        mult = float(d['scc'][t])
        ro = derivs(lambda: B.problem(81, eta, 'dice', TLO), x, warm, t)
        def mkfix():
            p = B.problem(81, eta, 'dice', TLO); fixed_policy(p, x); return p
        fp = derivs(mkfix, x, warm, t)
        # analytic multipliers for comparison of the pieces
        rec[str(yr)] = {'multiplier': mult, 'dVdE_mult': float(d['dVdE'][t]), 'dVdC_mult': float(d['dVdC'][t]),
                        'reopt': ro, 'fixed_policy': fp,
                        'gap_reopt_pct': 100 * (ro['scc_rich'] / mult - 1),
                        'gap_fixed_pct': 100 * (fp['scc_rich'] / mult - 1),
                        'h_consistency_reopt_pct': 100 * (ro['scc_h2'] / ro['scc_h'] - 1)}
        print(eta, yr, round(mult, 4), round(ro['scc_h'], 4), round(ro['scc_h2'], 4), round(ro['scc_rich'], 4),
              '| fixed', round(fp['scc_rich'], 4), flush=True)
    out.append(rec); IO.save(out, name)
