"""Peak decomposition, second version (paper 2, v0.7). Results: results/p3_decomp2.json (resumable).

For each run (calibration, first access year, variant) two counterfactual temperature paths with removal set to zero:
  econ : control rates and savings of the removal run, removal and its costs dropped (so output, investment and gross
         emissions change slightly); this is the counterfactual of versions 0.5 and 0.6.
  phys : every economic variable of the removal run kept, including removal costs and gross emissions; only the net
         removal flow (effectiveness x removal minus leakage) is taken out of the atmosphere's budget.
In both the carbon-cycle variable is re-solved. The script also records an indicator for the sign of the peak change:
  D = cumulative extra gross emissions from weaker abatement up to the new peak date, relative to no removal,
  Rn = cumulative net removal up to the new peak date, and the timing term P.
Variants: base, cost2 (removal cost floors and slopes x2), stor05 (biological and geological caps x0.5)."""
import os, sys, json
import numpy as np, jax, jax.numpy as jnp
import removal3 as R3, dice_jax as dj

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'results', 'p3_decomp2.json')
SC = {'pub': 9, 'd3': 3, 'd2': 2}; YR = R3.YR[:81]
VAR = {'base': {}, 'cost2': dict(aN0=0.0, aNinf=0.0, bN=200.0, aG0=1000.0, aGinf=200.0, bG=200.0, aM0=400.0, aMinf=160.0, bM=200.0, aG_floor=120.0),
       'stor05': dict(capN=150.0, capG=500.0)}
res = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    json.dump(res, open(OUT, 'w'))


def cf_paths(pr, x, r):
    """Temperature paths of the two counterfactuals."""
    N = pr.N; F = pr.F; qd, rp = pr.qd, pr.rp; z = jnp.zeros(N + 1); q = dict(pr.q, N=N)
    RN, RG, RM = [np.asarray(v) for v in pr.unpack(jnp.asarray(x))[1:4]]
    LN = np.asarray(r['LEAK'])
    net = rp['effN'] * RN + rp['effG'] * RG + rp['effM'] * RM - LN      # net removal flow by period, GtCO2/yr

    def solve_alpha(xx, eE):
        xj = jnp.asarray(xx)

        def resid(a):
            y = xj.at[5 * N + 1:6 * N].set(a)
            rr, al = F['run'](y, qd, rp, eE, z, z, z)
            return dj.irf_residual(q, rr, al)[1:]
        rj = jax.jit(resid); J = jax.jit(jax.jacfwd(resid)); a = jnp.asarray(np.asarray(xx)[5 * N + 1:6 * N])
        for _ in range(40):
            g = rj(a)
            if float(jnp.max(jnp.abs(g))) < 1e-10:
                break
            a = a - jnp.linalg.solve(J(a), g)
        rr, _ = F['run'](xj.at[5 * N + 1:6 * N].set(a), qd, rp, eE, z, z, z)
        return np.asarray(rr['TATM'])[1:], float(jnp.max(jnp.abs(rj(a)))), np.asarray(rr['ECO2'])[1:]
    xe = np.array(x, float); xe[N:4 * N] = 0.0
    Te, ee, Ee = solve_alpha(xe, z)                                   # econ counterfactual
    Tp, ep, Ep = solve_alpha(np.array(x, float), jnp.asarray(net))    # phys counterfactual: add net removal back
    return Te, Tp, max(ee, ep), net[1:], Ep


for v, cal0 in VAR.items():
    for c, sc in SC.items():
        if v != 'base' and c == 'd3':
            continue
        nk = f'{v}_{c}_none'
        if nk not in res:
            pr = R3.Problem(sc, cal=cal0, allow=''); x, info, r, d, dt = pr.solve()
            res[nk] = {'T': np.asarray(r['TATM'])[1:].tolist(), 'E': np.asarray(r['ECO2'])[1:].tolist(), 'status': int(info['status'])}; save()
        T0 = np.array(res[nk]['T']); E0 = np.array(res[nk]['E'])
        years = range(2030, 2160, 10) if v == 'base' else range(2030, 2160, 20)
        for y in years:
            key = f'{v}_{c}_{y}'
            if key in res:
                continue
            pr = R3.Problem(sc, cal=dict(cal0, startN=float(y), startG=float(y), startM=float(y)), allow='NGM')
            x, info, r, d, dt = pr.solve(); s = pr.summary(x, info, r, d)
            T1 = np.asarray(r['TATM'])[1:]; Te, Tp, err, net, Egross = cf_paths(pr, x, r)
            ip = int(np.argmax(T1)); m0 = T0.max()
            D = 5 * float((Egross[:ip + 1] - E0[:ip + 1]).sum()); Rn = 5 * float(net[:ip + 1].sum())
            res[key] = {'status': int(info['status']), 'scc2025': s['scc2025'], 'peakT': float(T1.max()), 'peak_year': int(YR[ip]),
                        'dpeak': float(T1.max() - m0), 'P': float(T0[ip] - m0),
                        'A_econ': float(Te[ip] - T0[ip]), 'R_econ': float(T1[ip] - Te[ip]),
                        'A_phys': float(Tp[ip] - T0[ip]), 'R_phys': float(T1[ip] - Tp[ip]),
                        'D': D, 'Rn': Rn, 'resid': err}
            save()
            q = res[key]
            print(f"{key}: st{q['status']} dpeak {q['dpeak']:+.3f} A {q['A_econ']:+.3f}/{q['A_phys']:+.3f} R {q['R_econ']:+.3f}/{q['R_phys']:+.3f} "
                  f"P {q['P']:+.3f} D {D:.0f} Rn {Rn:.0f}", flush=True)
print('done')
