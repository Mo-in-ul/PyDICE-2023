"""Peak indicator: out-of-sample test, climate-module coefficients, fixed non-CO2 mitigation, and peak interpolation
(paper 2, v0.8). Results: results/p3_decomp3.json (resumable)."""
import os, json
import numpy as np, jax, jax.numpy as jnp
import removal3 as R3, dice_jax as dj
from p3_decomp2 import cf_paths, SC, YR

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'results', 'p3_decomp3.json')
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
DC = json.load(open(os.path.join(HERE, 'results', 'p3_decomp2.json')))
FX = json.load(open(os.path.join(HERE, 'results', 'p3_fixes.json')))


def save():
    json.dump(res, open(OUT, 'w'))


def temp_with_eE(pr, x, eE):
    """Temperature path of plan x with an emissions perturbation eE (GtCO2/yr by period), carbon cycle re-solved."""
    N = pr.N; F = pr.F; qd, rp = pr.qd, pr.rp; z = jnp.zeros(N + 1); q = dict(pr.q, N=N); xj = jnp.asarray(x)

    def resid(a):
        rr, al = F['run'](xj.at[5 * N + 1:6 * N].set(a), qd, rp, jnp.asarray(eE), z, z, z)
        return dj.irf_residual(q, rr, al)[1:]
    rj = jax.jit(resid); J = jax.jit(jax.jacfwd(resid)); a = jnp.asarray(np.asarray(x)[5 * N + 1:6 * N])
    for _ in range(40):
        g = rj(a)
        if float(jnp.max(jnp.abs(g))) < 1e-10:
            break
        a = a - jnp.linalg.solve(J(a), g)
    rr, _ = F['run'](xj.at[5 * N + 1:6 * N].set(a), qd, rp, jnp.asarray(eE), z, z, z)
    return np.asarray(rr['TATM'])[1:]


# 1. climate-module coefficients: temperature response at a date to emissions spread over the preceding decades
if 'pulse' not in res:
    res['pulse'] = {}
    for c, sc in (('pub', 9), ('d2', 2)):
        pr = R3.Problem(sc, allow=''); x, *_ = pr.solve(); T0 = temp_with_eE(pr, x, np.zeros(pr.N + 1))
        for (a, b, y) in ((2025, 2090, 2090), (2025, 2060, 2060), (2070, 2090, 2090), (2050, 2060, 2060)):
            e = np.zeros(pr.N + 1); idx = np.where((YR >= a) & (YR <= b))[0] + 1; e[idx] = 1.0
            T1 = temp_with_eE(pr, x, e); i = int(np.where(YR == y)[0][0])
            res['pulse'][f'{c}_{a}_{b}'] = 1000 * float(T1[i] - T0[i]) / (5 * len(idx))
        print('pulse', c, {k: round(v, 3) for k, v in res['pulse'].items() if k.startswith(c)}, flush=True)
    save()

# 2. fixed non-CO2 mitigation: decomposition runs on a 20-year grid
for c, sc in (('pub', 9), ('d2', 2)):
    nk = f'nc_{c}_none'
    if nk not in res:
        pr0 = R3.Problem(sc, allow=''); x0, *_ = pr0.solve()
        mu0 = np.array(pr0.unpack(jnp.asarray(x0))[0]); mu0[1] = pr0.p['miu1']
        prn = R3.Problem(sc, cal=dict(munc=mu0.tolist()), allow=''); xn, i_, rn, d_, _ = prn.solve()
        res[nk] = {'munc': mu0.tolist(), 'T': np.asarray(rn['TATM'])[1:].tolist(), 'E': np.asarray(rn['ECO2'])[1:].tolist()}; save()
    mu0 = res[nk]['munc']; T0 = np.array(res[nk]['T']); E0 = np.array(res[nk]['E'])
    for y in range(2030, 2160, 20):
        key = f'nc_{c}_{y}'
        if key in res:
            continue
        pr = R3.Problem(sc, cal=dict(munc=mu0, startN=float(y), startG=float(y), startM=float(y)), allow='NGM')
        x, info, r, d, dt = pr.solve(); s = pr.summary(x, info, r, d)
        T1 = np.asarray(r['TATM'])[1:]; Te, Tp, err, net, Egross = cf_paths(pr, x, r); ip = int(np.argmax(T1)); m0 = T0.max()
        res[key] = {'status': int(info['status']), 'dpeak': float(T1.max() - m0), 'P': float(T0[ip] - m0), 'A_phys': float(Tp[ip] - T0[ip]),
                    'R_phys': float(T1[ip] - Tp[ip]), 'D': 5 * float((Egross[:ip + 1] - E0[:ip + 1]).sum()), 'Rn': 5 * float(net[:ip + 1].sum()),
                    'scc2025': s['scc2025']}
        save(); q = res[key]
        print(key, {a: round(b, 3) for a, b in q.items()}, flush=True)

# 3. peak interpolation on the 5-year grid (39 base runs of p3_fixes)
def peak_interp(T):
    i = int(np.argmax(T))
    if i == 0 or i == len(T) - 1:
        return float(T[i]), float(YR[i])
    a, b, cc = T[i - 1], T[i], T[i + 1]; den = a - 2 * b + cc
    if den == 0:
        return float(b), float(YR[i])
    d = 0.5 * (a - cc) / den
    return float(b - 0.25 * (a - cc) * d), float(YR[i] + 5 * d)


out = {}
for cn in ('pub', 'd3', 'd2'):
    T0 = np.array(FX['none_T'][cn]['T']); p0, y0 = peak_interp(T0)
    for y in range(2030, 2160, 10):
        T1 = np.array(FX['late'][f'{cn}_{y}']['T']); p1, y1 = peak_interp(T1)
        out[f'{cn}_{y}'] = {'dpeak_grid': float(T1.max() - T0.max()), 'dpeak_interp': p1 - p0, 'peak_year_interp': y1}
res['interp'] = out; save()
print('max |interp - grid|', max(abs(v['dpeak_interp'] - v['dpeak_grid']) for v in out.values()))
print('done')
