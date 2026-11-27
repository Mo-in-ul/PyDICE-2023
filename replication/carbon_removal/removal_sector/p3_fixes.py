"""Essential fixes for report v0.5 (referee report on v0.4). Results: results/p3_fixes.json (resumable).
A. First-order-condition audit of geological removal on the central runs (SCC, marginal cost, value of space, residual).
B. Late-access threshold: finer grid of first-access years, actual first deployment year, and a decomposition of the
   change in temperature into (i) the effect of the changed abatement path with removal set to zero and (ii) the effect
   of the removal itself. The counterfactual keeps control rates and savings and re-solves the carbon-cycle state.
C. Storage-value definitions: permanent relaxation from date t (the reported V), one-period relaxation, and the cap
   derivative (finite difference), for the central published run.
D. BECCS land share, finite mineral resource, and the horizon check (2420 against 2620)."""
import os, sys, json, copy
import numpy as np, jax, jax.numpy as jnp
import removal3 as R3, dice_jax as dj
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'results', 'p3_fixes.json')
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
YR = R3.YR[:81]; iy = lambda y: int(np.where(YR == y)[0][0])
SC = {'pub': 9, 'd3': 3, 'd2': 2}


def save():
    json.dump(res, open(OUT, 'w'))


def solve(sc, cal=None, allow='NGM', session=None):
    pr = R3.Problem(sc, cal=cal or {}, allow=allow, session=session)
    x, info, r, d, dt = pr.solve()
    return pr, x, pr.summary(x, info, r, d)


def simulate_without_removal(pr, x):
    """Same control rates and savings, removal set to zero; carbon-cycle alpha re-solved (Newton) to satisfy IRF100."""
    N = pr.N; F = pr.F; qd, rp = pr.qd, pr.rp; z = jnp.zeros(N + 1); q = dict(qd, N=N)
    xx = np.array(x, float); xx[N:4 * N] = 0.0
    if pr.nb == 7:
        xx[6 * N:7 * N] = 0.0
    xj = jnp.asarray(xx)

    def resid(a):
        y = xj.at[5 * N + 1:6 * N].set(a)
        r, al = F['run'](y, qd, rp, z, z, z, z)
        return dj.irf_residual(q, r, al)[1:]
    rj = jax.jit(resid); J = jax.jit(jax.jacfwd(resid))
    a = jnp.asarray(xx[5 * N + 1:6 * N])
    for _ in range(40):
        g = rj(a)
        if float(jnp.max(jnp.abs(g))) < 1e-10:
            break
        a = a - jnp.linalg.solve(J(a), g)
    y = xj.at[5 * N + 1:6 * N].set(a)
    r, _ = F['run'](y, qd, rp, z, z, z, z)
    return np.asarray(r['TATM'])[1:], float(jnp.max(jnp.abs(rj(a))))


# ---------- A. FOC audit ----------
if 'foc' not in res:
    res['foc'] = {}
    for c, sc in SC.items():
        pr, x, s = solve(sc)
        P = s['paths']; c_ = pr.cal; rows = []
        for y in (2075, 2090, 2100, 2110, 2120):
            i = iy(y)
            Rbar = c_['RG_inf'] / (1 + np.exp(-(y - c_['RG_mid']) / c_['RG_tau']))
            aG = c_['aGinf'] + (c_['aG0'] - c_['aGinf']) * np.exp(-(y - 2025) / c_['taucG'])
            RG = P['RG'][i]; mc = aG + c_['bG'] * (RG / Rbar) ** 2
            if RG < 1e-3:
                continue
            Vn = P['vG_pv'][i + 1] * (P['vG'][i] / P['vG_pv'][i])
            rows.append({'year': y, 'scc': P['scc'][i], 'mc': mc, 'V_next_in_t': Vn, 'V_t': P['vG'][i], 'resid': mc - (P['scc'][i] - Vn),
                         'cap_use': RG / Rbar})
        res['foc'][c] = rows
        print('foc', c, [(r['year'], round(r['resid'], 3)) for r in rows], flush=True)
    save()

# ---------- B. late-access threshold and decomposition ----------
res.setdefault('late', {})
for c, sc in SC.items():
    if c not in res.setdefault('none_T', {}):
        _, _, s0 = solve(sc, allow='')
        res['none_T'][c] = {'T': s0['paths']['T'], 'peakT': s0['peakT'], 'peakT_year': s0['peakT_year'], 'U': s0['U'], 'lamC2025': s0['lamC2025'],
                            'E': s0['paths']['E']}
        save()
    T0 = np.array(res['none_T'][c]['T']); E0 = np.array(res['none_T'][c]['E'])
    for y in range(2030, 2160, 10):
        key = f'{c}_{y}'
        if key in res['late']:
            continue
        cal = dict(startN=float(y), startG=float(y), startM=float(y))
        pr, x, s = solve(sc, cal)
        P = s['paths']; R = np.array(P['RN']) + np.array(P['RG']) + np.array(P['RM'])
        dep = np.where(R > 1e-3)[0]; first_dep = int(YR[dep[0]]) if len(dep) else None
        Tnr, err = simulate_without_removal(pr, x)
        T1 = np.array(P['T']); E1 = np.array(P['E'])
        ip = int(np.argmax(T1)); i0 = int(np.argmax(T0))
        kdep = dep[0] if len(dep) else len(YR)
        res['late'][key] = {'start': y, 'first_deploy': first_dep, 'peakT': float(T1.max()), 'peakT_year': int(YR[ip]), 'scc2025': s['scc2025'],
                            'U': s['U'], 'dpeak': float(T1.max() - T0.max()),
                            'dT_total_at_peak': float(T1[ip] - T0[ip]), 'dT_abatement_at_peak': float(Tnr[ip] - T0[ip]),
                            'dT_removal_at_peak': float(T1[ip] - Tnr[ip]),
                            'dT_abatement_at_nopeak': float(Tnr[i0] - T0[i0]), 'dT_removal_at_nopeak': float(T1[i0] - Tnr[i0]),
                            'extra_emissions_before_deploy': float(5 * (E1[:kdep] - E0[:kdep]).sum()),
                            'removal_before_nopeak': float(5 * R[:i0 + 1].sum()), 'removal_before_peak': float(5 * R[:ip + 1].sum()),
                            'sim_resid': err, 'T_noremoval_sim': Tnr.tolist(), 'T': T1.tolist(), 'status': s['status']}
        save()
        r = res['late'][key]
        print(f"late {key}: dep {first_dep} peak {r['peakT']:.3f}@{r['peakT_year']} dpeak {r['dpeak']:+.3f} | at peak: abat {r['dT_abatement_at_peak']:+.3f} "
              f"rem {r['dT_removal_at_peak']:+.3f} | extraE {r['extra_emissions_before_deploy']:.0f} | resid {err:.1e}", flush=True)

# ---------- C. storage-value definitions ----------
if 'vdef' not in res:
    pr, x, s = solve(9)
    P = s['paths']; out = {'vN_perm_pv': P['vN_pv'][:30], 'vG_perm_pv': P['vG_pv'][:30]}
    out['vN_oneperiod_pv'] = [P['vN_pv'][i] - P['vN_pv'][i + 1] for i in range(29)]
    out['vG_oneperiod_pv'] = [P['vG_pv'][i] - P['vG_pv'][i + 1] for i in range(29)]
    _, _, sN = solve(9, dict(capN=306.0)); _, _, sG = solve(9, dict(capG=1020.0))
    conv = s['lamC2025'] / 5
    out['capN_fd'] = 1000 * (sN['U'] - s['U']) / conv / 6.0
    out['capG_fd'] = 1000 * (sG['U'] - s['U']) / conv / 20.0
    out['vN2025'] = s['vN2025']; out['vG2025'] = s['vG2025']
    res['vdef'] = out; save()
    print('vdef cap FD', out['capN_fd'], out['capG_fd'], 'perm 2025', out['vN2025'], out['vG2025'], flush=True)

# ---------- D. BECCS share, finite mineral, horizon ----------
res.setdefault('misc', {})
jobs = []
for c in ('pub', 'd2'):
    jobs.append((f'{c}_land0.125', SC[c], dict(land_c=0.125), 'NGM', None))
    for cm in (300.0, 600.0):
        jobs.append((f'{c}_capM{cm:g}', SC[c], dict(capM=cm), 'NGM', None))
        jobs.append((f'{c}_capM{cm:g}_from2110', SC[c], dict(capM=cm, startN=2110.0, startG=2110.0, startM=2110.0), 'NGM', None))
for name, sc, cal, allow, ses in jobs:
    if name in res['misc']:
        continue
    _, _, s = solve(sc, cal, allow)
    res['misc'][name] = {k: s[k] for k in ('scc2025', 'peakT', 'peakT_year', 'T2100', 'T2200', 'cumN', 'cumG', 'cumM', 'cumN_2200', 'vN2025', 'vG2025', 'U', 'status')}
    save(); print('misc', name, s['status'], round(s['scc2025'], 3), round(s['peakT'], 3), round(s['cumM']), flush=True)
sys.path.insert(0, os.path.join(R3.PB, 'stage2'))
import s2_core as S2
for N in (81, 121):
    for lab, cal, allow in (('none', {}, ''), ('NGM', {}, 'NGM'), ('NGM_capM600', dict(capM=600.0), 'NGM'),
                            ('NGM_from2110', dict(startN=2110.0, startG=2110.0, startM=2110.0), 'NGM')):
        name = f'hz{N}_{lab}'
        if name in res['misc']:
            continue
        p2 = S2.problem(N, 0.95); ses = (copy.deepcopy(p2.p), copy.deepcopy(p2.b))
        _, _, s = solve(9, cal, allow, session=ses)
        res['misc'][name] = {k: s[k] for k in ('scc2025', 'peakT', 'peakT_year', 'T2100', 'cumM', 'vN2025', 'vG2025', 'status')}
        save(); print('misc', name, s['status'], round(s['scc2025'], 3), round(s['peakT'], 3), flush=True)
print('done')
