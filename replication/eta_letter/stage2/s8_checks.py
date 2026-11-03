"""Final interpretation checks after review 6 (no temperature floor, N = 81, sigma = 1%, cap 235 y, default rho_e).
A. Fine phi grid (1.15-1.45, step 0.0125): zero crossing of dSCC/dphi with two difference steps.
B. Fixed-policy SCC curve: policy held at the phi = 0.95 optimum (and at the phi = 1.3 optimum) while phi varies.
C. Interaction of the utility and precautionary channels at fixed policy (x at 0.95 and at 1.3).
D. Re-optimization term at phi in {1.2, 1.3, 1.4} split into (i) weights moving with the consumption path and
   (ii) changes in the marginal losses dC/dE; plus descriptive policy changes.
The fixed-policy SCC uses SCC_fp = -1000 sum_s omega_s dC_s with omega_s = cpc_s^-phi_u RR_s(phi_p) / (same at t = 2)."""
import numpy as np, s2_core as S, bv_core as B, json
from s7_split import dCdE
RE = float(np.exp(0.001 + 0.025) - 1); SIG = 0.01; N = 81
t = np.arange(1, N + 1); h = 5.0 * (t - 1); m = 5.0 * np.minimum(t - 1, 47)
RR = lambda pp: (1 + RE) ** (-h) * (1 - 0.5 * pp ** 2 * SIG ** 2 * m) ** (-h)
def sfp(pu, pp, cpc, dC):
    lam = cpc ** (-pu) * RR(pp); return float(-1000.0 * np.sum(lam / lam[1] * dC))
sol = {}
def solve(e, x0=None):
    e = round(e, 5)
    if e in sol: return sol[e]
    pr = S.problem(N, e, tlo=None)
    x, info, n, _ = B.solve(pr, pr.start_from_gams() if x0 is None else x0, max_iter=20000)
    scc, _, _ = pr.multiplier_scc(x, info['mult_g']); sol[e] = (float(scc[2]), x, pr, int(info['status'])); return sol[e]
out = {}
# A
grid = np.round(np.arange(1.15, 1.45001, 0.0125), 5); x = None
for e in grid:
    s, x, _, st = solve(e, x)
G = np.array([sol[round(e, 5)][0] for e in grid])
d1 = {round(grid[i], 4): (G[i + 2] - G[i - 2]) / 0.05 for i in range(2, len(grid) - 2)}      # step 0.025
d2 = {round(grid[i], 4): (G[i + 1] - G[i - 1]) / 0.025 for i in range(1, len(grid) - 1)}     # step 0.0125
def zero(d):
    k = sorted(d); v = [d[a] for a in k]
    for i in range(len(k) - 1):
        if v[i] < 0 <= v[i + 1]: return k[i] + (k[i + 1] - k[i]) * (-v[i]) / (v[i + 1] - v[i])
out['A'] = {'grid': grid.tolist(), 'scc': G.tolist(), 'deriv_0.025': d1, 'deriv_0.0125': d2, 'zero_0.025': zero(d1), 'zero_0.0125': zero(d2),
            'status': sorted({sol[round(e, 5)][3] for e in grid})}
print('A zero crossing', out['A']['zero_0.025'], out['A']['zero_0.0125'], flush=True)
# B and C
lossdata = {}
for e0 in (0.95, 1.3):
    s0, x0, pr0, _ = solve(e0, None if e0 == 0.95 else sol[1.3][1] if 1.3 in sol else None)
    dC, cpc = dCdE(pr0, x0); lossdata[e0] = (dC, cpc)
    phis = np.round(np.arange(0.5, 1.8001, 0.025), 3); curve = [sfp(p, p, cpc, dC) for p in phis]
    i = int(np.argmin(curve))
    C = {(pu, pp): sfp(pu, pp, cpc, dC) for pu in (0.95, 1.3, 1.75) for pp in (0.95, 1.3, 1.75)}
    inter = C[(1.3, 1.3)] - C[(1.3, 0.95)] - C[(0.95, 1.3)] + C[(0.95, 0.95)]
    out[f'B_x{e0}'] = {'phi': phis.tolist(), 'scc_fp': curve, 'phi_min': float(phis[i]), 'scc_min': curve[i], 'check_at_e0': sfp(e0, e0, cpc, dC), 'mult_at_e0': s0}
    out[f'C_x{e0}'] = {f'{k[0]}_{k[1]}': v for k, v in C.items()} | {'utility_effect': C[(1.3, 0.95)] - C[(0.95, 0.95)],
                       'prec_effect': C[(0.95, 1.3)] - C[(0.95, 0.95)], 'joint': C[(1.3, 1.3)] - C[(0.95, 0.95)], 'interaction': inter}
    print(f'B policy at {e0}: phi_min {phis[i]:.3f}, check {out[f"B_x{e0}"]["check_at_e0"]:.3f} vs {s0:.3f}', flush=True)
    print(f'C policy at {e0}:', {k: round(v, 2) for k, v in out[f'C_x{e0}'].items() if k in ('utility_effect', 'prec_effect', 'joint', 'interaction')}, flush=True)
# D
D = 0.0125; out['D'] = {}
for e in (1.2, 1.3, 1.4):
    sm, xm, prm, _ = solve(e - D); s0, x0, pr0, _ = solve(e); sp_, xp, prp, _ = solve(e + D)
    dC0, c0 = dCdE(pr0, x0); dCm, cm = dCdE(prm, xm); dCp, cp = dCdE(prp, xp)
    total = (sp_ - sm) / (2 * D)
    direct = (sfp(e + D, e + D, c0, dC0) - sfp(e - D, e - D, c0, dC0)) / (2 * D)
    path_w = (sfp(e + D, e + D, cp, dC0) - sfp(e - D, e - D, cm, dC0)) / (2 * D) - direct      # weights move with consumption path
    losses = total - direct - path_w                                                             # marginal losses change
    rp = lambda pr, x: pr.run(__import__('jax').numpy.asarray(x), pr.z, pr.z)[0]
    r0, rpl, rmi = rp(pr0, x0), rp(prp, xp), rp(prm, xm)
    desc = {'dMIU_2025_2100_avg': float(np.mean(xp[1:17] - xm[1:17]) / (2 * D)), 'dS_2025_2100_avg': float(np.mean(xp[N + 1:N + 17] - xm[N + 1:N + 17]) / (2 * D)),
            'dpeakT': float((np.max(np.asarray(rpl['TATM'])[1:41]) - np.max(np.asarray(rmi['TATM'])[1:41])) / (2 * D)),
            'dloss_sum': float(np.sum(dCp - dCm) / (2 * D))}
    out['D'][str(e)] = {'total': total, 'direct': direct, 'reopt': total - direct, 'reopt_path_weights': path_w, 'reopt_losses': losses} | desc
    print(f'D {e}: total {total:.2f} direct {direct:.2f} reopt {total-direct:.2f} = path-weights {path_w:.2f} + losses {losses:.2f}', desc, flush=True)
json.dump(out, open('results/s8_checks.json', 'w'), indent=1, default=float)
