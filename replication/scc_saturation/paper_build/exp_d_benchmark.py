"""Experiment D: computational benchmark of the exact SCC decomposition against finite differences.

Cost-benefit optimum (scenario 9) on GAMS controls. For every pulse year 2025-2200 (36 periods):
  * exact:  IFT decomposition (JAX), direct + feedback, all periods at once
  * FD-resp: perturbation SCC with alpha re-solved (Richardson, h = 0.1), the quantity the exact method computes
  * FD-fixed: perturbation SCC with alpha held fixed; a different quantity (the direct part only), timed for reference
Accuracy: relative difference exact vs FD-resp, and exact direct part vs FD-fixed. Timing: compile once, then
REPEATS timed calls of each; median, min, max; same process, one core, nothing else running.
Writes results/exp_d_benchmark.json."""
import time, platform
import numpy as np
import jax, jax.numpy as jnp
from common import dm, save_json, scenario_params, gams_controls
import mc_ift as M
import dice_jax as dj

REPEATS = 20
LAST = 37          # periods 2..37 = 2025..2200


def stats(ts):
    return {'median_s': float(np.median(ts)), 'min_s': float(np.min(ts)), 'max_s': float(np.max(ts)), 'n': len(ts)}


if __name__ == '__main__':
    jax.config.update('jax_platform_name', 'cpu')
    _, p = scenario_params(9)
    MIU, S, g = gams_controls(9, p)
    a, _ = dm.solve_alpha(p, MIU, S)
    qd = M.qdyn(p); MIUj, Sj = jnp.asarray(MIU), jnp.asarray(S); a1 = jnp.asarray(a[1:]); e0 = jnp.zeros(81)

    qf = M.full(qd); aj = jnp.asarray(a)
    LAMC = jax.jit(lambda m, s, al: dj.marginal_utility_C(qf, dj.trajectory(qf, m, s, al)))

    def exact():
        # the whole calculation: Jacobians and gradients (JAX, compiled), the linear solve, and the marginal
        # utility of consumption for the conversion to dollars (compiled trajectory pass)
        Ja = np.asarray(M.JA(a1, e0, MIUj, Sj, qd)); Je = np.asarray(M.JE(a1, e0, MIUj, Sj, qd))
        gUa = np.asarray(M.GUA(a1, e0, MIUj, Sj, qd)); gUE = np.asarray(M.GUE(a1, e0, MIUj, Sj, qd))
        dadE = -np.linalg.solve(Ja, Je)
        lamC = np.asarray(LAMC(MIUj, Sj, aj))
        return -1000 * gUE / lamC, -1000 * (gUa @ dadE) / lamC

    out = {'machine': {'cpu': platform.processor() or platform.machine(), 'python': platform.python_version(),
                       'jax': jax.__version__, 'threads': 'default'}, 'repeats': REPEATS}
    t0 = time.perf_counter(); d, f = exact(); out['exact_compile_and_first_call_s'] = time.perf_counter() - t0
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter(); exact(); ts.append(time.perf_counter() - t0)
    out['exact_all_periods'] = stats(ts)

    t0 = time.perf_counter(); fr = dm.compute_SCC(p, MIU, S, a, alpha_response=True, last_period=LAST); out['fd_resp_first_call_s'] = time.perf_counter() - t0
    ts = []
    for _ in range(5):
        t0 = time.perf_counter(); fr = dm.compute_SCC(p, MIU, S, a, alpha_response=True, last_period=LAST); ts.append(time.perf_counter() - t0)
    out['fd_responsive_2025_2200'] = stats(ts)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter(); ff = dm.compute_SCC(p, MIU, S, a, alpha_response=False, last_period=LAST); ts.append(time.perf_counter() - t0)
    out['fd_fixed_2025_2200'] = stats(ts)

    fr = np.asarray(fr); ff = np.asarray(ff)
    k = np.arange(1, LAST)                       # 0-based index into d, f: period 2 (2025) is index 1
    tot = d[k] + f[k]
    frv = fr[2:LAST + 1] if len(fr) > LAST else fr[k]
    ffv = ff[2:LAST + 1] if len(ff) > LAST else ff[k]
    n = min(len(tot), len(frv), len(ffv))
    rel_tot = np.abs(tot[:n] - frv[:n]) / np.abs(frv[:n]); rel_dir = np.abs(d[k][:n] - ffv[:n]) / np.abs(ffv[:n])
    out['accuracy'] = {'periods_compared': int(n), 'years': [2025, 2020 + 5 * n],
                       'exact_total_vs_fd_responsive_max_rel': float(rel_tot.max()),
                       'exact_total_vs_fd_responsive_2025_rel': float(rel_tot[0]),
                       'exact_direct_vs_fd_fixed_max_rel': float(rel_dir.max()),
                       'scc_2025': {'exact': float(tot[0]), 'fd_responsive': float(frv[0]), 'fd_fixed': float(ffv[0]), 'exact_direct': float(d[1])}}
    out['speedup_vs_fd_responsive'] = out['fd_responsive_2025_2200']['median_s'] / out['exact_all_periods']['median_s']
    save_json(out, 'exp_d_benchmark.json')
    for kk, v in out.items():
        print(kk, v)
