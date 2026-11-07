"""Wall-clock timing of every computation reported in the paper (single core, nothing else running).

Each item is timed after a warm-up call, so Numba and JAX compilation is excluded and reported
separately. Median of `REPEATS` runs unless stated. Cost-benefit optimal run (scenario 9).
Writes results/timing.json.
"""
import time, platform, os
import numpy as np
from common import dm, save_json, scenario_params, gams_controls, gams_session

REPEATS = 5


def timeit(f, repeats=REPEATS):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(np.min(ts)), float(np.max(ts))


if __name__ == '__main__':
    out = {'machine': {'cpu': 'Intel Xeon @ 2.80 GHz (cloud VM), 1 core used', 'python': platform.python_version()}}
    _, p = scenario_params(9)
    MIU, S, g = gams_controls(9, p)
    a, _ = dm.solve_alpha(p, MIU, S)

    # 1. forward model, Numba (compile, then a call)
    t0 = time.perf_counter(); dm.diceTrajectory(p, MIU, S, a); out['numba_compile_s'] = time.perf_counter() - t0
    out['forward_numba_s'] = timeit(lambda: dm.diceTrajectory(p, MIU, S, a), 200)
    # 2. alpha from the IRF condition for given controls
    out['solve_alpha_s'] = timeit(lambda: dm.solve_alpha(p, MIU, S))
    # 3. SLSQP optimization (as run_faithful.py), cold start
    import subprocess, json
    def slsqp():
        subprocess.run(['python3', 'run_faithful.py', '9'], capture_output=True)
    t = []
    for _ in range(3):
        t0 = time.perf_counter(); slsqp(); t.append(time.perf_counter() - t0)
    js = json.load(open(os.path.join('results', 'faith_scen9.json')))
    out['slsqp_optimize_total_process_s'] = float(np.median(t))
    out['slsqp_optimize_s'] = js['elapsed_s']
    out['slsqp_iterations'] = js['attempts'][-1]['nit']
    # 4. SCC by perturbation over the full horizon (80 periods, two pulse sizes each)
    out['scc_fd_fixed_all_s'] = timeit(lambda: dm.compute_SCC(p, MIU, S, a, alpha_response=False), 3)
    out['scc_fd_responsive_all_s'] = timeit(lambda: dm.compute_SCC(p, MIU, S, a, alpha_response=True), 3)
    out['scc_fd_responsive_one_year_s'] = timeit(lambda: dm.compute_SCC(p, MIU, S, a, alpha_response=True, last_period=2), 5)
    # 5. IFT decomposition (JAX, compiled once)
    import mc_ift as M, jax.numpy as jnp, dice_jax as dj
    qd = M.qdyn(p); MIUj, Sj = jnp.asarray(MIU), jnp.asarray(S); a1 = jnp.asarray(a[1:]); e0 = jnp.zeros(81)
    def ift():
        Ja = np.asarray(M.JA(a1, e0, MIUj, Sj, qd)); Je = np.asarray(M.JE(a1, e0, MIUj, Sj, qd))
        gUa = np.asarray(M.GUA(a1, e0, MIUj, Sj, qd)); gUE = np.asarray(M.GUE(a1, e0, MIUj, Sj, qd))
        return -np.linalg.solve(Ja, Je), gUa, gUE
    t0 = time.perf_counter(); ift(); out['ift_compile_s'] = time.perf_counter() - t0
    out['ift_all_periods_s'] = timeit(ift)
    # 6. IPOPT optimization + multiplier SCC (ipopt_model.py; includes building the JAX functions)
    from ipopt_model import solve
    t0 = time.perf_counter(); r = solve(9, warm='default'); out['ipopt_first_call_s'] = time.perf_counter() - t0
    out['ipopt_solve_only_cold_s'] = r['elapsed_s']; out['ipopt_iterations_cold'] = None
    r2 = solve(9, warm='gams'); out['ipopt_solve_only_warm_s'] = r2['elapsed_s']
    # 7. Monte Carlo per draw (uncontended, mc_fast.evaluate: SLSQP + IFT)
    import mc_fast as F
    keys, X = M.design()
    from ipopt_model import Problem
    xg = Problem(9).start_from_gams()
    F.evaluate(X[1], xg)
    ts = []
    for k in (11, 22, 33, 44, 55):
        t0 = time.perf_counter(); F.evaluate(X[k], xg); ts.append(time.perf_counter() - t0)
    out['mc_draw_s'] = (float(np.median(ts)), float(np.min(ts)), float(np.max(ts)))
    # 8. Gymnasium step
    from gym_replay import replay
    replay(9, 'contemporaneous'); o = replay(9, 'contemporaneous'); out['gym_ms_per_step'] = o['ms_per_step']
    save_json(out, 'timing.json')
    for k, v in out.items():
        print(k, v)
