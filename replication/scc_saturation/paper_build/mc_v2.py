"""Monte Carlo, version 2: eight uncertain parameters with distributions taken from the literature.

Changes from mc_fast.py (version 1, five parameters on uniform ranges):
  * adds equilibrium climate sensitivity (ECS), the elasticity of marginal utility (elasmu) and
    the damage exponent (a3);
  * replaces the uniform ranges for the discounting parameters and land emissions with
    distributions taken from published sources, centred so that the DICE-2023 calibration
    can be located inside each distribution;
  * drops the GAMS stability bound TATM >= 0.5 for every draw (in version 1 it bound in 253 of
    1000 draws and those draws had to be rerun), so the IFT split is exact for every draw.

Distributions (500-point Latin hypercube on the unit cube, then inverse CDFs; seed 2026):
  rho     pure rate of time preference: lognormal with median 0.5% and mean 1.1%, the median and
          mean of the 180 answers in the expert survey of Drupp et al. (2018, Table 1);
          truncated to [0.01%, 8%] (the survey's range is 0-8%). DICE-2023: 0.1%.
  elasmu  elasticity of marginal utility: lognormal with median 1.0 and mean 1.35 (Drupp et al.
          2018, 173 answers); truncated to [0.3, 1.5]. DICE-2023: 0.95. The upper limit is
          set by the DICE-2023 precautionary discount term -0.5 siggc1^2 h eta^2, which grows with
          eta^2 and above eta = 1.5 makes long-run discount rates fall towards zero, so the SCC rises
          with eta (eta_sweep.py). That region is reported by the sweep, not sampled here.
  ECS     equilibrium climate sensitivity: lognormal with median 3.0 C and 5-95% range about
          1.9-4.7 C, close to the IPCC AR6 very likely range of 2-5 C (best estimate 3 C).
          Applied by scaling teq1 and teq2 together (ECS = fco22x (teq1 + teq2)). DICE-2023: 3.0.
  psi2    damage coefficient at the reference exponent 2: lognormal with median at the DICE value
          0.003467 and a factor of 2 either side as the 5-95% range (assumption; the spread of
          published damage estimates is at least this wide).
  a3      damage exponent: uniform [1.5, 3.0]. The coefficient is rescaled so that damages at 3 C
          equal psi2 * 3^2 whatever the exponent: a2 = psi2 * 3^(2 - a3). The exponent therefore
          changes the curvature of damages, not their level at 3 C. DICE-2023: 2.
  irC     carbon-cycle sensitivity of the 100-year IRF to cumulative uptake: uniform [0.010, 0.030]
          yr/GtC (version 1 range; about +-50% around the FaIR v1 / DICE value 0.019).
  irT     temperature sensitivity of the IRF: uniform [2.5, 6.0] yr/K (version 1 range; DICE 4.165).
  eland0  land-use CO2 in 2020: normal with mean at the DICE value 5.9 GtCO2/yr and standard
          deviation 2.6 GtCO2/yr, the uncertainty the Global Carbon Budget 2023 gives for
          2013-2022 (4.7 +- 2.6 GtCO2/yr); truncated to [0.5, 12].
Each draw: SLSQP optimum of the cost-benefit run (GAMS opt settings without the 0.5 C floor),
then the exact IFT split of the SCC (mc_fast.evaluate).
Usage: python mc_v2.py <first> <last>  -> results/mc2/v2_<first>_<last>.json
"""
import os, sys
import numpy as np
from scipy.stats import qmc, norm, lognorm, truncnorm
from scipy.optimize import brentq
from common import dm, save_json, load_json, gams_session, RES, GAMS_ALPHA
import mc_ift as M
import mc_fast as F

N = 81
KEYS = ['rho', 'elasmu', 'ecs', 'psi2', 'a3', 'irC', 'irT', 'eland0']
DICE = {'rho': 0.001, 'elasmu': 0.95, 'ecs': 3.93 * (0.324 + 0.44), 'psi2': 0.003467, 'a3': 2.0,
        'irC': 0.019, 'irT': 4.165, 'eland0': 5.9}


def _lognorm_median_mean(median, mean):
    s = np.sqrt(2 * np.log(mean / median))
    return lognorm(s=s, scale=median)


def _trunc_ppf(dist, u, lo, hi):
    a, b = dist.cdf(lo), dist.cdf(hi)
    return dist.ppf(a + u * (b - a))


DIST = {
    'rho': lambda u: _trunc_ppf(_lognorm_median_mean(0.005, 0.011), u, 0.0001, 0.08),
    'elasmu': lambda u: _trunc_ppf(_lognorm_median_mean(1.0, 1.35), u, 0.3, 1.5),
    'ecs': lambda u: lognorm(s=0.28, scale=3.0).ppf(u),
    'psi2': lambda u: lognorm(s=np.log(2) / 1.645, scale=0.003467).ppf(u),
    'a3': lambda u: 1.5 + 1.5 * u,
    'irC': lambda u: 0.010 + 0.020 * u,
    'irT': lambda u: 2.5 + 3.5 * u,
    'eland0': lambda u: _trunc_ppf(norm(5.9, 2.6), u, 0.5, 12.0),
}
CDF = {   # percentile of the DICE value in each distribution (reported in the paper)
    'rho': lambda v: (lambda d: (d.cdf(v) - d.cdf(0.0001)) / (d.cdf(0.08) - d.cdf(0.0001)))(_lognorm_median_mean(0.005, 0.011)),
    'elasmu': lambda v: (lambda d: (d.cdf(v) - d.cdf(0.3)) / (d.cdf(1.5) - d.cdf(0.3)))(_lognorm_median_mean(1.0, 1.35)),
    'ecs': lambda v: lognorm(s=0.28, scale=3.0).cdf(v),
    'psi2': lambda v: lognorm(s=np.log(2) / 1.645, scale=0.003467).cdf(v),
    'a3': lambda v: (v - 1.5) / 1.5, 'irC': lambda v: (v - 0.010) / 0.020, 'irT': lambda v: (v - 2.5) / 3.5,
    'eland0': lambda v: (lambda d: (d.cdf(v) - d.cdf(0.5)) / (d.cdf(12) - d.cdf(0.5)))(norm(5.9, 2.6)),
}


def design(n=500, seed=2026):
    U = qmc.LatinHypercube(d=len(KEYS), seed=seed).random(n=n)
    return np.column_stack([DIST[k](U[:, j]) for j, k in enumerate(KEYS)])


def draw_params(row):
    """Cost-benefit optimal params (GAMS opt settings, no 0.5 C floor) with one draw applied."""
    d = dict(zip(KEYS, row))
    p, b = gams_session(9)
    b['Tlo'] = 0.01
    # alpha floor of the GAMS program (0.1), not the 0.2 default of the Numba model: the IPOPT bounds already use
    # 0.1, and the forward-model clamp and SLSQP bounds must match them (v5 fix). PYDICE_ALPHA_LO overrides.
    p['AlphaLowerBound'] = float(os.environ.get('PYDICE_ALPHA_LO', GAMS_ALPHA[0]))
    t = np.arange(1, N + 1)
    p['prstp'] = d['rho']; p['elasmu'] = d['elasmu']
    scale = d['ecs'] / DICE['ecs']
    p['teq1'] = 0.324 * scale; p['teq2'] = 0.44 * scale
    p['a3'] = d['a3']; p['a2base'] = d['psi2'] * 3.0 ** (2.0 - d['a3'])
    p['irC'] = d['irC']; p['irT'] = d['irT']; p['eland0'] = d['eland0']
    p['eland'][1:] = d['eland0'] * (1 - p['deland']) ** (t - 1)
    p['rartp'] = np.exp(p['prstp'] + p['betaclim'] * p['pi']) - 1
    p['rprecaut'][1:] = -0.5 * p['varpcc'][1:] * p['elasmu'] ** 2
    p['RR1'][1:] = 1 / ((1 + p['rartp']) ** (p['tstep'] * (t - 1)))
    p['RR'][1:] = p['RR1'][1:] * (1 + p['rprecaut'][1:]) ** (-p['tstep'] * (t - 1))
    lhs = p['irf0'] + p['irC'] * (p['CumEmiss0'] - (p['mat0'] - p['mateq'])) + p['irT'] * p['tatm0']
    p['a0'] = brentq(lambda a: dm.irf_rhs(p, a) - lhs, 0.05, 50.0, xtol=1e-14)
    return p, b


# mc_fast.evaluate calls M.draw_params(row) and labels params with M.BOUNDS
M.draw_params = draw_params
M.BOUNDS = {k: None for k in KEYS}


if __name__ == '__main__':
    a, z = int(sys.argv[1]), int(sys.argv[2])
    X = design()
    D = os.environ.get('MC_DIR', 'mc2')
    os.makedirs(os.path.join(RES, D), exist_ok=True)
    from ipopt_model import Problem
    xg = Problem(9).start_from_gams()
    name = f'{D}/v2_{a}_{z}.json'
    out = [o for o in load_json(name) if 'error' not in o] if os.path.exists(os.path.join(RES, name)) else []
    done = {o['draw'] for o in out}
    for k in range(a, z):
        if k in done:
            continue
        try:
            o, _ = F.evaluate(X[k], xg)
        except Exception as e:
            o = {'params': dict(zip(KEYS, map(float, X[k]))), 'error': repr(e)}
        o['draw'] = k; out.append(o)
        if k % 10 == 0 or k == z - 1:
            save_json(out, name)
        print(k, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in o.items()
                  if kk in ('success', 'scc_2025', 'share_2025', 'share_2100', 'elapsed_s', 'error')}, flush=True)
    save_json(out, name)
