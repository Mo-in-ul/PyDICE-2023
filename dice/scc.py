import numpy as np
import copy
import time
import warnings
import pandas as pd
from scipy.optimize import minimize, root
from numba import njit

from dice.model import diceTrajectory_numba, DiceFunc, recoverAllVars
from dice.params import LoadParams


@njit(cache=False)
def compute_SCC_numba(MIU, S, alpha, num_periods, tstep, dk, gama, eco2Param, sigma, eland, cost1tot, expcost2,
                      miuup, sLBounds, sUBounds, AlphaLowerBound, Fcoef1, Fcoef2, CO2E_GHGabateB, emshare0,
                      emshare1, emshare2, emshare3, tau0, tau1, tau2, tau3, mateq, fco22x, F_Misc, teq1, teq2,
                      d1, d2, a1, a2base, a3, res00, res10, res20, res30, tbox10, tbox20, CumEmiss0,
                      F_GHGabate2020, k0, tatm0, mat0, a0, L, RR, scale1, scale2, elasmu):

    pulse_zero = np.zeros(num_periods + 1, dtype=np.float64)
    base = diceTrajectory_numba(
        MIU, S, alpha, num_periods, tstep, dk, gama, eco2Param, sigma, eland,
        cost1tot, expcost2, miuup, sLBounds, sUBounds, AlphaLowerBound, Fcoef1,
        Fcoef2, CO2E_GHGabateB, emshare0, emshare1, emshare2, emshare3, tau0, tau1,
        tau2, tau3, mateq, fco22x, F_Misc, teq1, teq2, d1, d2, a1, a2base, a3,
        res00, res10, res20, res30, tbox10, tbox20, CumEmiss0, F_GHGabate2020,
        k0, tatm0, mat0, a0, L, RR, scale1, scale2, elasmu, pulse_zero)

    if np.isnan(base[0]):
        return np.zeros(num_periods + 1)

    C_base = base[1]
    CPC_base = 1000.0 * C_base[1:] / L[1:]
    PERIODU_base = ((CPC_base ** (1.0 - elasmu)) - 1.0) / (1.0 - elasmu) - 1.0
    TOTPERIODU_base = PERIODU_base * L[1:] * RR[1:]
    W0 = np.sum(TOTPERIODU_base)
    lambda_C = (CPC_base ** (-elasmu)) * RR[1:]

    SCC = np.zeros(num_periods + 1)

    pulse_h  = 5e-6
    pulse_h2 = pulse_h / 2

    for i in range(2, num_periods + 1):
        pulse_array_h = np.zeros(num_periods + 1, dtype=np.float64)
        pulse_array_h[i] = pulse_h

        alt_h = diceTrajectory_numba(
            MIU, S, alpha, num_periods, tstep, dk, gama, eco2Param, sigma, eland,
            cost1tot, expcost2, miuup, sLBounds, sUBounds, AlphaLowerBound, Fcoef1,
            Fcoef2, CO2E_GHGabateB, emshare0, emshare1, emshare2, emshare3, tau0, tau1,
            tau2, tau3, mateq, fco22x, F_Misc, teq1, teq2, d1, d2, a1, a2base, a3,
            res00, res10, res20, res30, tbox10, tbox20, CumEmiss0, F_GHGabate2020,
            k0, tatm0, mat0, a0, L, RR, scale1, scale2, elasmu, pulse_array_h)

        if np.isnan(alt_h[0]):
            SCC[i] = 0.0
            continue

        C_alt_h  = alt_h[1]
        CPC_alt_h = 1000.0 * C_alt_h[1:] / L[1:]
        W_h  = np.sum(((CPC_alt_h  ** (1.0 - elasmu)) - 1.0) / (1.0 - elasmu) - 1.0) * L[1:] * RR[1:]
        dW_h = W_h - W0

        pulse_array_h2 = np.zeros(num_periods + 1, dtype=np.float64)
        pulse_array_h2[i] = pulse_h2

        alt_h2 = diceTrajectory_numba(
            MIU, S, alpha, num_periods, tstep, dk, gama, eco2Param, sigma, eland,
            cost1tot, expcost2, miuup, sLBounds, sUBounds, AlphaLowerBound, Fcoef1,
            Fcoef2, CO2E_GHGabateB, emshare0, emshare1, emshare2, emshare3, tau0, tau1,
            tau2, tau3, mateq, fco22x, F_Misc, teq1, teq2, d1, d2, a1, a2base, a3,
            res00, res10, res20, res30, tbox10, tbox20, CumEmiss0, F_GHGabate2020,
            k0, tatm0, mat0, a0, L, RR, scale1, scale2, elasmu, pulse_array_h2)

        if np.isnan(alt_h2[0]):
            SCC[i] = 0.0
            continue

        C_alt_h2  = alt_h2[1]
        CPC_alt_h2 = 1000.0 * C_alt_h2[1:] / L[1:]
        W_h2  = np.sum(((CPC_alt_h2 ** (1.0 - elasmu)) - 1.0) / (1.0 - elasmu) - 1.0) * L[1:] * RR[1:]
        dW_h2 = W_h2 - W0

        if abs(lambda_C[i - 1]) > 1e-12:
            SCC_h  = -(dW_h  / pulse_h)  / lambda_C[i - 1]
            SCC_h2 = -(dW_h2 / pulse_h2) / lambda_C[i - 1]
            SCC[i] = (4.0 * SCC_h2 - SCC_h) / 3.0
        else:
            SCC[i] = 0.0

    SCC[1] = 0.0
    return SCC


def compute_SCC(params, MIU, S, alpha):
    """
    Compute the SCC trajectory for a given policy.

    Parameters
    ----------
    params : dict       Output of LoadParams (or apply_disc_prstp).
    MIU    : ndarray    Emission control rates, shape (num_periods+1,), 1-based.
    S      : ndarray    Savings rates,          shape (num_periods+1,), 1-based.
    alpha  : ndarray    IRF scaling,            shape (num_periods+1,), 1-based.

    Returns
    -------
    SCC : ndarray, shape (num_periods+1,), 1-based.
        Social cost of carbon in 2019 USD/tCO2.  SCC[1] = 0 by convention.
    """
    MIU   = np.ascontiguousarray(MIU,   dtype=np.float64).copy()
    S     = np.ascontiguousarray(S,     dtype=np.float64)
    alpha = np.ascontiguousarray(alpha, dtype=np.float64).copy()
    MIU[1]   = params['miu1']
    alpha[1] = params['a0']

    return compute_SCC_numba(
        MIU, S, alpha,
        params['num_periods'], params['tstep'], params['dk'], params['gama'],
        params['eco2Param'], params['sigma'], params['eland'], params['cost1tot'],
        params['expcost2'], params['miuup'], params['sLBounds'], params['sUBounds'],
        params['AlphaLowerBound'], params['Fcoef1'], params['Fcoef2'],
        params['CO2E_GHGabateB'], params['emshare0'], params['emshare1'],
        params['emshare2'], params['emshare3'], params['tau0'], params['tau1'],
        params['tau2'], params['tau3'], params['mateq'], params['fco22x'],
        params['F_Misc'], params['teq1'], params['teq2'], params['d1'], params['d2'],
        params['a1'], params['a2base'], params['a3'], params['res00'], params['res10'],
        params['res20'], params['res30'], params['tbox10'], params['tbox20'],
        params['CumEmiss0'], params['F_GHGabate2020'], params['k0'], params['tatm0'],
        params['mat0'], params['a0'], params['L'], params['RR'],
        params['scale1'], params['scale2'], params['elasmu'],
    )


def run_scc_fan(samples, num_periods=81):
    """
    Optimize DICE across a DataFrame of LHS parameter samples and collect
    per-sample SCC trajectories for fan chart construction.

    Parameters
    ----------
    samples : pd.DataFrame
        Columns: psi2, rho, irC, irT, eland0.  One row per sample.
        Typically the output of the LHS design in 03_monte_carlo.ipynb.
    num_periods : int
        Number of DICE periods (default 81).

    Returns
    -------
    fan_df : pd.DataFrame
        Long-form records with columns: sample_id, year, SCC, rho, psi2.
        Only finite, non-negative SCC values are included.
        Also returns (fan_df, n_success) as a tuple.
    """
    base_p  = LoadParams(num_periods)
    yr0     = base_p['yr0']
    tstep   = base_p['tstep']
    years   = np.arange(yr0, yr0 + tstep * num_periods, tstep)

    fan_records = []
    n_success   = 0
    t0          = time.time()
    n_total     = len(samples)

    print(f'Running {n_total} samples for SCC fan chart...')

    for idx, row in samples.iterrows():
        p = copy.deepcopy(base_p)
        p['a2base'] = float(row['psi2'])
        p['prstp']  = float(row['rho'])
        p['irC']    = float(row['irC'])
        p['irT']    = float(row['irT'])
        p['eland0'] = float(row['eland0'])

        for t in range(1, num_periods + 1):
            p['eland'][t] = p['eland0'] * (1 - p['deland']) ** (t - 1)

        def irf_eq(a0, p=p):
            lhs = (p['irf0']
                   + p['irC'] * (p['CumEmiss0'] - (p['mat0'] - p['mateq']))
                   + p['irT'] * p['tatm0'])
            rhs = sum(
                a0 * p[f'emshare{i}'] * p[f'tau{i}']
                * (1 - np.exp(-100 / (a0 * p[f'tau{i}'])))
                for i in range(4)
            )
            return lhs - rhs

        sol = root(irf_eq, p['a0'], method='hybr', options={'xtol': 1e-10})
        if not sol.success:
            continue
        p['a0'] = float(sol.x)

        p['rartp']    = np.exp(p['prstp'] + p['betaclim'] * p['pi']) - 1
        p['optlrsav'] = (
            (p['dk'] + 0.004) / (p['dk'] + 0.004 * p['elasmu'] + p['rartp'])
        ) * p['gama']
        for t in range(1, num_periods + 1):
            p['RR1'][t] = 1.0 / ((1 + p['rartp']) ** (p['tstep'] * (t - 1)))
            p['RR'][t]  = p['RR1'][t] * (1 + p['rprecaut'][t]) ** (-p['tstep'] * (t - 1))

        prob  = DiceFunc(num_periods, p, TempUpperConstraint=20.0, TempLowerConstraint=0.01)
        t_arr = np.arange(1, num_periods + 1)
        MIU0  = (0.05 + (p['miuup'][1:] - 0.05)
                 * (1 - np.exp(-0.05 * (t_arr - 1)))
                 / (1 - np.exp(-0.05 * num_periods)))
        MIU0[0] = 0.05
        S0      = np.full(num_periods, max(p['optlrsav'], 0.2))
        S0[p['FixSperiod'] - 1:] = p['FixSvalue']
        A0      = np.linspace(p['a0'], 0.425, num_periods)
        A0[0]   = p['a0']
        x0      = np.concatenate([MIU0, S0, A0])

        miu_b = ([(p['miu1'], p['miu1'])]
                 + [(p['MIULowerBound'], p['miuup'][t]) for t in range(2, num_periods + 1)])
        s_b   = [(p['sLBounds'][i], p['sUBounds'][i]) for i in range(1, num_periods + 1)]
        a_b   = [(p['AlphaLowerBound'], p['AlphaUpperBound']) for _ in range(num_periods)]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = minimize(
                prob.objective, x0,
                method='SLSQP',
                bounds=miu_b + s_b + a_b,
                constraints=[
                    {'type': 'eq',   'fun': prob.irf_residual},
                    {'type': 'ineq', 'fun': prob.temp_up},
                    {'type': 'ineq', 'fun': prob.temp_lo},
                ],
                options={'maxiter': 500, 'ftol': 1e-5, 'disp': False},
            )

        if np.isnan(res.fun) or res.fun > 1e14:
            continue

        output = recoverAllVars(res.x, p)
        SCC    = output[:, 39]

        for yr, scc in zip(years, SCC):
            if np.isfinite(scc) and scc >= 0:
                fan_records.append({
                    'sample_id': idx,
                    'year':      yr,
                    'SCC':       scc,
                    'rho':       float(row['rho']),
                    'psi2':      float(row['psi2']),
                })

        n_success += 1
        if (n_success % 100) == 0:
            print(f'  {idx + 1}/{n_total}  success={n_success}  '
                  f'elapsed={time.time() - t0:.1f}s')

    fan_df = pd.DataFrame(fan_records)
    print(f'\nDone in {time.time() - t0:.1f}s  |  success={n_success}/{n_total}  '
          f'|  records={len(fan_df)}')
    return fan_df, n_success
