# %% [markdown]
# # PyDICE-2023 (notebook version, corrected)
#
# Corrections relative to the "Nov4 FINAL" notebook, each checked against the GAMS reference output:
#
# 1. Cumulative emissions (CCATOT) accumulate the previous period's emissions, not the current one.
# 2. Other abatable GHG forcing (F_GHGabate) uses the previous period's base emissions and control rate.
# 3. Savings: lower bound 0 (GAMS goes below 0.2, e.g. ~0.14 at 5% discounting), and fixed at 0.28
#    from period 38 on (the notebook fixed it at optlrsav ≈ 0.24).
# 4. SCC: alpha re-adjusts to the emissions pulse through the IRF condition, as in GAMS. Holding alpha
#    fixed understates the GAMS SCC by 12-24% in 2025 (paper_build/scc_gams_controls.py). Richardson
#    extrapolation uses pulses h = 0.1 and h/2 with SCC = 2 f(h/2) - f(h), and the SCC[1] = 0.85·SCC[2]
#    fill-in is removed.
# 6. MAT floor at 10 GtC (GAMS MAT.LO = 10) instead of mateq: GAMS reservoirs are unconstrained.
# 5. 1.5°C scenario: a feasibility check (max abatement, lowest savings, alpha from IRF) and a
#    best-effort path when the limit cannot be met with equality capital accumulation.
#
# The land-use CO2 line was already right in this notebook (the control rate applies to land CO2).
#
# New: after every run, `compare_with_gams` loads the matching GAMS file, prints a summary and draws
# PyDICE-vs-GAMS figures. Usage in a notebook:
#
#     results = run_scenario(9)          # cost-benefit optimal
#     results['summary']
#
# Set GAMS_DIR below to the folder holding the GAMS .xlsx files (dice_scenario_opt.xlsx etc.).

# %%
import numpy as np
import pandas as pd
import os
import csv
from scipy.optimize import minimize, root, newton, brentq
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn
from numba import njit

seaborn.set(style='ticks')

GAMS_DIR = os.environ.get('PYDICE_GAMS_DIR', './gams_reference')   # folder with dice_scenario_*.xlsx
RESULTS_DIR = './results'

# %% Parameters
def LoadParams(num_periods=81, **kwargs):
    params = {}
    params['num_periods'] = num_periods
    params['tstep'] = kwargs.get('tstep', 5)
    params['gama'] = kwargs.get('gama', 0.300)
    params['pop1'] = kwargs.get('pop1', 7752.9)
    params['popadj'] = kwargs.get('popadj', 0.145)
    params['popasym'] = kwargs.get('popasym', 10825)
    params['dk'] = kwargs.get('dk', 0.100)
    params['q1'] = kwargs.get('q1', 135.7)
    params['AL1'] = kwargs.get('AL1', 5.84)
    params['gA1'] = kwargs.get('gA1', 0.066)
    params['delA'] = kwargs.get('delA', 0.0015)
    params['gsigma1'] = kwargs.get('gsigma1', -0.015)
    params['delgsig'] = kwargs.get('delgsig', 0.96)
    params['asymgsig'] = kwargs.get('asymgsig', -0.005)
    params['e1'] = kwargs.get('e1', 37.56)
    params['miu1'] = kwargs.get('miu1', 0.05)
    params['fosslim'] = kwargs.get('fosslim', 6000)
    params['CumEmiss0'] = kwargs.get('CumEmiss0', 633.5)
    params['a1'] = kwargs.get('a1', 0)
    params['a2base'] = kwargs.get('a2base', 0.003467)
    params['a3'] = kwargs.get('a3', 2.00)
    params['expcost2'] = kwargs.get('expcost2', 2.6)
    params['pback2050'] = kwargs.get('pback2050', 515)
    params['gback'] = kwargs.get('gback', -0.012)
    params['cprice1'] = kwargs.get('cprice1', 6)
    params['gcprice'] = kwargs.get('gcprice', 0.025)
    params['limmiu2070'] = kwargs.get('limmiu2070', 1.0)
    params['limmiu2120'] = kwargs.get('limmiu2120', 1.1)
    params['limmiu2200'] = kwargs.get('limmiu2200', 1.05)
    params['limmiu2300'] = kwargs.get('limmiu2300', 1.0)
    params['delmiumax'] = kwargs.get('delmiumax', 0.12)
    params['betaclim'] = kwargs.get('betaclim', 0.5)
    params['elasmu'] = kwargs.get('elasmu', 0.95)
    params['prstp'] = kwargs.get('prstp', 0.001)
    params['pi'] = kwargs.get('pi', 0.05)
    params['k0'] = kwargs.get('k0', 295)
    params['siggc1'] = kwargs.get('siggc1', 0.01)
    params['SRF'] = kwargs.get('SRF', 1e6)
    params['scale1'] = kwargs.get('scale1', 0.00891061)
    params['scale2'] = kwargs.get('scale2', -6275.91)
    params['eland0'] = kwargs.get('eland0', 5.9)
    params['deland'] = kwargs.get('deland', 0.1)
    params['F_Misc2020'] = kwargs.get('F_Misc2020', -0.054)
    params['F_Misc2100'] = kwargs.get('F_Misc2100', 0.265)
    params['F_GHGabate2020'] = kwargs.get('F_GHGabate2020', 0.518)
    params['F_GHGabate2100'] = kwargs.get('F_GHGabate2100', 0.957)
    params['ECO2eGHGB2020'] = kwargs.get('ECO2eGHGB2020', 9.96)
    params['ECO2eGHGB2100'] = kwargs.get('ECO2eGHGB2100', 15.5)
    params['emissrat2020'] = kwargs.get('emissrat2020', 1.40)
    params['emissrat2100'] = kwargs.get('emissrat2100', 1.21)
    params['Fcoef1'] = kwargs.get('Fcoef1', 0.00955)
    params['Fcoef2'] = kwargs.get('Fcoef2', 0.861)
    params['yr0'] = kwargs.get('yr0', 2020)
    params['emshare0'] = kwargs.get('emshare0', 0.2173)
    params['emshare1'] = kwargs.get('emshare1', 0.224)
    params['emshare2'] = kwargs.get('emshare2', 0.2824)
    params['emshare3'] = kwargs.get('emshare3', 0.2763)
    params['tau0'] = kwargs.get('tau0', 1e6)
    params['tau1'] = kwargs.get('tau1', 394.4)
    params['tau2'] = kwargs.get('tau2', 36.53)
    params['tau3'] = kwargs.get('tau3', 4.304)
    params['teq1'] = kwargs.get('teq1', 0.324)
    params['teq2'] = kwargs.get('teq2', 0.44)
    params['d1'] = kwargs.get('d1', 236)
    # Old file used 4.07; keep it for parity
    params['d2'] = kwargs.get('d2', 4.07)
    params['irf0'] = kwargs.get('irf0', 32.4)
    params['irC'] = kwargs.get('irC', 0.019)
    params['irT'] = kwargs.get('irT', 4.165)
    params['fco22x'] = kwargs.get('fco22x', 3.93)
    params['mat0'] = kwargs.get('mat0', 886.5128014)
    params['res00'] = kwargs.get('res00', 150.093)
    params['res10'] = kwargs.get('res10', 102.698)
    params['res20'] = kwargs.get('res20', 39.534)
    params['res30'] = kwargs.get('res30', 6.1865)
    params['mateq'] = kwargs.get('mateq', 588)
    params['tbox10'] = kwargs.get('tbox10', 0.1477)
    params['tbox20'] = kwargs.get('tbox20', 1.099454)
    params['tatm0'] = kwargs.get('tatm0', 1.24715)
    # GAMS lets the savings rate go below 0.2 (e.g. ~0.14 at 5% discounting)
    params['SLower'] = kwargs.get('SLower', 0.0)
    params['SUpper'] = kwargs.get('SUpper', 1.0)
    params['FixSperiod'] = kwargs.get('FixSperiod', 38)
    params['FixSvalue'] = kwargs.get('FixSvalue', 0.28)
    params['AlphaUpperBound'] = kwargs.get('AlphaUpperBound', np.inf)
    params['AlphaLowerBound'] = kwargs.get('AlphaLowerBound', 0.1)   # GAMS alpha.lo = 0.1 (FAIR-beta-4-3-1.gms l.99)
    params['MIULowerBound'] = kwargs.get('MIULowerBound', 0)

    # Solve for a0 from IRF equality at t=1
    def irf_eq(a0):
        IRF_LHS = (
            params['irf0'] +
            params['irC'] * (params['CumEmiss0'] - (params['mat0'] - params['mateq'])) +
            params['irT'] * params['tatm0']
        )
        IRF_RHS = (
            a0 * params['emshare0'] * params['tau0'] * (1 - np.exp(-100 / (a0 * params['tau0']))) +
            a0 * params['emshare1'] * params['tau1'] * (1 - np.exp(-100 / (a0 * params['tau1']))) +
            a0 * params['emshare2'] * params['tau2'] * (1 - np.exp(-100 / (a0 * params['tau2']))) +
            a0 * params['emshare3'] * params['tau3'] * (1 - np.exp(-100 / (a0 * params['tau3'])))
        )
        return IRF_LHS - IRF_RHS

    sol = root(irf_eq, 0.5, method='hybr', options={'xtol': 1e-12})
    params['a0'] = float(sol.x[0])

    params['rartp'] = np.exp(params['prstp'] + params['betaclim'] * params['pi']) - 1
    params['sig1'] = params['e1'] / (params['q1'] * (1 - params['miu1']))
    params['optlrsav'] = (params['dk'] + 0.004) / (params['dk'] + 0.004 * params['elasmu'] + params['rartp']) * params['gama']

    # Time series arrays (1..N used; index 0 unused)
    params['L'] = np.zeros(num_periods + 1)
    params['L'][1] = params['pop1']
    params['aL'] = np.zeros(num_periods + 1); params['aL'][1] = params['AL1']
    params['sigma'] = np.zeros(num_periods + 1); params['sigma'][1] = params['sig1']
    params['sigmatot'] = np.zeros(num_periods + 1)
    params['gA'] = np.zeros(num_periods + 1)
    params['gsig'] = np.zeros(num_periods + 1)
    params['eland'] = np.zeros(num_periods + 1)
    params['cost1tot'] = np.zeros(num_periods + 1)
    params['PBACKTIME'] = np.zeros(num_periods + 1)
    params['cpricebase'] = np.zeros(num_periods + 1)
    params['varpcc'] = np.zeros(num_periods + 1)
    params['rprecaut'] = np.zeros(num_periods + 1)
    params['RR1'] = np.zeros(num_periods + 1)
    params['RR'] = np.zeros(num_periods + 1)
    params['CO2E_GHGabateB'] = np.zeros(num_periods + 1)
    params['F_Misc'] = np.zeros(num_periods + 1)
    params['emissrat'] = np.zeros(num_periods + 1)

    for t in range(1, num_periods + 1):
        params['varpcc'][t] = min(params['siggc1'] ** 2 * 5 * (t - 1), params['siggc1'] ** 2 * 5 * 47)
        params['rprecaut'][t] = -0.5 * params['varpcc'][t] * params['elasmu'] ** 2
        params['RR1'][t] = 1 / ((1 + params['rartp']) ** (params['tstep'] * (t - 1)))
        params['RR'][t] = params['RR1'][t] * (1 + params['rprecaut'][t]) ** (-params['tstep'] * (t - 1))
        params['gA'][t] = params['gA1'] * np.exp(-params['delA'] * 5 * (t - 1))
        params['cpricebase'][t] = params['cprice1'] * (1 + params['gcprice']) ** (5 * (t - 1))
        params['PBACKTIME'][t] = params['pback2050'] * np.exp(-0.05 * (t - 7) if t <= 7 else -0.005 * (t - 7))
        params['gsig'][t] = min(params['gsigma1'] * params['delgsig'] ** (t - 1), params['asymgsig'])
        params['eland'][t] = params['eland0'] * (1 - params['deland']) ** (t - 1)

        if t <= 16:
            params['CO2E_GHGabateB'][t] = params['ECO2eGHGB2020'] + ((params['ECO2eGHGB2100'] - params['ECO2eGHGB2020']) / 16) * (t - 1)
            params['F_Misc'][t] = params['F_Misc2020'] + ((params['F_Misc2100'] - params['F_Misc2020']) / 16) * (t - 1)
            params['emissrat'][t] = params['emissrat2020'] + ((params['emissrat2100'] - params['emissrat2020']) / 16) * (t - 1)
        else:
            params['CO2E_GHGabateB'][t] = params['ECO2eGHGB2100']
            params['F_Misc'][t] = params['F_Misc2100']
            params['emissrat'][t] = params['emissrat2100']

        params['sigmatot'][t] = params['sigma'][t] * params['emissrat'][t]
        params['cost1tot'][t] = params['PBACKTIME'][t] * params['sigmatot'][t] / params['expcost2'] / 1000

        if t < num_periods:
            params['L'][t + 1] = params['L'][t] * (params['popasym'] / params['L'][t]) ** params['popadj']
            params['aL'][t + 1] = params['aL'][t] / (1 - params['gA'][t])
            params['sigma'][t + 1] = params['sigma'][t] * np.exp(5 * params['gsig'][t])

    # MIU caps
    params['miuup'] = np.zeros(num_periods + 1)
    params['miuup'][1] = 0.05
    params['miuup'][2] = 0.10
    for t in range(3, num_periods + 1):
        if t <= 8:
            params['miuup'][t] = params['delmiumax'] * (t - 1)
        elif t <= 11:
            params['miuup'][t] = 0.85 + 0.05 * (t - 8)
        elif t <= 20:
            params['miuup'][t] = params['limmiu2070']
        elif t <= 37:
            params['miuup'][t] = params['limmiu2120']
        elif t <= 57:
            params['miuup'][t] = params['limmiu2200']
        else:
            params['miuup'][t] = params['limmiu2300']

    params['sLBounds'] = np.full(num_periods + 1, params['SLower'])
    params['sUBounds'] = np.full(num_periods + 1, params['SUpper'])
    # GAMS fixes the savings rate at FixSvalue (0.28) from period FixSperiod on
    if params['FixSperiod'] <= num_periods:
        params['sLBounds'][params['FixSperiod']:] = params['FixSvalue']
        params['sUBounds'][params['FixSperiod']:] = params['FixSvalue']

    params['eco2Param'] = params['aL'] * (params['L'] / 1000) ** (1 - params['gama'])
    return params

def apply_disc_prstp(params, prstp_value, *, elasmu_value=0.001,
                     cap_after_t=None, cap_exp_value=None):
    """
    Build RR the GAMS DISC way:
      - Pure PRSTP discounting (no risk premium, no precaution)
      - Optional cap: for t.val > cap_after_t use exponent = cap_exp_value
      - Recompute optlrsav using PRSTP (not rartp)
      - Keep S fixed at FixSvalue after FixSperiod
    """
    params['prstp'] = float(prstp_value)
    params['elasmu'] = float(elasmu_value)
    #params['betaclim'] = 0.0  # kill climate risk premium

    N, step = int(params['num_periods']), int(params['tstep'])
    RR1 = np.zeros(N + 1); RR = np.zeros(N + 1)
    for t in range(1, N + 1):
        exp_pow = step * (t - 1)
        if cap_after_t is not None and cap_exp_value is not None and t > cap_after_t:
            exp_pow = cap_exp_value
        RR1[t] = 1.0 / ((1.0 + params['prstp']) ** exp_pow)
        RR[t]  = RR1[t]

    params['RR1'] = RR1
    params['RR']  = RR
    if 'rprecaut' in params and params['rprecaut'].size == (N + 1):
        params['rprecaut'][:] = 0.0  # ensure precaution doesn't creep back in
    params['rartp'] = params['prstp']  # for clarity; not used in RR

    params['optlrsav'] = (
        (params['dk'] + 0.004) /
        (params['dk'] + 0.004 * params['elasmu'] + params['prstp'])
    ) * params['gama']

    # Savings stay fixed at FixSvalue (0.28) after FixSperiod, as in GAMS
    if params['FixSperiod'] <= N:
        params['sLBounds'][params['FixSperiod']:] = params['FixSvalue']
        params['sUBounds'][params['FixSperiod']:] = params['FixSvalue']
    return params


# %% Forward model
@njit
def diceForward_numba(i, MIU, S, alpha, CCATOT, K, I, F_GHGabate, RES0, RES1, RES2, RES3, TBOX1, TBOX2,
                      tstep, dk, gama, eco2Param, sigma, eland, cost1tot, expcost2, miuup, sLBounds, sUBounds,
                      AlphaLowerBound, Fcoef1, Fcoef2, CO2E_GHGabateB, emshare0, emshare1, emshare2, emshare3,
                      tau0, tau1, tau2, tau3, mateq, fco22x, F_Misc, teq1, teq2, d1, d2, a1, a2base, a3, 
                      pulse_GtCO2_per_year, ECO2_prev, MIU_prev_i):
    # GAMS DICE-2023 timing (verified against the GAMS output files):
    #   CCATOT(t+1)     = CCATOT(t) + ECO2(t)*tstep/3.666
    #   F_GHGabate(t+1) = Fcoef2*F_GHGabate(t) + Fcoef1*CO2E_GHGabateB(t)*(1-MIU(t))
    #   RES_j(t+1) uses alpha(t+1) and ECO2(t+1)
    # ECO2_prev, MIU_prev_i: total emissions (incl. pulse) and control rate of period i-1
    # Clamp controls
    MIU_i = min(max(MIU[i], 0.0), miuup[i])
    S_i = min(max(S[i], sLBounds[i]), sUBounds[i])
    alpha_i = max(alpha[i], AlphaLowerBound)
    
    # Capital accumulation
    K = (1.0 - dk) ** tstep * K + tstep * I
    
    # Calculate CURRENT period gross output
    YGROSS = eco2Param[i] * (K ** gama)
    
    # Calculate CURRENT period emissions with CURRENT MIU
    ECO2 = (sigma[i] * YGROSS + eland[i]) * (1.0 - MIU_i)
    
    # Cumulative emissions accumulate the PREVIOUS period's emissions (GAMS timing)
    CCATOT = CCATOT + ECO2_prev * (tstep / 3.666)
    
    # Other abatable GHG forcing uses the PREVIOUS period's base emissions and control rate
    F_GHGabate = Fcoef2 * F_GHGabate + Fcoef1 * CO2E_GHGabateB[i - 1] * (1.0 - MIU_prev_i)
    
    # Carbon boxes - use same emissions as CCATOT for consistency
    total_flow = ECO2 + pulse_GtCO2_per_year[i]
    inflow_GtC_per_yr = total_flow / 3.667
    
    RES0 = (emshare0 * tau0 * alpha_i * inflow_GtC_per_yr) * (1.0 - np.exp(-tstep / (tau0 * alpha_i))) + RES0 * np.exp(-tstep / (tau0 * alpha_i))
    RES1 = (emshare1 * tau1 * alpha_i * inflow_GtC_per_yr) * (1.0 - np.exp(-tstep / (tau1 * alpha_i))) + RES1 * np.exp(-tstep / (tau1 * alpha_i))
    RES2 = (emshare2 * tau2 * alpha_i * inflow_GtC_per_yr) * (1.0 - np.exp(-tstep / (tau2 * alpha_i))) + RES2 * np.exp(-tstep / (tau2 * alpha_i))
    RES3 = (emshare3 * tau3 * alpha_i * inflow_GtC_per_yr) * (1.0 - np.exp(-tstep / (tau3 * alpha_i))) + RES3 * np.exp(-tstep / (tau3 * alpha_i))
    
    MAT = mateq + RES0 + RES1 + RES2 + RES3
    MAT = max(MAT, 10.0)   # GAMS: MAT.LO = 10 (FAIR-beta-4-3-1.gms l.96); reservoirs are unconstrained, so MAT may fall below mateq
    
    # Forcing & temperature
    FORC = fco22x * np.log(MAT / mateq) / np.log(2.0) + F_Misc[i] + F_GHGabate
    TBOX1 = TBOX1 * np.exp(-tstep / d1) + teq1 * FORC * (1.0 - np.exp(-tstep / d1))
    TBOX2 = TBOX2 * np.exp(-tstep / d2) + teq2 * FORC * (1.0 - np.exp(-tstep / d2))
    TATM = TBOX1 + TBOX2
    TATM = min(max(TATM, 0.01), 20.0)
    
    # Output, damages, abatement
    DAMFRAC = a1 * TATM + a2base * TATM ** a3
    ABATECOST = YGROSS * cost1tot[i] * (MIU_i ** expcost2)
    YNET = YGROSS * (1.0 - DAMFRAC)
    Y = YNET - ABATECOST
    
    # Investment and Consumption
    I = S_i * Y
    C = Y - I
    
    if not (np.isfinite(MAT) and np.isfinite(TATM) and np.isfinite(C) and np.isfinite(YGROSS)):
        return np.array([np.nan] * 20, dtype=np.float64)
    
    return np.array([C, CCATOT, K, I, F_GHGabate, RES0, RES1, RES2, RES3,
                     TBOX1, TBOX2, MAT, TATM, Y, YNET, YGROSS, DAMFRAC, ABATECOST,
                     total_flow, MIU_i], dtype=np.float64)
@njit
def diceTrajectory_numba(MIU, S, alpha, num_periods, tstep, dk, gama, eco2Param, sigma, eland, cost1tot, expcost2,
                         miuup, sLBounds, sUBounds, AlphaLowerBound, Fcoef1, Fcoef2, CO2E_GHGabateB, emshare0,
                         emshare1, emshare2, emshare3, tau0, tau1, tau2, tau3, mateq, fco22x, F_Misc, teq1, teq2,
                         d1, d2, a1, a2base, a3, res00, res10, res20, res30, tbox10, tbox20, CumEmiss0,
                         F_GHGabate2020, k0, tatm0, mat0, a0, L, RR, scale1, scale2, elasmu, pulse_GtCO2_per_year):

    C = np.zeros(num_periods + 1)
    K = np.zeros(num_periods + 1)
    CCATOT = np.zeros(num_periods + 1)
    MAT = np.zeros(num_periods + 1)
    TATM = np.zeros(num_periods + 1)
    Y = np.zeros(num_periods + 1)
    YNET = np.zeros(num_periods + 1)
    YGROSS = np.zeros(num_periods + 1)
    DAMFRAC = np.zeros(num_periods + 1)
    ABATECOST = np.zeros(num_periods + 1)
    RES0_arr = np.zeros(num_periods + 1)
    RES1_arr = np.zeros(num_periods + 1)
    RES2_arr = np.zeros(num_periods + 1)
    RES3_arr = np.zeros(num_periods + 1)
    TBOX1_arr = np.zeros(num_periods + 1)
    TBOX2_arr = np.zeros(num_periods + 1)
    F_GHGab_arr = np.zeros(num_periods + 1)

    RES0 = res00; RES1 = res10; RES2 = res20; RES3 = res30
    TBOX1 = tbox10; TBOX2 = tbox20
    CCATOT[1] = CumEmiss0
    F_GHGabate = F_GHGabate2020
    K[1] = k0
    TATM[1] = tatm0
    DAMFRAC[1] = a1 * TATM[1] + a2base * TATM[1] ** a3
    YGROSS[1] = eco2Param[1] * (K[1] ** gama)
    YNET[1] = YGROSS[1] * (1.0 - DAMFRAC[1])
    ABATECOST[1] = YGROSS[1] * cost1tot[1] * (MIU[1] ** expcost2)
    Y[1] = YNET[1] - ABATECOST[1]
    I = S[1] * Y[1]
    C[1] = Y[1] - I

    RES0_arr[1] = RES0; RES1_arr[1] = RES1; RES2_arr[1] = RES2; RES3_arr[1] = RES3
    TBOX1_arr[1] = TBOX1; TBOX2_arr[1] = TBOX2
    F_GHGab_arr[1] = F_GHGabate
    MAT[1] = mat0
    # DO NOT mutate alpha here (Numba broadcast risk) — set alpha[1]=a0 in Python wrapper

    # Period-1 emissions and control rate feed the lagged CCATOT and F_GHGabate updates
    MIU_prev = min(max(MIU[1], 0.0), miuup[1])
    ECO2_prev = (sigma[1] * YGROSS[1] + eland[1]) * (1.0 - MIU_prev) + pulse_GtCO2_per_year[1]

    for i in range(2, num_periods + 1):
        result = diceForward_numba(i, MIU, S, alpha, CCATOT[i - 1], K[i - 1], I, F_GHGabate, RES0, RES1, RES2, RES3, TBOX1, TBOX2,
                                   tstep, dk, gama, eco2Param, sigma, eland, cost1tot, expcost2, miuup, sLBounds, sUBounds,
                                   AlphaLowerBound, Fcoef1, Fcoef2, CO2E_GHGabateB, emshare0, emshare1, emshare2, emshare3,
                                   tau0, tau1, tau2, tau3, mateq, fco22x, F_Misc, teq1, teq2, d1, d2, a1, a2base, a3, pulse_GtCO2_per_year,
                                   ECO2_prev, MIU_prev)
        ECO2_prev = result[18]
        MIU_prev = result[19]
        if np.any(np.isnan(result)):
            # Return NaNs in all outputs so callers can detect failure
            nan = np.full(num_periods + 1, np.nan)
            return (np.nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan, nan, nan)

        C[i] = result[0]; CCATOT[i] = result[1]; K[i] = result[2]
        I = result[3]; F_GHGabate = result[4]
        RES0 = result[5]; RES1 = result[6]; RES2 = result[7]; RES3 = result[8]
        TBOX1 = result[9]; TBOX2 = result[10]
        MAT[i] = result[11]; TATM[i] = result[12]
        Y[i] = result[13]; YNET[i] = result[14]; YGROSS[i] = result[15]
        DAMFRAC[i] = result[16]
        ABATECOST[i] = result[17]

        RES0_arr[i] = RES0; RES1_arr[i] = RES1; RES2_arr[i] = RES2; RES3_arr[i] = RES3
        TBOX1_arr[i] = TBOX1; TBOX2_arr[i] = TBOX2
        F_GHGab_arr[i] = F_GHGabate

    # Welfare
    PERIODU = ((C[1:] * 1000.0 / L[1:]) ** (1.0 - elasmu) - 1.0) / (1.0 - elasmu) - 1.0
    TOTPERIODU = PERIODU * L[1:] * RR[1:]
    UTILITY = tstep * scale1 * np.sum(TOTPERIODU) + scale2

    return (UTILITY, C, CCATOT, MAT, TATM, K, Y, YNET, YGROSS, DAMFRAC, ABATECOST,
            RES0_arr, RES1_arr, RES2_arr, RES3_arr, TBOX1_arr, TBOX2_arr, F_GHGab_arr)

def diceTrajectory(params, MIU, S, alpha, pulse_GtCO2_per_year=None):
    # Ensure contiguous and **writable** arrays; set alpha[1] in Python (not inside @njit)
    MIU   = np.ascontiguousarray(MIU,   dtype=np.float64)
    S     = np.ascontiguousarray(S,     dtype=np.float64)
    alpha = np.ascontiguousarray(alpha, dtype=np.float64).copy()
    MIU[1]   = params['miu1']
    alpha[1] = params['a0']
    
    if pulse_GtCO2_per_year is None:
        pulse_GtCO2_per_year = np.zeros(params['num_periods'] + 1, dtype=np.float64)
    else:
        pulse_GtCO2_per_year = np.ascontiguousarray(pulse_GtCO2_per_year, dtype=np.float64)
        
    return diceTrajectory_numba(
        MIU, S, alpha,
        params['num_periods'], params['tstep'], params['dk'], params['gama'], params['eco2Param'],
        params['sigma'], params['eland'], params['cost1tot'], params['expcost2'], params['miuup'],
        params['sLBounds'], params['sUBounds'], params['AlphaLowerBound'], params['Fcoef1'], params['Fcoef2'], params['CO2E_GHGabateB'],
        params['emshare0'], params['emshare1'], params['emshare2'], params['emshare3'], params['tau0'], params['tau1'],
        params['tau2'], params['tau3'], params['mateq'], params['fco22x'], params['F_Misc'], params['teq1'], params['teq2'],
        params['d1'], params['d2'], params['a1'], params['a2base'], params['a3'], params['res00'], params['res10'],
        params['res20'], params['res30'], params['tbox10'], params['tbox20'], params['CumEmiss0'], params['F_GHGabate2020'],
        params['k0'], params['tatm0'], params['mat0'], params['a0'], params['L'], params['RR'], params['scale1'],
        params['scale2'], params['elasmu'], pulse_GtCO2_per_year
    )


# %% IRF, alpha solver and SCC
# ── IRF condition, alpha solver and social cost of carbon ─────────────────────

def irf_rhs(params, a):
    """Right-hand side of the IRF100 condition for a given alpha (scalar or array)."""
    out = 0.0
    for j in range(4):
        tau = params[f'tau{j}']
        out = out + a * params[f'emshare{j}'] * tau * (1.0 - np.exp(-100.0 / (a * tau)))
    return out


def irf_lhs(params, CCATOT, MAT, TATM):
    """Left-hand side of the IRF100 condition from the model state."""
    return params['irf0'] + params['irC'] * (CCATOT - (MAT - params['mateq'])) + params['irT'] * TATM


def solve_alpha(params, MIU, S, pulse=None, alpha_start=None, first_period=2):
    """
    Solve alpha(t) so the IRF condition holds in every period, for fixed MIU and S
    (1-based arrays of length N+1). The residual in period t depends only on
    alpha(1..t), so alpha is solved one period at a time.

    pulse        : optional emissions pulse (GtCO2/yr), 1-based array
    alpha_start  : starting guess; periods before first_period are kept from it
    Periods where alpha would fall below AlphaLowerBound (far future, large net-negative
    emissions) are held at the bound, matching the clamp in the forward model.
    Returns (alpha, max |residual| over periods not at the bound).
    """
    N = params['num_periods']
    lo, hi = max(params['AlphaLowerBound'], 1e-3), 50.0
    a = np.full(N + 1, params['a0']) if alpha_start is None else np.array(alpha_start, dtype=np.float64).copy()
    a[1] = params['a0']
    at_bound = np.zeros(N + 1, dtype=bool)

    def resid_k(val, k):
        a[k] = val
        o = diceTrajectory(params, MIU, S, a, pulse)
        return irf_rhs(params, val) - irf_lhs(params, o[2][k], o[3][k], o[4][k])

    for k in range(max(first_period, 2), N + 1):
        x0 = a[k] if alpha_start is not None else a[k - 1]
        try:
            # secant iteration from a good starting point is fast; fall back to bisection
            val = newton(resid_k, x0, args=(k,), tol=1e-13, maxiter=50)
            if not (lo <= val <= hi) or abs(resid_k(val, k)) > 1e-8:
                raise RuntimeError
            a[k] = val
        except (RuntimeError, OverflowError, ZeroDivisionError, FloatingPointError):
            g_lo, g_hi = resid_k(lo, k), resid_k(hi, k)
            if g_lo * g_hi < 0:
                a[k] = brentq(resid_k, lo, hi, args=(k,), xtol=1e-14)
            elif g_lo > 0:          # even the smallest allowed alpha is too large
                a[k] = lo; at_bound[k] = True
            else:
                a[k] = hi
    o = diceTrajectory(params, MIU, S, a, pulse)
    r = irf_rhs(params, a[1:]) - irf_lhs(params, o[2][1:], o[3][1:], o[4][1:])
    keep = ~at_bound[1:]
    return a, (float(np.max(np.abs(r[keep]))) if np.any(keep) else 0.0)


def _welfare_sum(params, C):
    cpc = 1000.0 * C[1:] / params['L'][1:]
    el = params['elasmu']
    return np.sum((((cpc ** (1.0 - el)) - 1.0) / (1.0 - el) - 1.0) * params['L'][1:] * params['RR'][1:]), cpc


def compute_SCC(params, MIU, S, alpha, alpha_response=True, pulse_size=0.1, last_period=None):
    """
    Social cost of carbon (2019 USD/tCO2) by period, from a small emissions pulse.

    alpha_response=True : alpha re-adjusts to the pulse through the IRF condition. This is
                          what the GAMS SCC measures (the marginal of the emissions equation
                          with the IRF condition binding) and reproduces it.
    alpha_response=False: alpha held at its baseline path (the older approach). This
                          understates the GAMS SCC (12-24% in 2025 across scenarios).

    Richardson extrapolation with pulses h and h/2 for a one-sided difference:
    SCC = 2*f(h/2) - f(h), error O(h^2). (The earlier (4*f(h/2) - f(h))/3 is the step for a
    central difference and leaves a third of the O(h) error.) h = 0.1 GtCO2/yr gives a
    relative error below 1e-7 against the exact derivative; below h ~ 1e-3 rounding in the
    welfare sum dominates (paper_build/check2_pulse_sweep.py).
    SCC[1] (2020) is set to 0: a pulse in period 1 cannot affect the period-1 state here.
    """
    N = params['num_periods']
    last = N if last_period is None else min(last_period, N)
    MIU = np.ascontiguousarray(MIU, dtype=np.float64)
    S = np.ascontiguousarray(S, dtype=np.float64)
    alpha = np.ascontiguousarray(alpha, dtype=np.float64)

    if alpha_response:
        alpha_base, _ = solve_alpha(params, MIU, S, alpha_start=alpha)
    else:
        alpha_base = alpha.copy()
    base = diceTrajectory(params, MIU, S, alpha_base)
    if np.isnan(base[0]):
        return np.zeros(N + 1)
    W0, cpc = _welfare_sum(params, base[1])

    def dW(i, h):
        pulse = np.zeros(N + 1); pulse[i] = h
        if alpha_response:
            a, _ = solve_alpha(params, MIU, S, pulse=pulse, alpha_start=alpha_base, first_period=i)
        else:
            a = alpha_base
        alt = diceTrajectory(params, MIU, S, a, pulse)
        if np.isnan(alt[0]):
            return np.nan
        return _welfare_sum(params, alt[1])[0] - W0

    SCC = np.zeros(N + 1)
    h, h2 = pulse_size, pulse_size / 2.0
    for i in range(2, last + 1):
        lam = (cpc[i - 1] ** (-params['elasmu'])) * params['RR'][i]
        if abs(lam) < 1e-300:
            continue
        s_h = -(dW(i, h) / h) / lam
        s_h2 = -(dW(i, h2) / h2) / lam
        SCC[i] = 2.0 * s_h2 - s_h if np.isfinite(s_h) and np.isfinite(s_h2) else 0.0
    return SCC


# %% Recover all variables
@njit
def recoverAllVars_numba(x, num_periods, tstep, dk, gama, eco2Param, sigma, eland, cost1tot, expcost2, miuup,
                         sLBounds, sUBounds, AlphaLowerBound, Fcoef1, Fcoef2, CO2E_GHGabateB, emshare0, emshare1,
                         emshare2, emshare3, tau0, tau1, tau2, tau3, mateq, fco22x, F_Misc, teq1, teq2, d1, d2,
                         a1, a2base, a3, res00, res10, res20, res30, tbox10, tbox20, CumEmiss0, F_GHGabate2020,
                         k0, tatm0, mat0, a0, L, RR, scale1, scale2, elasmu, PBACKTIME, irf0, irC, irT, SRF):
    MIU = np.zeros(num_periods + 1)
    S   = np.zeros(num_periods + 1)
    alpha = np.zeros(num_periods + 1)
    MIU[1:] = x[:num_periods]
    S[1:]   = x[num_periods:2 * num_periods]
    alpha[1:] = x[2 * num_periods:]
    alpha[1] = a0  # set here in Python wrapper normally; safe here for compiled path
    pulse_GtCO2_per_year = np.zeros(num_periods + 1, dtype=np.float64)
    result = diceTrajectory_numba(
        MIU, S, alpha, num_periods, tstep, dk, gama, eco2Param, sigma, eland, cost1tot, expcost2, miuup, sLBounds,
        sUBounds, AlphaLowerBound, Fcoef1, Fcoef2, CO2E_GHGabateB, emshare0, emshare1, emshare2, emshare3, tau0,
        tau1, tau2, tau3, mateq, fco22x, F_Misc, teq1, teq2, d1, d2, a1, a2base, a3, res00, res10, res20, res30,
        tbox10, tbox20, CumEmiss0, F_GHGabate2020, k0, tatm0, mat0, a0, L, RR, scale1, scale2, elasmu, pulse_GtCO2_per_year
    )
    if np.isnan(result[0]):
        return np.zeros((num_periods, 46))

    C = result[1]; CCATOT = result[2]; MAT = result[3]; TATM = result[4]; K = result[5]
    Y = result[6]; YNET = result[7]; YGROSS = result[8]; DAMFRAC = result[9]; ABATECOST = result[10]
    RES0_arr = result[11]; RES1_arr = result[12]; RES2_arr = result[13]; RES3_arr = result[14]
    TBOX1_arr = result[15]; TBOX2_arr = result[16]; F_GHGab_arr = result[17]

    safe_MAT = np.maximum(MAT, 10.0)   # same floor as the forward model (GAMS MAT.LO = 10)
    I = S * Y
    CACC = CCATOT - (MAT - mateq)
    DAMAGES = YGROSS * DAMFRAC
    CPRICE = PBACKTIME * (MIU ** (expcost2 - 1.0))
    # Industrial and total CO2 (the control rate applies to land-use CO2 too, as in GAMS)
    EIND = sigma * (eco2Param * (K ** gama)) * (1.0 - MIU)
    ECO2 = EIND + eland * (1.0 - MIU)
    ECO2E = ECO2 + CO2E_GHGabateB * (1.0 - MIU)
    FORC = fco22x * np.log(safe_MAT / mateq) / np.log(2.0) + F_Misc + F_GHGab_arr
    FORC_CO2 = fco22x * np.log(safe_MAT / mateq) / np.log(2.0)
    CPC = np.zeros(num_periods + 1)
    for t in range(1, num_periods + 1):
        CPC[t] = 1000.0 * C[t] / L[t]

    # RFACTLONG, RLONG, RSHORT
    RFACTLONG = np.full(num_periods + 1, SRF)
    for i in range(2, num_periods + 1):
        RFACTLONG[i] = SRF * (CPC[i - 1] / CPC[1]) ** (-elasmu) * RR[i]
    RLONG = np.zeros(num_periods + 1)
    RSHORT = np.zeros(num_periods + 1)
    for i in range(2, num_periods + 1):
        RLONG[i] = -np.log(RFACTLONG[i] / SRF) / (5.0 * (i - 1))
        RSHORT[i] = -np.log(RFACTLONG[i] / RFACTLONG[i - 1]) / 5.0

    # Welfare per period
    PERIODU = np.zeros(num_periods + 1)
    TOTPERIODU = np.zeros(num_periods + 1)
    for t in range(1, num_periods + 1):
        PERIODU[t] = ((C[t] * 1000.0 / L[t]) ** (1.0 - elasmu) - 1.0) / (1.0 - elasmu) - 1.0
        TOTPERIODU[t] = PERIODU[t] * L[t] * RR[t]

    IRFt = irf0 + irC * (CCATOT - (MAT - mateq)) + irT * TATM

    # SCC is computed outside Numba (see compute_SCC) and written into column 39 by recoverAllVars
    SCC = np.zeros(num_periods + 1)

    # Build output with columns aligned to header used in dump_state()
    output = np.zeros((num_periods, 46))
    for i in range(num_periods):
        t = i + 1
        # 0..9
        output[i, 0]  = EIND[t]                                  # EIND
        output[i, 1]  = ECO2[t]                                  # ECO2
        output[i, 2]  = MAT[t] / 2.13                            # CO2PPM
        output[i, 3]  = TATM[t]                                  # TATM
        output[i, 4]  = Y[t]                                     # Y
        output[i, 5]  = DAMFRAC[t]                               # DAMFRAC
        output[i, 6]  = CPC[t]                                   # CPC
        output[i, 7]  = CPRICE[t]                                # CPRICE
        output[i, 8]  = MIU[t]                                   # MIUopt
        output[i, 9]  = RSHORT[t]                                # RSHORT
        # 10..19
        output[i,10]  = ECO2E[t]                                 # ECO2E
        output[i,11]  = L[t]                                     # L
        output[i,12]  = eco2Param[t]                             # AL (proxy used in old code)
        output[i,13]  = YGROSS[t]                                # YGROSS
        output[i,14]  = K[t]                                     # K
        output[i,15]  = S[t]                                     # Sopt
        output[i,16]  = I[t]                                     # I
        output[i,17]  = YNET[t]                                  # YNET
        output[i,18]  = CCATOT[t]                                # CCATOT
        output[i,19]  = CACC[t]                                  # CACC
        # 20..29
        output[i,20]  = RES0_arr[t]                              # RES0
        output[i,21]  = RES1_arr[t]                              # RES1
        output[i,22]  = RES2_arr[t]                              # RES2
        output[i,23]  = RES3_arr[t]                              # RES3
        output[i,24]  = DAMAGES[t]                               # DAMAGES
        output[i,25]  = ABATECOST[t]                             # ABATECOST
        output[i,26]  = PBACKTIME[t] * (MIU[t] ** (expcost2 - 1.0))  # MCABATE
        output[i,27]  = C[t]                                     # C
        output[i,28]  = PERIODU[t]                               # PERIODU
        output[i,29]  = TOTPERIODU[t]                            # TOTPERIODU
        # 30..44
        output[i,30]  = MAT[t]                                   # MAT
        output[i,31]  = FORC[t]                                  # FORC
        output[i,32]  = TBOX1_arr[t]                             # TBOX1
        output[i,33]  = TBOX2_arr[t]                             # TBOX2
        output[i,34]  = F_GHGab_arr[t]                           # F_GHGABATE
        output[i,35]  = IRFt[t]                                  # IRFT
        output[i,36]  = alpha[t]                                 # ALPHA
        output[i,37]  = RFACTLONG[t]                             # RFACTLONG
        output[i,38]  = RLONG[t]                                 # RLONG
        output[i,39]  = SCC[t]                                   # SCC
        output[i,40]  = ABATECOST[t] / max(Y[t], 1e-12)          # ABATERAT
        output[i,41]  = MAT[t] / mat0                            # ATFRAC2020
        output[i,42]  = MAT[t] / mateq                           # ATFRAC1765
        output[i,43]  = FORC_CO2[t]                              # FORC_CO2
        output[i,44]  = RR[t]                                    # RR

    if not np.all(np.isfinite(output)):
        return np.zeros((num_periods, 46))
    return output

def recoverAllVars(x, params, scc_alpha_response=True):
    """Rebuild all reported variables from the decision vector x = [MIU, S, alpha].
    The SCC (column 39) is computed with compute_SCC; by default alpha re-adjusts to
    each emissions pulse through the IRF condition, which is how GAMS defines it."""
    num_periods = params['num_periods']
    output = _recover_core(x, params)
    if np.all(output == 0):
        return output
    MIU = np.zeros(num_periods + 1); S = np.zeros(num_periods + 1); alpha = np.zeros(num_periods + 1)
    MIU[1:] = x[:num_periods]; S[1:] = x[num_periods:2 * num_periods]; alpha[1:] = x[2 * num_periods:]
    SCC = compute_SCC(params, MIU, S, alpha, alpha_response=scc_alpha_response)
    output[:, 39] = SCC[1:]
    return output


def _recover_core(x, params):
    num_periods = params['num_periods']
    return recoverAllVars_numba(
        x, num_periods, params['tstep'], params['dk'], params['gama'], params['eco2Param'], params['sigma'],
        params['eland'], params['cost1tot'], params['expcost2'], params['miuup'], params['sLBounds'],
        params['sUBounds'], params['AlphaLowerBound'], params['Fcoef1'], params['Fcoef2'], params['CO2E_GHGabateB'],
        params['emshare0'], params['emshare1'], params['emshare2'], params['emshare3'], params['tau0'], params['tau1'],
        params['tau2'], params['tau3'], params['mateq'], params['fco22x'], params['F_Misc'], params['teq1'], params['teq2'],
        params['d1'], params['d2'], params['a1'], params['a2base'], params['a3'], params['res00'], params['res10'],
        params['res20'], params['res30'], params['tbox10'], params['tbox20'], params['CumEmiss0'], params['F_GHGabate2020'],
        params['k0'], params['tatm0'], params['mat0'], params['a0'], params['L'], params['RR'], params['scale1'],
        params['scale2'], params['elasmu'], params['PBACKTIME'], params['irf0'], params['irC'], params['irT'], params['SRF']
    )


# %% Optimisation problem
class DiceFunc:
    def __init__(self, num_periods, params, TempUpperConstraint=20, TempLowerConstraint=0.5):
        self.num_periods = num_periods
        self.params = params
        self.TempUpperConstraint = TempUpperConstraint
        self.TempLowerConstraint = TempLowerConstraint
        self.MIU = np.zeros(num_periods + 1)
        t = np.arange(1, num_periods + 1)
        self.MIU[1:] = 0.05 + (params['miuup'][1:] - 0.05) * (1 - np.exp(-0.05 * (t - 1))) / (1 - np.exp(-0.05 * num_periods))
        self.MIU[1] = 0.05
        self.S = np.full(num_periods + 1, max(params['optlrsav'], 0.2))
        self.S[params['FixSperiod']:] = params['FixSvalue']
        self.Alpha = np.linspace(params['a0'], 0.425, num_periods + 1)
        self.Alpha[1] = params['a0']

    def pack(self, x):
        return x[:self.num_periods], x[self.num_periods:2 * self.num_periods], x[2 * self.num_periods:]

    def objective(self, x):
        MIU, S, Alpha = self.pack(x)
        self.MIU[1:self.num_periods + 1] = MIU
        self.S[1:self.num_periods + 1] = S
        self.Alpha[1:self.num_periods + 1] = Alpha
        out = diceTrajectory(self.params, self.MIU, self.S, self.Alpha)
        UTILITY, C = out[0], out[1]
        if np.any(np.isnan(C)):
            return 1e15
        return -UTILITY

    def irf_residual(self, x):
        MIU, S, Alpha = self.pack(x)
        self.MIU[1:self.num_periods + 1] = MIU
        self.S[1:self.num_periods + 1] = S
        self.Alpha[1:self.num_periods + 1] = Alpha
        out = diceTrajectory(self.params, self.MIU, self.S, self.Alpha)
        CCATOT = out[2]; MAT = out[3]; TATM = out[4]
        if np.any(np.isnan(TATM)):
            return np.ones(self.num_periods) * 1e10
        IRF_LHS = (
            self.params['irf0'] +
            self.params['irC'] * (CCATOT[1:] - (MAT[1:] - self.params['mateq'])) +
            self.params['irT'] * TATM[1:]
        )
        IRF_RHS = (
            Alpha * self.params['emshare0'] * self.params['tau0'] * (1 - np.exp(-100 / (Alpha * self.params['tau0']))) +
            Alpha * self.params['emshare1'] * self.params['tau1'] * (1 - np.exp(-100 / (Alpha * self.params['tau1']))) +
            Alpha * self.params['emshare2'] * self.params['tau2'] * (1 - np.exp(-100 / (Alpha * self.params['tau2']))) +
            Alpha * self.params['emshare3'] * self.params['tau3'] * (1 - np.exp(-100 / (Alpha * self.params['tau3'])))
        )
        return IRF_LHS - IRF_RHS

    def temp_up(self, x):
        MIU, S, Alpha = self.pack(x)
        self.MIU[1:self.num_periods + 1] = MIU
        self.S[1:self.num_periods + 1] = S
        self.Alpha[1:self.num_periods + 1] = Alpha
        out = diceTrajectory(self.params, self.MIU, self.S, self.Alpha)
        TATM = out[4]
        if np.any(np.isnan(TATM)):
            return np.ones(self.num_periods) * -1e10
        return self.TempUpperConstraint - TATM[1:]

    def temp_lo(self, x):
        MIU, S, Alpha = self.pack(x)
        self.MIU[1:self.num_periods + 1] = MIU
        self.S[1:self.num_periods + 1] = S
        self.Alpha[1:self.num_periods + 1] = Alpha
        out = diceTrajectory(self.params, self.MIU, self.S, self.Alpha)
        TATM = out[4]
        if np.any(np.isnan(TATM)):
            return np.ones(self.num_periods) * -1e10
        return TATM[1:] - self.TempLowerConstraint


# %% Model
class Dice2023Model:
    def __init__(self, num_times, scenario):
        self.num_periods = num_times
        self.scenario = scenario
        self.params = LoadParams(num_times)
        self.TempUpperConstraint = 20.0 if scenario == 10 else (1.5 if scenario == 6 else (2.0 if scenario == 7 else 20.0))
        self.TempLowerConstraint = 0.01

    def check_temp_feasibility(self, temp_limit):
        """
        Lowest-temperature path the model allows: maximum abatement every period and the
        lowest allowed savings rate (least capital, output and emissions), with alpha solved
        from the IRF condition. If even this exceeds temp_limit, no control path can meet it
        while capital accumulation holds with equality.
        """
        p = self.params
        MIU_max = p['miuup'].copy(); MIU_max[1] = p['miu1']
        S_low = p['sLBounds'].copy()
        alpha, res = solve_alpha(p, MIU_max, S_low)
        TATM = diceTrajectory(p, MIU_max, S_low, alpha)[4]
        k = int(np.argmax(TATM[1:]))
        return {'feasible': bool(TATM[1:].max() <= temp_limit), 'peak_temp': float(TATM[1:].max()),
                'peak_year': int(p['yr0'] + p['tstep'] * k), 'irf_max_residual': res}

    def run_model(self):
        import time

        # Scenario tweaks (mirror old behavior)
        if self.scenario == 1:  # 1%
            self.params['k0'] = 420
            apply_disc_prstp(self.params, prstp_value=0.01)  # no cap line in GAMS

        elif self.scenario == 2:  # 2%
            self.params['k0'] = 409  # <-- matches GAMS
            apply_disc_prstp(self.params, prstp_value=0.02)  # no cap line

        elif self.scenario == 3:  # 3%
            self.params['k0'] = 370
            apply_disc_prstp(self.params, prstp_value=0.03)  # no cap line

        elif self.scenario == 4:  # 4%
            self.params['k0'] = 326
            # GAMS: cap after t.val>81; exponent = 5*80
            apply_disc_prstp(self.params, prstp_value=0.04,
                             cap_after_t=81, cap_exp_value=5*80)

        elif self.scenario == 5:  # 5%
            self.params['k0'] = 290
            # GAMS: cap after t.val>51; exponent = 5*51
            apply_disc_prstp(self.params, prstp_value=0.05,
                             cap_after_t=51, cap_exp_value=5*51)
        
        elif self.scenario == 6:
            self.TempUpperConstraint = 1.5   # feasibility is checked below before optimizing
        elif self.scenario == 7:
            self.TempUpperConstraint = 2.0
        elif self.scenario == 8:
            for i in range(1, self.num_periods + 1):
                self.params['miuup'][i] = min(0.05 + 0.04 * (i - 1) - 0.01 * max(0, i - 5), self.params['limmiu2070'])
        elif self.scenario == 9:
            self.params['prstp'] = 0.001
        elif self.scenario == 10:
            self.TempUpperConstraint = 15.0
            self.params['miuup'][:] = 1.0

        # Recompute a few derived params that depend on prstp/elasmu when changed
        #self.params['rartp'] = np.exp(self.params['prstp'] + self.params['betaclim'] * self.params['pi']) - 1
        #self.params['optlrsav'] = (self.params['dk'] + 0.004) / (self.params['dk'] + 0.004 * self.params['elasmu'] + self.params['rartp']) * self.params['gama']
        if self.scenario not in {1, 2, 3, 4, 5}:
            self.params['rartp'] = np.exp(self.params['prstp'] + self.params['betaclim'] * self.params['pi']) - 1
            self.params['optlrsav'] = (
                (self.params['dk'] + 0.004) /
                (self.params['dk'] + 0.004 * self.params['elasmu'] + self.params['rartp'])
            ) * self.params['gama']
                    
        prob = DiceFunc(self.num_periods, self.params, TempUpperConstraint=self.TempUpperConstraint, TempLowerConstraint=self.TempLowerConstraint)

        x0 = np.concatenate([
            prob.MIU[1:self.num_periods + 1],
            prob.S[1:self.num_periods + 1],
            prob.Alpha[1:self.num_periods + 1]
        ])

        miu_bounds = []
        for t in range(1, self.num_periods+1):
            if t == 1:
                miu_bounds.append((self.params['miu1'], self.params['miu1']))  # fix MIU(1)
            elif self.scenario == 10 and t > 57:
                miu_bounds.append((1.0, 1.0))
            else:
                miu_bounds.append((self.params['MIULowerBound'], self.params['miuup'][t]))


        s_bounds = [(self.params['sLBounds'][i], self.params['sUBounds'][i]) for i in range(1, self.num_periods + 1)]
        alpha_bounds = [(self.params['AlphaLowerBound'], self.params['AlphaUpperBound']) for _ in range(1, self.num_periods + 1)]

        constraints = [
            {'type': 'eq',   'fun': prob.irf_residual},
            {'type': 'ineq', 'fun': prob.temp_up},
            {'type': 'ineq', 'fun': prob.temp_lo},
        ]
        if self.scenario == 10:
            # Keep as inequality, like the old file
            def cprice_residual(x):
                MIU, _, _ = prob.pack(x)
                PBT = self.params['PBACKTIME'][1:]
                CPR = PBT * (MIU ** (self.params['expcost2'] - 1))
                # cpricebase[t] - CPRICE[t] >= 0  (or eq if you prefer tight fit)
                return np.array([self.params['cpricebase'][t] - CPR[t-1] for t in range(1, min(47, self.num_periods)+1)])
            constraints.append({'type': 'ineq', 'fun': cprice_residual})

        bounds = miu_bounds + s_bounds + alpha_bounds

        def callback(xk):
            obj = prob.objective(xk)
            irf_res = np.max(np.abs(prob.irf_residual(xk)))
            temp_upper = np.min(prob.temp_up(xk))
            temp_lower = np.min(prob.temp_lo(xk))
            print(f"Iteration: Objective={-obj:.2f}, IRF max residual={irf_res:.2e}, "
                  f"Temp Upper min={temp_upper:.2f}, Temp Lower min={temp_lower:.2f}")

        meta = {'scenario': self.scenario, 'infeasible': False}
        start = time.time()

        if self.scenario == 6:
            fcheck = self.check_temp_feasibility(1.5)
            meta['feasibility'] = fcheck
            if not fcheck['feasible']:
                # No control path reaches 1.5°C with equality capital accumulation. The GAMS
                # reference reaches it only by letting capital fall far below its depreciation
                # path (free disposal). Return the path that minimises the exceedance instead.
                print(f"[1.5°C] Infeasible: lowest attainable peak is {fcheck['peak_temp']:.3f}°C "
                      f"({fcheck['peak_year']}). Running best-effort optimisation with a soft penalty.")
                N = self.num_periods
                def penalized(x):
                    MIU, S, A = prob.pack(x)
                    prob.MIU[1:N + 1] = MIU; prob.S[1:N + 1] = S; prob.Alpha[1:N + 1] = A
                    out = diceTrajectory(self.params, prob.MIU, prob.S, prob.Alpha)
                    if np.any(np.isnan(out[4])):
                        return 1e15
                    excess = np.maximum(0.0, out[4][1:] - 1.5)
                    return -out[0] + 1e4 * float(np.sum(excess ** 2))
                x_start = np.concatenate([self.params['miuup'][1:N + 1], prob.S[1:N + 1], prob.Alpha[1:N + 1]])
                x_start[0] = self.params['miu1']
                res = minimize(penalized, x_start, method='SLSQP', bounds=bounds,
                               constraints=[constraints[0], constraints[2]],
                               options={'maxiter': 2000, 'ftol': 1e-7, 'disp': False})
                meta['infeasible'] = True

        if not meta['infeasible']:
            print("Starting optimization...")
            res = minimize(
                prob.objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 25000, 'ftol': 1e-5, 'disp': True, 'eps': 1e-6},
                callback=callback
            )
        print(f"Done in {time.time() - start:.1f}s | Success: {res.success}, Message: {res.message}")

        x_opt = res.x
        output = recoverAllVars(x_opt, self.params)
        meta.update({'success': bool(res.success), 'message': str(res.message),
                     'iterations': int(getattr(res, 'nit', -1)),
                     'irf_max_residual': float(np.max(np.abs(prob.irf_residual(x_opt)))),
                     'peak_temp': float(output[:, 3].max())})

        print("Optimized MIU range:", x_opt[:self.num_periods].min(), x_opt[:self.num_periods].max())
        print("Optimized S range:", x_opt[self.num_periods:2 * self.num_periods].min(), x_opt[self.num_periods:2 * self.num_periods].max())
        print("Optimized Alpha range:", x_opt[2 * self.num_periods:].min(), x_opt[2 * self.num_periods:].max())
        print("IRF max residual:", meta['irf_max_residual'])
        return x_opt, output, meta

    def dump_parameters(self):
        results_dir = RESULTS_DIR
        os.makedirs(results_dir, exist_ok=True)
        path = f"{results_dir}/dice2023_parameters_scen{self.scenario}.csv"
        with open(path, mode="w", newline="") as f:
            writer = csv.writer(f)
            header = [
                "PERIOD", "L", "gA", "aL", "gsig", "sigma", "sigmatot",
                "PBACKTIME", "cost1tot", "eland", "cpricebase",
                "F_Misc", "CO2E_GHGabateB", "emissrat", "RR1", "RR",
                "miuup"
            ]
            writer.writerow(header)
            for t in range(1, self.num_periods + 1):
                writer.writerow([
                    t,
                    self.params['L'][t],
                    self.params['gA'][t],
                    self.params['aL'][t],
                    self.params['gsig'][t],
                    self.params['sigma'][t],
                    self.params['sigmatot'][t],
                    self.params['PBACKTIME'][t],
                    self.params['cost1tot'][t],
                    self.params['eland'][t],
                    self.params['cpricebase'][t],
                    self.params['F_Misc'][t],
                    self.params['CO2E_GHGabateB'][t],
                    self.params['emissrat'][t],
                    self.params['RR1'][t],
                    self.params['RR'][t],
                    self.params['miuup'][t],
                ])
        print(f"Parameters written to {path}")

    def dump_state(self, years, output, filename, scenario):
        results_dir = RESULTS_DIR
        os.makedirs(results_dir, exist_ok=True)
        path = f"{results_dir}/dice2023_state_scen{scenario}.csv"
        with open(path, mode="w", newline='') as f:
            writer = csv.writer(f)
            header = ["EIND", "ECO2", "CO2PPM", "TATM", "Y", "DAMFRAC", "CPC", "CPRICE", "MIUopt", "RSHORT",
                      "ECO2E", "L", "AL", "YGROSS", "K", "Sopt", "I", "YNET", "CCATOT", "CACC",
                      "RES0", "RES1", "RES2", "RES3", "DAMAGES", "ABATECOST", "MCABATE", "C",
                      "PERIODU", "TOTPERIODU", "MAT", "FORC", "TBOX1", "TBOX2", "F_GHGABATE",
                      "IRFT", "ALPHA", "RFACTLONG", "RLONG", "SCC", "ABATERAT", "ATFRAC2020",
                      "ATFRAC1765", "FORC_CO2", "RR"]
            writer.writerow(['IPERIOD'] + header)
            for i in range(self.num_periods):
                writer.writerow([i + 1] + list(output[i, :len(header)]))
        print(f"State written to {path}")

    def plot_figure(self, x, y, xlabel, ylabel, title):
        fig = plt.figure(figsize=(8, 6), dpi=72, facecolor="white")
        plt.plot(x, y)
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=14)
        seaborn.despine()
        fig.tight_layout()
        return fig

    def plot_state_to_file(self, fileName, years, output, x):
        results_dir = RESULTS_DIR
        os.makedirs(results_dir, exist_ok=True)
        num_periods = output.shape[0]
        output_T = np.transpose(output)
        pp = PdfPages(fileName)
        all_columns = [
            (0, "Industrial CO2 Emissions (EIND)", "GtCO2/year"),
            (1, "Total CO2 Emissions (ECO2)", "GtCO2/year"),
            (2, "CO2 Concentration (CO2PPM)", "ppm"),
            (3, "Atmospheric Temperature (TATM)", "°C above 1765"),
            (4, "Net Output (Y)", "Trill USD 2019"),
            (5, "Damage Fraction (DAMFRAC)", "Fraction"),
            (6, "Consumption per Capita (CPC)", "Thousand USD 2019"),
            (7, "Carbon Price (CPRICE)", "USD 2019/tCO2"),
            (8, "Emission Control Rate (MIUopt)", "Rate"),
            (9, "RSHORT", "Unitless"),
            (10, "Total CO2e Emissions (ECO2E)", "GtCO2e/year"),
            (11, "Population (L)", "Millions"),
            (12, "Productivity (AL)", "Unitless"),
            (13, "Gross World Product (YGROSS)", "Trill USD 2019"),
            (14, "Capital (K)", "Trill USD 2019"),
            (15, "Saving Rate (Sopt)", "Rate"),
            (16, "Investment (I)", "Trill USD 2019"),
            (17, "Net Output After Damages (YNET)", "Trill USD 2019"),
            (18, "CCATOT", "GtC"),
            (19, "CACC", "GtC"),
            (20, "RES0", "GtC"),
            (21, "RES1", "GtC"),
            (22, "RES2", "GtC"),
            (23, "RES3", "GtC"),
            (24, "Damages (DAMAGES)", "Trill USD 2019/year"),
            (25, "Abatement Cost (ABATECOST)", "Trill USD 2019/year"),
            (26, "Marginal Abatement Cost (MCABATE)", "USD 2019/tCO2"),
            (27, "Consumption (C)", "Trill USD 2019/year"),
            (28, "PERIODU", "Utility per Period"),
            (29, "TOTPERIODU", "Discounted Utility"),
            (30, "Atmos. Carbon (MAT)", "GtC"),
            (31, "Radiative Forcing (FORC)", "W/m^2"),
            (32, "TBOX1 (Shallow Ocean Temp)", "°C above 1765"),
            (33, "TBOX2 (Deep Ocean Temp)", "°C above 1765"),
            (34, "F_GHGABATE", "W/m^2"),
            (35, "IRFT", "W/m^2"),
            (36, "ALPHA", "Unitless"),
            (37, "RFACTLONG", "Unitless"),
            (38, "RLONG", "Unitless"),
            (39, "Social Cost of Carbon (SCC)", "USD 2019/tCO2"),
            (40, "Abatement Ratio (ABATERAT)", "Fraction"),
            (41, "ATFRAC2020", "Unitless"),
            (42, "ATFRAC1765", "Unitless"),
            (43, "Forcing from CO2 (FORC_CO2)", "W/m^2"),
            (44, "Discount Factor (RR)", "Unitless"),
        ]
        for col_idx, title, ylabel in all_columns:
            fig = self.plot_figure(years, output_T[col_idx], "Years", ylabel, title)
            pp.savefig(fig)
            plt.close(fig)

        MIU = x[:num_periods]
        S = x[num_periods:2 * num_periods]
        ALPHA = x[2 * num_periods:]
        fig_miu = self.plot_figure(years, MIU, "Years", "Rate", "Optimized: Carbon Emission Control Rate (MIU)")
        pp.savefig(fig_miu); plt.close(fig_miu)
        fig_s = self.plot_figure(years, S, "Years", "Rate", "Optimized: Saving Rates (S)")
        pp.savefig(fig_s); plt.close(fig_s)
        fig_alpha = self.plot_figure(years, ALPHA, "Years", "Unitless", "Optimized: ALPHA")
        pp.savefig(fig_alpha); plt.close(fig_alpha)
        pp.close()
        print(f"Wrote plots to {fileName}")


# %% Comparison with GAMS
# ── Comparison with the GAMS reference solutions ─────────────────────────────

# PyDICE scenario number -> GAMS reference file in GAMS_DIR
GAMS_FILES = {
    1: 'dice_scenario_R1.xlsx', 2: 'dice_scenario_R2.xlsx', 3: 'dice_scenario_R3.xlsx',
    4: 'dice_scenario_R4.xlsx', 5: 'dice_scenario_R5.xlsx', 6: 'dice_scenario_T1.5.xlsx',
    7: 'dice_scenario_T2.xlsx', 8: 'dice_scenario_paris.xlsx', 9: 'dice_scenario_opt.xlsx',
    10: 'dice_scenario_base.xlsx',
}
SCENARIO_NAMES = {
    1: '1% discounting', 2: '2% discounting', 3: '3% discounting', 4: '4% discounting',
    5: '5% discounting', 6: '1.5°C limit', 7: '2°C limit', 8: 'Paris', 9: 'Cost-benefit optimal',
    10: 'Baseline',
}

# variable key: (GAMS column prefix, PyDICE output column, label, unit)
COMPARE_VARS = {
    'TATM':    ('Atmospheric temperatur',        3,  'Atmospheric temperature',  '°C'),
    'MIU':     ('Emissions control rate',        8,  'Emissions control rate',   'fraction'),
    'ECO2':    ('Total CO2 Emissions',           1,  'Total CO2 emissions',      'GtCO2/yr'),
    'CO2PPM':  ('Atmospheric concentration C',   2,  'CO2 concentration',        'ppm'),
    'FORC':    ('Total forcings',                31, 'Total forcing',            'W/m²'),
    'SCC':     ('Social cost of carbon',         39, 'Social cost of carbon',    '$/tCO2'),
    'CPRICE':  ('Carbon price',                  7,  'Carbon price',             '$/tCO2'),
    'S':       ('Savings rate',                  15, 'Savings rate',             'fraction'),
    'K':       ('Capital stock',                 14, 'Capital stock',            'trn $'),
    'CPC':     ('Consumption per capita',        6,  'Consumption per capita',   'thousand $'),
    'DAMFRAC': ('Climate damages, fraction',     5,  'Damages',                  'share of output'),
    'ALPHA':   ('Alpha',                         36, 'Carbon-cycle alpha',       '-'),
}


def load_gams(scenario, gams_dir=None):
    """Load the GAMS reference solution for a scenario. Returns a dict of 1-D arrays."""
    gams_dir = GAMS_DIR if gams_dir is None else gams_dir
    path = os.path.join(gams_dir, GAMS_FILES[scenario])
    if not os.path.exists(path):
        raise FileNotFoundError(f"GAMS reference not found: {path}. Set GAMS_DIR to the folder with the .xlsx files.")
    df = pd.read_excel(path, header=0)

    def col(prefix):
        hits = [c for c in df.columns if str(c).strip().lower().startswith(prefix.lower())]
        if not hits:
            raise KeyError(f"Column starting with '{prefix}' not found in {path}")
        return pd.to_numeric(df[hits[0]], errors='coerce').values

    g = {k: col(v[0]) for k, v in COMPARE_VARS.items()}
    g['Year'] = col('Year')
    g['C'] = pd.to_numeric(df[[c for c in df.columns if str(c).strip() == 'Consumption'][0]], errors='coerce').values
    g['I'] = col('Gross investment')
    return g


def _welfare(params, C):
    """PyDICE welfare function evaluated on a consumption path C (trillion $/yr, 0-based)."""
    n = len(C)
    cpc = 1000.0 * C / params['L'][1:n + 1]
    el = params['elasmu']
    U = ((cpc ** (1.0 - el)) - 1.0) / (1.0 - el) - 1.0
    return params['tstep'] * params['scale1'] * np.sum(U * params['L'][1:n + 1] * params['RR'][1:n + 1]) + params['scale2']


def model_only_check(params, gams):
    """
    Run the GAMS control and savings paths through the PyDICE forward model (alpha solved from
    the IRF condition). This separates differences in the model equations from differences
    in how the two optimisers converged. A small value means the equations agree.
    """
    N = params['num_periods']
    MIU = np.zeros(N + 1); S = np.zeros(N + 1)
    MIU[1:] = gams['MIU'][:N]; S[1:] = gams['S'][:N]
    alpha, res = solve_alpha(params, MIU, S)
    TATM = diceTrajectory(params, MIU, S, alpha)[4][1:]
    return TATM, res


def capital_disposal(params, gams, horizon=40):
    """Largest one-period shortfall of GAMS capital below the accumulation equation, as % of the stock."""
    K, I = gams['K'], gams['I']
    implied = (1.0 - params['dk']) ** params['tstep'] * K[:-1] + params['tstep'] * I[:-1]
    gap = implied - K[1:]
    share = 100.0 * gap[:horizon - 1] / np.maximum(implied[:horizon - 1], 1e-12)
    return float(np.max(share)), int(params['yr0'] + params['tstep'] * (np.argmax(share) + 1))


def compare_with_gams(model, output, meta=None, gams_dir=None, results_dir=None,
                      show=True, horizon_year=2200, save=True):
    """
    Compare a finished PyDICE run with the GAMS reference solution for the same scenario.

    Prints a summary table, and draws two figures:
      1. PyDICE (solid) vs GAMS (dashed) for 12 key variables
      2. PyDICE minus GAMS for the same variables
    Both are saved as PDF and PNG in results_dir. Returns (summary dict, list of figures).
    """
    results_dir = RESULTS_DIR if results_dir is None else results_dir
    os.makedirs(results_dir, exist_ok=True)
    p = model.params
    sc = model.scenario
    g = load_gams(sc, gams_dir)
    N = output.shape[0]
    years = p['yr0'] + p['tstep'] * np.arange(N)
    n_h = int(np.searchsorted(years, horizon_year, side='right'))
    i2030, i2050, i2100 = [int(np.searchsorted(years, y)) for y in (2030, 2050, 2100)]
    n2100 = i2100 + 1
    n2215 = min(40, N)

    py = {k: output[:, v[1]] for k, v in COMPARE_VARS.items()}
    T_model_only, irf_res_mo = model_only_check(p, g)
    disposal_pct, disposal_year = capital_disposal(p, g)

    W_py = _welfare(p, output[:, 27])
    W_g = _welfare(p, g['C'][:N])
    summary = {
        'scenario': sc,
        'name': SCENARIO_NAMES[sc],
        'welfare_gap_pct': 100.0 * (W_py - W_g) / abs(W_g),
        'max_dT_to_2100': float(np.max(np.abs(py['TATM'][:n2100] - g['TATM'][:n2100]))),
        'max_dT_to_2215': float(np.max(np.abs(py['TATM'][:n2215] - g['TATM'][:n2215]))),
        'peak_T_pydice': float(py['TATM'][:n2215].max()),
        'peak_T_gams': float(g['TATM'][:n2215].max()),
        'MIU_2030': (float(py['MIU'][i2030]), float(g['MIU'][i2030])),
        'MIU_2050': (float(py['MIU'][i2050]), float(g['MIU'][i2050])),
        'SCC_2025': (float(py['SCC'][1]), float(g['SCC'][1])),
        'SCC_2050': (float(py['SCC'][i2050]), float(g['SCC'][i2050])),
        'ECO2_2030': (float(py['ECO2'][i2030]), float(g['ECO2'][i2030])),
        'model_only_max_dT_to_2215': float(np.max(np.abs(T_model_only[:n2215] - g['TATM'][:n2215]))),
        'gams_capital_disposal_pct': disposal_pct,
        'gams_capital_disposal_year': disposal_year,
        'irf_max_residual': (meta or {}).get('irf_max_residual'),
        'infeasible': (meta or {}).get('infeasible', False),
    }

    # ── printed summary ──
    line = '─' * 64
    print(f"\n{line}\n PyDICE vs GAMS  |  scenario {sc}: {SCENARIO_NAMES[sc]}\n{line}")
    print(f" Welfare gap (PyDICE − GAMS)/|GAMS|      {summary['welfare_gap_pct']:+.4f} %")
    print(f" Max |ΔT| 2020–2100 / 2020–2215          {summary['max_dT_to_2100']:.4f} / {summary['max_dT_to_2215']:.4f} °C")
    print(f" Peak T (to 2215)  PyDICE / GAMS         {summary['peak_T_pydice']:.3f} / {summary['peak_T_gams']:.3f} °C")
    for k, lab in (('MIU_2030', 'Control rate 2030'), ('MIU_2050', 'Control rate 2050'),
                   ('SCC_2025', 'SCC 2025 ($/tCO2)'), ('SCC_2050', 'SCC 2050 ($/tCO2)'),
                   ('ECO2_2030', 'CO2 emissions 2030 (Gt)')):
        a, b = summary[k]
        print(f" {lab:<28}PyDICE / GAMS   {a:9.3f} / {b:9.3f}")
    print(f" Model-only check (GAMS controls in PyDICE): max |ΔT| {summary['model_only_max_dT_to_2215']:.4f} °C")
    if disposal_pct > 1.0:
        print(f" Note: GAMS capital falls {disposal_pct:.1f}% below its accumulation path in {disposal_year}"
              f" (free disposal of capital). PyDICE does not allow this.")
    if summary['infeasible']:
        print(" Note: this scenario is infeasible with equality capital; PyDICE returned a best-effort path.")
    print(line)

    # ── figures ──
    keys = list(COMPARE_VARS.keys())
    yrs = years[:n_h]
    title = (f"Scenario {sc}: {SCENARIO_NAMES[sc]}   |   welfare gap {summary['welfare_gap_pct']:+.4f}%   |   "
             f"max |ΔT| to 2100 {summary['max_dT_to_2100']:.3f}°C   |   peak T {summary['peak_T_pydice']:.2f} vs {summary['peak_T_gams']:.2f}°C")

    fig1, axes = plt.subplots(3, 4, figsize=(18, 11), facecolor='white')
    for ax, k in zip(axes.flat, keys):
        lab, unit = COMPARE_VARS[k][2], COMPARE_VARS[k][3]
        sl = slice(1, n_h) if k == 'SCC' else slice(0, n_h)     # SCC 2020 is not computed
        ax.plot(years[sl], py[k][sl], color='#1f5fa8', lw=2.0, label='PyDICE')
        ax.plot(years[sl], g[k][:N][sl], color='#d1495b', lw=1.6, ls='--', label='GAMS')
        if k == 'TATM' and sc in (6, 7):
            ax.axhline(1.5 if sc == 6 else 2.0, color='0.5', lw=0.8, ls=':')
        ax.set_title(lab, fontsize=11)
        ax.set_ylabel(unit, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=9, frameon=False)
    fig1.suptitle(title, fontsize=12)
    fig1.tight_layout(rect=[0, 0, 1, 0.96])

    fig2, axes2 = plt.subplots(3, 4, figsize=(18, 11), facecolor='white')
    for ax, k in zip(axes2.flat, keys):
        lab, unit = COMPARE_VARS[k][2], COMPARE_VARS[k][3]
        sl = slice(1, n_h) if k == 'SCC' else slice(0, n_h)
        d = py[k][sl] - g[k][:N][sl]
        ax.axhline(0, color='0.4', lw=0.8)
        ax.plot(years[sl], d, color='#2a7f62', lw=1.8)
        if k == 'TATM':
            ax.plot(years[:n_h], T_model_only[:n_h] - g['TATM'][:n_h], color='0.3', lw=1.2, ls=':',
                    label='GAMS controls in PyDICE')
            ax.legend(fontsize=8, frameon=False)
        ax.set_title(f"{lab}: PyDICE − GAMS", fontsize=11)
        ax.set_ylabel(unit, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
    fig2.suptitle(f"Scenario {sc}: {SCENARIO_NAMES[sc]}  |  differences (PyDICE − GAMS)", fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        base = os.path.join(results_dir, f"gams_comparison_scen{sc}")
        with PdfPages(base + '.pdf') as pdf:
            pdf.savefig(fig1); pdf.savefig(fig2)
        fig1.savefig(base + '_levels.png', dpi=110)
        fig2.savefig(base + '_differences.png', dpi=110)
        pd.DataFrame({'Year': yrs, **{f'{k}_pydice': py[k][:n_h] for k in keys},
                      **{f'{k}_gams': g[k][:n_h] for k in keys}}).to_csv(base + '_series.csv', index=False)
        print(f"Comparison saved: {base}.pdf, {base}_levels.png, {base}_differences.png, {base}_series.csv")
    if show:
        plt.show()
    else:
        plt.close(fig1); plt.close(fig2)
    return summary, [fig1, fig2]


def run_scenario(scenario, compare=True, show=True, plots=True):
    """Run one scenario end to end: optimise, save results, and compare with GAMS."""
    model = Dice2023Model(num_times=81, scenario=scenario)
    model.dump_parameters()
    years = np.arange(model.params['yr0'],
                      model.params['yr0'] + model.params['tstep'] * model.num_periods,
                      model.params['tstep'])
    x_opt, output, meta = model.run_model()
    model.dump_state(years, output, None, scenario)
    if plots:
        model.plot_state_to_file(os.path.join(RESULTS_DIR, f"dice2023_plots_scen{scenario}.pdf"), years, output, x_opt)
    summary = None
    if compare:
        summary, _ = compare_with_gams(model, output, meta, show=show)
    return {'model': model, 'x_opt': x_opt, 'output': output, 'meta': meta, 'summary': summary}


# %% Scenario menu
def display_scenarios():
    print("Select a scenario by entering the corresponding number:")
    for i, label in enumerate([
        "1% Discounting (R = 1%)", "2% Discounting", "3% Discounting", "4% Discounting",
        "5% Discounting", "Maximum Temp 1.5°C", "Maximum Temp 2°C", "Paris",
        "Optimal", "Base"
    ], start=1):
        print(f"{i}. {label}")
    while True:
        try:
            choice = int(input("Enter scenario number (1-10): "))
            if 1 <= choice <= 10:
                return choice
        except ValueError:
            pass
        print("Invalid. Please enter a number between 1 and 10.")


# %% Run
if __name__ == "__main__":
    scenario = display_scenarios()
    run_scenario(scenario, compare=True, show=True)
