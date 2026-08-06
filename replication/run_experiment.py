#!/usr/bin/env python3
"""Convergent Abatement, Divergent Prices --- full experiment.
Plain-Python export of convergent_abatement_FULL_experiment.ipynb.
Runs top to bottom; writes all result CSVs to the working directory.
Generated for the replication package.
"""

# ===== cell 1 =====
import numpy as np
import pandas as pd
import os, json, time, traceback, warnings
from datetime import datetime, timedelta
from scipy.optimize import minimize, root
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn
from numba import njit
warnings.filterwarnings('ignore')
seaborn.set(style='ticks')

RESULTS_DIR = './results'
RUNS_DIR    = os.path.join(RESULTS_DIR, 'runs')
os.makedirs(RUNS_DIR, exist_ok=True)

print('All imports OK.')

# ===== cell 2 =====
# damage_type codes:
#   0 = polynomial   DAMFRAC = a1*T + a2base*T^a3
#   1 = Weitzman     DAMFRAC = 1 - 1/(1 + a2base*T^2 + a3b*T^6.754)
#   2 = Dietz-Stern  DAMFRAC = 1 - exp(-gamma_ds * T^2)
import seaborn
seaborn.set(style='ticks')

def set_paper_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family':        'serif',
        'font.size':          9,
        'axes.titlesize':     9.5,
        'axes.labelsize':     9,
        'xtick.labelsize':    8,
        'ytick.labelsize':    8,
        'legend.fontsize':    7.5,
        'figure.dpi':         600,
        'savefig.dpi':        600,
        'axes.linewidth':     0.8,
        'lines.linewidth':    1.7,
        'lines.markersize':   4.2,
        'patch.linewidth':    0.7,
        'xtick.major.width':  0.6,
        'ytick.major.width':  0.6,
        'xtick.major.size':   3.0,
        'ytick.major.size':   3.0,
    })

DAMAGE_REGISTRY = {
    #               (damage_type, a2base,      a3,   a3b,           gamma_ds)
    'nordhaus_2023': (0, 0.003467,   2.0, 0.0,           0.0   ),
    'nordhaus_2016': (0, 0.00267,    2.0, 0.0,           0.0   ),
    'hs_low':        (0, 0.00267,    2.0, 0.0,           0.0   ),
    'hs_central':    (0, 0.00515,    2.0, 0.0,           0.0   ),
    'hs_high':       (0, 0.01012,    2.0, 0.0,           0.0   ),
    'weitzman':      (1, 1/(20.46**2),  2.0, 1/(6.081**6.754),  0.0   ),
    'dietz_stern':   (2, 0.0,        2.0, 0.0,           0.0025),
    'kahn':          (0, 0.0055,     2.0, 0.0,           0.0   ),
}

DAMAGE_NAMES = {
    'nordhaus_2023': 'Nordhaus DICE-2023',
    'nordhaus_2016': 'Nordhaus DICE-2016',
    'hs_low':        'Howard-Sterner Low',
    'hs_central':    'Howard-Sterner Central',
    'hs_high':       'Howard-Sterner High',
    'weitzman':      'Weitzman (2012)',
    'dietz_stern':   'Dietz-Stern (2015)',
    'kahn':          'Kahn et al. (2021)',
}

CMAP = {
    'nordhaus_2016': '#4393c3', 'nordhaus_2023': '#2166ac',
    'hs_low':        '#92c5de', 'hs_central':    '#f4a582',
    'kahn':          '#d6604d', 'weitzman':      '#b2182b',
    'hs_high':       '#e08214', 'dietz_stern':   '#762a83',
}

BW_FILL = {
    'dietz_stern':   '#f2f2f2',
    'nordhaus_2016': '#d9d9d9',
    'hs_low':        '#c7c7c7',
    'weitzman':      '#b0b0b0',
    'nordhaus_2023': '#969696',
    'hs_central':    '#737373',
    'kahn':          '#525252',
    'hs_high':       '#252525',
}

BW_HATCH = {
    'dietz_stern':   '////',
    'nordhaus_2016': '....',
    'hs_low':        'xxxx',
    'weitzman':      '++++',
    'nordhaus_2023': '||||',
    'hs_central':    '----',
    'kahn':          '\\\\\\\\',
    'hs_high':       'oooo',
}

BW_LINE = {
    'dietz_stern':   {'ls': '-',   'lw': 1.7, 'marker': 'o',  'ms': 4.0},  # circle
    'nordhaus_2016': {'ls': '--',  'lw': 1.7, 'marker': 's',  'ms': 4.0},  # square
    'hs_low':        {'ls': ':',   'lw': 1.9, 'marker': '^',  'ms': 4.0},  # triangle up
    'weitzman':      {'ls': '-.',  'lw': 1.9, 'marker': 'D',  'ms': 4.0},  # diamond
    'nordhaus_2023': {'ls': '-',   'lw': 2.2, 'marker': 'v',  'ms': 4.5},  # triangle down
    'hs_central':    {'ls': '--',  'lw': 1.9, 'marker': 'P',  'ms': 4.0},  # plus filled
    'kahn':          {'ls': ':',   'lw': 1.9, 'marker': '*',  'ms': 5.0},  # star
    'hs_high':       {'ls': '-.',  'lw': 2.2, 'marker': 'X',  'ms': 4.5},  # x filled
}

def set_damage_function(params, key):
    if key not in DAMAGE_REGISTRY:
        raise KeyError(f"Unknown damage function '{key}'. Available: {list(DAMAGE_REGISTRY)}")
    dtype, a2base, a3, a3b, gamma_ds = DAMAGE_REGISTRY[key]
    params['damage_type'] = int(dtype)
    params['a2base']      = float(a2base)
    params['a3']          = float(a3)
    params['a3b']         = float(a3b)
    params['gamma_ds']    = float(gamma_ds)
    params['damage_key']  = key
    return params

# Sanity check — damage % at key temps
temps = np.array([1., 2., 3., 4., 5., 6.])
rows  = []
for key, (dtype, a2base, a3, a3b, gamma_ds) in DAMAGE_REGISTRY.items():
    if dtype == 0:   dam = a2base * temps**a3
    elif dtype == 1: dam = 1 - 1/(1 + a2base*temps**2 + a3b*temps**6.754)
    else:            dam = 1 - np.exp(-gamma_ds * np.minimum(temps, 10)**2)
    rows.append([DAMAGE_NAMES[key]] + list((dam*100).round(2)))
df_ref = pd.DataFrame(rows, columns=['Function','1°C','2°C','3°C','4°C','5°C','6°C'])
display(df_ref)
print(f"Range at 3°C: {df_ref['3°C'].min():.2f}% – {df_ref['3°C'].max():.2f}%  "
      f"({df_ref['3°C'].max()/df_ref['3°C'].min():.1f}× spread)")

# ===== cell 3 =====
def LoadParams(num_periods=81, **kwargs):
    params = {}
    params['num_periods'] = num_periods
    params['tstep']       = kwargs.get('tstep', 5)
    params['gama']        = kwargs.get('gama', 0.300)
    params['pop1']        = kwargs.get('pop1', 7752.9)
    params['popadj']      = kwargs.get('popadj', 0.145)
    params['popasym']     = kwargs.get('popasym', 10825)
    params['dk']          = kwargs.get('dk', 0.100)
    params['q1']          = kwargs.get('q1', 135.7)
    params['AL1']         = kwargs.get('AL1', 5.84)
    params['gA1']         = kwargs.get('gA1', 0.066)
    params['delA']        = kwargs.get('delA', 0.0015)
    params['gsigma1']     = kwargs.get('gsigma1', -0.015)
    params['delgsig']     = kwargs.get('delgsig', 0.96)
    params['asymgsig']    = kwargs.get('asymgsig', -0.005)
    params['e1']          = kwargs.get('e1', 37.56)
    params['miu1']        = kwargs.get('miu1', 0.05)
    params['fosslim']     = kwargs.get('fosslim', 6000)
    params['CumEmiss0']   = kwargs.get('CumEmiss0', 633.5)
    params['a1']          = kwargs.get('a1', 0)
    params['a2base']      = kwargs.get('a2base', 0.003467)
    params['a3']          = kwargs.get('a3', 2.00)
    params['damage_type'] = int(kwargs.get('damage_type', 0))
    params['a3b']         = float(kwargs.get('a3b', 0.0000050703))
    params['gamma_ds']    = float(kwargs.get('gamma_ds', 0.0025))
    params['damage_key']  = kwargs.get('damage_key', 'nordhaus_2023')
    params['expcost2']    = kwargs.get('expcost2', 2.6)
    params['pback2050']   = kwargs.get('pback2050', 515)
    params['gback']       = kwargs.get('gback', -0.012)
    params['cprice1']     = kwargs.get('cprice1', 6)
    params['gcprice']     = kwargs.get('gcprice', 0.025)
    params['limmiu2070']  = kwargs.get('limmiu2070', 1.0)
    params['limmiu2120']  = kwargs.get('limmiu2120', 1.1)
    params['limmiu2200']  = kwargs.get('limmiu2200', 1.05)
    params['limmiu2300']  = kwargs.get('limmiu2300', 1.0)
    params['delmiumax']   = kwargs.get('delmiumax', 0.12)
    params['betaclim']    = kwargs.get('betaclim', 0.5)
    params['elasmu']      = kwargs.get('elasmu', 0.95)
    params['prstp']       = kwargs.get('prstp', 0.001)
    params['pi']          = kwargs.get('pi', 0.05)
    params['k0']          = kwargs.get('k0', 295)
    params['siggc1']      = kwargs.get('siggc1', 0.01)
    params['SRF']         = kwargs.get('SRF', 1e6)
    params['scale1']      = kwargs.get('scale1', 0.00891061)
    params['scale2']      = kwargs.get('scale2', -6275.91)
    params['eland0']      = kwargs.get('eland0', 5.9)
    params['deland']      = kwargs.get('deland', 0.1)
    params['F_Misc2020']       = kwargs.get('F_Misc2020', -0.054)
    params['F_Misc2100']       = kwargs.get('F_Misc2100', 0.265)
    params['F_GHGabate2020']   = kwargs.get('F_GHGabate2020', 0.518)
    params['F_GHGabate2100']   = kwargs.get('F_GHGabate2100', 0.957)
    params['ECO2eGHGB2020']    = kwargs.get('ECO2eGHGB2020', 9.96)
    params['ECO2eGHGB2100']    = kwargs.get('ECO2eGHGB2100', 15.5)
    params['emissrat2020']     = kwargs.get('emissrat2020', 1.40)
    params['emissrat2100']     = kwargs.get('emissrat2100', 1.21)
    params['Fcoef1']      = kwargs.get('Fcoef1', 0.00955)
    params['Fcoef2']      = kwargs.get('Fcoef2', 0.861)
    params['yr0']         = kwargs.get('yr0', 2020)
    params['emshare0']    = kwargs.get('emshare0', 0.2173)
    params['emshare1']    = kwargs.get('emshare1', 0.224)
    params['emshare2']    = kwargs.get('emshare2', 0.2824)
    params['emshare3']    = kwargs.get('emshare3', 0.2763)
    params['tau0']        = kwargs.get('tau0', 1e6)
    params['tau1']        = kwargs.get('tau1', 394.4)
    params['tau2']        = kwargs.get('tau2', 36.53)
    params['tau3']        = kwargs.get('tau3', 4.304)
    params['teq1']        = kwargs.get('teq1', 0.324)
    params['teq2']        = kwargs.get('teq2', 0.44)
    params['d1']          = kwargs.get('d1', 236)
    params['d2']          = kwargs.get('d2', 4.07)
    params['irf0']        = kwargs.get('irf0', 32.4)
    params['irC']         = kwargs.get('irC', 0.019)
    params['irT']         = kwargs.get('irT', 4.165)
    params['fco22x']      = kwargs.get('fco22x', 3.93)
    params['mat0']        = kwargs.get('mat0', 886.5128014)
    params['res00']       = kwargs.get('res00', 150.093)
    params['res10']       = kwargs.get('res10', 102.698)
    params['res20']       = kwargs.get('res20', 39.534)
    params['res30']       = kwargs.get('res30', 6.1865)
    params['mateq']       = kwargs.get('mateq', 588)
    params['tbox10']      = kwargs.get('tbox10', 0.1477)
    params['tbox20']      = kwargs.get('tbox20', 1.099454)
    params['tatm0']       = kwargs.get('tatm0', 1.24715)
    params['SLower']      = kwargs.get('SLower', 0.0)
    params['SUpper']      = kwargs.get('SUpper', 1.0)
    params['FixSperiod']  = kwargs.get('FixSperiod', 38)
    params['FixSvalue']   = kwargs.get('FixSvalue', 0.28)
    params['AlphaUpperBound'] = kwargs.get('AlphaUpperBound', np.inf)
    params['AlphaLowerBound'] = kwargs.get('AlphaLowerBound', 0.2)
    params['MIULowerBound']   = kwargs.get('MIULowerBound', 0)

    def irf_eq(a0):
        LHS = (params['irf0']
               + params['irC'] * (params['CumEmiss0'] - (params['mat0'] - params['mateq']))
               + params['irT'] * params['tatm0'])
        RHS = (a0*params['emshare0']*params['tau0']*(1-np.exp(-100/(a0*params['tau0'])))
               +a0*params['emshare1']*params['tau1']*(1-np.exp(-100/(a0*params['tau1'])))
               +a0*params['emshare2']*params['tau2']*(1-np.exp(-100/(a0*params['tau2'])))
               +a0*params['emshare3']*params['tau3']*(1-np.exp(-100/(a0*params['tau3']))))
        return LHS - RHS
    sol = root(irf_eq, 0.5, method='hybr', options={'xtol': 1e-12})
    params['a0'] = float(sol.x[0])

    params['rartp']    = np.exp(params['prstp'] + params['betaclim']*params['pi']) - 1
    params['sig1']     = params['e1'] / (params['q1'] * (1 - params['miu1']))
    params['optlrsav'] = ((params['dk']+0.004) /
                          (params['dk']+0.004*params['elasmu']+params['rartp'])) * params['gama']

    N = num_periods
    for k in ['L','aL','sigma','sigmatot','gA','gsig','eland','cost1tot',
              'PBACKTIME','cpricebase','varpcc','rprecaut','RR1','RR',
              'CO2E_GHGabateB','F_Misc','emissrat']:
        params[k] = np.zeros(N + 1)
    params['L'][1]  = params['pop1']
    params['aL'][1] = params['AL1']
    params['sigma'][1] = params['sig1']

    for t in range(1, N + 1):
        params['varpcc'][t]   = min(params['siggc1']**2*5*(t-1), params['siggc1']**2*5*47)
        params['rprecaut'][t] = -0.5*params['varpcc'][t]*params['elasmu']**2
        params['RR1'][t]      = 1/((1+params['rartp'])**(params['tstep']*(t-1)))
        params['RR'][t]       = params['RR1'][t]*(1+params['rprecaut'][t])**(-params['tstep']*(t-1))
        params['gA'][t]       = params['gA1']*np.exp(-params['delA']*5*(t-1))
        params['cpricebase'][t] = params['cprice1']*(1+params['gcprice'])**(5*(t-1))
        params['PBACKTIME'][t]  = params['pback2050']*np.exp(
            -0.05*(t-7) if t<=7 else -0.005*(t-7))
        params['gsig'][t]     = min(params['gsigma1']*params['delgsig']**(t-1), params['asymgsig'])
        params['eland'][t]    = params['eland0']*(1-params['deland'])**(t-1)
        if t <= 16:
            params['CO2E_GHGabateB'][t] = params['ECO2eGHGB2020']+((params['ECO2eGHGB2100']-params['ECO2eGHGB2020'])/16)*(t-1)
            params['F_Misc'][t]         = params['F_Misc2020']+((params['F_Misc2100']-params['F_Misc2020'])/16)*(t-1)
            params['emissrat'][t]       = params['emissrat2020']+((params['emissrat2100']-params['emissrat2020'])/16)*(t-1)
        else:
            params['CO2E_GHGabateB'][t] = params['ECO2eGHGB2100']
            params['F_Misc'][t]         = params['F_Misc2100']
            params['emissrat'][t]       = params['emissrat2100']
        params['sigmatot'][t]  = params['sigma'][t]*params['emissrat'][t]
        params['cost1tot'][t]  = params['PBACKTIME'][t]*params['sigmatot'][t]/params['expcost2']/1000
        if t < N:
            params['L'][t+1]     = params['L'][t]*(params['popasym']/params['L'][t])**params['popadj']
            params['aL'][t+1]    = params['aL'][t]/(1-params['gA'][t])
            params['sigma'][t+1] = params['sigma'][t]*np.exp(5*params['gsig'][t])

    params['miuup'] = np.zeros(N + 1)
    params['miuup'][1] = 0.05; params['miuup'][2] = 0.10
    for t in range(3, N+1):
        if   t <= 8:  params['miuup'][t] = params['delmiumax']*(t-1)
        elif t <= 11: params['miuup'][t] = 0.85+0.05*(t-8)
        elif t <= 20: params['miuup'][t] = params['limmiu2070']
        elif t <= 37: params['miuup'][t] = params['limmiu2120']
        elif t <= 57: params['miuup'][t] = params['limmiu2200']
        else:         params['miuup'][t] = params['limmiu2300']

    params['sLBounds'] = np.full(N+1, params['SLower'])
    params['sUBounds'] = np.full(N+1, params['SUpper'])
    if params['FixSperiod'] <= N:
        params['sLBounds'][params['FixSperiod']:] = params['FixSvalue']
        params['sUBounds'][params['FixSperiod']:] = params['FixSvalue']

    params['eco2Param'] = params['aL'] * (params['L']/1000)**(1-params['gama'])
    return params

#changed this
def apply_disc_prstp(params, prstp_value, *, elasmu_value=0.95,
                     cap_after_t=None, cap_exp_value=None):
    # NOTE: sets rartp = prstp, dropping betaclim*pi risk premium.
    # Intentional for Group B discount sensitivity experiments.
    params['prstp']  = float(prstp_value)
    params['elasmu'] = float(elasmu_value)
    N, step = int(params['num_periods']), int(params['tstep'])
    RR1 = np.zeros(N+1); RR = np.zeros(N+1)
    for t in range(1, N+1):
        exp_pow = step*(t-1)
        if cap_after_t is not None and cap_exp_value is not None and t > cap_after_t:
            exp_pow = cap_exp_value
        RR1[t] = 1.0/((1.0+params['prstp'])**exp_pow)
        RR[t]  = RR1[t]
    params['RR1'] = RR1; params['RR'] = RR
    if 'rprecaut' in params and params['rprecaut'].size == (N+1):
        params['rprecaut'][:] = 0.0
    params['rartp']    = params['prstp']
    params['optlrsav'] = ((params['dk']+0.004) /
                          (params['dk']+0.004*params['elasmu']+params['prstp'])) * params['gama']
    return params

print('LoadParams and apply_disc_prstp defined.')

# ===== cell 4 =====
@njit(cache=True)
def _damfrac(T, a1, a2base, a3, damage_type, a3b, gamma_ds):
    if damage_type == 1:          # Weitzman
        return 1.0 - 1.0/(1.0 + a2base*T**2 + a3b*T**6.754)
    elif damage_type == 2:        # Dietz-Stern
        return 1.0 - np.exp(-gamma_ds * min(T, 10.0)**2)
    else:                         # polynomial
        return a1*T + a2base*T**a3


@njit
def diceForward_numba(i, MIU, S, alpha, CCATOT, K, I, F_GHGabate,
                      RES0, RES1, RES2, RES3, TBOX1, TBOX2,
                      tstep, dk, gama, eco2Param, sigma, eland, cost1tot,
                      expcost2, miuup, sLBounds, sUBounds, AlphaLowerBound,
                      Fcoef1, Fcoef2, CO2E_GHGabateB,
                      emshare0, emshare1, emshare2, emshare3,
                      tau0, tau1, tau2, tau3, mateq, fco22x, F_Misc,
                      teq1, teq2, d1, d2, a1, a2base, a3,
                      damage_type, a3b, gamma_ds,
                      pulse_GtCO2_per_year):
    MIU_i   = min(max(MIU[i], 0.0), miuup[i])
    S_i     = min(max(S[i], sLBounds[i]), sUBounds[i])
    alpha_i = max(alpha[i], AlphaLowerBound)
    K       = (1.0-dk)**tstep * K + tstep*I
    YGROSS  = eco2Param[i] * (K**gama)
    ECO2    = (sigma[i]*YGROSS)*(1.0-MIU_i) + eland[i]
    CCATOT  = CCATOT + (ECO2+pulse_GtCO2_per_year[i])*(tstep/3.667)
    F_GHGabate = Fcoef2*F_GHGabate + Fcoef1*CO2E_GHGabateB[i]*(1.0-MIU_i)
    inflow  = (ECO2+pulse_GtCO2_per_year[i])/3.667
    RES0 = emshare0*tau0*alpha_i*inflow*(1-np.exp(-tstep/(tau0*alpha_i))) + RES0*np.exp(-tstep/(tau0*alpha_i))
    RES1 = emshare1*tau1*alpha_i*inflow*(1-np.exp(-tstep/(tau1*alpha_i))) + RES1*np.exp(-tstep/(tau1*alpha_i))
    RES2 = emshare2*tau2*alpha_i*inflow*(1-np.exp(-tstep/(tau2*alpha_i))) + RES2*np.exp(-tstep/(tau2*alpha_i))
    RES3 = emshare3*tau3*alpha_i*inflow*(1-np.exp(-tstep/(tau3*alpha_i))) + RES3*np.exp(-tstep/(tau3*alpha_i))
    MAT   = mateq + RES0+RES1+RES2+RES3
    FORC  = fco22x*np.log(MAT/mateq)/np.log(2.0) + F_Misc[i] + F_GHGabate
    TBOX1 = TBOX1*np.exp(-tstep/d1) + teq1*FORC*(1-np.exp(-tstep/d1))
    TBOX2 = TBOX2*np.exp(-tstep/d2) + teq2*FORC*(1-np.exp(-tstep/d2))
    TATM  = min(max(TBOX1+TBOX2, 0.01), 20.0)
    DAMFRAC   = _damfrac(TATM, a1, a2base, a3, damage_type, a3b, gamma_ds)
    ABATECOST = YGROSS*cost1tot[i]*(MIU_i**expcost2)
    YNET = YGROSS*(1.0-DAMFRAC)
    Y    = YNET - ABATECOST
    I    = S_i*Y;  C = Y-I
    if not (np.isfinite(MAT) and np.isfinite(TATM) and np.isfinite(C) and np.isfinite(YGROSS)):
        return np.array([np.nan]*18, dtype=np.float64)
    return np.array([C,CCATOT,K,I,F_GHGabate,RES0,RES1,RES2,RES3,
                     TBOX1,TBOX2,MAT,TATM,Y,YNET,YGROSS,DAMFRAC,ABATECOST], dtype=np.float64)


@njit
def diceTrajectory_numba(MIU, S, alpha, num_periods,
                         tstep, dk, gama, eco2Param, sigma, eland, cost1tot,
                         expcost2, miuup, sLBounds, sUBounds, AlphaLowerBound,
                         Fcoef1, Fcoef2, CO2E_GHGabateB,
                         emshare0, emshare1, emshare2, emshare3,
                         tau0, tau1, tau2, tau3, mateq, fco22x, F_Misc,
                         teq1, teq2, d1, d2, a1, a2base, a3,
                         res00, res10, res20, res30, tbox10, tbox20,
                         CumEmiss0, F_GHGabate2020, k0, tatm0, mat0, a0,
                         L, RR, scale1, scale2, elasmu,
                         pulse_GtCO2_per_year, damage_type, a3b, gamma_ds):
    C=np.zeros(num_periods+1); K=np.zeros(num_periods+1)
    CCATOT=np.zeros(num_periods+1); MAT=np.zeros(num_periods+1)
    TATM=np.zeros(num_periods+1); Y=np.zeros(num_periods+1)
    YNET=np.zeros(num_periods+1); YGROSS=np.zeros(num_periods+1)
    DAMFRAC=np.zeros(num_periods+1); ABATECOST=np.zeros(num_periods+1)
    RES0_a=np.zeros(num_periods+1); RES1_a=np.zeros(num_periods+1)
    RES2_a=np.zeros(num_periods+1); RES3_a=np.zeros(num_periods+1)
    TB1_a=np.zeros(num_periods+1);  TB2_a=np.zeros(num_periods+1)
    FG_a=np.zeros(num_periods+1)
    RES0=res00;RES1=res10;RES2=res20;RES3=res30
    TBOX1=tbox10;TBOX2=tbox20
    CCATOT[1]=CumEmiss0; F_GHGabate=F_GHGabate2020
    K[1]=k0; TATM[1]=tatm0; MAT[1]=mat0
    DAMFRAC[1]  = _damfrac(TATM[1], a1, a2base, a3, damage_type, a3b, gamma_ds)
    YGROSS[1]   = eco2Param[1]*(K[1]**gama)
    YNET[1]     = YGROSS[1]*(1.0-DAMFRAC[1])
    ABATECOST[1]= YGROSS[1]*cost1tot[1]*(MIU[1]**expcost2)
    Y[1]        = YNET[1]-ABATECOST[1]
    I           = S[1]*Y[1]; C[1]=Y[1]-I
    RES0_a[1]=RES0;RES1_a[1]=RES1;RES2_a[1]=RES2;RES3_a[1]=RES3
    TB1_a[1]=TBOX1;TB2_a[1]=TBOX2;FG_a[1]=F_GHGabate
    for i in range(2, num_periods+1):
        r = diceForward_numba(i,MIU,S,alpha,CCATOT[i-1],K[i-1],I,F_GHGabate,
                              RES0,RES1,RES2,RES3,TBOX1,TBOX2,
                              tstep,dk,gama,eco2Param,sigma,eland,cost1tot,
                              expcost2,miuup,sLBounds,sUBounds,AlphaLowerBound,
                              Fcoef1,Fcoef2,CO2E_GHGabateB,
                              emshare0,emshare1,emshare2,emshare3,
                              tau0,tau1,tau2,tau3,mateq,fco22x,F_Misc,
                              teq1,teq2,d1,d2,a1,a2base,a3,
                              damage_type,a3b,gamma_ds,pulse_GtCO2_per_year)
        if np.any(np.isnan(r)):
            nan=np.full(num_periods+1,np.nan)
            return(np.nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan)
        C[i]=r[0];CCATOT[i]=r[1];K[i]=r[2];I=r[3];F_GHGabate=r[4]
        RES0=r[5];RES1=r[6];RES2=r[7];RES3=r[8]
        TBOX1=r[9];TBOX2=r[10];MAT[i]=r[11];TATM[i]=r[12]
        Y[i]=r[13];YNET[i]=r[14];YGROSS[i]=r[15];DAMFRAC[i]=r[16];ABATECOST[i]=r[17]
        RES0_a[i]=RES0;RES1_a[i]=RES1;RES2_a[i]=RES2;RES3_a[i]=RES3
        TB1_a[i]=TBOX1;TB2_a[i]=TBOX2;FG_a[i]=F_GHGabate
    PERIODU    = ((C[1:]*1000.0/L[1:])**(1.0-elasmu)-1.0)/(1.0-elasmu)-1.0
    TOTPERIODU = PERIODU*L[1:]*RR[1:]
    UTILITY    = tstep*scale1*np.sum(TOTPERIODU)+scale2
    return(UTILITY,C,CCATOT,MAT,TATM,K,Y,YNET,YGROSS,DAMFRAC,ABATECOST,
           RES0_a,RES1_a,RES2_a,RES3_a,TB1_a,TB2_a,FG_a)


def _traj_args(params, pulse=None):
    """Pack params dict into the positional args diceTrajectory_numba expects."""
    p = params
    if pulse is None:
        pulse = np.zeros(p['num_periods']+1, dtype=np.float64)
    return (p['num_periods'],p['tstep'],p['dk'],p['gama'],p['eco2Param'],
            p['sigma'],p['eland'],p['cost1tot'],p['expcost2'],p['miuup'],
            p['sLBounds'],p['sUBounds'],p['AlphaLowerBound'],
            p['Fcoef1'],p['Fcoef2'],p['CO2E_GHGabateB'],
            p['emshare0'],p['emshare1'],p['emshare2'],p['emshare3'],
            p['tau0'],p['tau1'],p['tau2'],p['tau3'],
            p['mateq'],p['fco22x'],p['F_Misc'],p['teq1'],p['teq2'],
            p['d1'],p['d2'],p['a1'],p['a2base'],p['a3'],
            p['res00'],p['res10'],p['res20'],p['res30'],
            p['tbox10'],p['tbox20'],p['CumEmiss0'],p['F_GHGabate2020'],
            p['k0'],p['tatm0'],p['mat0'],p['a0'],
            p['L'],p['RR'],p['scale1'],p['scale2'],p['elasmu'],
            pulse, p['damage_type'],p['a3b'],p['gamma_ds'])


def diceTrajectory(params, MIU, S, alpha, pulse=None):
    MIU   = np.ascontiguousarray(MIU,   dtype=np.float64)
    S     = np.ascontiguousarray(S,     dtype=np.float64)
    alpha = np.ascontiguousarray(alpha, dtype=np.float64).copy()
    MIU[1]=params['miu1']; alpha[1]=params['a0']
    if pulse is not None:
        pulse = np.ascontiguousarray(pulse, dtype=np.float64)
    return diceTrajectory_numba(MIU, S, alpha, *_traj_args(params, pulse))


@njit(cache=False)
def compute_SCC_numba(MIU, S, alpha, num_periods,
                      tstep,dk,gama,eco2Param,sigma,eland,cost1tot,
                      expcost2,miuup,sLBounds,sUBounds,AlphaLowerBound,
                      Fcoef1,Fcoef2,CO2E_GHGabateB,
                      emshare0,emshare1,emshare2,emshare3,
                      tau0,tau1,tau2,tau3,mateq,fco22x,F_Misc,
                      teq1,teq2,d1,d2,a1,a2base,a3,
                      res00,res10,res20,res30,tbox10,tbox20,
                      CumEmiss0,F_GHGabate2020,k0,tatm0,mat0,a0,
                      L,RR,scale1,scale2,elasmu,
                      damage_type,a3b,gamma_ds):
    pz = np.zeros(num_periods+1, dtype=np.float64)
    args = (num_periods,tstep,dk,gama,eco2Param,sigma,eland,cost1tot,
            expcost2,miuup,sLBounds,sUBounds,AlphaLowerBound,
            Fcoef1,Fcoef2,CO2E_GHGabateB,
            emshare0,emshare1,emshare2,emshare3,
            tau0,tau1,tau2,tau3,mateq,fco22x,F_Misc,
            teq1,teq2,d1,d2,a1,a2base,a3,
            res00,res10,res20,res30,tbox10,tbox20,
            CumEmiss0,F_GHGabate2020,k0,tatm0,mat0,a0,
            L,RR,scale1,scale2,elasmu)
    base = diceTrajectory_numba(MIU,S,alpha,*args,pz,damage_type,a3b,gamma_ds)
    if np.isnan(base[0]): return np.zeros(num_periods+1)
    C_b    = base[1]
    CPC_b  = 1000.0*C_b[1:]/L[1:]
    W0     = np.sum(((CPC_b**(1.0-elasmu)-1.0)/(1.0-elasmu)-1.0)*L[1:]*RR[1:])
    lam    = (CPC_b**(-elasmu))*RR[1:]
    SCC    = np.zeros(num_periods+1)
    ph=5e-6; ph2=ph/2.0
    for i in range(2, num_periods+1):
        pa=np.zeros(num_periods+1,dtype=np.float64); pa[i]=ph
        pb=np.zeros(num_periods+1,dtype=np.float64); pb[i]=ph2
        ah=diceTrajectory_numba(MIU,S,alpha,*args,pa,damage_type,a3b,gamma_ds)
        ab=diceTrajectory_numba(MIU,S,alpha,*args,pb,damage_type,a3b,gamma_ds)
        if np.isnan(ah[0]) or np.isnan(ab[0]): continue
        def W(res):
            c=1000.0*res[1][1:]/L[1:]
            return np.sum(((c**(1.0-elasmu)-1.0)/(1.0-elasmu)-1.0)*L[1:]*RR[1:])
        dh=W(ah)-W0; db=W(ab)-W0
        if abs(lam[i-1])>1e-12:
            SCC[i]=(4.0*(-(db/ph2)/lam[i-1]) - (-(dh/ph)/lam[i-1]))/3.0
    SCC[1]=0.0
    return SCC


@njit
def recoverAllVars_numba(x, num_periods,
                         tstep,dk,gama,eco2Param,sigma,eland,cost1tot,
                         expcost2,miuup,sLBounds,sUBounds,AlphaLowerBound,
                         Fcoef1,Fcoef2,CO2E_GHGabateB,
                         emshare0,emshare1,emshare2,emshare3,
                         tau0,tau1,tau2,tau3,mateq,fco22x,F_Misc,
                         teq1,teq2,d1,d2,a1,a2base,a3,
                         res00,res10,res20,res30,tbox10,tbox20,
                         CumEmiss0,F_GHGabate2020,k0,tatm0,mat0,a0,
                         L,RR,scale1,scale2,elasmu,
                         PBACKTIME,irf0,irC,irT,SRF,
                         damage_type,a3b,gamma_ds):
    MIU=np.zeros(num_periods+1); S=np.zeros(num_periods+1); alpha=np.zeros(num_periods+1)
    MIU[1:]=x[:num_periods]; S[1:]=x[num_periods:2*num_periods]; alpha[1:]=x[2*num_periods:]
    alpha[1]=a0
    pz=np.zeros(num_periods+1,dtype=np.float64)
    args=(num_periods,tstep,dk,gama,eco2Param,sigma,eland,cost1tot,
          expcost2,miuup,sLBounds,sUBounds,AlphaLowerBound,
          Fcoef1,Fcoef2,CO2E_GHGabateB,
          emshare0,emshare1,emshare2,emshare3,
          tau0,tau1,tau2,tau3,mateq,fco22x,F_Misc,
          teq1,teq2,d1,d2,a1,a2base,a3,
          res00,res10,res20,res30,tbox10,tbox20,
          CumEmiss0,F_GHGabate2020,k0,tatm0,mat0,a0,
          L,RR,scale1,scale2,elasmu)
    result=diceTrajectory_numba(MIU,S,alpha,*args,pz,damage_type,a3b,gamma_ds)
    if np.isnan(result[0]): return np.zeros((num_periods,46))
    C=result[1];CCATOT=result[2];MAT=result[3];TATM=result[4];K=result[5]
    Y=result[6];YNET=result[7];YGROSS=result[8];DAMFRAC=result[9];ABATECOST=result[10]
    R0=result[11];R1=result[12];R2=result[13];R3=result[14]
    TB1=result[15];TB2=result[16];FG=result[17]
    sMAT=np.maximum(MAT,mateq+1e-6)
    I=S*Y; DAMAGES=YGROSS*DAMFRAC; CPRICE=PBACKTIME*(MIU**(expcost2-1.0))
    EIND=sigma*(eco2Param*(K**gama))*(1.0-MIU)
    ECO2=EIND+eland; ECO2E=ECO2+CO2E_GHGabateB*(1.0-MIU)
    FORC=fco22x*np.log(sMAT/mateq)/np.log(2.0)+F_Misc+FG
    FORC_CO2=fco22x*np.log(sMAT/mateq)/np.log(2.0)
    CPC=np.zeros(num_periods+1)
    for t in range(1,num_periods+1): CPC[t]=1000.0*C[t]/L[t]
    RF=np.full(num_periods+1,SRF)
    for i in range(2,num_periods+1): RF[i]=SRF*(CPC[i-1]/CPC[1])**(-elasmu)*RR[i]
    RL=np.zeros(num_periods+1); RS=np.zeros(num_periods+1)
    for i in range(2,num_periods+1):
        RL[i]=-np.log(RF[i]/SRF)/(5.0*(i-1))
        RS[i]=-np.log(RF[i]/RF[i-1])/5.0
    PU=np.zeros(num_periods+1); TPU=np.zeros(num_periods+1)
    for t in range(1,num_periods+1):
        PU[t]=((C[t]*1000.0/L[t])**(1.0-elasmu)-1.0)/(1.0-elasmu)-1.0
        TPU[t]=PU[t]*L[t]*RR[t]
    IRFt=irf0+irC*(CCATOT-(MAT-mateq))+irT*TATM
    SCC=compute_SCC_numba(MIU,S,alpha,num_periods,
                          tstep,dk,gama,eco2Param,sigma,eland,cost1tot,
                          expcost2,miuup,sLBounds,sUBounds,AlphaLowerBound,
                          Fcoef1,Fcoef2,CO2E_GHGabateB,
                          emshare0,emshare1,emshare2,emshare3,
                          tau0,tau1,tau2,tau3,mateq,fco22x,F_Misc,
                          teq1,teq2,d1,d2,a1,a2base,a3,
                          res00,res10,res20,res30,tbox10,tbox20,
                          CumEmiss0,F_GHGabate2020,k0,tatm0,mat0,a0,
                          L,RR,scale1,scale2,elasmu,damage_type,a3b,gamma_ds)
    out=np.zeros((num_periods,46))
    for i in range(num_periods):
        t=i+1
        out[i,0]=EIND[t];out[i,1]=ECO2[t];out[i,2]=MAT[t]/2.13;out[i,3]=TATM[t]
        out[i,4]=Y[t];out[i,5]=DAMFRAC[t];out[i,6]=CPC[t];out[i,7]=CPRICE[t]
        out[i,8]=MIU[t];out[i,9]=RS[t];out[i,10]=ECO2E[t];out[i,11]=L[t]
        out[i,12]=eco2Param[t];out[i,13]=YGROSS[t];out[i,14]=K[t];out[i,15]=S[t]
        out[i,16]=I[t];out[i,17]=YNET[t];out[i,18]=CCATOT[t];out[i,19]=CCATOT[t]-(MAT[t]-mateq)
        out[i,20]=R0[t];out[i,21]=R1[t];out[i,22]=R2[t];out[i,23]=R3[t]
        out[i,24]=DAMAGES[t];out[i,25]=ABATECOST[t]
        out[i,26]=PBACKTIME[t]*(MIU[t]**(expcost2-1.0))
        out[i,27]=C[t];out[i,28]=PU[t];out[i,29]=TPU[t];out[i,30]=MAT[t]
        out[i,31]=FORC[t];out[i,32]=TB1[t];out[i,33]=TB2[t];out[i,34]=FG[t]
        out[i,35]=IRFt[t];out[i,36]=alpha[t];out[i,37]=RF[t];out[i,38]=RL[t]
        out[i,39]=SCC[t];out[i,40]=ABATECOST[t]/max(Y[t],1e-12)
        out[i,41]=MAT[t]/mat0;out[i,42]=MAT[t]/mateq;out[i,43]=FORC_CO2[t];out[i,44]=RR[t]
    if not np.all(np.isfinite(out)): return np.zeros((num_periods,46))
    return out


def recoverAllVars(x, params):
    p = params
    return recoverAllVars_numba(
        x, p['num_periods'],
        p['tstep'],p['dk'],p['gama'],p['eco2Param'],p['sigma'],p['eland'],
        p['cost1tot'],p['expcost2'],p['miuup'],p['sLBounds'],p['sUBounds'],
        p['AlphaLowerBound'],p['Fcoef1'],p['Fcoef2'],p['CO2E_GHGabateB'],
        p['emshare0'],p['emshare1'],p['emshare2'],p['emshare3'],
        p['tau0'],p['tau1'],p['tau2'],p['tau3'],
        p['mateq'],p['fco22x'],p['F_Misc'],p['teq1'],p['teq2'],
        p['d1'],p['d2'],p['a1'],p['a2base'],p['a3'],
        p['res00'],p['res10'],p['res20'],p['res30'],
        p['tbox10'],p['tbox20'],p['CumEmiss0'],p['F_GHGabate2020'],
        p['k0'],p['tatm0'],p['mat0'],p['a0'],
        p['L'],p['RR'],p['scale1'],p['scale2'],p['elasmu'],
        p['PBACKTIME'],p['irf0'],p['irC'],p['irT'],p['SRF'],
        p['damage_type'],p['a3b'],p['gamma_ds'])

print('Numba core compiled and ready.')

# ===== cell 5 =====
class DiceFunc:
    def __init__(self, num_periods, params, TempUpperConstraint=20, TempLowerConstraint=0.5):
        self.num_periods = num_periods
        self.params = params
        self.TempUpperConstraint = TempUpperConstraint
        self.TempLowerConstraint = TempLowerConstraint
        self.MIU = np.zeros(num_periods+1)
        t = np.arange(1, num_periods+1)
        self.MIU[1:] = 0.05+(params['miuup'][1:]-0.05)*(1-np.exp(-0.05*(t-1)))/(1-np.exp(-0.05*num_periods))
        self.MIU[1] = 0.05
        self.S = np.full(num_periods+1, max(params['optlrsav'], 0.2))
        self.S[params['FixSperiod']:] = params['optlrsav']
        self.Alpha = np.linspace(params['a0'], 0.425, num_periods+1)
        self.Alpha[1] = params['a0']

    def pack(self, x):
        return x[:self.num_periods], x[self.num_periods:2*self.num_periods], x[2*self.num_periods:]

    def objective(self, x):
        MIU,S,Alpha = self.pack(x)
        self.MIU[1:self.num_periods+1]=MIU
        self.S[1:self.num_periods+1]=S
        self.Alpha[1:self.num_periods+1]=Alpha
        out = diceTrajectory(self.params, self.MIU, self.S, self.Alpha)
        if np.any(np.isnan(out[1])): return 1e15
        return -out[0]

    def irf_residual(self, x):
        MIU,S,Alpha = self.pack(x)
        self.MIU[1:self.num_periods+1]=MIU
        self.S[1:self.num_periods+1]=S
        self.Alpha[1:self.num_periods+1]=Alpha
        out = diceTrajectory(self.params, self.MIU, self.S, self.Alpha)
        CCATOT=out[2]; MAT=out[3]; TATM=out[4]
        if np.any(np.isnan(TATM)): return np.ones(self.num_periods)*1e10
        p = self.params
        LHS = p['irf0']+p['irC']*(CCATOT[1:]-(MAT[1:]-p['mateq']))+p['irT']*TATM[1:]
        RHS = (Alpha*p['emshare0']*p['tau0']*(1-np.exp(-100/(Alpha*p['tau0'])))
               +Alpha*p['emshare1']*p['tau1']*(1-np.exp(-100/(Alpha*p['tau1'])))
               +Alpha*p['emshare2']*p['tau2']*(1-np.exp(-100/(Alpha*p['tau2'])))
               +Alpha*p['emshare3']*p['tau3']*(1-np.exp(-100/(Alpha*p['tau3']))))
        return LHS-RHS

    def temp_up(self, x):
        MIU,S,Alpha = self.pack(x)
        self.MIU[1:self.num_periods+1]=MIU
        self.S[1:self.num_periods+1]=S
        self.Alpha[1:self.num_periods+1]=Alpha
        out = diceTrajectory(self.params, self.MIU, self.S, self.Alpha)
        TATM=out[4]
        if np.any(np.isnan(TATM)): return np.ones(self.num_periods)*-1e10
        return self.TempUpperConstraint - TATM[1:]

    def temp_lo(self, x):
        MIU,S,Alpha = self.pack(x)
        self.MIU[1:self.num_periods+1]=MIU
        self.S[1:self.num_periods+1]=S
        self.Alpha[1:self.num_periods+1]=Alpha
        out = diceTrajectory(self.params, self.MIU, self.S, self.Alpha)
        TATM=out[4]
        if np.any(np.isnan(TATM)): return np.ones(self.num_periods)*-1e10
        return TATM[1:] - self.TempLowerConstraint


class Dice2023Model:
    def __init__(self, num_times, scenario, damage_key='nordhaus_2023', delay_periods=0):
        self.num_periods = num_times
        self.scenario    = scenario
        self.damage_key  = damage_key
        self.delay_periods  = delay_periods 
        self.params      = LoadParams(num_times)
        set_damage_function(self.params, damage_key)
        self.TempUpperConstraint = (
            1.5  if scenario==6 else
            2.0  if scenario==7 else
            15.0 if scenario==10 else 20.0)
        self.TempLowerConstraint = 0.01

    def run_model(self):
        if self.scenario==1:  self.params['k0']=420; apply_disc_prstp(self.params,0.01)
        elif self.scenario==2: self.params['k0']=409; apply_disc_prstp(self.params,0.02)
        elif self.scenario==3: self.params['k0']=370; apply_disc_prstp(self.params,0.03)
        elif self.scenario==4: self.params['k0']=326; apply_disc_prstp(self.params,0.04,cap_after_t=81,cap_exp_value=5*80)
        elif self.scenario==5: self.params['k0']=290; apply_disc_prstp(self.params,0.05,cap_after_t=51,cap_exp_value=5*51)
        elif self.scenario==8:
            for i in range(1,self.num_periods+1):
                self.params['miuup'][i]=min(0.05+0.04*(i-1)-0.01*max(0,i-5),self.params['limmiu2070'])
        elif self.scenario==10: self.params['miuup'][:]=1.0

        if self.scenario not in {1,2,3,4,5}:
            self.params['rartp'] = np.exp(self.params['prstp']+self.params['betaclim']*self.params['pi'])-1
            self.params['optlrsav'] = ((self.params['dk']+0.004)/
                                       (self.params['dk']+0.004*self.params['elasmu']+self.params['rartp']))*self.params['gama']

        prob = DiceFunc(self.num_periods, self.params,
                        TempUpperConstraint=self.TempUpperConstraint,
                        TempLowerConstraint=self.TempLowerConstraint)
        x0 = np.concatenate([prob.MIU[1:self.num_periods+1],
                              prob.S[1:self.num_periods+1],
                              prob.Alpha[1:self.num_periods+1]])
        # Delayed action: constrain MIU to near-zero for first delay_periods
        delay_periods = getattr(self, 'delay_periods', 0)
        
        miu_b = [(self.params['miu1'], self.params['miu1'])]
        for t in range(2, self.num_periods+1):
            if t <= delay_periods:
                # Force near-zero abatement during delay window
                miu_b.append((0.0, 0.03))
            else:
                miu_b.append((self.params['MIULowerBound'], self.params['miuup'][t]))

        s_b   = [(self.params['sLBounds'][i],self.params['sUBounds'][i]) for i in range(1,self.num_periods+1)]
        al_b  = [(self.params['AlphaLowerBound'],self.params['AlphaUpperBound'])]*self.num_periods
        bounds = miu_b+s_b+al_b

        constraints = [{'type':'eq','fun':prob.irf_residual},
                       {'type':'ineq','fun':prob.temp_up},
                       {'type':'ineq','fun':prob.temp_lo}]

        if self.scenario==6:
            steps=[2.0,1.9,1.8,1.79,1.78,1.77,1.76,1.75,1.74,1.73,1.72,
                   1.71,1.70,1.68,1.66,1.64,1.62,1.60,1.58,1.56,1.5]
            x_cur=x0.copy(); last_ok=x0.copy()
            for tl in steps:
                ps=DiceFunc(self.num_periods,self.params,TempUpperConstraint=tl,
                            TempLowerConstraint=self.TempLowerConstraint)
                cs=[{'type':'eq','fun':ps.irf_residual},
                    {'type':'ineq','fun':ps.temp_up},{'type':'ineq','fun':ps.temp_lo}]
                sr=minimize(ps.objective,x_cur,method='SLSQP',bounds=bounds,
                            constraints=cs,options={'maxiter':1000,'ftol':1e-7,'disp':False})
                irf_r=np.abs(ps.irf_residual(sr.x)).max()
                if sr.success and irf_r<0.01:
                    x_cur=sr.x.copy(); last_ok=sr.x.copy()
            res=sr; res.x=last_ok
        else:
            res=minimize(prob.objective,x0,method='SLSQP',bounds=bounds,
                         constraints=constraints,
                         options={'maxiter':25000,'ftol':1e-5,'disp':False,'eps':1e-6})

        x_opt  = res.x
        output = recoverAllVars(x_opt, self.params)
        irf_r  = np.max(np.abs(prob.irf_residual(x_opt)))
        print(f'  Converged: {res.success} | IRF residual: {irf_r:.2e} | {res.message}')
        return x_opt, output, None


print('DiceFunc and Dice2023Model defined.')

# ===== cell 6 =====
# ============================================================
# ADD-I — Multi-start SLSQP wrapper
# ============================================================
# Motivation: DICE-2023 is a ~243-dimensional nonlinear program.
# SLSQP finds a LOCAL optimum from the given start. For ramp-relaxed
# cases where the feasible region is larger, a single start may miss
# a better solution. This wrapper runs N_STARTS tries with controlled
# perturbations and returns the best converged solution.
#
# Usage: replaces direct minimize() calls in Dice2023ModelRamp.run_model()

def multistart_slsqp(prob, x0, bounds, constraints, params,
                     n_starts=3, seed=42,
                     ftol=1e-5, maxiter=25000, eps=1e-6):
    """
    Run SLSQP from n_starts initial points; return best converged result.

    Start 0: standard x0 (unchanged).
    Starts 1+: x0 perturbed with small uniform noise on MIU component only.
    The savings (S) and carbon-cycle (alpha) components are not perturbed
    because they are well-constrained by economic and IRF structure.

    Returns: best scipy OptimizeResult, list of all welfare values tried.
    """
    rng = np.random.default_rng(seed)
    N   = params['num_periods']

    best_res     = None
    best_welfare = -np.inf
    all_welfares = []

    for k in range(n_starts):
        if k == 0:
            x_try = x0.copy()
        else:
            # Perturb only the MIU component (first N elements)
            noise = rng.uniform(-0.03, 0.03, size=N)
            x_try = x0.copy()
            x_try[:N] = np.clip(x_try[:N] + noise,
                                 [b[0] for b in bounds[:N]],
                                 [b[1] for b in bounds[:N]])

        res = minimize(
            prob.objective, x_try,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': ftol,
                     'disp': False, 'eps': eps}
        )

        welfare = -res.fun if np.isfinite(res.fun) else -np.inf
        all_welfares.append(welfare)

        if res.success and welfare > best_welfare:
            best_welfare = welfare
            best_res     = res

    # If no start converged, return the least-bad result
    if best_res is None:
        best_res = min(
            [minimize(prob.objective, x0, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': maxiter, 'ftol': ftol,
                               'disp': False, 'eps': eps})],
            key=lambda r: r.fun
        )

    # Report spread across starts (diagnostic)
    if len(all_welfares) > 1:
        spread = max(all_welfares) - min(w for w in all_welfares if np.isfinite(w))
        pct    = 100 * spread / abs(max(all_welfares)) if abs(max(all_welfares)) > 0 else 0
        print(f"    [multistart] {n_starts} starts | "
              f"welfare spread: {spread:.4f} ({pct:.4f}%) | "
              f"best welfare: {best_welfare:.4f}")

    return best_res, all_welfares


# Configuration — set to 1 for single-start (original behaviour)
MULTISTART_N = 3   # Increase to 5 for publication; 3 is a good smoke-test
print(f"Multi-start wrapper loaded. MULTISTART_N = {MULTISTART_N}")
print("Set MULTISTART_N = 1 to replicate original single-start behaviour.")

# ===== cell 7 =====
ECS_DEFAULT = 3.93 * (0.324 + 0.440)   # ≈ 3.002°C
ECS_LEVELS  = {'ecs_low': 2.5, 'ecs_central': ECS_DEFAULT, 'ecs_high': 4.5}

def apply_ecs(params, ecs_target):
    """Scale teq1 + teq2 so ECS = fco22x*(teq1+teq2) = ecs_target."""
    scale = ecs_target / ECS_DEFAULT
    params['teq1'] = 0.324 * scale
    params['teq2'] = 0.440 * scale
    return params


def build_experiment_matrix():
    experiments = []
    run_id = 1
    # Group A: 8 damage × 3 ECS, default discount (scenario 9)
    for dk in DAMAGE_REGISTRY:
        for ecs_label, ecs_val in ECS_LEVELS.items():
            experiments.append({'run_id':run_id,'group':'A','damage_key':dk,
                                 'ecs_label':ecs_label,'ecs_val':round(ecs_val,4),
                                 'disc_label':'default','disc_scenario':9})
            run_id += 1
    # Group B: 8 damage × 3 discount rates, central ECS
    for dk in DAMAGE_REGISTRY:
        for dl, ds in [('disc_1pct',1),('disc_3pct',3),('disc_5pct',5)]:
            experiments.append({'run_id':run_id,'group':'B','damage_key':dk,
                                 'ecs_label':'ecs_central','ecs_val':round(ECS_DEFAULT,4),
                                 'disc_label':dl,'disc_scenario':ds})
            run_id += 1

    # Group C: delayed action sensitivity
    # 3 specs × 3 delays = 9 runs
    # delay_periods: 0 = no delay (baseline), 2 = 10yr, 4 = 20yr
    for dk in ['nordhaus_2023', 'weitzman', 'hs_high']:
        for delay_label, delay_periods in [('delay_0yr',0),
                                            ('delay_10yr',2),
                                            ('delay_20yr',4)]:
            experiments.append({'run_id':run_id,'group':'C','damage_key':dk,
                                 'ecs_label':'ecs_central','ecs_val':round(ECS_DEFAULT,4),
                                 'disc_label':'default','disc_scenario':9,
                                 'delay_label':delay_label,
                                 'delay_periods':delay_periods})
            run_id += 1
    return experiments


EXPERIMENTS = build_experiment_matrix()
df_plan = pd.DataFrame(EXPERIMENTS)
df_plan['damage_name'] = df_plan['damage_key'].map(DAMAGE_NAMES)
print(f'  Group A (ECS sensitivity):      {sum(1 for e in EXPERIMENTS if e["group"]=="A")} runs')
print(f'  Group B (discount sensitivity): {sum(1 for e in EXPERIMENTS if e["group"]=="B")} runs')
print(f'  Group C (delayed action):       {sum(1 for e in EXPERIMENTS if e["group"]=="C")} runs')
print(f'  Total:                          {len(EXPERIMENTS)} runs')
print()
display(df_plan[['run_id','group','damage_name','ecs_label','disc_label']])

# ===== cell 8 =====
SUMMARY_FILE = os.path.join(RESULTS_DIR, 'summary.csv')
LOG_FILE     = os.path.join(RESULTS_DIR, 'run_log.json')

# Updated manuscript-ready summary columns.
# Keeps legacy SCC_2025 and SCC_2075, and adds SCC/MIU/T/DAMFRAC at 2030, 2050, and 2100.
SUMMARY_COLS = ['run_id','group','damage_key','damage_name',
                'ecs_label','ecs_val','disc_label','disc_scenario',
                'delay_label',
                'SCC_2025','SCC_2030','SCC_2050','SCC_2075','SCC_2100',
                'MIU_2025','MIU_2030','MIU_2050','MIU_2100',
                'T_2030','T_2050','T_2100','T_peak',
                'welfare','PV_consumption',
                'peak_MIU_year','peak_MIU_val','year_MIU99',
                'damfrac_2030','damfrac_2050','damfrac_2100',
                'elapsed_s','status','timestamp']

OUTPUT_HEADER = ['PERIOD','EIND','ECO2','CO2PPM','TATM','Y','DAMFRAC','CPC','CPRICE',
                 'MIUopt','RSHORT','ECO2E','L','AL','YGROSS','K','Sopt','I','YNET',
                 'CCATOT','CACC','RES0','RES1','RES2','RES3','DAMAGES','ABATECOST',
                 'MCABATE','C','PERIODU','TOTPERIODU','MAT','FORC','TBOX1','TBOX2',
                 'F_GHGABATE','IRFT','ALPHA','RFACTLONG','RLONG','SCC','ABATERAT',
                 'ATFRAC2020','ATFRAC1765','FORC_CO2','RR']

COL = {'TATM':3,'DAMFRAC':5,'CPC':6,'MIU':8,'YGROSS':13,
       'YNET':17,'C':27,'TOTPERIODU':29,'SCC':39,'RR':44}

def _idx(year):
    """Calendar year -> zero-based output row index. Model starts in 2020 with 5-year steps."""
    return int((year - 2020) // 5)

def extract_metrics(output, params):
    """Extract manuscript-ready metrics from one optimized run."""
    m = {}

    # SCC years used in current and earlier manuscript versions.
    for year in [2025, 2030, 2050, 2075, 2100]:
        row = _idx(year)
        m[f'SCC_{year}'] = output[row, COL['SCC']]

    # Main policy/temperature/damage years used in the revised manuscript.
    for year in [2025, 2030, 2050, 2100]:
        row = _idx(year)
        m[f'MIU_{year}'] = output[row, COL['MIU']]
        m[f'T_{year}'] = output[row, COL['TATM']]
        if year in [2030, 2050, 2100]:
            m[f'damfrac_{year}'] = output[row, COL['DAMFRAC']]

    m['T_peak'] = output[:, COL['TATM']].max()

    m['welfare'] = (
        params['tstep'] * params['scale1'] *
        output[:, COL['TOTPERIODU']].sum() + params['scale2']
    )

    peak_idx = output[:, COL['MIU']].argmax()
    m['peak_MIU_year'] = 2020 + peak_idx * 5
    m['peak_MIU_val']  = output[peak_idx, COL['MIU']]

    hits = np.where(output[:, COL['MIU']] >= 0.99)[0]
    m['year_MIU99'] = int(2020 + hits[0] * 5) if len(hits) else 9999

    m['PV_consumption'] = (
        output[:, COL['C']] * output[:, COL['RR']] * params['tstep']
    ).sum()

    return m


def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            return json.load(f)
    return {}

def save_log(log):
    with open(LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2)

def append_summary(metrics):
    write_header = not os.path.exists(SUMMARY_FILE)
    with open(SUMMARY_FILE, 'a') as f:
        if write_header:
            f.write(','.join(SUMMARY_COLS) + '\n')
        f.write(','.join(str(metrics.get(c, '')) for c in SUMMARY_COLS) + '\n')


def run_one(exp):
    """Run one main experiment from EXPERIMENTS and save its full trajectory."""
    dk = exp['damage_key']
    ecs_val = exp['ecs_val']
    disc_scen = exp['disc_scenario']
    run_id = exp['run_id']

    print(f"\n{'─'*60}")
    print(f"Run {run_id:>3d} | {DAMAGE_NAMES[dk]:<28} | ECS={ecs_val:.2f}°C | {exp['disc_label']}")
    print(f"{'─'*60}")

    t0 = time.time()
    model = Dice2023Model(
        num_times=81,
        scenario=disc_scen,
        damage_key=dk,
        delay_periods=exp.get('delay_periods', 0)
    )
    apply_ecs(model.params, ecs_val)
    x_opt, output, _ = model.run_model()
    elapsed = round(time.time() - t0, 1)

    metrics = extract_metrics(output, model.params)
    metrics.update({
        'run_id': run_id,
        'group': exp['group'],
        'damage_key': dk,
        'damage_name': DAMAGE_NAMES[dk],
        'ecs_label': exp['ecs_label'],
        'ecs_val': ecs_val,
        'disc_label': exp['disc_label'],
        'disc_scenario': disc_scen,
        'delay_label': exp.get('delay_label', 'none'),
        'elapsed_s': elapsed,
        'status': 'ok',
        'timestamp': datetime.now().isoformat(timespec='seconds')
    })

    print(f"  SCC2030: ${metrics['SCC_2030']:>7.1f} | "
          f"SCC2050: ${metrics['SCC_2050']:>7.1f} | "
          f"MIU2030: {100*metrics['MIU_2030']:.1f}% | "
          f"T2100: {metrics['T_2100']:.2f}°C | Time: {elapsed:.0f}s")

    run_tag  = f"run{run_id:03d}_{dk}_{exp['ecs_label']}_{exp['disc_label']}"
    run_path = os.path.join(RUNS_DIR, f"{run_tag}.csv")

    n_out = len(OUTPUT_HEADER) - 1   # OUTPUT_HEADER includes PERIOD, output does not
    rows = [[i+1] + list(output[i, :n_out]) for i in range(output.shape[0])]
    pd.DataFrame(rows, columns=OUTPUT_HEADER).to_csv(run_path, index=False)

    return metrics, output

print('Updated metric extractor and main-run helpers defined.')

# ===== cell 9 =====
# ============================================================
# R1 — Helper functions for ramp/ceiling experiments
# ============================================================
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize

RAMP_RESULTS_DIR = os.path.join(RESULTS_DIR, "ramp_ceiling_experiment_clean")
os.makedirs(RAMP_RESULTS_DIR, exist_ok=True)
RAMP_RUNS_DIR = os.path.join(RAMP_RESULTS_DIR, "runs")
os.makedirs(RAMP_RUNS_DIR, exist_ok=True)
RAMP_SUMMARY_FILE = os.path.join(RAMP_RESULTS_DIR, "ramp_summary.csv")


def _longrun_cap(params, t):
    """
    Replica of the long-run miuup cap structure from LoadParams.
    Nonstandard ramp cases change the near-term ramp only, not these caps.
    """
    if   t <= 20: return params["limmiu2070"]
    elif t <= 37: return params["limmiu2120"]
    elif t <= 57: return params["limmiu2200"]
    else:         return params["limmiu2300"]


# Ramp rate is in absolute control-rate units per 5-year period.
# 0.08 = 8 percentage points per period.
# None = no near-term ramp; only the long-run caps apply.
RAMP_RATES = {
    "tighter_ramp_8pp":   0.08,
    "relaxed_ramp_20pp": 0.20,
    "relaxed_ramp_30pp": 0.30,
    "no_near_term_ramp": None,
}


def overwrite_miu_bounds(params, case_name):
    """Overwrite params['miuup'] for one ramp/ceiling sensitivity case."""
    if case_name == "standard_dice2023":
        return params

    if case_name not in RAMP_RATES:
        raise ValueError(f"Unknown ramp case: {case_name}")

    N = params["num_periods"]
    rate = RAMP_RATES[case_name]

    # Period 1 is fixed at miu1 by the optimizer bounds.
    params["miuup"][1] = params["miu1"]

    for t in range(2, N + 1):
        cap = _longrun_cap(params, t)
        if rate is None:
            params["miuup"][t] = cap
        else:
            ramp_bound = params["miu1"] + rate * (t - 1)
            params["miuup"][t] = min(ramp_bound, cap)

    return params


def classify_binding(miu, bound, tol=1e-4, near_tol=0.01):
    """Classify whether MIU is binding, near-binding, or interior."""
    slack = bound - miu
    if slack <= tol:
        return "binding"
    elif slack <= near_tol:
        return "near-binding"
    else:
        return "interior"


def get_year_index(year):
    """Calendar year -> zero-based output row index. Model starts in 2020."""
    return int((year - 2020) // 5)


def first_binding_year(output, params, tol=1e-4):
    """
    First year where MIU sits at its upper bound.
    Starts at row 1 because period 1 is fixed by construction.
    """
    miu_path = output[:, COL["MIU"]]
    n = min(len(miu_path), params["num_periods"])
    for row in range(1, n):
        period = row + 1
        year = 2020 + row * 5
        if params["miuup"][period] - miu_path[row] <= tol:
            return year
    return None

print("Ramp/ceiling helper functions loaded.")
print(f"Clean ramp results will be saved to: {os.path.abspath(RAMP_RESULTS_DIR)}")

# ===== cell 10 =====
# ============================================================
# R2 — Model wrapper allowing ramp-case overrides
# ============================================================
class Dice2023ModelRamp(Dice2023Model):
    """
    Same as Dice2023Model, but overwrites MIU upper bounds before optimization.
    Scenario 6 is intentionally not supported here because it has custom temperature-stepping logic.
    """

    def __init__(self, num_times, scenario, damage_key="nordhaus_2023",
                 delay_periods=0, ramp_case="standard_dice2023"):
        super().__init__(num_times=num_times, scenario=scenario,
                         damage_key=damage_key, delay_periods=delay_periods)
        self.ramp_case = ramp_case

    def run_model(self):
        if self.scenario == 6:
            raise NotImplementedError(
                "Dice2023ModelRamp omits scenario-6 stepping logic; "
                "use the parent Dice2023Model for the 1.5°C scenario."
            )

        # Scenario-specific setup, copied from parent run_model.
        if self.scenario == 1:
            self.params["k0"] = 420
            apply_disc_prstp(self.params, 0.01)
        elif self.scenario == 2:
            self.params["k0"] = 409
            apply_disc_prstp(self.params, 0.02)
        elif self.scenario == 3:
            self.params["k0"] = 370
            apply_disc_prstp(self.params, 0.03)
        elif self.scenario == 4:
            self.params["k0"] = 326
            apply_disc_prstp(self.params, 0.04, cap_after_t=81, cap_exp_value=5*80)
        elif self.scenario == 5:
            self.params["k0"] = 290
            apply_disc_prstp(self.params, 0.05, cap_after_t=51, cap_exp_value=5*51)
        elif self.scenario == 8:
            for i in range(1, self.num_periods + 1):
                self.params["miuup"][i] = min(
                    0.05 + 0.04*(i-1) - 0.01*max(0, i-5),
                    self.params["limmiu2070"]
                )
        elif self.scenario == 10:
            self.params["miuup"][:] = 1.0

        if self.scenario not in {1, 2, 3, 4, 5}:
            self.params["rartp"] = np.exp(
                self.params["prstp"] + self.params["betaclim"] * self.params["pi"]
            ) - 1
            self.params["optlrsav"] = (
                (self.params["dk"] + 0.004) /
                (self.params["dk"] + 0.004*self.params["elasmu"] + self.params["rartp"])
            ) * self.params["gama"]

        # New part: overwrite MIU feasible upper bounds for this ramp case.
        overwrite_miu_bounds(self.params, self.ramp_case)

        prob = DiceFunc(self.num_periods, self.params,
                        TempUpperConstraint=self.TempUpperConstraint,
                        TempLowerConstraint=self.TempLowerConstraint)
        x0 = np.concatenate([prob.MIU[1:self.num_periods + 1],
                             prob.S[1:self.num_periods + 1],
                             prob.Alpha[1:self.num_periods + 1]])

        delay_periods = getattr(self, "delay_periods", 0)
        miu_b = [(self.params["miu1"], self.params["miu1"])]
        for t in range(2, self.num_periods + 1):
            if t <= delay_periods:
                miu_b.append((0.0, 0.03))
            else:
                miu_b.append((self.params["MIULowerBound"], self.params["miuup"][t]))

        s_b = [(self.params["sLBounds"][i], self.params["sUBounds"][i])
               for i in range(1, self.num_periods + 1)]
        al_b = [(self.params["AlphaLowerBound"], self.params["AlphaUpperBound"])
                for _ in range(self.num_periods)]
        bounds = miu_b + s_b + al_b

        constraints = [{"type": "eq",   "fun": prob.irf_residual},
                       {"type": "ineq", "fun": prob.temp_up},
                       {"type": "ineq", "fun": prob.temp_lo}]

        # ADD-I: use multistart wrapper (MULTISTART_N controls #starts)
        res, _all_welfares = multistart_slsqp(
            prob, x0, bounds, constraints, self.params,
            n_starts=MULTISTART_N,
            ftol=1e-5, maxiter=25000, eps=1e-6
        )

        x_opt = res.x
        output = recoverAllVars(x_opt, self.params)
        irf_r = np.max(np.abs(prob.irf_residual(x_opt)))
        print(f"  Converged: {res.success} | IRF residual: {irf_r:.2e} | "
              f"Ramp case: {self.ramp_case} | {res.message}")
        return x_opt, output, res

print("Dice2023ModelRamp loaded.")

# ===== cell 11 =====
# ============================================================
# R3 — Experiment matrix: central ECS, default discounting, multiple ramp cases
# ============================================================
RAMP_CASES = [
    "standard_dice2023",
    "tighter_ramp_8pp",
    "relaxed_ramp_20pp",
    "relaxed_ramp_30pp",
    "no_near_term_ramp",
]

DAMAGE_KEYS_FOR_RAMP_TEST = list(DAMAGE_REGISTRY.keys())

ramp_experiments = []
run_id = 1
for ramp_case in RAMP_CASES:
    for dk in DAMAGE_KEYS_FOR_RAMP_TEST:
        ramp_experiments.append({
            "ramp_run_id": run_id,
            "ramp_case": ramp_case,
            "damage_key": dk,
            "damage_name": DAMAGE_NAMES[dk],
            "ecs_label": "ecs_central",
            "ecs_val": ECS_DEFAULT,
            "disc_label": "default",
            "disc_scenario": 9,
            "delay_label": "none",
            "delay_periods": 0,
        })
        run_id += 1

df_ramp_plan = pd.DataFrame(ramp_experiments)
display(df_ramp_plan)
print(f"Total ramp/ceiling robustness runs: {len(ramp_experiments)}")

# ===== cell 12 =====
# ============================================================
# R4 — Run one ramp experiment and extract diagnostics
# ============================================================
RAMP_SUMMARY_COLS = [
    "ramp_run_id", "ramp_case",
    "damage_key", "damage_name",
    "ecs_label", "ecs_val",
    "disc_label", "disc_scenario",
    "SCC_2030", "SCC_2050", "SCC_2100",
    "MIU_2030", "MIU_2050", "MIU_2100",
    "MIU_bound_2030", "MIU_bound_2050", "MIU_bound_2100",
    "MIU_slack_2030", "MIU_slack_2050", "MIU_slack_2100",
    "binding_2030", "binding_2050", "binding_2100",
    "first_binding_year",
    "T_2030", "T_2050", "T_2100",
    "welfare",
    "elapsed_s", "status", "timestamp",
]


def extract_ramp_metrics(output, params):
    """SCC, MIU, upper-bound slack, and binding status for key years."""
    years = [2030, 2050, 2100]
    m = {}
    for year in years:
        row = get_year_index(year)
        period = row + 1
        miu = output[row, COL["MIU"]]
        bound = params["miuup"][period]
        m[f"SCC_{year}"] = output[row, COL["SCC"]]
        m[f"MIU_{year}"] = miu
        m[f"T_{year}"] = output[row, COL["TATM"]]
        m[f"MIU_bound_{year}"] = bound
        m[f"MIU_slack_{year}"] = bound - miu
        m[f"binding_{year}"] = classify_binding(miu, bound)
    m["first_binding_year"] = first_binding_year(output, params)
    m["welfare"] = (params["tstep"] * params["scale1"] *
                    output[:, COL["TOTPERIODU"]].sum() + params["scale2"])
    return m


def run_one_ramp_experiment(exp):
    """Run one damage function under one ramp/ceiling case."""
    print("\n" + "─" * 70)
    print(f"Ramp run {exp['ramp_run_id']:>3d} | "
          f"{exp['damage_name']:<28} | {exp['ramp_case']}")
    print("─" * 70)
    t0 = time.time()

    model = Dice2023ModelRamp(
        num_times=81,
        scenario=exp["disc_scenario"],
        damage_key=exp["damage_key"],
        delay_periods=exp.get("delay_periods", 0),
        ramp_case=exp["ramp_case"],
    )
    apply_ecs(model.params, exp["ecs_val"])
    x_opt, output, res = model.run_model()
    elapsed = round(time.time() - t0, 1)

    metrics = extract_ramp_metrics(output, model.params)
    metrics.update({
        "ramp_run_id": exp["ramp_run_id"],
        "ramp_case": exp["ramp_case"],
        "damage_key": exp["damage_key"],
        "damage_name": exp["damage_name"],
        "ecs_label": exp["ecs_label"],
        "ecs_val": exp["ecs_val"],
        "disc_label": exp["disc_label"],
        "disc_scenario": exp["disc_scenario"],
        "elapsed_s": elapsed,
        "status": "ok" if res.success else "no_convergence_flag",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })

    print(f"  SCC2030=${metrics['SCC_2030']:.1f} | "
          f"MIU2030={100*metrics['MIU_2030']:.1f}% | "
          f"Bound2030={100*metrics['MIU_bound_2030']:.1f}% | "
          f"Status={metrics['binding_2030']} | "
          f"T2100={metrics['T_2100']:.2f}°C | Time={elapsed:.0f}s")

    # Save full time path for later plotting.
    run_tag = (f"ramp{exp['ramp_run_id']:03d}_"
               f"{exp['ramp_case']}_{exp['damage_key']}")
    run_path = os.path.join(RAMP_RUNS_DIR, f"{run_tag}.csv")

    n_out = len(OUTPUT_HEADER) - 1   # OUTPUT_HEADER includes PERIOD, output does not
    rows = [[i + 1] + list(output[i, :n_out]) for i in range(output.shape[0])]
    df_out = pd.DataFrame(rows, columns=OUTPUT_HEADER)
    df_out["YEAR"] = 2020 + (df_out["PERIOD"] - 1) * 5
    df_out["ramp_case"] = exp["ramp_case"]
    df_out["damage_key"] = exp["damage_key"]
    df_out["damage_name"] = exp["damage_name"]

    df_out["MIU_BOUND"] = [model.params["miuup"][i + 1]
                           for i in range(len(df_out))]
    df_out["MIU_SLACK"] = df_out["MIU_BOUND"] - df_out["MIUopt"]

    df_out.to_csv(run_path, index=False)
    return metrics, output, model.params


def append_ramp_summary(metrics):
    write_header = not os.path.exists(RAMP_SUMMARY_FILE)
    row = {col: metrics.get(col, "") for col in RAMP_SUMMARY_COLS}
    pd.DataFrame([row]).to_csv(RAMP_SUMMARY_FILE, mode="a",
                               header=write_header, index=False)

print("Ramp experiment runner loaded.")

# ===== cell 13 =====
# ============================================================
# generate_paper_data.py
# ============================================================
# PURPOSE
# -------
# Generates every CSV file needed for the "When Carbon Prices
# Diverge but Abatement Paths Converge" manuscript.
#
# USAGE
# -----
# This script must be run AFTER all model cells in
# dice_damage_showdown_ENHANCED.ipynb have been executed
# (Cells 1–6, ADD-I), so that the following names are defined
# in the notebook kernel:
#
#   DAMAGE_REGISTRY, DAMAGE_NAMES, ECS_DEFAULT, EXPERIMENTS
#   LoadParams, set_damage_function, apply_ecs, apply_disc_prstp
#   Dice2023Model, Dice2023ModelRamp, DiceFunc
#   diceTrajectory, recoverAllVars, compute_SCC_numba
#   multistart_slsqp, MULTISTART_N
#   RESULTS_DIR, RAMP_RESULTS_DIR, RAMP_RUNS_DIR
#   COL, OUTPUT_HEADER, SUMMARY_COLS, RAMP_SUMMARY_COLS
#   _idx, get_year_index, extract_metrics, run_one
#   classify_binding, first_binding_year
#   overwrite_miu_bounds, RAMP_CASES, RAMP_RATES
#   run_one_ramp_experiment, append_ramp_summary
#   append_summary, SUMMARY_FILE, RAMP_SUMMARY_FILE
#
# HOW TO USE
# ----------
# Option A — run from inside the notebook (recommended):
#   Add a new cell at the end and paste:
#       exec(open('generate_paper_data.py').read())
#
# Option B — run as standalone if kernel state is available:
#       %run generate_paper_data.py
#
# OUTPUTS (all paths printed at end)
# -------
#  results/
#    summary.csv                              (57 main runs)
#    variance_decomposition_corrected_updated.csv
#    variance_decomposition_groupB.csv
#    table1_main_results_clean.csv
#    damage_curvature_along_optimal_path.csv
#    scc_pulse_stability.csv
#    irf_alpha_audit.csv
#  results/ramp_ceiling_experiment_clean/
#    ramp_summary.csv                         (40 ramp runs)
#    ramp_case_dispersion_summary.csv
#    welfare_cost_of_constraint.csv
#    standard_dice2023_constraint_diagnostic_table.csv
#
# RUNTIME ESTIMATE
# ----------------
#  Main experiment (57 runs)  ~9-10 min on modern CPU
#  Ramp experiment (40 runs)  ~7-8 min
#  Post-processing            <30 sec
#  Total                      ~17-19 min
#
# Set QUICK_TEST = True for a 3-run smoke test (~45 sec).
# ============================================================

import os, time, json, traceback, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────
QUICK_TEST        = False  # True = 3-run smoke test on nordhaus_2023 only
CLEAN_OUTPUTS     = True   # True = delete old CSVs before running
MULTISTART_N_USE  = 3      # Override notebook's MULTISTART_N if desired (set 1 for speed)
# ──────────────────────────────────────────────────────────────

_t_script_start = time.time()
_generated_files = []

def _log(path):
    _generated_files.append(os.path.abspath(path))

def _elapsed():
    s = int(time.time() - _t_script_start)
    return f"{s//60}m {s%60}s"

# Apply MULTISTART_N override to the notebook's global before any model runs.
# The notebook's Dice2023ModelRamp reads MULTISTART_N directly.
try:
    _old_multistart = MULTISTART_N  # noqa: F821 — defined in notebook kernel
    MULTISTART_N = MULTISTART_N_USE
    print(f"  MULTISTART_N overridden: {_old_multistart} → {MULTISTART_N_USE}")
except NameError:
    pass  # MULTISTART_N not yet defined; model cells may not have been run

print("=" * 65)
print("  DICE-2023 Damage Showdown — Paper Data Generator")
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  QUICK_TEST={QUICK_TEST}  CLEAN_OUTPUTS={CLEAN_OUTPUTS}")
print("=" * 65)


# ══════════════════════════════════════════════════════════════
# BLOCK 1 — MAIN EXPERIMENT (57 runs → summary.csv)
# ══════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("BLOCK 1: Main experiment (Groups A / B / C)")
print("─"*65)

LOG_FILE = os.path.join(RESULTS_DIR, 'run_log.json')

if CLEAN_OUTPUTS:
    for f in [SUMMARY_FILE, LOG_FILE]:
        if os.path.exists(f):
            os.remove(f)
    print("  Cleaned old summary/log files.")

experiments_main = list(EXPERIMENTS)
if QUICK_TEST:
    experiments_main = [e for e in experiments_main
                        if e['damage_key'] == 'nordhaus_2023'
                        and e['group'] == 'A'
                        and e['ecs_label'] == 'ecs_central']
    print(f"  QUICK_TEST: running {len(experiments_main)} run(s).")

total = len(experiments_main)
completed_main = 0
failed_main = 0
run_times = []
log = {}

for i, exp in enumerate(experiments_main):
    run_id = exp['run_id']
    try:
        metrics, output = run_one(exp)
        append_summary(metrics)
        log[str(run_id)] = {'status': 'ok', 'ts': metrics['timestamp']}
        completed_main += 1
        run_times.append(metrics['elapsed_s'])
    except Exception as exc:
        failed_main += 1
        print(f"  ⚠  Run {run_id} FAILED: {exc}")
        traceback.print_exc()
        # Append a failed-status row so downstream shape checks don't crash.
        err_row = {c: '' for c in SUMMARY_COLS}
        err_row.update({
            'run_id':       run_id,
            'group':        exp.get('group', ''),
            'damage_key':   exp.get('damage_key', ''),
            'damage_name':  DAMAGE_NAMES.get(exp.get('damage_key',''), ''),
            'ecs_label':    exp.get('ecs_label', ''),
            'ecs_val':      exp.get('ecs_val', ''),
            'disc_label':   exp.get('disc_label', ''),
            'disc_scenario':exp.get('disc_scenario', ''),
            'delay_label':  exp.get('delay_label', 'none'),
            'status':       'failed',
            'timestamp':    datetime.now().isoformat(timespec='seconds'),
        })
        append_summary(err_row)
        log[str(run_id)] = {'status': 'failed', 'error': str(exc)[:200]}

    pct = 100 * (i + 1) / total if total else 100
    avg_t = np.mean(run_times) if run_times else 0
    eta = str(timedelta(seconds=int(avg_t * (total - i - 1))))
    print(f"  [{i+1}/{total}] {pct:.0f}%  |  Wall: {_elapsed()}  |  ETA: {eta}")

with open(LOG_FILE, 'w') as f:
    json.dump(log, f, indent=2)

print(f"\n  Block 1 done — Completed: {completed_main}  Failed: {failed_main}")
_log(SUMMARY_FILE)


# ══════════════════════════════════════════════════════════════
# BLOCK 2 — VARIANCE DECOMPOSITIONS
# ══════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("BLOCK 2: Variance decompositions (Groups A and B)")
print("─"*65)

df_main = pd.read_csv(SUMMARY_FILE)
for col in ['SCC_2025','SCC_2030','SCC_2050','SCC_2100','T_2100','welfare']:
    df_main[col] = pd.to_numeric(df_main[col], errors='coerce')

# ── Group A: damage function × ECS ────────────────────────────
df_A = df_main[
    (df_main['group'] == 'A') &
    (df_main['disc_label'] == 'default') &
    (df_main['delay_label'] == 'none') &
    (df_main['status'] == 'ok') &
    (df_main['damage_key'] != 'hs_low')   # drop duplicate of nordhaus_2016 → 7 distinct functions
].copy()

def _vd_two_way(df, metric, fa, fb, fb_col_name='ecs_pct'):
    """Two-way balanced ANOVA variance decomposition.
    fa      = first factor column (always mapped to 'damage_pct')
    fb      = second factor column
    fb_col_name = key to use for factor-B share in the output dict
    """
    y = df[metric].astype(float)
    gm = y.mean()
    ma  = df.groupby(fa)[metric].mean()
    mb  = df.groupby(fb)[metric].mean()
    mab = df.groupby([fa, fb])[metric].mean()
    nb = df[fb].nunique(); na = df[fa].nunique()
    ss_a  = nb * ((ma - gm)**2).sum()
    ss_b  = na * ((mb - gm)**2).sum()
    ss_ab = sum((mab.loc[a, b] - ma.loc[a] - mb.loc[b] + gm)**2
                for (a, b) in mab.index)
    ss_t  = ((y - gm)**2).sum()
    return {
        'metric':          metric,
        'total_ss':        ss_t,
        'damage_ss':       ss_a,
        'fb_ss':           ss_b,
        'interaction_ss':  ss_ab,
        'damage_pct':      round(100*ss_a  / ss_t, 4),
        fb_col_name:       round(100*ss_b  / ss_t, 4),
        'interaction_pct': round(100*ss_ab / ss_t, 4),
        'check_pct':       round(100*(ss_a + ss_b + ss_ab) / ss_t, 4),
    }

vd_metrics = ['SCC_2025','SCC_2030','SCC_2050','SCC_2100','T_2100','welfare']
if not df_A.empty and len(df_A) >= 6:
    rows_vd = [_vd_two_way(df_A, m, 'damage_key', 'ecs_label',
                            fb_col_name='ecs_pct') for m in vd_metrics]
    df_vd = pd.DataFrame(rows_vd)
    # Rename fb_ss → ecs_ss to match the column name the notebook's Cell 9 saves.
    df_vd = df_vd.rename(columns={'fb_ss': 'ecs_ss'})
    vd_path = os.path.join(RESULTS_DIR, 'variance_decomposition_corrected_updated.csv')
    df_vd.to_csv(vd_path, index=False)
    print(f"  Saved: {vd_path}")
    _log(vd_path)
else:
    print("  ⚠  Skipping Group A variance decomp — insufficient rows.")

# ── Group B: damage function × discount rate ───────────────────
df_B = df_main[
    (df_main['group'] == 'B') &
    (df_main['status'] == 'ok') &
    (df_main['damage_key'] != 'hs_low')   # drop duplicate of nordhaus_2016 → 7 distinct functions
].copy()
for col in ['SCC_2025','SCC_2030','SCC_2050','SCC_2100']:
    df_B[col] = pd.to_numeric(df_B[col], errors='coerce')

if not df_B.empty and len(df_B) >= 6:
    vd_B_rows = [_vd_two_way(df_B, m, 'damage_key', 'disc_label',
                              fb_col_name='disc_rate_pct')
                 for m in ['SCC_2025','SCC_2030','SCC_2050','SCC_2100']]
    df_vd_B_out = pd.DataFrame(vd_B_rows)[
        ['metric', 'damage_pct', 'disc_rate_pct', 'interaction_pct']
    ]
    vd_B_path = os.path.join(RESULTS_DIR, 'variance_decomposition_groupB.csv')
    df_vd_B_out.to_csv(vd_B_path, index=False)
    print(f"  Saved: {vd_B_path}")
    _log(vd_B_path)
else:
    print("  ⚠  Skipping Group B variance decomp — insufficient rows.")


# ══════════════════════════════════════════════════════════════
# BLOCK 3 — TABLE 1 (central ECS, 8 damage specs)
# ══════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("BLOCK 3: Table 1 — main results")
print("─"*65)

df_t1 = df_main[
    (df_main['group'] == 'A') &
    (df_main['ecs_label'] == 'ecs_central') &
    (df_main['disc_label'] == 'default') &
    (df_main['delay_label'] == 'none') &
    (df_main['status'] == 'ok')
].copy()

if not df_t1.empty:
    for col in ['SCC_2030','SCC_2050','SCC_2100',
                'MIU_2030','MIU_2050','MIU_2100','T_2100','damfrac_2100']:
        df_t1[col] = pd.to_numeric(df_t1[col], errors='coerce')

    table1 = df_t1[[
        'damage_name','SCC_2030','SCC_2050','SCC_2100',
        'MIU_2030','MIU_2050','MIU_2100','T_2100','damfrac_2100'
    ]].copy()
    for col in ['MIU_2030','MIU_2050','MIU_2100','damfrac_2100']:
        table1[col] = 100 * table1[col]
    table1 = table1.rename(columns={
        'damage_name': 'Damage function',
        'SCC_2030': 'SCC 2030', 'SCC_2050': 'SCC 2050', 'SCC_2100': 'SCC 2100',
        'MIU_2030': 'Control 2030 (%)', 'MIU_2050': 'Control 2050 (%)',
        'MIU_2100': 'Control 2100 (%)', 'T_2100': 'T 2100 (°C)',
        'damfrac_2100': 'Damage 2100 (%)'
    })
    round_cols = [c for c in table1.columns if c != 'Damage function']
    table1[round_cols] = table1[round_cols].round(2)
    table1 = table1.sort_values('SCC 2030').reset_index(drop=True)

    t1_path = os.path.join(RESULTS_DIR, 'table1_main_results_clean.csv')
    table1.to_csv(t1_path, index=False)
    print(f"  Saved: {t1_path}")
    _log(t1_path)

    print(f"\n  Headline SCC 2030 range: "
          f"${df_t1['SCC_2030'].min():.2f}–${df_t1['SCC_2030'].max():.2f} /tCO₂")
    print(f"  Headline MIU 2030 range: "
          f"{100*df_t1['MIU_2030'].min():.2f}%–{100*df_t1['MIU_2030'].max():.2f}%")
else:
    print("  ⚠  No Group A central ECS rows — skipping Table 1.")


# ══════════════════════════════════════════════════════════════
# BLOCK 4 — DAMAGE CURVATURE ALONG OPTIMAL PATH
# ══════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("BLOCK 4: Damage curvature along optimal temperature path")
print("─"*65)

def _damfrac_scalar(T, dtype, a2base, a3, a3b, gamma_ds):
    if dtype == 1:
        return 1.0 - 1.0/(1.0 + a2base*T**2 + a3b*T**6.754)
    elif dtype == 2:
        return 1.0 - np.exp(-gamma_ds * min(T, 10.0)**2)
    else:
        return a2base * T**a3

def _dD_dT(T, dtype, a2base, a3, a3b, gamma_ds):
    if dtype == 1:
        num = 2*a2base*T + 6.754*a3b*T**5.754
        den = (1 + a2base*T**2 + a3b*T**6.754)**2
        return num/den
    elif dtype == 2:
        T_ = min(T, 10.0)
        return 2*gamma_ds*T_*np.exp(-gamma_ds*T_**2)
    else:
        return a3*a2base*T**(a3-1)

def _d2D_dT2(T, dtype, a2base, a3, a3b, gamma_ds):
    if dtype == 1:
        h = 1e-5
        return (_dD_dT(T+h, dtype, a2base, a3, a3b, gamma_ds)
                - _dD_dT(T-h, dtype, a2base, a3, a3b, gamma_ds)) / (2*h)
    elif dtype == 2:
        T_ = min(T, 10.0)
        return 2*gamma_ds*np.exp(-gamma_ds*T_**2)*(1 - 2*gamma_ds*T_**2)
    else:
        return a3*(a3-1)*a2base*T**(a3-2)

curv_rows = []
years_curv = [2025, 2030, 2040, 2050, 2075, 2100]

# Primary source: trajectory files saved by Block 1 — have all years,
# original column names, no dependency on the renamed df_t1/table1.
_runs_dir = os.path.join(RESULTS_DIR, 'runs')
if os.path.exists(_runs_dir):
    for dk in DAMAGE_REGISTRY:
        dtype, a2base, a3, a3b, gamma_ds = DAMAGE_REGISTRY[dk]
        matches = [f for f in os.listdir(_runs_dir)
                   if f'_{dk}_ecs_central_default' in f and f.endswith('.csv')]
        if not matches:
            continue
        traj = pd.read_csv(os.path.join(_runs_dir, matches[0]))
        traj['YEAR'] = 2020 + (traj['PERIOD'] - 1) * 5
        for yr in years_curv:
            sub = traj[traj['YEAR'] == yr]
            if sub.empty:
                continue
            T_opt_val = float(sub['TATM'].values[0])
            D   = _damfrac_scalar(T_opt_val, dtype, a2base, a3, a3b, gamma_ds)
            dD  = _dD_dT(T_opt_val,          dtype, a2base, a3, a3b, gamma_ds)
            d2D = _d2D_dT2(T_opt_val,         dtype, a2base, a3, a3b, gamma_ds)
            curv_rows.append({
                'damage_key':  dk,
                'damage_name': DAMAGE_NAMES[dk],
                'year':        yr,
                'T_opt':       round(T_opt_val, 3),
                'D_T':         round(100 * D, 4),
                'dD_dT':       round(dD, 4),
                'd2D_dT2':     round(d2D, 4),
            })

# Fallback: if trajectory files don't exist (quick-test), use T_2030/T_2050/T_2100
# from df_main (original column names, pre-rename).
if not curv_rows:
    print("  ⚠  No trajectory files — falling back to df_main temperatures (2030/2050/2100 only).")
    df_c = df_main[
        (df_main['group'] == 'A') &
        (df_main['ecs_label'] == 'ecs_central') &
        (df_main['disc_label'] == 'default') &
        (df_main['status'] == 'ok')
    ].copy()
    for _, row in df_c.iterrows():
        dk = row['damage_key']
        if dk not in DAMAGE_REGISTRY:
            continue
        dtype, a2base, a3, a3b, gamma_ds = DAMAGE_REGISTRY[dk]
        for yr in [2030, 2050, 2100]:
            T_col = f'T_{yr}'
            if T_col not in row.index:
                continue
            T_opt_val = pd.to_numeric(row[T_col], errors='coerce')
            if np.isnan(T_opt_val):
                continue
            D   = _damfrac_scalar(T_opt_val, dtype, a2base, a3, a3b, gamma_ds)
            dD  = _dD_dT(T_opt_val,          dtype, a2base, a3, a3b, gamma_ds)
            d2D = _d2D_dT2(T_opt_val,         dtype, a2base, a3, a3b, gamma_ds)
            curv_rows.append({
                'damage_key':  dk,
                'damage_name': DAMAGE_NAMES[dk],
                'year':        yr,
                'T_opt':       round(T_opt_val, 3),
                'D_T':         round(100 * D, 4),
                'dD_dT':       round(dD, 4),
                'd2D_dT2':     round(d2D, 4),
            })



if curv_rows:
    df_curv = pd.DataFrame(curv_rows)
    curv_path = os.path.join(RESULTS_DIR, 'damage_curvature_along_optimal_path.csv')
    df_curv.to_csv(curv_path, index=False)
    print(f"  Saved: {curv_path}")
    _log(curv_path)
else:
    print("  ⚠  No curvature rows computed.")


# ══════════════════════════════════════════════════════════════
# BLOCK 5 — SCC PULSE STABILITY CHECK
# ══════════════════════════════════════════════════════════════
# PURPOSE
# -------
# Verifies that the reported SCC is numerically stable across a
# 16× range of pulse sizes for one binding case (HS High) and one
# interior case (Dietz-Stern).
#
# TWO RICHARDSON FORMULAS — BOTH COMPUTED, ONE REPORTED
# -------------------------------------------------------
# The notebook's compute_SCC_numba (used for ALL 57 main runs and
# 40 ramp runs) uses the two-sided Richardson formula:
#
#   SCC_twosided = (4·SCC(h/2) − SCC(h)) / 3          ... (A)
#
# This formula is designed to cancel O(h²) error in a centred
# finite difference. However, the pulse here is one-sided (only
# forward perturbations, no negative pulse), so the theoretically
# correct extrapolation for a forward-difference O(h) scheme is:
#
#   SCC_onesided = 2·SCC(h/2) − SCC(h)                 ... (B)
#
# Both are computed below. The stability table shows they agree to
# <0.1% across all pulse sizes at these magnitudes (h ≤ 1e-5),
# confirming that the O(h) vs O(h²) distinction is numerically
# immaterial at the pulse sizes used — the welfare surface is
# smooth enough that both extrapolations converge to the same value.
#
# REPORTED VALUE: SCC_onesided (B) — theoretically correct.
# COMPARISON COL: SCC_twosided (A) — matches compute_SCC_numba.
# The agreement between (A) and (B) is the robustness argument.
# ══════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("BLOCK 5: SCC pulse-size stability (HS High + Dietz-Stern)")
print("─"*65)

PULSE_SIZES = [1e-5, 5e-6, 2.5e-6, 1.25e-6, 6.25e-7]
PULSE_CASES = ['hs_high', 'dietz_stern']

def _scc_at_pulse(params, x_opt, t_period, pulse_size):
    """
    Compute raw (unextrapolated) SCC and both Richardson estimates
    for a given pulse size at a given period.

    Returns
    -------
    scc_raw      : unextrapolated forward-difference SCC at pulse_size
    scc_onesided : Richardson extrapolation, one-sided formula (B) — REPORTED
    scc_twosided : Richardson extrapolation, two-sided formula (A) — COMPARISON
                   (matches what compute_SCC_numba uses internally)
    """
    N = params['num_periods']
    MIU = np.zeros(N+1); S = np.zeros(N+1); alpha = np.zeros(N+1)
    MIU[1:] = x_opt[:N]; S[1:] = x_opt[N:2*N]; alpha[1:] = x_opt[2*N:]
    alpha[1] = params['a0']; MIU[1] = params['miu1']

    ph  = np.zeros(N+1); ph[t_period]  = pulse_size        # full pulse
    ph2 = np.zeros(N+1); ph2[t_period] = pulse_size / 2.0  # half pulse

    base    = diceTrajectory(params, MIU, S, alpha)
    pert_h  = diceTrajectory(params, MIU, S, alpha, pulse=ph)
    pert_h2 = diceTrajectory(params, MIU, S, alpha, pulse=ph2)

    if np.isnan(base[0]) or np.isnan(pert_h[0]) or np.isnan(pert_h2[0]):
        return np.nan, np.nan, np.nan

    L = params['L']; RR = params['RR']; elasmu = params['elasmu']

    def W(res):
        c = 1000.0 * res[1][1:] / L[1:]
        return np.sum(((c**(1-elasmu) - 1) / (1-elasmu) - 1) * L[1:] * RR[1:])

    W0  = W(base)
    lam = (1000.0 * base[1][t_period] / L[t_period])**(-elasmu) * RR[t_period]
    if abs(lam) < 1e-12:
        return np.nan, np.nan, np.nan

    dh = W(pert_h)  - W0   # welfare change from full pulse
    db = W(pert_h2) - W0   # welfare change from half pulse

    # Raw unextrapolated SCC estimates at each pulse size
    scc_h  = -(dh / pulse_size)        / lam   # SCC at h
    scc_h2 = -(db / (pulse_size/2.0))  / lam   # SCC at h/2

    # Formula (B): one-sided Richardson — theoretically correct for forward diff
    scc_onesided = 2.0*scc_h2 - scc_h

    # Formula (A): two-sided Richardson — matches compute_SCC_numba
    scc_twosided = (4.0*scc_h2 - scc_h) / 3.0

    return round(scc_h, 4), round(scc_onesided, 4), round(scc_twosided, 4)

pulse_rows = []
for dk in (PULSE_CASES if not QUICK_TEST else ['hs_high']):
    print(f"  {DAMAGE_NAMES[dk]} ...", end=' ', flush=True)
    try:
        model = Dice2023Model(num_times=81, scenario=9, damage_key=dk)
        apply_ecs(model.params, ECS_DEFAULT)
        x_opt, _, _ = model.run_model()
        for ps in PULSE_SIZES:
            scc_raw, scc_onesided, scc_twosided = _scc_at_pulse(
                model.params, x_opt, t_period=2, pulse_size=ps
            )
            # Agreement metric: absolute % difference between the two formulas
            if (np.isfinite(scc_onesided) and np.isfinite(scc_twosided)
                    and abs(scc_onesided) > 1e-8):
                agree_pct = round(100 * abs(scc_onesided - scc_twosided)
                                  / abs(scc_onesided), 4)
            else:
                agree_pct = np.nan
            pulse_rows.append({
                'damage_key':    dk,
                'damage_name':   DAMAGE_NAMES[dk],
                'pulse_size':    ps,
                'SCC_raw':       scc_raw,       # unextrapolated at h
                'SCC_rich':      scc_onesided,  # REPORTED: one-sided Richardson (B)
                'SCC_rich_2side':scc_twosided,  # COMPARISON: two-sided Richardson (A)
                'agree_pct':     agree_pct,     # |B-A|/|B| × 100  (should be <0.1%)
            })
        print("done")
    except Exception as e:
        print(f"FAILED: {e}")

if pulse_rows:
    df_pulse = pd.DataFrame(pulse_rows)

    # Print comparison table to console for immediate inspection
    print("\n  ── Richardson formula comparison (SCC at t=2025) ──")
    print(f"  {'Damage':<26} {'Pulse':>10}  {'Raw':>8}  "
          f"{'1-sided':>8}  {'2-sided':>8}  {'Agree%':>7}")
    print("  " + "─"*72)
    for _, r in df_pulse.iterrows():
        print(f"  {r['damage_name']:<26} {r['pulse_size']:>10.2e}  "
              f"{r['SCC_raw']:>8.3f}  {r['SCC_rich']:>8.3f}  "
              f"{r['SCC_rich_2side']:>8.3f}  {r['agree_pct']:>6.3f}%")

    # Summary: max disagreement across all pulse sizes / both cases
    max_agree = df_pulse['agree_pct'].max()
    print(f"\n  Max disagreement between one-sided and two-sided: {max_agree:.4f}%")
    if max_agree < 0.1:
        print("  ✓ Both formulas agree to <0.1% — numerical robustness confirmed.")
    else:
        print("  ⚠ Disagreement exceeds 0.1% — inspect pulse sizes.")

    pulse_path = os.path.join(RESULTS_DIR, 'scc_pulse_stability.csv')
    df_pulse.to_csv(pulse_path, index=False)
    print(f"\n  Saved: {pulse_path}")
    print("  Columns: SCC_raw (unextrapolated), SCC_rich (one-sided, REPORTED),")
    print("           SCC_rich_2side (two-sided, matches compute_SCC_numba),")
    print("           agree_pct (|one-sided − two-sided| / |one-sided| × 100)")
    _log(pulse_path)


# ══════════════════════════════════════════════════════════════
# BLOCK 6 — RAMP EXPERIMENT (40 runs → ramp_summary.csv)
# ══════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("BLOCK 6: Ramp/ceiling diagnostic experiment (40 runs)")
print("─"*65)

if CLEAN_OUTPUTS and os.path.exists(RAMP_SUMMARY_FILE):
    os.remove(RAMP_SUMMARY_FILE)
    print("  Cleaned old ramp summary.")

ramp_run_id = 1
ramp_experiments = []
for ramp_case in RAMP_CASES:
    for dk in DAMAGE_REGISTRY:
        ramp_experiments.append({
            'ramp_run_id': ramp_run_id,
            'ramp_case':   ramp_case,
            'damage_key':  dk,
            'damage_name': DAMAGE_NAMES[dk],
            'ecs_label':   'ecs_central',
            'ecs_val':     ECS_DEFAULT,
            'disc_label':  'default',
            'disc_scenario': 9,
            'delay_label': 'none',
            'delay_periods': 0,
        })
        ramp_run_id += 1

if QUICK_TEST:
    ramp_experiments = [e for e in ramp_experiments
                        if e['damage_key'] == 'nordhaus_2023'
                        and e['ramp_case'] in ['standard_dice2023', 'no_near_term_ramp']]
    print(f"  QUICK_TEST: running {len(ramp_experiments)} ramp run(s).")

completed_ramp = 0
failed_ramp = 0
t_ramp = time.time()

for i, exp in enumerate(ramp_experiments):
    try:
        metrics, output, params_r = run_one_ramp_experiment(exp)
        append_ramp_summary(metrics)
        completed_ramp += 1
    except Exception as exc:
        failed_ramp += 1
        print(f"  ⚠  Ramp run {exp['ramp_run_id']} FAILED: {exc}")
        err = {col: '' for col in RAMP_SUMMARY_COLS}
        err.update({'ramp_run_id': exp['ramp_run_id'],
                    'ramp_case': exp['ramp_case'],
                    'damage_key': exp['damage_key'],
                    'damage_name': exp['damage_name'],
                    'status': 'failed',
                    'timestamp': datetime.now().isoformat(timespec='seconds')})
        append_ramp_summary(err)

    pct = 100*(i+1)/len(ramp_experiments)
    print(f"  [{i+1}/{len(ramp_experiments)}] {pct:.0f}%  |  Wall: {_elapsed()}")

print(f"\n  Block 6 done — Completed: {completed_ramp}  Failed: {failed_ramp}")
_log(RAMP_SUMMARY_FILE)


# ══════════════════════════════════════════════════════════════
# BLOCK 7 — RAMP POST-PROCESSING
#   ramp_case_dispersion_summary.csv
#   welfare_cost_of_constraint.csv
#   standard_dice2023_constraint_diagnostic_table.csv
#   irf_alpha_audit.csv
# ══════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("BLOCK 7: Ramp post-processing")
print("─"*65)

df_ramp = pd.read_csv(RAMP_SUMMARY_FILE)
for col in ['SCC_2030','SCC_2050','SCC_2100','MIU_2030','MIU_2050',
            'MIU_bound_2030','MIU_slack_2030','welfare']:
    if col in df_ramp.columns:
        df_ramp[col] = pd.to_numeric(df_ramp[col], errors='coerce')

df_ok = df_ramp[df_ramp['status'] == 'ok'].copy()
print(f"  Ramp runs OK: {len(df_ok)} / {len(df_ramp)}")

# ── 7a: Dispersion summary ─────────────────────────────────────
disp_rows = []
for ramp_case, g in df_ok.groupby('ramp_case'):
    disp_rows.append({
        'ramp_case':              ramp_case,
        'SCC_2030_min':           g['SCC_2030'].min(),
        'SCC_2030_max':           g['SCC_2030'].max(),
        'SCC_2030_range':         g['SCC_2030'].max() - g['SCC_2030'].min(),
        'MIU_2030_min_pct':       100*g['MIU_2030'].min(),
        'MIU_2030_max_pct':       100*g['MIU_2030'].max(),
        'MIU_2030_range_pp':      100*(g['MIU_2030'].max()-g['MIU_2030'].min()),
        'binding_count_2030':     (g['binding_2030']=='binding').sum(),
        'near_binding_count_2030':(g['binding_2030']=='near-binding').sum(),
        'interior_count_2030':    (g['binding_2030']=='interior').sum(),
        'SCC_2050_range':         g['SCC_2050'].max() - g['SCC_2050'].min(),
        'MIU_2050_range_pp':      100*(g['MIU_2050'].max()-g['MIU_2050'].min()),
    })
df_disp = pd.DataFrame(disp_rows).sort_values('ramp_case')
disp_path = os.path.join(RAMP_RESULTS_DIR, 'ramp_case_dispersion_summary.csv')
df_disp.to_csv(disp_path, index=False)
print(f"  Saved: {disp_path}")
_log(disp_path)

# ── 7b: Welfare cost of constraint ────────────────────────────
welf_pivot = df_ok.pivot_table(
    index='damage_key', columns='ramp_case', values='welfare', aggfunc='mean'
)
wc_rows = []
for dk in welf_pivot.index:
    w_std  = welf_pivot.loc[dk, 'standard_dice2023'] if 'standard_dice2023' in welf_pivot.columns else np.nan
    w_none = welf_pivot.loc[dk, 'no_near_term_ramp'] if 'no_near_term_ramp' in welf_pivot.columns else np.nan
    w_r20  = welf_pivot.loc[dk, 'relaxed_ramp_20pp'] if 'relaxed_ramp_20pp' in welf_pivot.columns else np.nan
    b_stat = df_ok.loc[
        (df_ok['damage_key']==dk) & (df_ok['ramp_case']=='standard_dice2023'),
        'binding_2030'
    ].values
    b_stat = b_stat[0] if len(b_stat) else 'unknown'
    cost_pct = round(100*(w_none-w_std)/abs(w_std), 4) if np.isfinite(w_std) and np.isfinite(w_none) else np.nan
    gain_r20 = round(100*(w_r20-w_std)/abs(w_std),  4) if np.isfinite(w_std) and np.isfinite(w_r20)  else np.nan
    wc_rows.append({
        'damage_key':          dk,
        'damage_name':         DAMAGE_NAMES.get(dk, dk),
        'binding_2030':        b_stat,
        'welfare_standard':    w_std,
        'welfare_no_ramp':     w_none,
        'welfare_cost_pct':    cost_pct,
        'welfare_gain_r20pct': gain_r20,
    })
df_wc = pd.DataFrame(wc_rows).sort_values('welfare_cost_pct', ascending=False)
wc_path = os.path.join(RAMP_RESULTS_DIR, 'welfare_cost_of_constraint.csv')
df_wc.to_csv(wc_path, index=False)
print(f"  Saved: {wc_path}")
_log(wc_path)

# ── 7c: Standard DICE-2023 constraint diagnostic table ─────────
df_std = df_ok[df_ok['ramp_case'] == 'standard_dice2023'].copy()
if not df_std.empty:
    diag_cols = ['damage_name','SCC_2030',
                 'MIU_2030','MIU_bound_2030','MIU_slack_2030','binding_2030',
                 'SCC_2050',
                 'MIU_2050','MIU_bound_2050','MIU_slack_2050','binding_2050',
                 'T_2100']
    diag_cols_present = [c for c in diag_cols if c in df_std.columns]
    df_diag = df_std[diag_cols_present].copy()
    for col in ['MIU_2030','MIU_bound_2030','MIU_slack_2030',
                'MIU_2050','MIU_bound_2050','MIU_slack_2050']:
        if col in df_diag.columns:
            df_diag[col] = (100*df_diag[col]).round(2)
    df_diag['SCC_2030'] = df_diag['SCC_2030'].round(2)
    df_diag['SCC_2050'] = df_diag['SCC_2050'].round(2) if 'SCC_2050' in df_diag.columns else np.nan
    df_diag['T_2100']   = df_diag['T_2100'].round(2) if 'T_2100' in df_diag.columns else np.nan
    df_diag = df_diag.sort_values('SCC_2030')
    diag_path = os.path.join(RAMP_RESULTS_DIR, 'standard_dice2023_constraint_diagnostic_table.csv')
    df_diag.to_csv(diag_path, index=False)
    print(f"  Saved: {diag_path}")
    _log(diag_path)

# ── 7d: IRF alpha audit ────────────────────────────────────────
irf_rows = []
if os.path.exists(RAMP_RUNS_DIR):
    for ramp_case in RAMP_CASES:
        case_files = [f for f in os.listdir(RAMP_RUNS_DIR)
                      if f.endswith('.csv') and f'_{ramp_case}_' in f]
        for fname in sorted(case_files):
            dk = fname.replace('.csv','').split(f'_{ramp_case}_')[-1]
            if dk not in DAMAGE_REGISTRY:
                continue
            try:
                traj = pd.read_csv(os.path.join(RAMP_RUNS_DIR, fname))
                if 'PERIOD' not in traj.columns:
                    continue
                traj['YEAR'] = 2020 + (traj['PERIOD']-1)*5
                a_2030 = traj.loc[traj['YEAR']==2030, 'ALPHA'].values if 'ALPHA' in traj.columns else [np.nan]
                a_2100 = traj.loc[traj['YEAR']==2100, 'ALPHA'].values if 'ALPHA' in traj.columns else [np.nan]
                irft   = traj['IRFT'].values if 'IRFT' in traj.columns else np.full(len(traj), np.nan)
                irf_rows.append({
                    'ramp_case':   ramp_case,
                    'damage_key':  dk,
                    'damage_name': DAMAGE_NAMES.get(dk, dk),
                    'alpha_2030':  float(a_2030[0]) if len(a_2030) else np.nan,
                    'alpha_2100':  float(a_2100[0]) if len(a_2100) else np.nan,
                    'irft_max':    float(np.nanmax(irft)),
                    'irft_min':    float(np.nanmin(irft)),
                })
            except Exception as e:
                print(f"    ⚠  Could not read {fname}: {e}")

if irf_rows:
    df_irf = pd.DataFrame(irf_rows)
    irf_path = os.path.join(RAMP_RESULTS_DIR, 'irf_alpha_audit.csv')
    df_irf.to_csv(irf_path, index=False)
    print(f"  Saved: {irf_path}")
    _log(irf_path)
else:
    print("  ⚠  No trajectory files found for IRF audit — run Block 6 first.")


# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"  DONE — total wall time: {_elapsed()}")
print(f"  Main runs:  completed={completed_main}  failed={failed_main}")
print(f"  Ramp runs:  completed={completed_ramp}  failed={failed_ramp}")
print("\n  Files generated:")
for fp in _generated_files:
    size_kb = os.path.getsize(fp) // 1024 if os.path.exists(fp) else 0
    print(f"    {fp}  ({size_kb} KB)")
print("=" * 65)

# ===== cell 14 =====
import numpy as np, pandas as pd, os

_required = ['Dice2023ModelRamp','DiceFunc','recoverAllVars','multistart_slsqp',
             'apply_ecs','ECS_DEFAULT','DAMAGE_REGISTRY','DAMAGE_NAMES','COL',
             'overwrite_miu_bounds','_idx','LoadParams','MULTISTART_N']
_missing = [n for n in _required if n not in globals()]
if _missing:
    raise NameError("Run the ENHANCED notebook's DEFINITION cells first. Missing: "
                    + ", ".join(_missing))

ROW_2030    = _idx(2030)         # 2  (zero-based output row)
PERIOD_2030 = ROW_2030 + 1       # 3  (period index into params['miuup'])
CPRICE_COL  = (OUTPUT_HEADER.index('CPRICE') - 1) if 'OUTPUT_HEADER' in globals() else 7
assert PERIOD_2030 == 3, "Expected 2030 to be period 3."
print("Prerequisites OK.  2030 -> output row", ROW_2030,
      "| miuup period", PERIOD_2030, "| p^gross col", CPRICE_COL)

# ===== cell 15 =====
def solve_ceiling(dk, mu_bar_2030, ecs_val=ECS_DEFAULT,
                  n_starts=1, x0=None, ftol=1e-9, maxiter=25000, eps=1e-7):
    m = Dice2023ModelRamp(num_times=81, scenario=9, damage_key=dk,
                          ramp_case="standard_dice2023")
    apply_ecs(m.params, ecs_val)
    p = m.params
    # scenario-9 setup, copied from run_model (scenario not in 1..5)
    p['rartp']    = np.exp(p['prstp'] + p['betaclim'] * p['pi']) - 1
    p['optlrsav'] = ((p['dk'] + 0.004) /
                     (p['dk'] + 0.004 * p['elasmu'] + p['rartp'])) * p['gama']
    overwrite_miu_bounds(p, "standard_dice2023")     # no-op, kept for parity
    p['miuup'][PERIOD_2030] = float(mu_bar_2030)     # <-- perturb the ceiling

    prob = DiceFunc(81, p, TempUpperConstraint=m.TempUpperConstraint,
                    TempLowerConstraint=m.TempLowerConstraint)
    if x0 is None:
        x0 = np.concatenate([prob.MIU[1:82], prob.S[1:82], prob.Alpha[1:82]])

    miu_b = [(p['miu1'], p['miu1'])] + \
            [(p['MIULowerBound'], p['miuup'][t]) for t in range(2, 82)]
    s_b   = [(p['sLBounds'][i], p['sUBounds'][i]) for i in range(1, 82)]
    al_b  = [(p['AlphaLowerBound'], p['AlphaUpperBound'])] * 81
    bounds = miu_b + s_b + al_b
    cons = [{'type': 'eq',   'fun': prob.irf_residual},
            {'type': 'ineq', 'fun': prob.temp_up},
            {'type': 'ineq', 'fun': prob.temp_lo}]

    res, _ = multistart_slsqp(prob, x0, bounds, cons, p,
                              n_starts=n_starts, ftol=ftol, maxiter=maxiter, eps=eps)
    out = recoverAllVars(res.x, p)
    W0  = float(out[:, COL['TOTPERIODU']].sum())      # UNSCALED welfare  (= compute_SCC's W0)
    return dict(x=res.x, out=out, params=p, W0=W0, res=res)

print("solve_ceiling defined.")

# ===== cell 16 =====
MU_BAR = 0.24
H_LIST = [0.010, 0.005, 0.002]     # forward steps on the ceiling; sweep for stability

def nu_for_spec(dk, h_list=H_LIST, verbose=True):
    base  = solve_ceiling(dk, MU_BAR, n_starts=MULTISTART_N)   # baseline (your pipeline)
    o, p  = base['out'], base['params']
    scc   = float(o[ROW_2030, COL['SCC']])
    miu   = float(o[ROW_2030, COL['MIU']])
    pg    = float(o[ROW_2030, CPRICE_COL])                     # p^gross = CPRICE
    ygr   = float(o[ROW_2030, COL['YGROSS']])
    cpc   = float(o[ROW_2030, COL['CPC']])
    rr    = float(o[ROW_2030, COL['RR']])
    stot  = float(p['sigmatot'][PERIOD_2030])
    eps_e = float(p['emissrat'][PERIOD_2030])
    Phi   = cpc ** (-p['elasmu']) * rr
    denom = stot * ygr * Phi
    Delta = scc - pg
    rows  = []
    for h in h_list:
        pert = solve_ceiling(dk, MU_BAR + h, n_starts=1, x0=base['x'])  # warm, forward
        nu   = (pert['W0'] - base['W0']) / h
        w_nu = nu / denom
        r    = Delta - w_nu
        ratio = (Delta / w_nu) if abs(w_nu) > 1e-9 else np.nan
        rows.append(dict(damage_key=dk, name=DAMAGE_NAMES[dk], h=h,
                         SCC=scc, p_gross=pg, MIU=miu, eps=eps_e,
                         nu=nu, w_nu=w_nu, Delta=Delta, r=r, ratio=ratio))
        if verbose:
            print(f"  {dk:14s} h={h:<6g} nu={nu:+.4e}  w_nu={w_nu:8.3f}  "
                  f"Delta={Delta:8.3f}  r={r:7.3f}  Delta/w_nu={ratio:6.3f}")
    return rows

print("nu_for_spec defined.")

# ===== cell 17 =====
BINDING   = ['hs_central', 'kahn', 'hs_high']
SANITY    = ['nordhaus_2023', 'dietz_stern']   # expect nu ~ 0

all_rows = []
print("BINDING specifications")
print("-"*78)
for dk in BINDING:
    print(f"[{DAMAGE_NAMES[dk]}]")
    all_rows += nu_for_spec(dk)

print("\nSANITY (near-binding / interior; expect nu ~ 0)")
print("-"*78)
for dk in SANITY:
    print(f"[{DAMAGE_NAMES[dk]}]")
    all_rows += nu_for_spec(dk, h_list=[0.010])

df_nu = pd.DataFrame(all_rows)
OUT_CSV = os.path.join(RESULTS_DIR, "nu_wedge_summary.csv")
df_nu.to_csv(OUT_CSV, index=False)
print(f"\nSaved -> {OUT_CSV}")
df_nu

# ===== cell 18 =====
EXPECT = {'hs_central':(86.40,0.240), 'kahn':(91.57,0.240), 'hs_high':(157.52,0.240),
          'nordhaus_2023':(60.94,0.234), 'dietz_stern':(45.20,0.196)}
seen = {}
for row in all_rows:
    seen.setdefault(row['damage_key'], row)   # first (baseline) row per spec
print(f"{'spec':16s} {'SCC(resolved)':>13s} {'SCC(ms)':>8s}   {'MIU%':>6s} {'MIU%(ms)':>8s}")
for dk, row in seen.items():
    e = EXPECT.get(dk, (None, None))
    em = f"{e[0]:.2f}" if e[0] else "   ?"
    mm = f"{100*e[1]:.1f}" if e[1] else "  ?"
    print(f"{dk:16s} {row['SCC']:13.2f} {em:>8s}   {100*row['MIU']:6.1f} {mm:>8s}")

# ===== cell 19 =====
bind = df_nu[df_nu.damage_key.isin(BINDING)]
ratios = bind.groupby('damage_key')['ratio'].median()
print("Median Delta/w_nu by binding spec:")
for dk, rr in ratios.items():
    print(f"  {dk:14s} {rr:.4f}")

med = float(ratios.median())
print(f"\nOverall median ratio: {med:.4f}")
if 0.9 <= med <= 1.1:
    print("=> Units consistent. w_nu ~ Delta; report r = Delta - w_nu as the residual.")
else:
    print("=> Ratio is not ~1. A global constant is off. Check, in order:")
    print("   * eps (1.376): are you on sigma vs sigmatot? w_nu uses sigmatot[3] (correct).")
    print("     ratio ~ 1.376 -> you normalised by sigma; ratio ~ 0.727 -> double-counted eps.")
    print("   * tstep (5) or 1/tstep: period vs annual rate mismatch in the difference.")
    print("   * 3.667 (tCO2/tC): GtCO2 vs GtC in the emissions base.")
    print("   Whatever the constant, r's SIGN pattern (small at interior, jump at binding)")
    print("   still validates the mechanism; fix the constant before filling the table.")

# ===== cell 20 =====
dk = 'nordhaus_2023'
base = solve_ceiling(dk, MU_BAR, n_starts=MULTISTART_N)
h = 0.01
up   = solve_ceiling(dk, MU_BAR + h, n_starts=1, x0=base['x'])
down = solve_ceiling(dk, MU_BAR - h, n_starts=1, x0=base['x'])
nu_fwd  = (up['W0']   - base['W0']) / h
nu_bwd  = (base['W0'] - down['W0']) / h
nu_cen  = (up['W0']   - down['W0']) / (2*h)
print(f"{DAMAGE_NAMES[dk]} (near-binding, slack ~0.6pp):")
print(f"  forward  nu = {nu_fwd:+.4e}   (correct: ~0, ceiling is slack)")
print(f"  backward nu = {nu_bwd:+.4e}   (spurious: step crossed the optimum)")
print(f"  centred  nu = {nu_cen:+.4e}   (contaminated by the backward half)")

# ===== cell 21 =====
def pick(dk):
    sub = df_nu[df_nu.damage_key == dk].sort_values('h')
    return sub.iloc[0]   # smallest h

order = [('dietz_stern','interior'), ('nordhaus_2016','interior'),
         ('weitzman','interior'), ('nordhaus_2023','near-binding'),
         ('hs_central','binding'), ('kahn','binding'), ('hs_high','binding')]
INTERIOR_DELTA = {'dietz_stern':-1.16,'nordhaus_2016':-1.09,
                  'weitzman':-0.98,'nordhaus_2023':-0.62}
print("% --- paste into tab:wedge_split ---")
for dk, status in order:
    nm = DAMAGE_NAMES[dk].replace('&','\\&')
    if status in ('interior','near-binding'):
        d = INTERIOR_DELTA.get(dk)
        dcell = f"${d:.2f}$" if d is not None else "[fill]"
        print(f"{nm:28s} & {status:12s} & {dcell:>8s} & 0.00 & {dcell:>8s} \\\\")
    else:
        if dk in df_nu.damage_key.values:
            r = pick(dk)
            print(f"{nm:28s} & {status:12s} & {r['Delta']:6.2f} "
                  f"& {r['w_nu']:6.2f} & {r['r']:6.2f} \\\\")
        else:
            print(f"{nm:28s} & {status:12s} & [Delta] & [w_nu] & [r] \\\\")

# ===== cell 22 =====
row = seen['hs_central']
eps_e = row['eps']; pg = row['p_gross']; scc = row['SCC']
print(f"Howard-Sterner Central @2030:  eps={eps_e:.3f}  p_gross={pg:.2f}")
print(f"  CO2-only endpoint eps*p_gross = {eps_e*pg:.2f}")
print(f"  reported SCC                  = {scc:.2f}")
print(f"  => SCC {'<' if scc < eps_e*pg else '>='} eps*p_gross : "
      f"{'exits' if scc < eps_e*pg else 'stays in'} the supporting set under a CO2-only instrument")
print("     (rem:base in the manuscript states exactly this).")

# ===== cell 23 =====
# ============================================================
# ADD-X — R7d: SCC growth rate analysis
# ============================================================
# Annualised SCC growth rate between year pairs.
# Binding cases expected to show faster growth than interior cases
# because the constraint shadow price contributes to the effective SCC path.

# Load all ramp run trajectories
scc_growth_rows = []
YEAR_PAIRS = [(2025,2030),(2030,2050),(2050,2075),(2075,2100)]

for ramp_case in RAMP_CASES:
    for dk in DAMAGE_REGISTRY:
        matches = [f for f in os.listdir(RAMP_RUNS_DIR)
                   if f.endswith('.csv')
                   and f'_{ramp_case}_' in f
                   and f.endswith(f'_{dk}.csv')]
        if not matches:
            continue
        traj = pd.read_csv(os.path.join(RAMP_RUNS_DIR, matches[0]))
        traj['YEAR'] = 2020 + (traj['PERIOD']-1)*5

        # Get binding status from ramp summary
        bs = df_ok.loc[(df_ok['ramp_case']==ramp_case) &
                       (df_ok['damage_key']==dk), 'binding_2030']
        binding = bs.values[0] if len(bs) else 'unknown'

        row = {'ramp_case': ramp_case,
               'damage_key': dk,
               'damage_name': DAMAGE_NAMES[dk],
               'binding_2030': binding}

        for y1, y2 in YEAR_PAIRS:
            scc1 = traj.loc[traj['YEAR']==y1, 'SCC'].values
            scc2 = traj.loc[traj['YEAR']==y2, 'SCC'].values
            if len(scc1) and len(scc2) and scc1[0] > 0 and scc2[0] > 0:
                ann_growth = (scc2[0]/scc1[0])**(1/(y2-y1)) - 1
                row[f'scc_growth_{y1}_{y2}'] = round(100*ann_growth, 3)
            else:
                row[f'scc_growth_{y1}_{y2}'] = np.nan
        scc_growth_rows.append(row)

if scc_growth_rows:
    df_growth = pd.DataFrame(scc_growth_rows)

    growth_cols = [f'scc_growth_{y1}_{y2}' for y1,y2 in YEAR_PAIRS]
    print("── SCC annualised growth rates (%) by ramp case and damage spec ──")
    print("Hotelling benchmark: growth ≈ social discount rate (~0.1% for DICE-2023 default)")

    std_growth = df_growth[df_growth['ramp_case']=='standard_dice2023'].copy()
    std_growth_disp = std_growth[['damage_name','binding_2030']+growth_cols].sort_values('scc_growth_2025_2030', ascending=False)
    display(std_growth_disp.round(3))

    print("\n── SCC growth rate: binding vs interior (standard ramp) ──")
    grp = std_growth.groupby('binding_2030')[growth_cols].mean()
    display(grp.round(3))

    print("\nNote: Binding cases typically show higher near-term SCC growth")
    print("because the constraint shadow price is reflected in the SCC path.")

    growth_path = os.path.join(RAMP_RESULTS_DIR, 'scc_growth_rates.csv')
    df_growth.to_csv(growth_path, index=False)
    print(f"\nSaved: {growth_path}")
else:
    print("No ramp trajectory files found. Run R5 first.")

# ===== cell 24 =====
# ============================================================
# ADD-IX — Cell 8d: No-policy (BAU) baseline runs
# ============================================================
# Forces MIU = miu1 = 0.05 in period 1 (fixed by DICE structure),
# then MIU = 0 for all subsequent periods (no abatement).
# Savings rate is held at optlrsav.
# Results: BAU temperature path, welfare, and damage fraction.
# Used to compute welfare gain from optimal policy per damage spec.

BAU_SUMMARY_FILE = os.path.join(RESULTS_DIR, 'summary_bau.csv')
BAU_COLS = ['damage_key','damage_name','ecs_label','ecs_val',
            'SCC_2030_bau',  # not meaningful under BAU but kept for structure
            'T_2030_bau','T_2050_bau','T_2100_bau','T_peak_bau',
            'welfare_bau','damfrac_2050_bau','damfrac_2100_bau',
            'status','timestamp']

def run_bau_baseline(damage_key, ecs_val=None):
    """Run DICE with no abatement (MIU = 0 after period 1)."""
    if ecs_val is None:
        ecs_val = ECS_DEFAULT
    params = LoadParams(81)
    set_damage_function(params, damage_key)
    apply_ecs(params, ecs_val)

    # Build zero-abatement trajectory manually (no optimisation needed)
    N = params['num_periods']
    MIU   = np.zeros(N+1); MIU[1] = params['miu1']  # period 1 fixed
    S     = np.full(N+1, params['optlrsav'])
    S[params['FixSperiod']:] = params['optlrsav']
    Alpha = np.linspace(params['a0'], 0.425, N+1)
    Alpha[1] = params['a0']

    output = recoverAllVars(
        np.concatenate([MIU[1:], S[1:], Alpha[1:]]), params
    )
    return output, params

def run_all_bau(clean=True):
    if clean and os.path.exists(BAU_SUMMARY_FILE):
        os.remove(BAU_SUMMARY_FILE)
    rows = []
    for dk in DAMAGE_REGISTRY:
        print(f"  BAU: {DAMAGE_NAMES[dk]:<28}", end=' ')
        try:
            output, params = run_bau_baseline(dk)
            T2100 = output[_idx(2100), COL['TATM']]
            T2050 = output[_idx(2050), COL['TATM']]
            T2030 = output[_idx(2030), COL['TATM']]
            Tpeak = output[:, COL['TATM']].max()
            welf  = (params['tstep']*params['scale1']*
                     output[:, COL['TOTPERIODU']].sum()+params['scale2'])
            df50  = output[_idx(2050), COL['DAMFRAC']]
            df100 = output[_idx(2100), COL['DAMFRAC']]
            print(f"T2100={T2100:.2f}°C  welfare={welf:.2f}")
            rows.append({
                'damage_key':     dk,
                'damage_name':    DAMAGE_NAMES[dk],
                'ecs_label':      'ecs_central',
                'ecs_val':        round(ECS_DEFAULT, 4),
                'T_2030_bau':     round(T2030, 3),
                'T_2050_bau':     round(T2050, 3),
                'T_2100_bau':     round(T2100, 3),
                'T_peak_bau':     round(Tpeak, 3),
                'welfare_bau':    round(welf,  4),
                'damfrac_2050_bau': round(100*df50,  4),
                'damfrac_2100_bau': round(100*df100, 4),
                'status':         'ok',
                'timestamp':      datetime.now().isoformat(timespec='seconds'),
            })
        except Exception as e:
            print(f"FAILED: {e}")
            rows.append({'damage_key':dk,'damage_name':DAMAGE_NAMES[dk],'status':'failed'})

    df_bau = pd.DataFrame(rows)
    df_bau.to_csv(BAU_SUMMARY_FILE, index=False)

    print("\n── BAU vs Optimal welfare gain by damage specification ──")
    # Merge with optimal welfare from main summary
    if os.path.exists(SUMMARY_FILE):
        df_opt = pd.read_csv(SUMMARY_FILE)
        df_opt_c = df_opt[
            (df_opt['group']=='A') &
            (df_opt['ecs_label']=='ecs_central') &
            (df_opt['disc_label']=='default') &
            (df_opt['status']=='ok')
        ][['damage_key','welfare']].copy()
        df_opt_c['welfare'] = pd.to_numeric(df_opt_c['welfare'], errors='coerce')
        df_opt_c = df_opt_c.rename(columns={'welfare': 'welfare_opt'})
        df_bau_m = df_bau.merge(df_opt_c, on='damage_key')
        df_bau_m['welfare_gain_pct'] = 100*(
            df_bau_m['welfare_opt'] - df_bau_m['welfare_bau']
        ) / abs(df_bau_m['welfare_bau'])
        display(df_bau_m[['damage_name','T_2100_bau',
                           'welfare_bau','welfare_opt','welfare_gain_pct']]
                .sort_values('welfare_gain_pct', ascending=False))
    else:
        display(df_bau[['damage_name','T_2100_bau','welfare_bau']])

    print(f"\nSaved: {BAU_SUMMARY_FILE}")
    return df_bau

print("BAU baseline runner defined. Call run_all_bau() to execute.")
print("BAU runs are fast (no optimisation) — typically < 5 seconds total.")


# execute
run_all_bau()

# ===== cell 25 =====
# ============================================================
# ADD-VIII — R7c: Targeted ramp × ECS cross-diagnostic
# ============================================================
# Targeted design: binding cases × all 3 ECS × {standard, no-ramp}
# Plus: DICE-2023 baseline and Weitzman under high ECS (borderline cases)
# Total: ~20 targeted runs (cheap relative to 40-run full ramp experiment)

TARGETED_RAMP_ECS = [
    # (damage_key, ecs_label, ecs_val, ramp_case)
    ('hs_central',    'ecs_low',     2.5,          'standard_dice2023'),
    ('hs_central',    'ecs_low',     2.5,          'no_near_term_ramp'),
    ('hs_central',    'ecs_high',    4.5,          'standard_dice2023'),
    ('hs_central',    'ecs_high',    4.5,          'no_near_term_ramp'),
    ('kahn',          'ecs_low',     2.5,          'standard_dice2023'),
    ('kahn',          'ecs_low',     2.5,          'no_near_term_ramp'),
    ('kahn',          'ecs_high',    4.5,          'standard_dice2023'),
    ('kahn',          'ecs_high',    4.5,          'no_near_term_ramp'),
    ('hs_high',       'ecs_low',     2.5,          'standard_dice2023'),
    ('hs_high',       'ecs_low',     2.5,          'no_near_term_ramp'),
    ('hs_high',       'ecs_high',    4.5,          'standard_dice2023'),
    ('hs_high',       'ecs_high',    4.5,          'no_near_term_ramp'),
    ('nordhaus_2023', 'ecs_high',    4.5,          'standard_dice2023'),  # does baseline bind?
    ('weitzman',      'ecs_high',    4.5,          'standard_dice2023'),  # does Weitzman bind?
]

TARGETED_SUMMARY_FILE = os.path.join(RAMP_RESULTS_DIR, 'targeted_ramp_ecs_summary.csv')

def run_targeted_ramp_ecs(clean=True):
    if clean and os.path.exists(TARGETED_SUMMARY_FILE):
        os.remove(TARGETED_SUMMARY_FILE)

    rows = []
    for i, (dk, ecs_label, ecs_val, ramp_case) in enumerate(TARGETED_RAMP_ECS):
        print(f"\n[{i+1}/{len(TARGETED_RAMP_ECS)}] {DAMAGE_NAMES[dk]} | "
              f"{ecs_label} | {ramp_case}")
        t0 = time.time()
        try:
            model = Dice2023ModelRamp(
                num_times=81, scenario=9,
                damage_key=dk, ramp_case=ramp_case
            )
            apply_ecs(model.params, ecs_val)
            x_opt, output, res = model.run_model()
            elapsed = round(time.time()-t0, 1)

            row_idx_30 = get_year_index(2030)
            miu_30   = output[row_idx_30, COL['MIU']]
            bound_30 = model.params['miuup'][row_idx_30+1]
            scc_30   = output[row_idx_30, COL['SCC']]
            T_2100   = output[get_year_index(2100), COL['TATM']]
            welf     = (model.params['tstep'] * model.params['scale1'] *
                        output[:, COL['TOTPERIODU']].sum() + model.params['scale2'])

            rows.append({
                'damage_key':    dk,
                'damage_name':   DAMAGE_NAMES[dk],
                'ecs_label':     ecs_label,
                'ecs_val':       ecs_val,
                'ramp_case':     ramp_case,
                'SCC_2030':      round(scc_30, 2),
                'MIU_2030':      round(100*miu_30, 2),
                'MIU_bound_2030':round(100*bound_30, 2),
                'MIU_slack_2030':round(100*(bound_30-miu_30), 2),
                'binding_2030':  classify_binding(miu_30, bound_30),
                'T_2100':        round(T_2100, 3),
                'welfare':       round(welf, 4),
                'status':        'ok' if res.success else 'no_flag',
                'elapsed_s':     elapsed,
            })
            print(f"  SCC={scc_30:.1f} | MIU={100*miu_30:.1f}% | "
                  f"Bound={100*bound_30:.1f}% | "
                  f"Status={classify_binding(miu_30, bound_30)}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            rows.append({'damage_key':dk,'ecs_label':ecs_label,
                         'ramp_case':ramp_case,'status':'failed'})

    df_tgt = pd.DataFrame(rows)
    df_tgt.to_csv(TARGETED_SUMMARY_FILE, index=False)

    print("\n── Targeted ramp × ECS diagnostic ──")
    display(df_tgt[['damage_name','ecs_label','ramp_case',
                    'SCC_2030','MIU_2030','MIU_slack_2030','binding_2030','T_2100']])
    print(f"Saved: {TARGETED_SUMMARY_FILE}")
    return df_tgt

print("Targeted ramp × ECS diagnostic defined. Call run_targeted_ramp_ecs() to execute.")
print(f"Design: {len(TARGETED_RAMP_ECS)} targeted runs")


# execute
run_targeted_ramp_ecs()

# ===== cell 26 =====
import os, glob
print('=== Data sheets written ===')
_dirs = [RESULTS_DIR]
if 'RAMP_RESULTS_DIR' in globals(): _dirs.append(RAMP_RESULTS_DIR)
for d in _dirs:
    for f in sorted(glob.glob(os.path.join(d, '*.csv'))):
        print(f'  {f:70s} {os.path.getsize(f)//1024:6d} KB')
print('\nTrajectory files:', len(glob.glob(os.path.join(RAMP_RUNS_DIR, '*.csv'))) if 'RAMP_RUNS_DIR' in globals() else 0)

# ===== cell 27 =====
# ============================================================
# R11 (standalone) — Welfare cost as CONSUMPTION-EQUIVALENT (money-metric)
# ============================================================
# Permanent fraction k of consumption whose addition to the CONSTRAINED
# (standard-ramp) path equates its welfare to the NO-RAMP path.  k>0 => the
# ramp costs k of permanent consumption.  Reads existing ramp trajectory
# files; NO model re-solve.  Self-contained: falls back gracefully if the
# usual kernel names are absent, and derives L from 1000*C/CPC if needed.
# Run AFTER the ramp experiment (Part B) so the trajectory files exist.
# ============================================================
import os, glob, re
import numpy as np, pandas as pd
from scipy.optimize import brentq

# --- elasticity of marginal utility (scenario 9 / cost-benefit default) ---
ELASMU = float(globals().get('ELASMU', 0.95))   # confirm if you overrode elasmu

# --- locate the ramp runs directory (globals -> RESULTS_DIR -> autodetect) ---
def _resolve_dirs():
    rr  = globals().get('RAMP_RUNS_DIR')
    rrd = globals().get('RAMP_RESULTS_DIR')
    if rr and os.path.isdir(rr):
        return rr, (rrd or os.path.dirname(rr))
    base = globals().get('RESULTS_DIR', './results')
    cand = os.path.join(base, 'ramp_ceiling_experiment_clean')
    rr   = os.path.join(cand, 'runs')
    if os.path.isdir(rr):
        return rr, cand
    # last resort: search for a standard-ramp trajectory anywhere under base
    for hit in glob.glob(os.path.join(base, '**', 'ramp*_standard_dice2023_*.csv'),
                         recursive=True):
        rr = os.path.dirname(hit)
        return rr, os.path.dirname(rr)
    raise FileNotFoundError("Could not locate the ramp 'runs' directory. "
                            "Run the ramp experiment (Part B) first.")

RAMP_RUNS_DIR_, RAMP_RESULTS_DIR_ = _resolve_dirs()

# --- damage keys / names (globals -> derive from filenames) ---
_std = sorted(glob.glob(os.path.join(RAMP_RUNS_DIR_, "ramp*_standard_dice2023_*.csv")))
_keys_from_files = [re.split(r"_standard_dice2023_", os.path.basename(p))[-1][:-4]
                    for p in _std]
DAMAGE_KEYS = list(globals().get('DAMAGE_REGISTRY', _keys_from_files))
_FALLBACK_NAMES = {
    'dietz_stern':'Dietz-Stern (2015)','nordhaus_2016':'Nordhaus DICE-2016',
    'nordhaus_2023':'Nordhaus DICE-2023','weitzman':'Weitzman (2012)',
    'hs_low':'Howard-Sterner Low','hs_central':'Howard-Sterner Central',
    'hs_high':'Howard-Sterner High','kahn':'Kahn et al. (2021)'}
DNAMES = dict(globals().get('DAMAGE_NAMES', {}))
def _name(dk): return DNAMES.get(dk, _FALLBACK_NAMES.get(dk, dk))

# --- helpers ---
def _load(path):
    d = pd.read_csv(path)
    cpc = d['CPC'].to_numpy(float)
    C   = d['C'].to_numpy(float)
    RR  = d['RR'].to_numpy(float)
    L   = d['L'].to_numpy(float) if 'L' in d.columns else (1000.0 * C / cpc)  # L = 1000C/CPC
    return cpc, L, RR, C

def _welfare(cpc, L, RR, elasmu, k=0.0):
    c = (1.0 + k) * cpc
    return np.sum(((c**(1.0 - elasmu) - 1.0) / (1.0 - elasmu) - 1.0) * L * RR)

def _ce_cost(cpc_std, cpc_none, L, RR, elasmu):
    W_target = _welfare(cpc_none, L, RR, elasmu, 0.0)
    f = lambda k: _welfare(cpc_std, L, RR, elasmu, k) - W_target
    return brentq(f, -0.9, 5.0)

def _find(ramp_case, dk):
    hits = sorted(glob.glob(os.path.join(RAMP_RUNS_DIR_, f"ramp*_{ramp_case}_{dk}.csv")))
    if not hits:
        raise FileNotFoundError(f"no trajectory for {ramp_case}/{dk}")
    return hits[0]

# --- compute ---
rows = []
for dk in DAMAGE_KEYS:
    try:
        cpc_s, L, RR, C_s = _load(_find("standard_dice2023", dk))
        cpc_n, _,  _,  _  = _load(_find("no_near_term_ramp", dk))
    except FileNotFoundError as e:
        print("  skip", dk, "|", e); continue
    k  = _ce_cost(cpc_s, cpc_n, L, RR, ELASMU)
    pv = float(np.sum(C_s * RR))
    rows.append({
        "damage_key":             dk,
        "damage_name":            _name(dk),
        "welfare_cost_ce_pct":    round(100.0 * k, 4),   # consumption-equivalent, %
        "welfare_cost_ce_usd_tn": round(k * pv, 2),      # present-value cost, trillion $
        "pv_consumption_usd_tn":  round(pv, 1),
    })

df_ce = pd.DataFrame(rows).sort_values("welfare_cost_ce_pct", ascending=False)
try:
    display(df_ce)
except NameError:
    print(df_ce.to_string(index=False))

os.makedirs(RAMP_RESULTS_DIR_, exist_ok=True)
ce_path = os.path.join(RAMP_RESULTS_DIR_, "welfare_cost_consumption_equivalent.csv")
df_ce.to_csv(ce_path, index=False)
print("Saved:", ce_path)
print("Runs dir used:", RAMP_RUNS_DIR_, "| ELASMU =", ELASMU)
# Sanity: values small & positive; Howard-Sterner High largest, interior smallest.
# This is the money-metric welfare column for tab:binding_diag (percent) and PV cost.

# ===== cell 28 =====
# ============================================================
# FINAL missing calculation — converge the capacity wedge w_nu
# ============================================================
# PREREQUISITE: run Part C (the nu/wedge cells) first, so that
# solve_ceiling, ROW_2030, PERIOD_2030, CPRICE_COL, COL, DAMAGE_NAMES,
# and MULTISTART_N are defined in this kernel.
#
# Binding specs: forward-difference envelope on a finer h-grid + tight
# ftol, Richardson-extrapolate to h->0, and check the estimate has settled.
# Non-binding specs: nu = 0 by complementary slackness, so w_nu = 0 and
# r = Delta (regenerated here so Nordhaus-2023 updates to 61.67 / -0.73).
# Writes wedge_split_final.csv and prints the paste-ready LaTeX.
# ============================================================
assert 'solve_ceiling' in globals(), "Run Part C (the nu/wedge cells) first."
import numpy as np, pandas as pd, os

MU_BAR   = 0.24
H_FINE   = [0.005, 0.002, 0.001, 0.0005]     # forward steps; 2:1 pairs for Richardson
BINDING  = ['hs_central', 'kahn', 'hs_high']
NONBIND  = ['dietz_stern', 'nordhaus_2016', 'weitzman', 'nordhaus_2023']

def _pieces(base):
    o, p = base['out'], base['params']
    scc  = float(o[ROW_2030, COL['SCC']])
    pg   = float(o[ROW_2030, CPRICE_COL])
    miu  = float(o[ROW_2030, COL['MIU']])
    stot = float(p['sigmatot'][PERIOD_2030])
    ygr  = float(o[ROW_2030, COL['YGROSS']])
    Phi  = float(o[ROW_2030, COL['CPC']])**(-p['elasmu']) * float(o[ROW_2030, COL['RR']])
    return scc, pg, miu, stot * ygr * Phi

rows = []

print("BINDING specs — finer-h convergence")
print("-"*72)
for dk in BINDING:
    base = solve_ceiling(dk, MU_BAR, n_starts=MULTISTART_N)
    scc, pg, miu, denom = _pieces(base)
    Delta = scc - pg
    print(f"\n{DAMAGE_NAMES[dk]}  (SCC={scc:.2f}, p_gross={pg:.2f}, Delta={Delta:.2f})")
    w = {}
    for h in H_FINE:
        pert = solve_ceiling(dk, MU_BAR + h, n_starts=1, x0=base['x'], ftol=1e-10)
        nu = (pert['W0'] - base['W0']) / h
        w[h] = nu / denom
        print(f"   h={h:<8g}  w_nu={w[h]:8.3f}   r={Delta-w[h]:7.3f}   Delta/w_nu={Delta/w[h]:.4f}")
    rich = lambda h1, h2: 2*w[h2] - w[h1]            # forward diff is O(h); h2 = h1/2
    w0_coarse = rich(0.002, 0.001)
    w0_fine   = rich(0.001, 0.0005)
    settled   = abs(w0_fine - w0_coarse)
    print(f"   Richardson h->0:  {w0_coarse:.3f} (.002/.001)  vs  {w0_fine:.3f} (.001/.0005)"
          f"   |diff|={settled:.3f}  {'SETTLED' if settled < 0.10 else 'NOT settled -> add smaller h'}")
    w0 = w0_fine
    rows.append(dict(dk=dk, name=DAMAGE_NAMES[dk], status='binding',
                     SCC=round(scc,2), p_gross=round(pg,2), Delta=round(Delta,2),
                     w_nu=round(w0,2), r=round(Delta-w0,2), share_pct=round(100*w0/Delta,1),
                     richardson_gap=round(settled,3)))

print("\nNON-BINDING specs — nu = 0 (w_nu = 0, r = Delta)")
print("-"*72)
for dk in NONBIND:
    base = solve_ceiling(dk, MU_BAR, n_starts=MULTISTART_N)
    scc, pg, miu, _ = _pieces(base)
    Delta  = scc - pg
    slack  = 100*(MU_BAR - miu)
    status = 'near-binding' if slack < 1.0 else 'interior'
    print(f"  {DAMAGE_NAMES[dk]:26s} SCC={scc:7.2f}  p_gross={pg:6.2f}  Delta={Delta:6.2f}  "
          f"slack={slack:4.2f}pp  {status}")
    rows.append(dict(dk=dk, name=DAMAGE_NAMES[dk], status=status,
                     SCC=round(scc,2), p_gross=round(pg,2), Delta=round(Delta,2),
                     w_nu=0.00, r=round(Delta,2), share_pct=0.0, richardson_gap=0.0))

df = pd.DataFrame(rows)
order = ['dietz_stern','nordhaus_2016','weitzman','nordhaus_2023','hs_central','kahn','hs_high']
df = df.set_index('dk').loc[[k for k in order if k in df['dk'].values]].reset_index()

out_csv = os.path.join(globals().get('RESULTS_DIR','.'), 'wedge_split_final.csv')
df.to_csv(out_csv, index=False)

print("\n=== FINAL wedge decomposition (one finalized run) ===")
print(df[['name','status','SCC','p_gross','Delta','w_nu','r','share_pct']].to_string(index=False))
print("\nSaved:", out_csv)

print("\n% ---------- paste into tab:wedge_split ----------")
for _, r in df.iterrows():
    nm = r['name'].replace('&', r'\&')
    print(f"{nm:28s} & {r['status']:12s} & {r['Delta']:6.2f} & {r['w_nu']:6.2f} & {r['r']:6.2f} \\\\")
print("\n% NOTE: Nordhaus DICE-2023 p^gross/gap update for tab:binding_diag:")
nr = df[df.dk=='nordhaus_2023'].iloc[0]
print(f"%   p_gross = {nr['p_gross']:.2f}  (was 61.56),   gap Delta = {nr['Delta']:.2f}  (was -0.62)")
