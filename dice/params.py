import numpy as np
from scipy.optimize import root


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
    params['SLower'] = kwargs.get('SLower', 0.0)
    params['SUpper'] = kwargs.get('SUpper', 1.0)
    params['FixSperiod'] = kwargs.get('FixSperiod', 38)
    params['FixSvalue'] = kwargs.get('FixSvalue', 0.28)
    params['AlphaUpperBound'] = kwargs.get('AlphaUpperBound', np.inf)
    params['AlphaLowerBound'] = kwargs.get('AlphaLowerBound', 0.2)
    params['MIULowerBound'] = kwargs.get('MIULowerBound', 0)

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
    if params['FixSperiod'] <= num_periods:
        params['sLBounds'][params['FixSperiod']:] = params['FixSvalue']
        params['sUBounds'][params['FixSperiod']:] = params['FixSvalue']

    params['eco2Param'] = params['aL'] * (params['L'] / 1000) ** (1 - params['gama'])
    return params


def apply_disc_prstp(params, prstp_value, *, elasmu_value=0.001,
                     cap_after_t=None, cap_exp_value=None):
    params['prstp'] = float(prstp_value)
    params['elasmu'] = float(elasmu_value)

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
        params['rprecaut'][:] = 0.0
    params['rartp'] = params['prstp']

    params['optlrsav'] = (
        (params['dk'] + 0.004) /
        (params['dk'] + 0.004 * params['elasmu'] + params['prstp'])
    ) * params['gama']

    return params
