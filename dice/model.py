import numpy as np
import os
import csv
import time
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn
from numba import njit

from dice.params import LoadParams, apply_disc_prstp
from dice.scc import compute_SCC_numba


seaborn.set(style='ticks')


@njit
def diceForward_numba(i, MIU, S, alpha, CCATOT, K, I, F_GHGabate, RES0, RES1, RES2, RES3, TBOX1, TBOX2,
                      tstep, dk, gama, eco2Param, sigma, eland, cost1tot, expcost2, miuup, sLBounds, sUBounds,
                      AlphaLowerBound, Fcoef1, Fcoef2, CO2E_GHGabateB, emshare0, emshare1, emshare2, emshare3,
                      tau0, tau1, tau2, tau3, mateq, fco22x, F_Misc, teq1, teq2, d1, d2, a1, a2base, a3,
                      pulse_GtCO2_per_year):

    MIU_i   = min(max(MIU[i], 0.0), miuup[i])
    S_i     = min(max(S[i], sLBounds[i]), sUBounds[i])
    alpha_i = max(alpha[i], AlphaLowerBound)

    K = (1.0 - dk) ** tstep * K + tstep * I
    YGROSS = eco2Param[i] * (K ** gama)
    ECO2   = (sigma[i] * YGROSS) * (1.0 - MIU_i) + eland[i]
    CCATOT = CCATOT + (ECO2 + pulse_GtCO2_per_year[i]) * (tstep / 3.667)
    F_GHGabate = Fcoef2 * F_GHGabate + Fcoef1 * CO2E_GHGabateB[i] * (1.0 - MIU_i)

    total_flow       = ECO2 + pulse_GtCO2_per_year[i]
    inflow_GtC_per_yr = total_flow / 3.667

    RES0 = (emshare0 * tau0 * alpha_i * inflow_GtC_per_yr) * (1.0 - np.exp(-tstep / (tau0 * alpha_i))) + RES0 * np.exp(-tstep / (tau0 * alpha_i))
    RES1 = (emshare1 * tau1 * alpha_i * inflow_GtC_per_yr) * (1.0 - np.exp(-tstep / (tau1 * alpha_i))) + RES1 * np.exp(-tstep / (tau1 * alpha_i))
    RES2 = (emshare2 * tau2 * alpha_i * inflow_GtC_per_yr) * (1.0 - np.exp(-tstep / (tau2 * alpha_i))) + RES2 * np.exp(-tstep / (tau2 * alpha_i))
    RES3 = (emshare3 * tau3 * alpha_i * inflow_GtC_per_yr) * (1.0 - np.exp(-tstep / (tau3 * alpha_i))) + RES3 * np.exp(-tstep / (tau3 * alpha_i))

    MAT  = mateq + RES0 + RES1 + RES2 + RES3
    FORC = fco22x * np.log(MAT / mateq) / np.log(2.0) + F_Misc[i] + F_GHGabate
    TBOX1 = TBOX1 * np.exp(-tstep / d1) + teq1 * FORC * (1.0 - np.exp(-tstep / d1))
    TBOX2 = TBOX2 * np.exp(-tstep / d2) + teq2 * FORC * (1.0 - np.exp(-tstep / d2))
    TATM  = min(max(TBOX1 + TBOX2, 0.01), 20.0)

    DAMFRAC   = a1 * TATM + a2base * TATM ** a3
    ABATECOST = YGROSS * cost1tot[i] * (MIU_i ** expcost2)
    YNET = YGROSS * (1.0 - DAMFRAC)
    Y    = YNET - ABATECOST
    I    = S_i * Y
    C    = Y - I

    if not (np.isfinite(MAT) and np.isfinite(TATM) and np.isfinite(C) and np.isfinite(YGROSS)):
        return np.array([np.nan] * 18, dtype=np.float64)

    return np.array([C, CCATOT, K, I, F_GHGabate, RES0, RES1, RES2, RES3,
                     TBOX1, TBOX2, MAT, TATM, Y, YNET, YGROSS, DAMFRAC, ABATECOST], dtype=np.float64)


@njit
def diceTrajectory_numba(MIU, S, alpha, num_periods, tstep, dk, gama, eco2Param, sigma, eland, cost1tot, expcost2,
                         miuup, sLBounds, sUBounds, AlphaLowerBound, Fcoef1, Fcoef2, CO2E_GHGabateB, emshare0,
                         emshare1, emshare2, emshare3, tau0, tau1, tau2, tau3, mateq, fco22x, F_Misc, teq1, teq2,
                         d1, d2, a1, a2base, a3, res00, res10, res20, res30, tbox10, tbox20, CumEmiss0,
                         F_GHGabate2020, k0, tatm0, mat0, a0, L, RR, scale1, scale2, elasmu, pulse_GtCO2_per_year):

    C = np.zeros(num_periods + 1); K = np.zeros(num_periods + 1)
    CCATOT = np.zeros(num_periods + 1); MAT = np.zeros(num_periods + 1)
    TATM = np.zeros(num_periods + 1); Y = np.zeros(num_periods + 1)
    YNET = np.zeros(num_periods + 1); YGROSS = np.zeros(num_periods + 1)
    DAMFRAC = np.zeros(num_periods + 1); ABATECOST = np.zeros(num_periods + 1)
    RES0_arr = np.zeros(num_periods + 1); RES1_arr = np.zeros(num_periods + 1)
    RES2_arr = np.zeros(num_periods + 1); RES3_arr = np.zeros(num_periods + 1)
    TBOX1_arr = np.zeros(num_periods + 1); TBOX2_arr = np.zeros(num_periods + 1)
    F_GHGab_arr = np.zeros(num_periods + 1)

    RES0 = res00; RES1 = res10; RES2 = res20; RES3 = res30
    TBOX1 = tbox10; TBOX2 = tbox20
    CCATOT[1] = CumEmiss0
    F_GHGabate = F_GHGabate2020
    K[1] = k0; TATM[1] = tatm0
    DAMFRAC[1]  = a1 * TATM[1] + a2base * TATM[1] ** a3
    YGROSS[1]   = eco2Param[1] * (K[1] ** gama)
    YNET[1]     = YGROSS[1] * (1.0 - DAMFRAC[1])
    ABATECOST[1]= YGROSS[1] * cost1tot[1] * (MIU[1] ** expcost2)
    Y[1]        = YNET[1] - ABATECOST[1]
    I           = S[1] * Y[1]
    C[1]        = Y[1] - I

    RES0_arr[1]=RES0; RES1_arr[1]=RES1; RES2_arr[1]=RES2; RES3_arr[1]=RES3
    TBOX1_arr[1]=TBOX1; TBOX2_arr[1]=TBOX2
    F_GHGab_arr[1]=F_GHGabate; MAT[1]=mat0

    for i in range(2, num_periods + 1):
        result = diceForward_numba(
            i, MIU, S, alpha, CCATOT[i-1], K[i-1], I, F_GHGabate,
            RES0, RES1, RES2, RES3, TBOX1, TBOX2,
            tstep, dk, gama, eco2Param, sigma, eland, cost1tot, expcost2,
            miuup, sLBounds, sUBounds, AlphaLowerBound, Fcoef1, Fcoef2,
            CO2E_GHGabateB, emshare0, emshare1, emshare2, emshare3,
            tau0, tau1, tau2, tau3, mateq, fco22x, F_Misc, teq1, teq2,
            d1, d2, a1, a2base, a3, pulse_GtCO2_per_year)

        if np.any(np.isnan(result)):
            nan = np.full(num_periods + 1, np.nan)
            return (np.nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan, nan, nan)

        C[i]=result[0]; CCATOT[i]=result[1]; K[i]=result[2]
        I=result[3]; F_GHGabate=result[4]
        RES0=result[5]; RES1=result[6]; RES2=result[7]; RES3=result[8]
        TBOX1=result[9]; TBOX2=result[10]
        MAT[i]=result[11]; TATM[i]=result[12]
        Y[i]=result[13]; YNET[i]=result[14]; YGROSS[i]=result[15]
        DAMFRAC[i]=result[16]; ABATECOST[i]=result[17]
        RES0_arr[i]=RES0; RES1_arr[i]=RES1; RES2_arr[i]=RES2; RES3_arr[i]=RES3
        TBOX1_arr[i]=TBOX1; TBOX2_arr[i]=TBOX2; F_GHGab_arr[i]=F_GHGabate

    PERIODU   = ((C[1:] * 1000.0 / L[1:]) ** (1.0 - elasmu) - 1.0) / (1.0 - elasmu) - 1.0
    TOTPERIODU = PERIODU * L[1:] * RR[1:]
    UTILITY   = tstep * scale1 * np.sum(TOTPERIODU) + scale2

    return (UTILITY, C, CCATOT, MAT, TATM, K, Y, YNET, YGROSS, DAMFRAC, ABATECOST,
            RES0_arr, RES1_arr, RES2_arr, RES3_arr, TBOX1_arr, TBOX2_arr, F_GHGab_arr)


def diceTrajectory(params, MIU, S, alpha, pulse_GtCO2_per_year=None):
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
        params['sLBounds'], params['sUBounds'], params['AlphaLowerBound'], params['Fcoef1'], params['Fcoef2'],
        params['CO2E_GHGabateB'], params['emshare0'], params['emshare1'], params['emshare2'], params['emshare3'],
        params['tau0'], params['tau1'], params['tau2'], params['tau3'], params['mateq'], params['fco22x'],
        params['F_Misc'], params['teq1'], params['teq2'], params['d1'], params['d2'], params['a1'],
        params['a2base'], params['a3'], params['res00'], params['res10'], params['res20'], params['res30'],
        params['tbox10'], params['tbox20'], params['CumEmiss0'], params['F_GHGabate2020'], params['k0'],
        params['tatm0'], params['mat0'], params['a0'], params['L'], params['RR'], params['scale1'],
        params['scale2'], params['elasmu'], pulse_GtCO2_per_year
    )


@njit(cache=False)
def compute_SCC_numba(MIU, S, alpha, num_periods, tstep, dk, gama, eco2Param, sigma, eland, cost1tot, expcost2,
                      miuup, sLBounds, sUBounds, AlphaLowerBound, Fcoef1, Fcoef2, CO2E_GHGabateB, emshare0,
                      emshare1, emshare2, emshare3, tau0, tau1, tau2, tau3, mateq, fco22x, F_Misc, teq1, teq2,
                      d1, d2, a1, a2base, a3, res00, res10, res20, res30, tbox10, tbox20, CumEmiss0,
                      F_GHGabate2020, k0, tatm0, mat0, a0, L, RR, scale1, scale2, elasmu):

    from dice.model import diceTrajectory_numba

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

        C_alt_h   = alt_h[1]
        CPC_alt_h = 1000.0 * C_alt_h[1:] / L[1:]
        W_h  = np.sum(((CPC_alt_h ** (1.0 - elasmu)) - 1.0) / (1.0 - elasmu) - 1.0) * L[1:] * RR[1:]
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

        C_alt_h2   = alt_h2[1]
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



@njit
def recoverAllVars_numba(x, num_periods, tstep, dk, gama, eco2Param, sigma, eland, cost1tot, expcost2, miuup,
                         sLBounds, sUBounds, AlphaLowerBound, Fcoef1, Fcoef2, CO2E_GHGabateB, emshare0, emshare1,
                         emshare2, emshare3, tau0, tau1, tau2, tau3, mateq, fco22x, F_Misc, teq1, teq2, d1, d2,
                         a1, a2base, a3, res00, res10, res20, res30, tbox10, tbox20, CumEmiss0, F_GHGabate2020,
                         k0, tatm0, mat0, a0, L, RR, scale1, scale2, elasmu, PBACKTIME, irf0, irC, irT, SRF):
    MIU = np.zeros(num_periods + 1)
    S   = np.zeros(num_periods + 1)
    alpha = np.zeros(num_periods + 1)
    MIU[1:]   = x[:num_periods]
    S[1:]     = x[num_periods:2 * num_periods]
    alpha[1:] = x[2 * num_periods:]
    alpha[1]  = a0
    pulse_GtCO2_per_year = np.zeros(num_periods + 1, dtype=np.float64)
    result = diceTrajectory_numba(
        MIU, S, alpha, num_periods, tstep, dk, gama, eco2Param, sigma, eland, cost1tot, expcost2, miuup, sLBounds,
        sUBounds, AlphaLowerBound, Fcoef1, Fcoef2, CO2E_GHGabateB, emshare0, emshare1, emshare2, emshare3, tau0,
        tau1, tau2, tau3, mateq, fco22x, F_Misc, teq1, teq2, d1, d2, a1, a2base, a3, res00, res10, res20, res30,
        tbox10, tbox20, CumEmiss0, F_GHGabate2020, k0, tatm0, mat0, a0, L, RR, scale1, scale2, elasmu,
        pulse_GtCO2_per_year)
    if np.isnan(result[0]):
        return np.zeros((num_periods, 46))

    C=result[1]; CCATOT=result[2]; MAT=result[3]; TATM=result[4]; K=result[5]
    Y=result[6]; YNET=result[7]; YGROSS=result[8]; DAMFRAC=result[9]; ABATECOST=result[10]
    RES0_arr=result[11]; RES1_arr=result[12]; RES2_arr=result[13]; RES3_arr=result[14]
    TBOX1_arr=result[15]; TBOX2_arr=result[16]; F_GHGab_arr=result[17]

    safe_MAT = np.maximum(MAT, mateq + 1e-6)
    I        = S * Y
    CACC     = CCATOT - (MAT - mateq)
    DAMAGES  = YGROSS * DAMFRAC
    CPRICE   = PBACKTIME * (MIU ** (expcost2 - 1.0))
    EIND     = sigma * (eco2Param * (K ** gama)) * (1.0 - MIU)
    ECO2     = EIND + eland
    ECO2E    = ECO2 + CO2E_GHGabateB * (1.0 - MIU)
    FORC     = fco22x * np.log(safe_MAT / mateq) / np.log(2.0) + F_Misc + F_GHGab_arr
    FORC_CO2 = fco22x * np.log(safe_MAT / mateq) / np.log(2.0)
    CPC = np.zeros(num_periods + 1)
    for t in range(1, num_periods + 1):
        CPC[t] = 1000.0 * C[t] / L[t]

    RFACTLONG = np.full(num_periods + 1, SRF)
    for i in range(2, num_periods + 1):
        RFACTLONG[i] = SRF * (CPC[i-1] / CPC[1]) ** (-elasmu) * RR[i]
    RLONG  = np.zeros(num_periods + 1)
    RSHORT = np.zeros(num_periods + 1)
    for i in range(2, num_periods + 1):
        RLONG[i]  = -np.log(RFACTLONG[i] / SRF) / (5.0 * (i - 1))
        RSHORT[i] = -np.log(RFACTLONG[i] / RFACTLONG[i-1]) / 5.0

    PERIODU   = np.zeros(num_periods + 1)
    TOTPERIODU = np.zeros(num_periods + 1)
    for t in range(1, num_periods + 1):
        PERIODU[t]   = ((C[t] * 1000.0 / L[t]) ** (1.0 - elasmu) - 1.0) / (1.0 - elasmu) - 1.0
        TOTPERIODU[t] = PERIODU[t] * L[t] * RR[t]

    IRFt = irf0 + irC * (CCATOT - (MAT - mateq)) + irT * TATM

    SCC = compute_SCC_numba(
        MIU, S, alpha, num_periods, tstep, dk, gama, eco2Param, sigma, eland, cost1tot, expcost2, miuup, sLBounds,
        sUBounds, AlphaLowerBound, Fcoef1, Fcoef2, CO2E_GHGabateB, emshare0, emshare1, emshare2, emshare3, tau0,
        tau1, tau2, tau3, mateq, fco22x, F_Misc, teq1, teq2, d1, d2, a1, a2base, a3, res00, res10, res20, res30,
        tbox10, tbox20, CumEmiss0, F_GHGabate2020, k0, tatm0, mat0, a0, L, RR, scale1, scale2, elasmu)

    output = np.zeros((num_periods, 46))
    for i in range(num_periods):
        t = i + 1
        output[i,0]=EIND[t]; output[i,1]=ECO2[t]; output[i,2]=MAT[t]/2.13
        output[i,3]=TATM[t]; output[i,4]=Y[t]; output[i,5]=DAMFRAC[t]
        output[i,6]=CPC[t]; output[i,7]=CPRICE[t]; output[i,8]=MIU[t]
        output[i,9]=RSHORT[t]; output[i,10]=ECO2E[t]; output[i,11]=L[t]
        output[i,12]=eco2Param[t]; output[i,13]=YGROSS[t]; output[i,14]=K[t]
        output[i,15]=S[t]; output[i,16]=I[t]; output[i,17]=YNET[t]
        output[i,18]=CCATOT[t]; output[i,19]=CACC[t]; output[i,20]=RES0_arr[t]
        output[i,21]=RES1_arr[t]; output[i,22]=RES2_arr[t]; output[i,23]=RES3_arr[t]
        output[i,24]=DAMAGES[t]; output[i,25]=ABATECOST[t]
        output[i,26]=PBACKTIME[t]*(MIU[t]**(expcost2-1.0)); output[i,27]=C[t]
        output[i,28]=PERIODU[t]; output[i,29]=TOTPERIODU[t]; output[i,30]=MAT[t]
        output[i,31]=FORC[t]; output[i,32]=TBOX1_arr[t]; output[i,33]=TBOX2_arr[t]
        output[i,34]=F_GHGab_arr[t]; output[i,35]=IRFt[t]; output[i,36]=alpha[t]
        output[i,37]=RFACTLONG[t]; output[i,38]=RLONG[t]; output[i,39]=SCC[t]
        output[i,40]=ABATECOST[t]/max(Y[t],1e-12); output[i,41]=MAT[t]/mat0
        output[i,42]=MAT[t]/mateq; output[i,43]=FORC_CO2[t]; output[i,44]=RR[t]

    if not np.all(np.isfinite(output)):
        return np.zeros((num_periods, 46))
    return output


def recoverAllVars(x, params):
    return recoverAllVars_numba(
        x, params['num_periods'], params['tstep'], params['dk'], params['gama'], params['eco2Param'],
        params['sigma'], params['eland'], params['cost1tot'], params['expcost2'], params['miuup'],
        params['sLBounds'], params['sUBounds'], params['AlphaLowerBound'], params['Fcoef1'], params['Fcoef2'],
        params['CO2E_GHGabateB'], params['emshare0'], params['emshare1'], params['emshare2'], params['emshare3'],
        params['tau0'], params['tau1'], params['tau2'], params['tau3'], params['mateq'], params['fco22x'],
        params['F_Misc'], params['teq1'], params['teq2'], params['d1'], params['d2'], params['a1'],
        params['a2base'], params['a3'], params['res00'], params['res10'], params['res20'], params['res30'],
        params['tbox10'], params['tbox20'], params['CumEmiss0'], params['F_GHGabate2020'], params['k0'],
        params['tatm0'], params['mat0'], params['a0'], params['L'], params['RR'], params['scale1'],
        params['scale2'], params['elasmu'], params['PBACKTIME'], params['irf0'], params['irC'],
        params['irT'], params['SRF'])


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
        self.S[params['FixSperiod']:] = params['optlrsav']
        self.Alpha = np.linspace(params['a0'], 0.425, num_periods + 1)
        self.Alpha[1] = params['a0']

    def pack(self, x):
        return x[:self.num_periods], x[self.num_periods:2*self.num_periods], x[2*self.num_periods:]

    def objective(self, x):
        MIU, S, Alpha = self.pack(x)
        self.MIU[1:self.num_periods+1] = MIU
        self.S[1:self.num_periods+1]   = S
        self.Alpha[1:self.num_periods+1] = Alpha
        out = diceTrajectory(self.params, self.MIU, self.S, self.Alpha)
        UTILITY, C = out[0], out[1]
        if np.any(np.isnan(C)):
            return 1e15
        return -UTILITY

    def irf_residual(self, x):
        MIU, S, Alpha = self.pack(x)
        self.MIU[1:self.num_periods+1] = MIU
        self.S[1:self.num_periods+1]   = S
        self.Alpha[1:self.num_periods+1] = Alpha
        out   = diceTrajectory(self.params, self.MIU, self.S, self.Alpha)
        CCATOT = out[2]; MAT = out[3]; TATM = out[4]
        if np.any(np.isnan(TATM)):
            return np.ones(self.num_periods) * 1e10
        IRF_LHS = (self.params['irf0'] + self.params['irC'] * (CCATOT[1:] - (MAT[1:] - self.params['mateq'])) + self.params['irT'] * TATM[1:])
        IRF_RHS = (Alpha * self.params['emshare0'] * self.params['tau0'] * (1 - np.exp(-100 / (Alpha * self.params['tau0']))) +
                   Alpha * self.params['emshare1'] * self.params['tau1'] * (1 - np.exp(-100 / (Alpha * self.params['tau1']))) +
                   Alpha * self.params['emshare2'] * self.params['tau2'] * (1 - np.exp(-100 / (Alpha * self.params['tau2']))) +
                   Alpha * self.params['emshare3'] * self.params['tau3'] * (1 - np.exp(-100 / (Alpha * self.params['tau3']))))
        return IRF_LHS - IRF_RHS

    def temp_up(self, x):
        MIU, S, Alpha = self.pack(x)
        self.MIU[1:self.num_periods+1] = MIU
        self.S[1:self.num_periods+1]   = S
        self.Alpha[1:self.num_periods+1] = Alpha
        out  = diceTrajectory(self.params, self.MIU, self.S, self.Alpha)
        TATM = out[4]
        if np.any(np.isnan(TATM)):
            return np.ones(self.num_periods) * -1e10
        return self.TempUpperConstraint - TATM[1:]

    def temp_lo(self, x):
        MIU, S, Alpha = self.pack(x)
        self.MIU[1:self.num_periods+1] = MIU
        self.S[1:self.num_periods+1]   = S
        self.Alpha[1:self.num_periods+1] = Alpha
        out  = diceTrajectory(self.params, self.MIU, self.S, self.Alpha)
        TATM = out[4]
        if np.any(np.isnan(TATM)):
            return np.ones(self.num_periods) * -1e10
        return TATM[1:] - self.TempLowerConstraint


class Dice2023Model:
    def __init__(self, num_times, scenario):
        self.num_periods = num_times
        self.scenario    = scenario
        self.params      = LoadParams(num_times)
        self.TempUpperConstraint = 20.0
        self.TempLowerConstraint = 0.01
        if scenario == 6:
            self.TempUpperConstraint = 1.5
        elif scenario == 7:
            self.TempUpperConstraint = 2.0

    def run_model(self):
        if self.scenario == 1:
            self.params['k0'] = 420
            apply_disc_prstp(self.params, prstp_value=0.01)
        elif self.scenario == 2:
            self.params['k0'] = 409
            apply_disc_prstp(self.params, prstp_value=0.02)
        elif self.scenario == 3:
            self.params['k0'] = 370
            apply_disc_prstp(self.params, prstp_value=0.03)
        elif self.scenario == 4:
            self.params['k0'] = 326
            apply_disc_prstp(self.params, prstp_value=0.04, cap_after_t=81, cap_exp_value=5*80)
        elif self.scenario == 5:
            self.params['k0'] = 290
            apply_disc_prstp(self.params, prstp_value=0.05, cap_after_t=51, cap_exp_value=5*51)
        elif self.scenario == 8:
            for i in range(1, self.num_periods + 1):
                self.params['miuup'][i] = min(0.05 + 0.04*(i-1) - 0.01*max(0, i-5), self.params['limmiu2070'])
        elif self.scenario == 10:
            self.TempUpperConstraint = 15.0
            self.params['miuup'][:] = 1.0

        if self.scenario not in {1, 2, 3, 4, 5}:
            self.params['rartp'] = np.exp(self.params['prstp'] + self.params['betaclim'] * self.params['pi']) - 1
            self.params['optlrsav'] = ((self.params['dk'] + 0.004) / (self.params['dk'] + 0.004 * self.params['elasmu'] + self.params['rartp'])) * self.params['gama']

        prob = DiceFunc(self.num_periods, self.params,
                        TempUpperConstraint=self.TempUpperConstraint,
                        TempLowerConstraint=self.TempLowerConstraint)

        x0 = np.concatenate([prob.MIU[1:self.num_periods+1],
                              prob.S[1:self.num_periods+1],
                              prob.Alpha[1:self.num_periods+1]])

        miu_bounds = []
        for t in range(1, self.num_periods + 1):
            if t == 1:
                miu_bounds.append((self.params['miu1'], self.params['miu1']))
            elif self.scenario == 10 and t > 57:
                miu_bounds.append((1.0, 1.0))
            else:
                miu_bounds.append((self.params['MIULowerBound'], self.params['miuup'][t]))

        s_bounds     = [(self.params['sLBounds'][i], self.params['sUBounds'][i]) for i in range(1, self.num_periods+1)]
        alpha_bounds = [(self.params['AlphaLowerBound'], self.params['AlphaUpperBound']) for _ in range(self.num_periods)]
        bounds       = miu_bounds + s_bounds + alpha_bounds

        constraints = [
            {'type': 'eq',   'fun': prob.irf_residual},
            {'type': 'ineq', 'fun': prob.temp_up},
            {'type': 'ineq', 'fun': prob.temp_lo},
        ]

        if self.scenario == 6:
            temp_steps = [2.0, 1.9, 1.8, 1.79, 1.78, 1.77, 1.76, 1.75,
                          1.74, 1.73, 1.72, 1.71, 1.70,
                          1.68, 1.66, 1.64, 1.62, 1.60,
                          1.58, 1.56, 1.5]
            x_current = x0.copy(); last_success = x0.copy()
            for temp_limit in temp_steps:
                print(f"\n  Homotopy step: TempUpperConstraint = {temp_limit}°C")
                prob_step = DiceFunc(self.num_periods, self.params,
                                     TempUpperConstraint=temp_limit,
                                     TempLowerConstraint=self.TempLowerConstraint)
                cons_step = [{'type': 'eq',   'fun': prob_step.irf_residual},
                             {'type': 'ineq', 'fun': prob_step.temp_up},
                             {'type': 'ineq', 'fun': prob_step.temp_lo}]
                step_res = minimize(prob_step.objective, x_current, method='SLSQP',
                                    bounds=bounds, constraints=cons_step,
                                    options={'maxiter': 1000, 'ftol': 1e-7, 'disp': True})
                irf_res = np.abs(prob_step.irf_residual(step_res.x)).max()
                if step_res.success and irf_res < 0.01:
                    x_current = step_res.x.copy(); last_success = step_res.x.copy()
            res = step_res; res.x = last_success
        else:
            def callback(xk):
                obj     = prob.objective(xk)
                irf_res = np.max(np.abs(prob.irf_residual(xk)))
                print(f"Objective={-obj:.2f}, IRF max residual={irf_res:.2e}")

            start = time.time()
            print("Starting optimization...")
            res = minimize(prob.objective, x0, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': 25000, 'ftol': 1e-5, 'disp': True, 'eps': 1e-6},
                           callback=callback)
            print(f"Done in {time.time()-start:.1f}s | Success: {res.success}")

        x_opt  = res.x
        output = recoverAllVars(x_opt, self.params)
        return x_opt, output, None

    def dump_parameters(self):
        os.makedirs("./results", exist_ok=True)
        path = f"./results/dice2023_parameters_scen{self.scenario}.csv"
        with open(path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["PERIOD","L","gA","aL","gsig","sigma","sigmatot",
                              "PBACKTIME","cost1tot","eland","cpricebase",
                              "F_Misc","CO2E_GHGabateB","emissrat","RR1","RR","miuup"])
            for t in range(1, self.num_periods + 1):
                writer.writerow([t, self.params['L'][t], self.params['gA'][t],
                                  self.params['aL'][t], self.params['gsig'][t],
                                  self.params['sigma'][t], self.params['sigmatot'][t],
                                  self.params['PBACKTIME'][t], self.params['cost1tot'][t],
                                  self.params['eland'][t], self.params['cpricebase'][t],
                                  self.params['F_Misc'][t], self.params['CO2E_GHGabateB'][t],
                                  self.params['emissrat'][t], self.params['RR1'][t],
                                  self.params['RR'][t], self.params['miuup'][t]])
        print(f"Parameters written to {path}")

    def dump_state(self, years, output, filename, scenario):
        os.makedirs("./results", exist_ok=True)
        path = f"./results/dice2023_state_scen{scenario}.csv"
        with open(path, mode="w", newline='') as f:
            writer = csv.writer(f)
            header = ["EIND","ECO2","CO2PPM","TATM","Y","DAMFRAC","CPC","CPRICE","MIUopt","RSHORT",
                      "ECO2E","L","AL","YGROSS","K","Sopt","I","YNET","CCATOT","CACC",
                      "RES0","RES1","RES2","RES3","DAMAGES","ABATECOST","MCABATE","C",
                      "PERIODU","TOTPERIODU","MAT","FORC","TBOX1","TBOX2","F_GHGABATE",
                      "IRFT","ALPHA","RFACTLONG","RLONG","SCC","ABATERAT","ATFRAC2020",
                      "ATFRAC1765","FORC_CO2","RR"]
            writer.writerow(['IPERIOD'] + header)
            for i in range(self.num_periods):
                writer.writerow([i+1] + list(output[i, :len(header)]))
        print(f"State written to {path}")

    def plot_state_to_file(self, fileName, years, output, x):
        os.makedirs("./results", exist_ok=True)
        num_periods = output.shape[0]
        output_T = np.transpose(output)
        pp = PdfPages(fileName)
        all_columns = [
            (3,  "Atmospheric Temperature (TATM)", "°C above 1765"),
            (1,  "Total CO2 Emissions (ECO2)", "GtCO2/year"),
            (6,  "Consumption per Capita (CPC)", "Thousand USD 2019"),
            (7,  "Carbon Price (CPRICE)", "USD 2019/tCO2"),
            (8,  "Emission Control Rate (MIUopt)", "Rate"),
            (15, "Saving Rate (Sopt)", "Rate"),
            (39, "Social Cost of Carbon (SCC)", "USD 2019/tCO2"),
        ]
        for col_idx, title, ylabel in all_columns:
            fig = plt.figure(figsize=(8, 6), dpi=72, facecolor="white")
            plt.plot(years, output_T[col_idx])
            plt.title(title, fontsize=16)
            plt.xlabel("Years", fontsize=12)
            plt.ylabel(ylabel, fontsize=14)
            seaborn.despine()
            fig.tight_layout()
            pp.savefig(fig); plt.close(fig)
        pp.close()
        print(f"Wrote plots to {fileName}")


def display_scenarios():
    print("Select a scenario:")
    for i, label in enumerate(["1% Discounting","2% Discounting","3% Discounting",
                                "4% Discounting","5% Discounting","Max Temp 1.5°C",
                                "Max Temp 2°C","Paris","Optimal","Base"], start=1):
        print(f"{i}. {label}")
    while True:
        try:
            choice = int(input("Enter scenario number (1-10): "))
            if 1 <= choice <= 10:
                return choice
        except ValueError:
            pass
        print("Invalid. Please enter a number between 1 and 10.")


if __name__ == "__main__":
    scenario = display_scenarios()
    model = Dice2023Model(num_times=81, scenario=scenario)
    model.dump_parameters()
    years = np.arange(model.params['yr0'],
                      model.params['yr0'] + model.params['tstep'] * model.num_periods,
                      model.params['tstep'])
    x_opt, output, _ = model.run_model()
    model.dump_state(years, output, "./results/dice2023_state.csv", scenario)
    model.plot_state_to_file(f"./results/dice2023_plots_scen{scenario}.pdf", years, output, x_opt)
