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

seaborn.set(style='ticks')


@njit
def diceForward_numba(i, MIU, S, alpha, CCATOT, K, I, F_GHGabate, RES0, RES1, RES2, RES3, TBOX1, TBOX2,
                      tstep, dk, gama, eco2Param, sigma, eland, cost1tot, expcost2, miuup, sLBounds, sUBounds,
                      AlphaLowerBound, Fcoef1, Fcoef2, CO2E_GHGabateB, emshare0, emshare1, emshare2, emshare3,
                      tau0, tau1, tau2, tau3, mateq, fco22x, F_Misc, teq1, teq2, d1, d2, a1, a2base, a3, 
                      pulse_GtCO2_per_year):

    # Clamp controls
    MIU_i = min(max(MIU[i], 0.0), miuup[i])
    S_i = min(max(S[i], sLBounds[i]), sUBounds[i])
    alpha_i = max(alpha[i], AlphaLowerBound)
    
    # Capital accumulation
    K = (1.0 - dk) ** tstep * K + tstep * I
    
    # Calculate CURRENT period gross output
    YGROSS = eco2Param[i] * (K ** gama)
    
    # Calculate CURRENT period emissions with CURRENT MIU
    ECO2 = (sigma[i] * YGROSS ) * (1.0 - MIU_i) + eland[i]
    
    # CORRECTED: Update CCATOT with current period emissions
    CCATOT = CCATOT + (ECO2 + pulse_GtCO2_per_year[i]) * (tstep / 3.667)
    
    # CORRECTED: F_GHGabate uses current period values
    F_GHGabate = Fcoef2 * F_GHGabate + Fcoef1 * CO2E_GHGabateB[i] * (1.0 - MIU_i)
    
    # Carbon boxes - use same emissions as CCATOT for consistency
    total_flow = ECO2 + pulse_GtCO2_per_year[i]
    inflow_GtC_per_yr = total_flow / 3.667
    
    RES0 = (emshare0 * tau0 * alpha_i * inflow_GtC_per_yr) * (1.0 - np.exp(-tstep / (tau0 * alpha_i))) + RES0 * np.exp(-tstep / (tau0 * alpha_i))
    RES1 = (emshare1 * tau1 * alpha_i * inflow_GtC_per_yr) * (1.0 - np.exp(-tstep / (tau1 * alpha_i))) + RES1 * np.exp(-tstep / (tau1 * alpha_i))
    RES2 = (emshare2 * tau2 * alpha_i * inflow_GtC_per_yr) * (1.0 - np.exp(-tstep / (tau2 * alpha_i))) + RES2 * np.exp(-tstep / (tau2 * alpha_i))
    RES3 = (emshare3 * tau3 * alpha_i * inflow_GtC_per_yr) * (1.0 - np.exp(-tstep / (tau3 * alpha_i))) + RES3 * np.exp(-tstep / (tau3 * alpha_i))
    
    MAT = mateq + RES0 + RES1 + RES2 + RES3
    #MAT = max(MAT, mateq + 1e-6)
    
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
        return np.array([np.nan] * 18, dtype=np.float64)

    
    return np.array([C, CCATOT, K, I, F_GHGabate, RES0, RES1, RES2, RES3,
                     TBOX1, TBOX2, MAT, TATM, Y, YNET, YGROSS, DAMFRAC, ABATECOST], dtype=np.float64)



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

    for i in range(2, num_periods + 1):
        result = diceForward_numba(i, MIU, S, alpha, CCATOT[i - 1], K[i - 1], I, F_GHGabate, RES0, RES1, RES2, RES3, TBOX1, TBOX2,
                                   tstep, dk, gama, eco2Param, sigma, eland, cost1tot, expcost2, miuup, sLBounds, sUBounds,
                                   AlphaLowerBound, Fcoef1, Fcoef2, CO2E_GHGabateB, emshare0, emshare1, emshare2, emshare3,
                                   tau0, tau1, tau2, tau3, mateq, fco22x, F_Misc, teq1, teq2, d1, d2, a1, a2base, a3, pulse_GtCO2_per_year)

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



@njit(cache=False)
def compute_SCC_numba(MIU, S, alpha, num_periods, tstep, dk, gama, eco2Param, sigma, eland, cost1tot, expcost2,
                      miuup, sLBounds, sUBounds, AlphaLowerBound, Fcoef1, Fcoef2, CO2E_GHGabateB, emshare0,
                      emshare1, emshare2, emshare3, tau0, tau1, tau2, tau3, mateq, fco22x, F_Misc, teq1, teq2,
                      d1, d2, a1, a2base, a3, res00, res10, res20, res30, tbox10, tbox20, CumEmiss0,
                      F_GHGabate2020, k0, tatm0, mat0, a0, L, RR, scale1, scale2, elasmu):
    
    # Baseline with no pulse
    pulse_zero = np.zeros(num_periods + 1, dtype=np.float64)
    base = diceTrajectory_numba(
        MIU, S, alpha, num_periods, tstep, dk, gama, eco2Param, sigma, eland, 
        cost1tot, expcost2, miuup, sLBounds, sUBounds, AlphaLowerBound, Fcoef1, 
        Fcoef2, CO2E_GHGabateB, emshare0, emshare1, emshare2, emshare3, tau0, tau1, 
        tau2, tau3, mateq, fco22x, F_Misc, teq1, teq2, d1, d2, a1, a2base, a3, 
        res00, res10, res20, res30, tbox10, tbox20, CumEmiss0, F_GHGabate2020, 
        k0, tatm0, mat0, a0, L, RR, scale1, scale2, elasmu,
        pulse_zero)
    
    if np.isnan(base[0]):
        return np.zeros(num_periods + 1)
    
    C_base = base[1]
    CPC_base = 1000.0 * C_base[1:] / L[1:]
    PERIODU_base = ((CPC_base ** (1.0 - elasmu)) - 1.0) / (1.0 - elasmu) - 1.0
    TOTPERIODU_base = PERIODU_base * L[1:] * RR[1:]
    W0 = np.sum(TOTPERIODU_base)
    lambda_C = (CPC_base ** (-elasmu)) * RR[1:]
    
    SCC = np.zeros(num_periods + 1)
    
    # Richardson extrapolation: use two pulse sizes
    pulse_h = 5e-6    # GtCO2/year (larger pulse)
    pulse_h2 = pulse_h/2   # GtCO2/year (smaller pulse, h/2)
    
    for i in range(2, num_periods + 1):
        # Compute SCC with larger pulse (h)
        pulse_array_h = np.zeros(num_periods + 1, dtype=np.float64)
        pulse_array_h[i] = pulse_h
        
        alt_h = diceTrajectory_numba(
            MIU, S, alpha, num_periods, tstep, dk, gama, eco2Param, sigma, eland, 
            cost1tot, expcost2, miuup, sLBounds, sUBounds, AlphaLowerBound, Fcoef1, 
            Fcoef2, CO2E_GHGabateB, emshare0, emshare1, emshare2, emshare3, tau0, tau1, 
            tau2, tau3, mateq, fco22x, F_Misc, teq1, teq2, d1, d2, a1, a2base, a3, 
            res00, res10, res20, res30, tbox10, tbox20, CumEmiss0, F_GHGabate2020, 
            k0, tatm0, mat0, a0, L, RR, scale1, scale2, elasmu,
            pulse_array_h)
        
        if np.isnan(alt_h[0]):
            SCC[i] = 0.0
            continue
        
        C_alt_h = alt_h[1]
        CPC_alt_h = 1000.0 * C_alt_h[1:] / L[1:]
        PERIODU_alt_h = ((CPC_alt_h ** (1.0 - elasmu)) - 1.0) / (1.0 - elasmu) - 1.0
        TOTPERIODU_alt_h = PERIODU_alt_h * L[1:] * RR[1:]
        W_h = np.sum(TOTPERIODU_alt_h)
        dW_h = W_h - W0
        
        # Compute SCC with smaller pulse (h/2)
        pulse_array_h2 = np.zeros(num_periods + 1, dtype=np.float64)
        pulse_array_h2[i] = pulse_h2
        
        alt_h2 = diceTrajectory_numba(
            MIU, S, alpha, num_periods, tstep, dk, gama, eco2Param, sigma, eland, 
            cost1tot, expcost2, miuup, sLBounds, sUBounds, AlphaLowerBound, Fcoef1, 
            Fcoef2, CO2E_GHGabateB, emshare0, emshare1, emshare2, emshare3, tau0, tau1, 
            tau2, tau3, mateq, fco22x, F_Misc, teq1, teq2, d1, d2, a1, a2base, a3, 
            res00, res10, res20, res30, tbox10, tbox20, CumEmiss0, F_GHGabate2020, 
            k0, tatm0, mat0, a0, L, RR, scale1, scale2, elasmu,
            pulse_array_h2)
        
        if np.isnan(alt_h2[0]):
            SCC[i] = 0.0
            continue
        
        C_alt_h2 = alt_h2[1]
        CPC_alt_h2 = 1000.0 * C_alt_h2[1:] / L[1:]
        PERIODU_alt_h2 = ((CPC_alt_h2 ** (1.0 - elasmu)) - 1.0) / (1.0 - elasmu) - 1.0
        TOTPERIODU_alt_h2 = PERIODU_alt_h2 * L[1:] * RR[1:]
        W_h2 = np.sum(TOTPERIODU_alt_h2)
        dW_h2 = W_h2 - W0
        
        if abs(lambda_C[i-1]) > 1e-12:
            # Compute SCC estimates with each pulse size
            SCC_h = -(dW_h / pulse_h) / lambda_C[i-1]
            SCC_h2 = -(dW_h2 / pulse_h2) / lambda_C[i-1]
            
            # Richardson extrapolation: (4*f(h/2) - f(h)) / 3
            SCC[i] = (4.0 * SCC_h2 - SCC_h) / 3.0
        else:
            SCC[i] = 0.0
    
    #SCC[1] = SCC[2] * 0.85 if np.isfinite(SCC[2]) else 0.0
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

    safe_MAT = np.maximum(MAT, mateq + 1e-6)
    I = S * Y
    CACC = CCATOT - (MAT - mateq)
    DAMAGES = YGROSS * DAMFRAC
    CPRICE = PBACKTIME * (MIU ** (expcost2 - 1.0))
    # Separate EIND vs ECO2 (old logic still uses MIU on total in the forward, but for reporting split here)
    EIND = sigma * (eco2Param * (K ** gama)) * (1.0 - MIU)
    ECO2 = EIND + eland 
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

    SCC = compute_SCC_numba(
        MIU, S, alpha, num_periods, tstep, dk, gama, eco2Param, sigma, eland, cost1tot, expcost2, miuup, sLBounds,
        sUBounds, AlphaLowerBound, Fcoef1, Fcoef2, CO2E_GHGabateB, emshare0, emshare1, emshare2, emshare3, tau0,
        tau1, tau2, tau3, mateq, fco22x, F_Misc, teq1, teq2, d1, d2, a1, a2base, a3, res00, res10, res20, res30,
        tbox10, tbox20, CumEmiss0, F_GHGabate2020, k0, tatm0, mat0, a0, L, RR, scale1, scale2, elasmu
    )

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


def recoverAllVars(x, params):
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
        #self.S[params['FixSperiod']:] = 0.28
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


class Dice2023Model:
    def __init__(self, num_times, scenario):
        self.num_periods = num_times
        self.scenario = scenario
        self.params = LoadParams(num_times)
        self.TempUpperConstraint = 20.0 if scenario == 10 else (1.5 if scenario == 6 else (2.0 if scenario == 7 else 20.0))
        self.TempLowerConstraint = 0.01

    # ── NEW METHOD 1 ──────────────────────────────────────────────────────────
    def check_temp_feasibility(self, temp_limit):
        """
        Run one forward pass at maximum abatement every period.
        Returns dict: {feasible, peak_temp, peak_period, peak_year, temp_limit,
                       TATM_maxabate}

        Cost: one compiled forward pass (~0.05 ms). No optimization.
        Call before any temperature-constrained optimization to detect
        infeasibility early and avoid silent constraint violation.
        """
        p = self.params
        N = self.num_periods

        MIU_max    = p['miuup'].copy()
        MIU_max[1] = p['miu1']                            # period 1 always fixed

        S_flat = np.full(N + 1, max(p['optlrsav'], 0.2))
        S_flat[p['FixSperiod']:] = p['FixSvalue']

        Alpha_init    = np.linspace(p['a0'], 0.425, N + 1)
        Alpha_init[1] = p['a0']

        out  = diceTrajectory(p, MIU_max, S_flat, Alpha_init)
        TATM = out[4]                                      # (N+1,), 1-based

        peak_idx    = int(np.argmax(TATM[1:]))
        peak_temp   = float(TATM[1:][peak_idx])
        peak_period = peak_idx + 1
        peak_year   = int(p['yr0'] + p['tstep'] * peak_idx)

        return {
            'feasible':      peak_temp <= temp_limit,
            'peak_temp':     peak_temp,
            'peak_period':   peak_period,
            'peak_year':     peak_year,
            'temp_limit':    temp_limit,
            'TATM_maxabate': TATM.copy(),
        }

    # ── NEW METHOD 2 ──────────────────────────────────────────────────────────
    def _run_best_effort(self, temp_limit, feasibility_info):
        """
        Optimize with a quadratic temperature penalty instead of a hard
        inequality constraint.  Always well-defined, converges to the
        physically consistent trajectory closest to the temperature target.

        Objective:  -Welfare + penalty_weight * sum(max(0, TATM - limit)^2)

        Returns (x_opt, output, result_meta) — identical signature to
        run_model(), so dump_state() and plot_state_to_file() work unchanged.
        result_meta['infeasible'] = True flags the result for downstream code.
        """
        import time
        p = self.params
        N = self.num_periods

        # penalty_weight=1e4: 0.25°C excess over 81 periods costs ~50,000
        # welfare units — two orders of magnitude above scenario welfare
        # differences.  Drives temperature as low as physically possible.
        penalty_weight = 1e4

        prob = DiceFunc(N, p,
                        TempUpperConstraint=temp_limit,
                        TempLowerConstraint=self.TempLowerConstraint)

        def penalized_objective(x):
            MIU, S, Alpha = prob.pack(x)
            prob.MIU[1:N + 1]   = MIU
            prob.S[1:N + 1]     = S
            prob.Alpha[1:N + 1] = Alpha
            out  = diceTrajectory(p, prob.MIU, prob.S, prob.Alpha)
            W    = out[0]
            TATM = out[4]
            if np.any(np.isnan(TATM)):
                return 1e15
            excess = np.maximum(0.0, TATM[1:] - temp_limit)
            return -W + penalty_weight * float(np.sum(excess ** 2))

        # Warm-start from max abatement — best physical initialization
        x0    = np.concatenate([p['miuup'][1:N + 1], prob.S[1:N + 1], prob.Alpha[1:N + 1]])
        x0[0] = p['miu1']

        miu_bounds   = [(p['miu1'], p['miu1'])] + \
                       [(p['MIULowerBound'], p['miuup'][t]) for t in range(2, N + 1)]
        s_bounds     = [(p['sLBounds'][i], p['sUBounds'][i]) for i in range(1, N + 1)]
        alpha_bounds = [(p['AlphaLowerBound'], p['AlphaUpperBound']) for _ in range(N)]
        bounds       = miu_bounds + s_bounds + alpha_bounds

        # IRF equality and lower temperature floor — no hard upper bound
        constraints = [
            {'type': 'eq',   'fun': prob.irf_residual},
            {'type': 'ineq', 'fun': prob.temp_lo},
        ]

        print("  Running best-effort optimization (soft temperature penalty)...")
        t0  = time.time()
        res = minimize(
            penalized_objective, x0,
            method      = 'SLSQP',
            bounds      = bounds,
            constraints = constraints,
            options     = {'maxiter': 2000, 'ftol': 1e-7, 'disp': False},
        )
        elapsed = time.time() - t0

        output = recoverAllVars(res.x, p)

        # Recover final TATM for reporting
        MIU_o, S_o, Alpha_o = prob.pack(res.x)
        prob.MIU[1:N + 1]   = MIU_o
        prob.S[1:N + 1]     = S_o
        prob.Alpha[1:N + 1] = Alpha_o
        TATM_final = diceTrajectory(p, prob.MIU, prob.S, prob.Alpha)[4]

        result_meta = {
            'infeasible':            True,
            'temp_limit':            temp_limit,
            'feasibility_info':      feasibility_info,
            'best_effort_peak_temp': float(TATM_final[1:].max()),
            'best_effort_peak_year': int(p['yr0'] + p['tstep'] *
                                         int(np.argmax(TATM_final[1:]))),
            'optimizer_success':     res.success,
            'optimizer_message':     res.message,
            'elapsed_s':             elapsed,
        }

        print(f"  Best-effort peak temperature : "
              f"{result_meta['best_effort_peak_temp']:.3f}°C "
              f"(year {result_meta['best_effort_peak_year']})")
        print(f"  Optimizer : {res.message} | {elapsed:.1f}s")

        return res.x, output, result_meta

    # ── run_model (scenario 6 block modified; return signature updated) ───────
    def run_model(self):
        import time

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
            apply_disc_prstp(self.params, prstp_value=0.04,
                             cap_after_t=81, cap_exp_value=5 * 80)

        elif self.scenario == 5:
            self.params['k0'] = 290
            apply_disc_prstp(self.params, prstp_value=0.05,
                             cap_after_t=51, cap_exp_value=5 * 51)

        elif self.scenario == 6:
            self.TempUpperConstraint = 1.5
            # ── Feasibility pre-check ─────────────────────────────────────
            # Barrage & Nordhaus (2024, PNAS) explicitly state the 1.5°C
            # scenario "is best thought of as infeasible" within the
            # technologies considered by DICE-2023.  We verify this
            # computationally with a single max-abatement forward pass
            # before committing to the homotopy continuation.
            fcheck = self.check_temp_feasibility(1.5)
            if not fcheck['feasible']:
                print(
                    f"\n[PyDICE-2023] INFEASIBILITY DETECTED — scenario 6 (1.5°C)\n"
                    f"  Maximum abatement at every period yields a peak of "
                    f"{fcheck['peak_temp']:.3f}°C in {fcheck['peak_year']}.\n"
                    f"  The 1.5°C ceiling cannot be reached under sequentially\n"
                    f"  consistent DFAIR dynamics regardless of the control trajectory.\n"
                    f"  Consistent with Barrage & Nordhaus (2024, PNAS): the 1.5°C\n"
                    f"  scenario 'is best thought of as infeasible' within the\n"
                    f"  technologies considered (excluding geoengineering).\n"
                    f"  Returning best-effort trajectory (minimize exceedance).\n"
                )
                return self._run_best_effort(1.5, fcheck)
            # Feasible under current params — fall through to homotopy
            self.use_homotopy = True

        elif self.scenario == 7:
            self.TempUpperConstraint = 2.0

        elif self.scenario == 8:
            for i in range(1, self.num_periods + 1):
                self.params['miuup'][i] = min(
                    0.05 + 0.04 * (i - 1) - 0.01 * max(0, i - 5),
                    self.params['limmiu2070'])

        elif self.scenario == 9:
            self.params['prstp'] = 0.001

        elif self.scenario == 10:
            self.TempUpperConstraint = 15.0
            self.params['miuup'][:] = 1.0

        if self.scenario not in {1, 2, 3, 4, 5}:
            self.params['rartp'] = np.exp(
                self.params['prstp'] + self.params['betaclim'] * self.params['pi']) - 1
            self.params['optlrsav'] = (
                (self.params['dk'] + 0.004) /
                (self.params['dk'] + 0.004 * self.params['elasmu'] + self.params['rartp'])
            ) * self.params['gama']

        prob = DiceFunc(self.num_periods, self.params,
                        TempUpperConstraint=self.TempUpperConstraint,
                        TempLowerConstraint=self.TempLowerConstraint)

        x0 = np.concatenate([
            prob.MIU[1:self.num_periods + 1],
            prob.S[1:self.num_periods + 1],
            prob.Alpha[1:self.num_periods + 1]
        ])

        miu_bounds = []
        for t in range(1, self.num_periods + 1):
            if t == 1:
                miu_bounds.append((self.params['miu1'], self.params['miu1']))
            elif self.scenario == 10 and t > 57:
                miu_bounds.append((1.0, 1.0))
            else:
                miu_bounds.append((self.params['MIULowerBound'], self.params['miuup'][t]))

        s_bounds     = [(self.params['sLBounds'][i], self.params['sUBounds'][i]) for i in range(1, self.num_periods + 1)]
        alpha_bounds = [(self.params['AlphaLowerBound'], self.params['AlphaUpperBound']) for _ in range(1, self.num_periods + 1)]

        constraints = [
            {'type': 'eq',   'fun': prob.irf_residual},
            {'type': 'ineq', 'fun': prob.temp_up},
            {'type': 'ineq', 'fun': prob.temp_lo},
        ]
        if self.scenario == 10:
            def cprice_residual(x):
                MIU, _, _ = prob.pack(x)
                PBT = self.params['PBACKTIME'][1:]
                CPR = PBT * (MIU ** (self.params['expcost2'] - 1))
                return np.array([self.params['cpricebase'][t] - CPR[t - 1]
                                  for t in range(1, min(47, self.num_periods) + 1)])
            constraints.append({'type': 'ineq', 'fun': cprice_residual})

        bounds = miu_bounds + s_bounds + alpha_bounds

        def callback(xk):
            obj      = prob.objective(xk)
            irf_res  = np.max(np.abs(prob.irf_residual(xk)))
            temp_up  = np.min(prob.temp_up(xk))
            temp_lo  = np.min(prob.temp_lo(xk))
            print(f"Iteration: Objective={-obj:.2f}, IRF max residual={irf_res:.2e}, "
                  f"Temp Upper min={temp_up:.2f}, Temp Lower min={temp_lo:.2f}")

        print("Starting optimization...")
        start = time.time()

        if self.scenario == 6:
            # Scenario 6 reaches here only if check_temp_feasibility passed
            temp_steps = [2.0, 1.9, 1.8, 1.79, 1.78, 1.77, 1.76, 1.75,
                          1.74, 1.73, 1.72, 1.71, 1.70,
                          1.68, 1.66, 1.64, 1.62, 1.60,
                          1.58, 1.56, 1.5]
            x_current   = x0.copy()
            last_success = x0.copy()

            for temp_limit in temp_steps:
                print(f"\n  Homotopy step: TempUpperConstraint = {temp_limit}°C")
                prob_step = DiceFunc(self.num_periods, self.params,
                                     TempUpperConstraint=temp_limit,
                                     TempLowerConstraint=self.TempLowerConstraint)
                cons_step = [
                    {'type': 'eq',   'fun': prob_step.irf_residual},
                    {'type': 'ineq', 'fun': prob_step.temp_up},
                    {'type': 'ineq', 'fun': prob_step.temp_lo},
                ]
                step_res = minimize(
                    prob_step.objective, x_current,
                    method='SLSQP', bounds=bounds, constraints=cons_step,
                    options={'maxiter': 1000, 'ftol': 1e-7, 'disp': True})

                MIU_c, S_c, Alpha_c = prob_step.pack(step_res.x)
                prob_step.MIU[1:self.num_periods + 1]   = MIU_c
                prob_step.S[1:self.num_periods + 1]     = S_c
                prob_step.Alpha[1:self.num_periods + 1] = Alpha_c
                out_internal  = diceTrajectory(self.params, prob_step.MIU,
                                               prob_step.S, prob_step.Alpha)
                TATM_internal = out_internal[4]
                irf_res       = np.abs(prob_step.irf_residual(step_res.x)).max()
                temp_up_min   = prob_step.temp_up(step_res.x).min()

                print(f"  Success      : {step_res.success}")
                print(f"  IRF residual : {irf_res:.6f}")
                print(f"  Max TATM     : {TATM_internal[1:].max():.4f}°C")
                print(f"  temp_up min  : {temp_up_min:.6f}")

                if step_res.success and irf_res < 0.01:
                    x_current    = step_res.x.copy()
                    last_success = step_res.x.copy()
                    continue

                    if irf_res < 0.01 and temp_up_min >= -0.01:
                        print("  Accepting despite convergence warning")
                        x_current    = step_res.x.copy()
                        last_success = step_res.x.copy()
                        continue

                    print("  Step failed — retrying...")
                    step_res2 = minimize(
                        prob_step.objective, step_res.x,
                        method='SLSQP', bounds=bounds, constraints=cons_step,
                        options={'maxiter': 5000, 'ftol': 1e-8, 'disp': True})
                    irf_res2     = np.abs(prob_step.irf_residual(step_res2.x)).max()
                    temp_up_min2 = prob_step.temp_up(step_res2.x).min()

                    if irf_res2 < 0.01 and temp_up_min2 >= -0.01:
                        x_current    = step_res2.x.copy()
                        last_success = step_res2.x.copy()
                        print(f"  Retry accepted | IRF: {irf_res2:.6f}")
                    else:
                        print(f"  Physical barrier at {temp_limit}°C — using last good solution")
                        break

            res   = step_res
            res.x = last_success
        else:
            res = minimize(
                prob.objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 25000, 'ftol': 1e-5, 'disp': True, 'eps': 1e-6},
                callback=callback)

        print(f"Done in {time.time() - start:.1f}s | Success: {res.success}, Message: {res.message}")

        x_opt  = res.x
        output = recoverAllVars(x_opt, self.params)

        print("Optimized MIU range:",   x_opt[:self.num_periods].min(),            x_opt[:self.num_periods].max())
        print("Optimized S range:",     x_opt[self.num_periods:2*self.num_periods].min(), x_opt[self.num_periods:2*self.num_periods].max())
        print("Optimized Alpha range:", x_opt[2*self.num_periods:].min(),          x_opt[2*self.num_periods:].max())
        print("IRF max residual:",      np.max(np.abs(prob.irf_residual(x_opt))))

        # ── Changed from `return x_opt, output, None` ────────────────────
        result_meta = {'infeasible': False, 'scenario': self.scenario}
        return x_opt, output, result_meta

    def dump_parameters(self):
        results_dir = "./results"
        os.makedirs(results_dir, exist_ok=True)
        path = f"{results_dir}/dice2023_parameters_scen{self.scenario}.csv"
        with open(path, mode="w", newline="") as f:
            writer = csv.writer(f)
            header = ["PERIOD", "L", "gA", "aL", "gsig", "sigma", "sigmatot",
                      "PBACKTIME", "cost1tot", "eland", "cpricebase",
                      "F_Misc", "CO2E_GHGabateB", "emissrat", "RR1", "RR", "miuup"]
            writer.writerow(header)
            for t in range(1, self.num_periods + 1):
                writer.writerow([t, self.params['L'][t], self.params['gA'][t], self.params['aL'][t],
                                  self.params['gsig'][t], self.params['sigma'][t], self.params['sigmatot'][t],
                                  self.params['PBACKTIME'][t], self.params['cost1tot'][t], self.params['eland'][t],
                                  self.params['cpricebase'][t], self.params['F_Misc'][t],
                                  self.params['CO2E_GHGabateB'][t], self.params['emissrat'][t],
                                  self.params['RR1'][t], self.params['RR'][t], self.params['miuup'][t]])
        print(f"Parameters written to {path}")

    def dump_state(self, years, output, filename, scenario):
        results_dir = "./results"
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
        results_dir = "./results"
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

        MIU   = x[:num_periods]
        S     = x[num_periods:2 * num_periods]
        ALPHA = x[2 * num_periods:]
        fig_miu = self.plot_figure(years, MIU,   "Years", "Rate",     "Optimized: Carbon Emission Control Rate (MIU)")
        pp.savefig(fig_miu); plt.close(fig_miu)
        fig_s   = self.plot_figure(years, S,     "Years", "Rate",     "Optimized: Saving Rates (S)")
        pp.savefig(fig_s);   plt.close(fig_s)
        fig_a   = self.plot_figure(years, ALPHA, "Years", "Unitless", "Optimized: ALPHA")
        pp.savefig(fig_a);   plt.close(fig_a)
        pp.close()
        print(f"Wrote plots to {fileName}")



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


if __name__ == "__main__":
    scenario = display_scenarios()
    model = Dice2023Model(num_times=81, scenario=scenario)
    model.dump_parameters()
    years = np.arange(
        model.params['yr0'],
        model.params['yr0'] + model.params['tstep'] * model.num_periods,
        model.params['tstep']
    )
    x_opt, output, result_meta = model.run_model()

    if result_meta.get('infeasible'):
        print(f"\n[Note] Scenario returned infeasible result.")
        print(f"  Best-effort peak temperature : {result_meta['best_effort_peak_temp']:.3f}°C")
        print(f"  Peak year                    : {result_meta['best_effort_peak_year']}")

    model.dump_state(years, output, "./results/dice2023_state.csv", scenario)
    model.plot_state_to_file(
        f"./results/dice2023_plots_scen{scenario}.pdf", years, output, x_opt)


if __name__ == "__main__":
    scenario = display_scenarios()
    model = Dice2023Model(num_times=81, scenario=scenario)
    model.dump_parameters()
    years = np.arange(
        model.params['yr0'],
        model.params['yr0'] + model.params['tstep'] * model.num_periods,
        model.params['tstep'],
    )
    x_opt, output, _ = model.run_model()
    model.dump_state(years, output, "./results/dice2023_state.csv", scenario)
    model.plot_state_to_file(
        f"./results/dice2023_plots_scen{scenario}.pdf", years, output, x_opt
    )
