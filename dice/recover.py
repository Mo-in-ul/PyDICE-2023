import numpy as np
from numba import njit

from dice.model import diceTrajectory_numba
from dice.scc import compute_SCC_numba


# Column index → variable name mapping for the output array (shape: num_periods × 46)
COLUMNS = [
    "EIND", "ECO2", "CO2PPM", "TATM", "Y", "DAMFRAC", "CPC", "CPRICE", "MIUopt", "RSHORT",
    "ECO2E", "L", "AL", "YGROSS", "K", "Sopt", "I", "YNET", "CCATOT", "CACC",
    "RES0", "RES1", "RES2", "RES3", "DAMAGES", "ABATECOST", "MCABATE", "C",
    "PERIODU", "TOTPERIODU", "MAT", "FORC", "TBOX1", "TBOX2", "F_GHGABATE",
    "IRFT", "ALPHA", "RFACTLONG", "RLONG", "SCC", "ABATERAT", "ATFRAC2020",
    "ATFRAC1765", "FORC_CO2", "RR",
]


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
    alpha[1] = a0
    pulse_GtCO2_per_year = np.zeros(num_periods + 1, dtype=np.float64)
    result = diceTrajectory_numba(
        MIU, S, alpha, num_periods, tstep, dk, gama, eco2Param, sigma, eland, cost1tot, expcost2, miuup, sLBounds,
        sUBounds, AlphaLowerBound, Fcoef1, Fcoef2, CO2E_GHGabateB, emshare0, emshare1, emshare2, emshare3, tau0,
        tau1, tau2, tau3, mateq, fco22x, F_Misc, teq1, teq2, d1, d2, a1, a2base, a3, res00, res10, res20, res30,
        tbox10, tbox20, CumEmiss0, F_GHGabate2020, k0, tatm0, mat0, a0, L, RR, scale1, scale2, elasmu,
        pulse_GtCO2_per_year
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
    EIND = sigma * (eco2Param * (K ** gama)) * (1.0 - MIU)
    ECO2 = EIND + eland
    ECO2E = ECO2 + CO2E_GHGabateB * (1.0 - MIU)
    FORC = fco22x * np.log(safe_MAT / mateq) / np.log(2.0) + F_Misc + F_GHGab_arr
    FORC_CO2 = fco22x * np.log(safe_MAT / mateq) / np.log(2.0)
    CPC = np.zeros(num_periods + 1)
    for t in range(1, num_periods + 1):
        CPC[t] = 1000.0 * C[t] / L[t]

    RFACTLONG = np.full(num_periods + 1, SRF)
    for i in range(2, num_periods + 1):
        RFACTLONG[i] = SRF * (CPC[i - 1] / CPC[1]) ** (-elasmu) * RR[i]
    RLONG = np.zeros(num_periods + 1)
    RSHORT = np.zeros(num_periods + 1)
    for i in range(2, num_periods + 1):
        RLONG[i] = -np.log(RFACTLONG[i] / SRF) / (5.0 * (i - 1))
        RSHORT[i] = -np.log(RFACTLONG[i] / RFACTLONG[i - 1]) / 5.0

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

    output = np.zeros((num_periods, 46))
    for i in range(num_periods):
        t = i + 1
        output[i, 0]  = EIND[t]
        output[i, 1]  = ECO2[t]
        output[i, 2]  = MAT[t] / 2.13
        output[i, 3]  = TATM[t]
        output[i, 4]  = Y[t]
        output[i, 5]  = DAMFRAC[t]
        output[i, 6]  = CPC[t]
        output[i, 7]  = CPRICE[t]
        output[i, 8]  = MIU[t]
        output[i, 9]  = RSHORT[t]
        output[i, 10] = ECO2E[t]
        output[i, 11] = L[t]
        output[i, 12] = eco2Param[t]
        output[i, 13] = YGROSS[t]
        output[i, 14] = K[t]
        output[i, 15] = S[t]
        output[i, 16] = I[t]
        output[i, 17] = YNET[t]
        output[i, 18] = CCATOT[t]
        output[i, 19] = CACC[t]
        output[i, 20] = RES0_arr[t]
        output[i, 21] = RES1_arr[t]
        output[i, 22] = RES2_arr[t]
        output[i, 23] = RES3_arr[t]
        output[i, 24] = DAMAGES[t]
        output[i, 25] = ABATECOST[t]
        output[i, 26] = PBACKTIME[t] * (MIU[t] ** (expcost2 - 1.0))
        output[i, 27] = C[t]
        output[i, 28] = PERIODU[t]
        output[i, 29] = TOTPERIODU[t]
        output[i, 30] = MAT[t]
        output[i, 31] = FORC[t]
        output[i, 32] = TBOX1_arr[t]
        output[i, 33] = TBOX2_arr[t]
        output[i, 34] = F_GHGab_arr[t]
        output[i, 35] = IRFt[t]
        output[i, 36] = alpha[t]
        output[i, 37] = RFACTLONG[t]
        output[i, 38] = RLONG[t]
        output[i, 39] = SCC[t]
        output[i, 40] = ABATECOST[t] / max(Y[t], 1e-12)
        output[i, 41] = MAT[t] / mat0
        output[i, 42] = MAT[t] / mateq
        output[i, 43] = FORC_CO2[t]
        output[i, 44] = RR[t]

    if not np.all(np.isfinite(output)):
        return np.zeros((num_periods, 46))
    return output


def recoverAllVars(x, params):
    return recoverAllVars_numba(
        x, params['num_periods'], params['tstep'], params['dk'], params['gama'],
        params['eco2Param'], params['sigma'], params['eland'], params['cost1tot'],
        params['expcost2'], params['miuup'], params['sLBounds'], params['sUBounds'],
        params['AlphaLowerBound'], params['Fcoef1'], params['Fcoef2'], params['CO2E_GHGabateB'],
        params['emshare0'], params['emshare1'], params['emshare2'], params['emshare3'],
        params['tau0'], params['tau1'], params['tau2'], params['tau3'],
        params['mateq'], params['fco22x'], params['F_Misc'], params['teq1'], params['teq2'],
        params['d1'], params['d2'], params['a1'], params['a2base'], params['a3'],
        params['res00'], params['res10'], params['res20'], params['res30'],
        params['tbox10'], params['tbox20'], params['CumEmiss0'], params['F_GHGabate2020'],
        params['k0'], params['tatm0'], params['mat0'], params['a0'],
        params['L'], params['RR'], params['scale1'], params['scale2'], params['elasmu'],
        params['PBACKTIME'], params['irf0'], params['irC'], params['irT'], params['SRF'],
    )