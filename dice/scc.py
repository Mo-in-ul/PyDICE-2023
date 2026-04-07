import numpy as np
from numba import njit


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
