"""JAX copy of the PyDICE-2023 forward model (notebook/dice2023_corrected.py).

Purpose: exact derivatives (automatic differentiation) for the SCC checks.
  * check3_ift.py uses them for the implicit-function-theorem SCC decomposition.
  * ipopt_model.py uses them to solve the model with IPOPT and read Lagrange multipliers.

The recursion is the same as diceTrajectory_numba, line by line:
  CCATOT(t+1)     = CCATOT(t) + ECO2(t) * tstep / 3.666
  F_GHGabate(t+1) = Fcoef2 * F_GHGabate(t) + Fcoef1 * CO2E_GHGabateB(t) * (1 - MIU(t))
  RES_j(t+1)      uses alpha(t+1) and ECO2(t+1)
  MAT             floored at 10 GtC (GAMS MAT.LO = 10)
Two perturbation inputs are added, both zero by default:
  eps_E[t] : additive term in the emissions equation ECO2(t) (the GAMS eco2eq), GtCO2/yr
  eps_C[t] : additive term in the consumption equation C(t) (the GAMS cc), trillion $/yr
and one optional control:
  D[t]     : capital disposal, K(t) = (1-dk)^tstep K(t-1) + tstep I(t-1) - D(t), D >= 0
             (the GAMS capital equation is =L=, i.e. disposal is free)
Arrays are 1-based with length N+1 (index 0 unused), as in the Numba code.

Switches for the gap decomposition (decomposition.py), all off by default (= current model):
  q['old_land']   : ECO2 = sigma*YGROSS*(1-MIU) + eland     (control rate not applied to land CO2)
  q['old_ccatot'] : CCATOT(t) = CCATOT(t-1) + ECO2(t)*tstep/3.667   (current, not previous, emissions)
  q['old_fghg']   : F_GHGabate(t) uses CO2E_GHGabateB(t) and MIU(t)  (current, not previous, period)
  q['mat_floor']  : floor on atmospheric carbon, default 10 (GAMS MAT.LO); the old model used mateq
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

PKEYS = ['tstep', 'dk', 'gama', 'expcost2', 'AlphaLowerBound', 'Fcoef1', 'Fcoef2',
         'emshare0', 'emshare1', 'emshare2', 'emshare3', 'tau0', 'tau1', 'tau2', 'tau3',
         'mateq', 'fco22x', 'teq1', 'teq2', 'd1', 'd2', 'a1', 'a2base', 'a3',
         'res00', 'res10', 'res20', 'res30', 'tbox10', 'tbox20', 'CumEmiss0', 'F_GHGabate2020',
         'k0', 'tatm0', 'mat0', 'a0', 'scale1', 'scale2', 'elasmu', 'miu1',
         'irf0', 'irC', 'irT']
AKEYS = ['eco2Param', 'sigma', 'eland', 'cost1tot', 'miuup', 'sLBounds', 'sUBounds',
         'CO2E_GHGabateB', 'F_Misc', 'L', 'RR', 'PBACKTIME']


def pack_params(p):
    """Turn the PyDICE params dict into a JAX-friendly dict of floats and arrays."""
    q = {k: float(p[k]) for k in PKEYS}
    q.update({k: jnp.asarray(np.asarray(p[k], dtype=np.float64)) for k in AKEYS})
    q['N'] = int(p['num_periods'])
    return q


def trajectory(q, MIU, S, alpha, eps_E=None, eps_C=None, D=None, clamp=True, T_shift=None, E_fix=None):
    """Forward model. Returns a dict of 1-based arrays plus UTILITY.
    clamp=True reproduces the Numba clamps (MIU to [0, miuup], S to its bounds,
    alpha >= AlphaLowerBound, TATM to [0.01, 20]); use clamp=False inside an optimizer
    whose variable bounds already enforce them, so derivatives are not cut at the clamps."""
    N = q['N']; ts = q['tstep']
    z = jnp.zeros(N + 1)
    eps_E = z if eps_E is None else eps_E
    eps_C = z if eps_C is None else eps_C
    D = z if D is None else D
    T_shift = z if T_shift is None else T_shift   # added to TATM in the damage function only (fair_benchmark.py)
    if not q.get('free_miu1', False):   # GAMS caps MIU(1) at miuup(1) = 0.05; only the base run frees it
        MIU = MIU.at[1].set(q['miu1'])
    alpha = alpha.at[1].set(q['a0'])
    if clamp:
        MIUc = jnp.clip(MIU, 0.0, q['miuup'])
        Sc = jnp.clip(S, q['sLBounds'], q['sUBounds'])
        alc = jnp.maximum(alpha, q['AlphaLowerBound'])
    else:
        MIUc, Sc, alc = MIU, S, alpha

    # period 1
    K1 = q['k0']
    T1 = q['tatm0']
    DAM1 = q['a1'] * (T1 + T_shift[1]) + q['a2base'] * (T1 + T_shift[1]) ** q['a3']
    YG1 = q['eco2Param'][1] * K1 ** q['gama']
    AB1 = YG1 * q['cost1tot'][1] * MIUc[1] ** q['expcost2']
    Y1 = YG1 * (1 - DAM1) - AB1
    I1 = Sc[1] * Y1
    C1 = Y1 - I1 + eps_C[1]
    old_land = q.get('old_land', False); old_cc = q.get('old_ccatot', False); old_fg = q.get('old_fghg', False)
    mat_floor = q.get('mat_floor', 10.0)
    def emis(i, YG, M):
        return (q['sigma'][i] * YG * (1 - M) + q['eland'][i]) if old_land else (q['sigma'][i] * YG + q['eland'][i]) * (1 - M)
    if E_fix is not None:   # emissions held at a given path (fair_benchmark.py welfare weights)
        emis = lambda i, YG, M: E_fix[i]
    E1 = emis(1, YG1, MIUc[1]) + eps_E[1]

    carry0 = (K1, I1, q['CumEmiss0'], q['F_GHGabate2020'], q['res00'], q['res10'], q['res20'],
              q['res30'], q['tbox10'], q['tbox20'], E1, MIUc[1])
    em = [q['emshare0'], q['emshare1'], q['emshare2'], q['emshare3']]
    tau = [q['tau0'], q['tau1'], q['tau2'], q['tau3']]

    def step(carry, i):
        K, I, CC, FG, R0, R1, R2, R3, B1, B2, Eprev, Mprev = carry
        K = (1 - q['dk']) ** ts * K + ts * I - D[i]
        YG = q['eco2Param'][i] * K ** q['gama']
        E = emis(i, YG, MIUc[i]) + eps_E[i]
        CC = CC + (E * (ts / 3.667) if old_cc else Eprev * (ts / 3.666))
        FG = q['Fcoef2'] * FG + (q['Fcoef1'] * q['CO2E_GHGabateB'][i] * (1 - MIUc[i]) if old_fg
                                 else q['Fcoef1'] * q['CO2E_GHGabateB'][i - 1] * (1 - Mprev))
        a = alc[i]; inflow = E / 3.667
        Rs = []
        for j, Rj in enumerate((R0, R1, R2, R3)):
            dec = jnp.exp(-ts / (tau[j] * a))
            Rs.append(em[j] * tau[j] * a * inflow * (1 - dec) + Rj * dec)
        MAT = q['mateq'] + Rs[0] + Rs[1] + Rs[2] + Rs[3]
        MAT = jnp.maximum(MAT, mat_floor)
        F = q['fco22x'] * jnp.log(MAT / q['mateq']) / jnp.log(2.0) + q['F_Misc'][i] + FG
        B1 = B1 * jnp.exp(-ts / q['d1']) + q['teq1'] * F * (1 - jnp.exp(-ts / q['d1']))
        B2 = B2 * jnp.exp(-ts / q['d2']) + q['teq2'] * F * (1 - jnp.exp(-ts / q['d2']))
        T = B1 + B2
        if clamp:
            T = jnp.clip(T, 0.01, 20.0)
        Td = T + T_shift[i]
        DAM = q['a1'] * Td + q['a2base'] * Td ** q['a3']
        AB = YG * q['cost1tot'][i] * MIUc[i] ** q['expcost2']
        if 'rem_kappa' in q:   # removal cost premium: units of control above 1 cost rem_kappa times more (hidden/removal_core.py)
            Mi = MIUc[i]
            AB = AB + (q['rem_kappa'] - 1.0) * YG * q['cost1tot'][i] * (Mi ** q['expcost2'] - 1.0) * jax.nn.sigmoid((Mi - 1.0) / q.get('rem_w', 0.005))
        Y = YG * (1 - DAM) - AB
        I = Sc[i] * Y
        C = Y - I + eps_C[i]
        out = (C, K, CC, MAT, T, Y, YG, E, I, F)
        return (K, I, CC, FG, Rs[0], Rs[1], Rs[2], Rs[3], B1, B2, E, MIUc[i]), out

    _, outs = jax.lax.scan(step, carry0, jnp.arange(2, N + 1))
    first = (C1, K1, q['CumEmiss0'], q['mat0'], T1, Y1, YG1, E1, I1, 0.0)
    names = ['C', 'K', 'CCATOT', 'MAT', 'TATM', 'Y', 'YGROSS', 'ECO2', 'I', 'FORC']
    res = {}
    for k, f, o in zip(names, first, outs):
        res[k] = jnp.concatenate([jnp.zeros(1), jnp.atleast_1d(jnp.asarray(f, dtype=jnp.float64)), o])
    C = res['C'][1:]
    cpc = 1000.0 * C / q['L'][1:]
    el = q['elasmu']
    U = ((cpc ** (1 - el)) - 1) / (1 - el) - 1
    res['UTILITY'] = ts * q['scale1'] * jnp.sum(U * q['L'][1:] * q['RR'][1:]) + q['scale2']
    return res


def irf_rhs(q, a):
    out = 0.0
    for j in range(4):
        tau = q[f'tau{j}']
        out = out + a * q[f'emshare{j}'] * tau * (1 - jnp.exp(-100.0 / (a * tau)))
    return out


def irf_residual(q, res, alpha):
    """IRF100 residual for periods 1..N (RHS(alpha) - LHS(state)); alpha is the 1-based vector."""
    lhs = q['irf0'] + q['irC'] * (res['CCATOT'][1:] - (res['MAT'][1:] - q['mateq'])) + q['irT'] * res['TATM'][1:]
    return irf_rhs(q, alpha[1:]) - lhs


def marginal_utility_C(q, res):
    """dUTILITY/dC(t) for t = 1..N (the GAMS cc.m), from the closed form."""
    cpc = 1000.0 * res['C'][1:] / q['L'][1:]
    return q['tstep'] * q['scale1'] * 1000.0 * cpc ** (-q['elasmu']) * q['RR'][1:]
