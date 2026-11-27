"""Explicit carbon dioxide removal in the corrected DICE-2023, version 3 (paper 2, after the referee review of v0.3).

Three removal families, each with its own capacity, cost and storage:
  N  biological (afforestation/reforestation, soil carbon, biochar, wetlands, agroforestry, forest management):
     stored in vegetation and soils; living store S_N capped (saturation, land); leaks at rate delta_N(T) per year.
  G  geological injection (DACCS, BECCS): stored in sedimentary basins; cumulative store S_G capped by geological
     storage, which is shared with an exogenous point-source CCS injection path (Direction B).
  M  mineral (enhanced weathering): stored as carbonates/bicarbonates; not limited by basin storage; optional cap.

Net CO2 emissions
  E = (sigma Y + eland)(1 - mu) - eff_N R_N - eff_G R_G - eff_M R_M + L_N
with the leakage flow L_N chosen so that carbon is conserved exactly within each five-year period:
  S_N(t+1) = k(t) S_N(t) + 5 R_N(t),  k(t) = (1 - delta_N(t))^5,  L_N(t) = S_N(t)(1 - k(t)) / 5,
  delta_N(t) = delta0 * max(0, 1 + gamma (T(t-1) - Tref))       (gamma = 0: constant leakage)
Removal made during period t starts leaking in period t+1 (stated convention, applied to stock and flow alike).

Marginal cost of family k: MC_k = a_k(t) + b_k (R_k/Rbar_k(t))^beta (+ surcharge s_k (O_k/Oref_k)^nu if soft limits),
with Rbar_k(t) logistic and a_k(t) falling from a_k0 to a_kinf (calendar learning), or for G with learning by doing on
cumulative S_G. Optional capacity investment for G: capacity stock K, investment I (Gt/yr of capacity per year),
K(t+1) = (1-dK)^5 K(t) + 5 I(t), R_G <= K, growth K(t+1) <= (1+g)^5 K(t) + seed, capital cost capex(t) I(t) with
capex = phi a_G(t) / crf and operating cost (1-phi) a_G(t) R_G, so capital is not counted twice.

Storage occupancy (used in the caps and surcharges): O_N = S_N - Z_N, O_G = S_G + C_CCS - Z_G, where C_CCS is cumulative
point-source CCS injection and Z_k(t) = cumulative exogenous freeing of storage space (zero in all runs; its derivative
gives the value of one tonne of free storage space at each date, the same economic object under hard and soft limits).
Land coupling (optional): O_N + land_c * S_G <= cap_N, a reduced form for BECCS taking land from biological removal.

Decision vector: x = [MU, RN, RG, RM, S, alpha] (6N) or [..., I] (7N) with investment."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); PB = os.path.join(HERE, '..', 'paper_build')
sys.path[:0] = [PB, os.path.join(PB, 'stage2'), os.path.join(PB, 'blockV')]
import numpy as np, jax, jax.numpy as jnp, cyipopt
jax.config.update('jax_enable_x64', True)
import dice_jax as dj
from common import gams_session, dm, GAMS_ALPHA

YR = 2020 + 5 * np.arange(200)
CPCAP = {5: 500.0, 4: 1000.0, 3: 1000.0, 2: 1000.0, 1: 1000.0}

CENTRAL = dict(
    beta=2.0,
    # biological
    RN_inf=5.0, RN_mid=2040.0, RN_tau=7.0, aN0=0.0, aNinf=0.0, bN=100.0, capN=300.0,
    delta0=0.005, gamma=0.0, Tref=1.2, effN=1.0,
    # geological injection (DACCS + BECCS)
    RG_inf=17.0, RG_mid=2060.0, RG_tau=8.0, aG0=500.0, aGinf=100.0, taucG=15.0, bG=100.0, capG=1000.0, effG=1.0,
    # mineral (enhanced weathering)
    RM_inf=4.0, RM_mid=2055.0, RM_tau=8.0, aM0=200.0, aMinf=80.0, taucM=15.0, bM=100.0, capM=None, effM=1.0,
    # soft limits (off): surcharge s (O/Oref)^nu, use with capN = capG = None
    sN=0.0, sG=0.0, nu=4.0, OrefN=300.0, OrefG=1000.0,
    # learning by doing for G (off)
    learn=0.0, LR=0.15, X0=0.01, aG_floor=60.0,
    # capacity investment for G (off)
    invest=False, phi=0.5, crf=0.0710, dK=0.04, g=0.20, seed=0.05,
    # Direction B: exogenous point-source CCS injection into the shared geological store
    ccs_cum2100=0.0, ccs_mid=2050.0, ccs_tau=8.0,
    # land coupling: biological capacity lost per tonne of geological removal (BECCS share x land intensity)
    land_c=0.0,
    # first year removal is allowed, per family
    startN=2025.0, startG=2025.0, startM=2025.0,
)


def logistic(y, mid, tau):
    return 1.0 / (1.0 + np.exp(-(y - mid) / tau))


def ccs_path(cal, N):
    """Point-source CCS injection (GtCO2/yr), logistic, scaled to the stated cumulative amount 2020-2100, constant
    after 2100 at the 2100 rate until the store is full (handled by the shared cap, which the planner must respect)."""
    y = np.concatenate([[YR[0]], YR[:N]]).astype(float)
    shape = logistic(y, cal['ccs_mid'], cal['ccs_tau'])
    k = y <= 2100
    tot = 5 * shape[1:][k[1:]].sum()
    q = shape * (cal['ccs_cum2100'] / tot) if tot > 0 else np.zeros_like(shape)
    q[y > 2100] = 0.0                                  # CCS assumed to end with fossil use by 2100
    return q


def paths(cal, N):
    y = np.concatenate([[YR[0]], YR[:N]]).astype(float)
    P = {}
    for k in 'NGM':
        P['Rbar' + k] = cal[f'R{k}_inf'] * logistic(y, cal[f'R{k}_mid'], cal[f'R{k}_tau'])
    P['aN'] = np.full(N + 1, cal['aN0'])
    for k in 'GM':
        P['a' + k] = cal[f'a{k}inf'] + (cal[f'a{k}0'] - cal[f'a{k}inf']) * np.exp(-np.maximum(y - 2025.0, 0) / cal[f'tauc{k}'])
    P['ccs'] = ccs_path(cal, N)
    return P


def mu_ceiling(b, p, sc, ramp=True, cpcap=True):
    up = b['miu_up'].copy(); N = p['num_periods']
    for i in range(1, N + 1):
        if YR[i - 1] >= 2070 or not ramp:
            up[i] = 1.0
    up[1:] = np.minimum(up[1:], 1.0)
    if sc in CPCAP and cpcap:
        up[1:] = np.minimum(up[1:], (CPCAP[sc] / p['PBACKTIME'][1:]) ** (1 / (p['expcost2'] - 1)))
    return up


def trajectory(q, MU, RN, RG, RM, S, alpha, IG, rp, eE, eC, zN, zG):
    N = q['N']; ts = q['tstep']; th = q['expcost2']; be = rp['beta']
    MU = MU.at[1].set(q['miu1']); alpha = alpha.at[1].set(q['a0'])
    ZN = jnp.cumsum(zN); ZG = jnp.cumsum(zG)

    def aG_eff(i, SG):
        lbd = rp['aGfl'] + (rp['aG0'] - rp['aGfl']) * (1.0 + SG / rp['X0']) ** (-rp['bL'])
        return (1.0 - rp['wL']) * rp['aG'][i] + rp['wL'] * lbd

    def remcost(i, ON, OG, SG):
        xN = RN[i] / rp['RbarN'][i]; xG = RG[i] / rp['RbarG'][i]; xM = RM[i] / rp['RbarM'][i]
        aG = aG_eff(i, SG)
        opexG = (1.0 - rp['phi'] * rp['inv']) * aG
        c = (rp['aN'][i] * RN[i] + rp['bN'] * rp['RbarN'][i] / (be + 1) * xN ** (be + 1)
             + opexG * RG[i] + rp['bG'] * rp['RbarG'][i] / (be + 1) * xG ** (be + 1)
             + rp['aM'][i] * RM[i] + rp['bM'] * rp['RbarM'][i] / (be + 1) * xM ** (be + 1)
             + rp['sN'] * (jnp.maximum(ON, 0.0) / rp['OrefN']) ** rp['nu'] * RN[i]
             + rp['sG'] * (jnp.maximum(OG, 0.0) / rp['OrefG']) ** rp['nu'] * RG[i]
             + rp['inv'] * rp['phi'] * aG / rp['crf'] * IG[i])
        return 0.001 * c

    def emis(i, YG, LN):
        return ((q['sigma'][i] * YG + q['eland'][i]) * (1 - MU[i])
                - rp['effN'] * RN[i] - rp['effG'] * RG[i] - rp['effM'] * RM[i] + LN)

    def keep(T):
        d = rp['delta0'] * jnp.maximum(0.0, 1.0 + rp['gamma'] * (T - rp['Tref']))
        return (1.0 - d) ** ts

    def dmg(T):
        # DICE-2023 damages plus two optional low-temperature variants (p5, 11 Nov 2026; both off by default):
        # dam_lt1 > 0: damages continue linearly below dam_lt1 with the slope they have there (marginal damage held
        # constant below dam_lt1). dam_vt1 > 0 (quadratic damages only): below dam_vt1 damages are a2 (t1^2/2 + 2 (T - t1/2)^2),
        # which matches level and slope at t1 and makes marginal damage reach zero at t1/2 instead of at 0.
        a2, a3, lt, vt = q['a2base'], q['a3'], q['dam_lt1'], q['dam_vt1']
        Ts = jnp.maximum(T, 1e-6)
        core = a2 * Ts ** a3
        core = jnp.where((lt > 0) & (T < lt), a2 * (lt ** a3 + a3 * lt ** (a3 - 1) * (T - lt)), core)
        core = jnp.where((vt > 0) & (T < vt), a2 * (vt ** 2 / 2 + 2 * (T - vt / 2) ** 2), core)
        return q['a1'] * T + core + q['dam_g'] * jnp.maximum(T - q['dam_t0'], 0.0) ** 3

    K1 = q['k0']; T1 = q['tatm0']
    DAM1 = dmg(T1)
    YG1 = q['eco2Param'][1] * K1 ** q['gama']
    AB1 = YG1 * q['cost1tot'][1] * MU[1] ** th
    RC1 = remcost(1, 0.0 - ZN[1], rp['ccscum'][1] - ZG[1], 0.0)
    Y1 = YG1 * (1 - DAM1) - AB1 - RC1
    I1 = S[1] * Y1; C1 = Y1 - I1 + eC[1]
    E1 = emis(1, YG1, 0.0) + eE[1]
    KC1 = 0.0
    carry0 = (K1, I1, q['CumEmiss0'], q['F_GHGabate2020'], q['res00'], q['res10'], q['res20'], q['res30'],
              q['tbox10'], q['tbox20'], E1, MU[1], 0.0, 0.0, 0.0, KC1, T1, 1.0)
    em = [q['emshare0'], q['emshare1'], q['emshare2'], q['emshare3']]
    tau = [q['tau0'], q['tau1'], q['tau2'], q['tau3']]

    def step(carry, i):
        K, I, CC, FG, R0, R1, R2, R3, B1, B2, Eprev, MUprev, SN, SG, SM, KC, Tprev, kL = carry
        # stocks at the start of period i: last period's stock after its leakage (factor kL) plus last period's removal
        SN = SN * kL + ts * RN[i - 1]
        kL = keep(Tprev)                                        # leakage factor for period i, set by T at its start
        SG = SG + ts * RG[i - 1]; SM = SM + ts * RM[i - 1]
        KC = (1 - rp['dK']) ** ts * KC + ts * IG[i - 1]
        K = (1 - q['dk']) ** ts * K + ts * I
        YG = q['eco2Param'][i] * K ** q['gama']
        LN = SN * (1.0 - kL) / ts                                # leakage flow during period i (conserves carbon)
        E = emis(i, YG, LN) + eE[i]
        CC = CC + Eprev * (ts / 3.666)
        MUnc = rp['ncfix'] * rp['munc'][i - 1] + (1.0 - rp['ncfix']) * MUprev      # non-CO2 abatement: own control or fixed path (v0.8)
        FG = q['Fcoef2'] * FG + q['Fcoef1'] * q['CO2E_GHGabateB'][i - 1] * (1 - MUnc)
        a = alpha[i]; inflow = E / 3.667; Rs = []
        for j, Rj in enumerate((R0, R1, R2, R3)):
            dec = jnp.exp(-ts / (tau[j] * a)); Rs.append(em[j] * tau[j] * a * inflow * (1 - dec) + Rj * dec)
        MAT = jnp.maximum(q['mateq'] + Rs[0] + Rs[1] + Rs[2] + Rs[3], 10.0)
        F = q['fco22x'] * jnp.log(MAT / q['mateq']) / jnp.log(2.0) + q['F_Misc'][i] + FG
        B1 = B1 * jnp.exp(-ts / q['d1']) + q['teq1'] * F * (1 - jnp.exp(-ts / q['d1']))
        B2 = B2 * jnp.exp(-ts / q['d2']) + q['teq2'] * F * (1 - jnp.exp(-ts / q['d2']))
        T = B1 + B2
        DAM = dmg(T)                                              # extra curvature above dam_t0 (v0.7); low-T variants (p5)
        AB = YG * q['cost1tot'][i] * MU[i] ** th
        ON = SN - ZN[i]; OG = SG + rp['ccscum'][i] - ZG[i]
        RC = remcost(i, ON, OG, SG)
        Y = YG * (1 - DAM) - AB - RC
        I = S[i] * Y; C = Y - I + eC[i]
        return ((K, I, CC, FG, Rs[0], Rs[1], Rs[2], Rs[3], B1, B2, E, MU[i], SN, SG, SM, KC, T, kL),
                (C, K, CC, MAT, T, Y, YG, E, AB, RC, SN, SG, SM, KC, LN, ON, OG))

    _, outs = jax.lax.scan(step, carry0, jnp.arange(2, N + 1))
    first = (C1, K1, q['CumEmiss0'], q['mat0'], T1, Y1, YG1, E1, AB1, RC1, 0.0, 0.0, 0.0, KC1, 0.0,
             0.0 - ZN[1], rp['ccscum'][1] - ZG[1])
    keys = ['C', 'K', 'CCATOT', 'MAT', 'TATM', 'Y', 'YGROSS', 'ECO2', 'ABATECOST', 'REMCOST', 'SN', 'SG', 'SM', 'KCAP',
            'LEAK', 'ON', 'OG']
    res = {}
    for k, f, o in zip(keys, first, outs):
        res[k] = jnp.concatenate([jnp.zeros(1), jnp.atleast_1d(jnp.asarray(f, dtype=jnp.float64)), o])
    cpc = 1000.0 * res['C'][1:] / q['L'][1:]; el = q['elasmu']
    U = ((cpc ** (1 - el)) - 1) / (1 - el) - 1
    res['UTILITY'] = ts * q['scale1'] * jnp.sum(U * q['L'][1:] * q['RR'][1:]) + q['scale2']
    return res


_CACHE = {}


def _functions(N, capN, capG, capM, invest):
    key = (N, capN, capG, capM, invest)
    if key in _CACHE:
        return _CACHE[key]
    nb = 7 if invest else 6

    def unpack(x):
        z1 = jnp.zeros(1)
        v = [jnp.concatenate([z1, x[k * N:(k + 1) * N]]) for k in range(nb)]
        if not invest:
            v.append(jnp.zeros(N + 1))
        return v                                               # MU, RN, RG, RM, S, alpha, IG

    def run(x, qd, rp, eE, eC, zN, zG):
        MU, RN, RG, RM, S, al, IG = unpack(x)
        return trajectory(dict(qd, N=N), MU, RN, RG, RM, S, al, IG, rp, eE, eC, zN, zG), al

    def f(x, qd, rp, eE, eC, zN, zG):
        return -run(x, qd, rp, eE, eC, zN, zG)[0]['UTILITY']

    def g(x, qd, rp, eE, eC, zN, zG):
        r, al = run(x, qd, rp, eE, eC, zN, zG)
        parts = [dj.irf_residual(dict(qd, N=N), r, al), r['TATM'][2:]]
        if capN:
            parts.append(r['ON'][2:] + rp['land_c'] * r['SG'][2:])
        if capG:
            parts.append(r['OG'][2:])
        if capM:
            parts.append(r['SM'][2:])
        if invest:
            MU, RN, RG, RM, S, al2, IG = unpack(x)
            parts.append(RG[2:] - r['KCAP'][2:])                                      # removal within capacity
            KC = r['KCAP']
            parts.append(KC[3:] - (1 + rp['g']) ** 5 * KC[2:-1] - rp['seed'])      # growth limit
        return jnp.concatenate(parts)
    z = jnp.zeros(N + 1)
    A = lambda fn: (lambda x, qd, rp: fn(x, qd, rp, z, z, z, z))
    F = {'unpack': unpack, 'run': run, 'nb': nb, 'f_raw': f, 'g_raw': g,
         'f': jax.jit(A(f)), 'grad': jax.jit(jax.grad(A(f))), 'g': jax.jit(A(g)), 'jac': jax.jit(jax.jacfwd(A(g))),
         'hess': jax.jit(jax.hessian(lambda x, lam, of, qd, rp: of * f(x, qd, rp, z, z, z, z) + lam @ g(x, qd, rp, z, z, z, z)))}
    pert = {'E': lambda e, x, qd, rp, fn: fn(x, qd, rp, e, z, z, z), 'C': lambda e, x, qd, rp, fn: fn(x, qd, rp, z, e, z, z),
            'N': lambda e, x, qd, rp, fn: fn(x, qd, rp, z, z, e, z), 'G': lambda e, x, qd, rp, fn: fn(x, qd, rp, z, z, z, e)}
    for k, pf in pert.items():
        F['df' + k] = jax.jit(jax.grad(lambda e, x, qd, rp, pf=pf: pf(e, x, qd, rp, f)))
        F['dg' + k] = jax.jit(jax.jacfwd(lambda e, x, qd, rp, pf=pf: pf(e, x, qd, rp, g)))
    _CACHE[key] = F
    return F


class Problem:
    def __init__(self, scenario=9, cal=None, allow='NGM', tlo=0.5, ramp=True, cpcap=True, session=None):
        self.sc = scenario; self.allow = allow; self.tlo = tlo; self.ramp = ramp; self.cpcap = cpcap
        self.cal = cal = dict(CENTRAL, **(cal or {}))
        if cal.get('ramp_set') is not None:                      # near-term control-rate ceiling on or off (v0.8)
            self.ramp = bool(cal['ramp_set'])
        if cal.get('tlo_set') is not None:                       # override the lower temperature bound (v0.7 mechanism test)
            self.tlo = float(cal['tlo_set'])
        if session is None:
            session = gams_session(scenario) if cpcap else gams_session(scenario, flags=('tlo', 'miulo', 'k0', 'miu1'))
        self.p, self.b = session
        p = self.p; self.N = N = p['num_periods']
        self.q = q = dj.pack_params(p); q['free_miu1'] = False
        self.qd = {k: v for k, v in q.items() if k not in ('N', 'free_miu1')}
        # extra convexity above T0 with level, slope and curvature unchanged at and below T0: dam_g * a2 * max(T - T0, 0)^3
        self.qd['dam_g'] = float(cal.get('dam_g') or 0.0) * float(q['a2base']); self.qd['dam_t0'] = float(cal.get('dam_t0', 2.0))
        self.qd['dam_lt1'] = float(cal.get('dam_lin_t1') or 0.0); self.qd['dam_vt1'] = float(cal.get('dam_van_t1') or 0.0)
        if cal.get('dam_exp') is not None or cal.get('dam_mult') is not None:      # damage curvature / level cases (v0.6)
            a3 = float(q['a3']); a3n = float(cal.get('dam_exp') or a3); tr = float(cal.get('dam_tref', 2.5)); m = float(cal.get('dam_mult') or 1.0)
            self.qd['a2base'] = m * float(q['a2base']) * tr ** a3 / tr ** a3n; self.qd['a3'] = a3n
        self.up = mu_ceiling(self.b, p, scenario, self.ramp, cpcap)
        self.P = P = paths(cal, N)
        self.invest = bool(cal['invest'])
        rp = {k: jnp.asarray(P[k]) for k in ('RbarN', 'RbarG', 'RbarM', 'aN', 'aG', 'aM')}
        rp['ccscum'] = jnp.asarray(np.concatenate([[0.0], np.cumsum(5 * P['ccs'][:-1])]))   # injected before period i
        for k in ('beta', 'bN', 'bG', 'bM', 'delta0', 'gamma', 'Tref', 'effN', 'effG', 'effM', 'sN', 'sG', 'nu', 'OrefN',
                  'OrefG', 'phi', 'crf', 'dK', 'g', 'seed', 'land_c', 'X0', 'aG0'):
            rp[k] = float(cal[k])
        rp['aGfl'] = float(cal['aG_floor']); rp['wL'] = float(cal['learn']); rp['bL'] = float(-np.log2(1.0 - cal['LR']))
        rp['inv'] = 1.0 if self.invest else 0.0
        rp['munc'] = jnp.asarray(np.asarray(cal['munc'], float) if cal.get('munc') is not None else np.zeros(N + 1))
        rp['ncfix'] = 1.0 if cal.get('munc') is not None else 0.0
        self.rp = rp
        self.capN, self.capG, self.capM = cal['capN'] is not None, cal['capG'] is not None, cal['capM'] is not None
        self.F = _functions(N, self.capN, self.capG, self.capM, self.invest)
        self.nb = self.F['nb']; self.nv = self.nb * N
        self.unpack = self.F['unpack']
        self.z = jnp.zeros(N + 1)
        self.hrow, self.hcol = np.tril_indices(self.nv)
        self.fixed = None                                          # (x_fix, first free period index) for re-optimisation

    def run(self, x, eE=None, eC=None, zN=None, zG=None):
        z = self.z
        return self.F['run'](jnp.asarray(x), self.qd, self.rp, z if eE is None else eE, z if eC is None else eC,
                             z if zN is None else zN, z if zG is None else zG)

    def bounds(self):
        N, p, c, P = self.N, self.p, self.cal, self.P
        y = YR[:N]
        mu_lo = self.b['miu_lo'][1:].copy(); mu_up = self.up[1:].copy(); mu_lo[0] = mu_up[0] = p['miu1']
        ups = {}
        for k in 'NGM':
            u = P['Rbar' + k][1:].copy() if k in self.allow else np.zeros(N)
            if k == 'G' and self.invest and k in self.allow:
                u = np.full(N, c['RG_inf'])
            u[y < c['start' + k]] = 0.0
            u[0] = 0.0; u[-1] = 0.0
            ups[k] = u
        xl = [mu_lo, np.zeros(N), np.zeros(N), np.zeros(N), p['sLBounds'][1:], np.full(N, GAMS_ALPHA[0])]
        xu = [mu_up, ups['N'], ups['G'], ups['M'], p['sUBounds'][1:], np.full(N, GAMS_ALPHA[1])]
        if self.invest:
            iu = np.full(N, 50.0) if 'G' in self.allow else np.zeros(N); iu[-1] = 0.0
            xl.append(np.zeros(N)); xu.append(iu)
        xl = np.concatenate(xl); xu = np.concatenate(xu)
        xl[5 * N] = xu[5 * N] = p['a0']
        gl = [np.zeros(N), np.full(N - 1, self.tlo)]; gu = [np.zeros(N), np.full(N - 1, self.b['Tup'])]
        for cap, key in ((self.capN, 'capN'), (self.capG, 'capG'), (self.capM, 'capM')):
            if cap:
                gl.append(np.full(N - 1, -1e4)); gu.append(np.full(N - 1, c[key]))
        if self.invest:
            gl.append(np.full(N - 1, -1e4)); gu.append(np.zeros(N - 1))
            gl.append(np.full(N - 2, -1e4)); gu.append(np.zeros(N - 2))
        if self.fixed is not None:                                 # hold periods before t0 at a previous solution
            xf, t0 = self.fixed
            for blk in getattr(self, 'fixed_blocks', range(self.nb)):
                sl = slice(blk * N, blk * N + t0)
                xl[sl] = xu[sl] = xf[sl]
        return xl, xu, np.concatenate(gl), np.concatenate(gu)

    def objective(self, x): return float(self.F['f'](jnp.asarray(x), self.qd, self.rp))
    def gradient(self, x): return np.asarray(self.F['grad'](jnp.asarray(x), self.qd, self.rp))
    def constraints(self, x): return np.asarray(self.F['g'](jnp.asarray(x), self.qd, self.rp))
    def jacobian(self, x): return np.asarray(self.F['jac'](jnp.asarray(x), self.qd, self.rp)).ravel()
    def hessianstructure(self): return self.hrow, self.hcol
    def hessian(self, x, lam, of):
        return np.asarray(self.F['hess'](jnp.asarray(x), jnp.asarray(lam), of, self.qd, self.rp))[self.hrow, self.hcol]

    def start(self, x0=None):
        N = self.N
        if x0 is None:
            gm = dm.load_gams(self.sc)
            pad = lambda v: np.concatenate([v[:N], np.full(max(N - len(v), 0), v[min(len(v), N) - 1])])
            P = self.P
            blocks = [pad(gm['MIU']), 0.3 * P['RbarN'][1:], 0.1 * P['RbarG'][1:], 0.2 * P['RbarM'][1:], pad(gm['S']), pad(gm['ALPHA'])]
            if self.invest:
                blocks.append(np.full(N, 0.1))
            x0 = np.concatenate(blocks)
        xl, xu, _, _ = self.bounds()
        return np.clip(x0, xl, xu)

    def duals(self, x, lam):
        """SCC and the value of one tonne of free storage space (biological, geological) by period, each in dollars of
        same-period consumption per tonne (current value), plus dV/dC for discounting to 2025."""
        xj = jnp.asarray(x); lj = jnp.asarray(lam); z = self.z; F = self.F; qd, rp = self.qd, self.rp
        dV = {k: np.asarray(F['df' + k](z, xj, qd, rp)) + np.asarray(lj @ F['dg' + k](z, xj, qd, rp)) for k in 'ECNG'}
        # eE and eC are flows over a five-year period (5 GtCO2 and 5 trillion $ per unit); zN and zG are one-off stock
        # changes of 1 GtCO2, hence the factor 5 against the consumption flow
        out = {'scc': -1000.0 * dV['E'] / dV['C'], 'vN': 5000.0 * dV['N'] / dV['C'], 'vG': 5000.0 * dV['G'] / dV['C'], 'dVdC': dV['C']}
        return out

    def solve(self, x0=None, tol=1e-10, max_iter=5000):
        x0 = self.start(x0)
        xl, xu, gl, gu = self.bounds()
        nlp = cyipopt.Problem(n=self.nv, m=len(gl), problem_obj=self, lb=xl, ub=xu, cl=gl, cu=gu)
        U0 = abs(self.objective(x0))
        for k, v in [('tol', tol), ('acceptable_tol', tol * 100), ('print_level', 0), ('max_iter', max_iter),
                     ('obj_scaling_factor', 1.0 / max(U0, 1.0)), ('mu_strategy', 'adaptive'), ('bound_relax_factor', 0.0),
                     ('honor_original_bounds', 'yes'), ('linear_solver', 'mumps')]:
            nlp.add_option(k, v)
        t0 = time.time(); x, info = nlp.solve(x0)
        r, _ = self.run(x)
        d = self.duals(x, info['mult_g'])
        return x, info, r, d, time.time() - t0

    def summary(self, x, info, r, d):
        N = self.N; c = self.cal
        MU, RN, RG, RM, S, al, IG = [np.array(v)[1:] for v in self.unpack(jnp.asarray(x))]
        MU[0] = self.p['miu1']
        g = lambda k: np.asarray(r[k])[1:]
        T, E, MAT, YGv = g('TATM'), g('ECO2'), g('MAT'), g('YGROSS')
        SN, SG, SM, LN, ON, OG, KC = g('SN'), g('SG'), g('SM'), g('LEAK'), g('ON'), g('OG'), g('KCAP')
        Bse = np.asarray(self.q['sigma'][1:]) * YGv + np.asarray(self.q['eland'][1:]); ABT = Bse * MU
        NCO2 = np.asarray(self.q['CO2E_GHGabateB'][1:]) * MU                    # non-CO2 abated through mu, GtCO2e/yr
        iy = lambda yy: int(np.where(YR == yy)[0][0])
        i21 = iy(2100) + 1; i22 = iy(2200) + 1
        nz = np.where(E <= 0)[0]
        cs = lambda v, k: float(5 * v[:k].sum())
        scc = d['scc'][1:]; vN = d['vN'][1:]; vG = d['vG'][1:]; dVdC = d['dVdC'][1:]
        pv = lambda v: (v * dVdC / dVdC[1]).tolist()                   # discount to 2025 consumption dollars
        o = {'sc': self.sc, 'allow': self.allow, 'status': int(info['status']), 'U': float(r['UTILITY']),
             'scc2025': float(scc[1]), 'scc2030': float(scc[2]), 'scc2050': float(scc[6]),
             'miu2030': float(MU[2]), 'miu2050': float(MU[6]), 'netzero_year': int(YR[nz[0]]) if len(nz) else None,
             'peakT': float(T.max()), 'peakT_year': int(YR[int(np.argmax(T))]), 'T2100': float(T[iy(2100)]), 'T2200': float(T[iy(2200)]),
             'cumN_2100': cs(RN, i21), 'cumG_2100': cs(RG, i21), 'cumM_2100': cs(RM, i21),
             'cumN_2200': cs(RN, i22), 'cumG_2200': cs(RG, i22), 'cumM_2200': cs(RM, i22),
             'cumN': cs(RN, N), 'cumG': cs(RG, N), 'cumM': cs(RM, N), 'cumLeak_2200': cs(LN, i22), 'cumLeak': cs(LN, N),
             'cumABT_2100': cs(ABT, i21), 'cumABT_2200': cs(ABT, i22), 'cumE_2020_2100': cs(E, i21),
             'cumNCO2_2100': cs(NCO2, i21), 'cumNCO2_2200': cs(NCO2, i22),
             'vN2025': float(vN[1]), 'vG2025': float(vG[1]), 'vN2025_pv': float(vN[1]), 'vG2025_pv': float(vG[1]),
             'lamC2025': float(-dVdC[1]),
             'paths': {'MU': MU.tolist(), 'RN': RN.tolist(), 'RG': RG.tolist(), 'RM': RM.tolist(), 'IG': IG.tolist(),
                       'T': T.tolist(), 'E': E.tolist(), 'SN': SN.tolist(), 'SG': SG.tolist(), 'SM': SM.tolist(),
                       'ON': ON.tolist(), 'OG': OG.tolist(), 'KCAP': KC.tolist(), 'LEAK': LN.tolist(),
                       'scc': scc.tolist(), 'vN': vN.tolist(), 'vG': vG.tolist(), 'vN_pv': pv(vN), 'vG_pv': pv(vG),
                       'ABT': ABT.tolist(), 'ccs': self.P['ccs'][1:].tolist()}}
        return o
