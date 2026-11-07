"""Benchmark: how large is the saturation feedback in FaIR v2 itself?

DFAIR, the DICE-2023 carbon cycle, is a simplified FaIR. Two things differ from FaIR v2 as used
in the IPCC AR6 (fair v2.2, default 'ar6' species configuration):
  * the closure: DICE solves the 100-year IRF condition for alpha exactly in each 5-year
    period; FaIR v2 uses the closed-form approximation alpha = g0 exp(iIRF / g1) on 1-year steps;
  * the parameters of the IRF target iIRF = r0 + rU G_U + rT T + rA G_A (G_U cumulative uptake,
    G_A airborne stock): DICE uses the FaIR v1 values r0 = 32.4 yr, rU = 0.019 yr/GtC,
    rT = 4.165 yr/K, rA = 0; FaIR v2 AR6 uses r0 = 29 yr, rU = 0.00846 yr/GtCO2
    (0.031 yr/GtC), rT = 4.0 yr/K, rA = 0.000819 yr/GtCO2 (0.0030 yr/GtC).
The four carbon boxes (partition fractions and lifetimes) are the same in both.

Method. The SCC is a welfare-weighted sum of the temperature response to a pulse,
    SCC(t) = -1000 sum_s w(s) dT(s)/dE(t) / lambda_C(t),
where w(s) = dU/dT(s) is the welfare effect of a change in temperature in period s through the
damage function, holding the DICE-2023 optimal controls fixed (dice_jax T_shift hook), and
lambda_C(t) is the marginal utility of consumption (GAMS cc.m). The weights come from DICE-2023;
only the carbon cycle that maps a pulse into dT(s) changes. Every variant uses the DICE forcing
and two-box temperature model and the DICE-2023 optimal emissions, so differences come from the
carbon cycle alone. Each variant is run twice: with alpha responding to the pulse ("full") and
with alpha held at its unperturbed path ("frozen"). The feedback share is 1 - frozen / full.

Variants
  dice      DFAIR as in DICE-2023 (5-year steps, exact alpha); computed in JAX with the IFT.
            Check: must reproduce the optimization-based split (13.0% in 2025).
  dice_fc   DICE parameters, FaIR closure and 1-year steps (isolates the closure).
  fair_ar6  FaIR v2 AR6 default CO2 parameters, FaIR closure, 1-year steps.
  fair_noA  as fair_ar6 with rA = 0 (isolates the airborne term).
  fair_rU   as dice_fc but with the FaIR v2 uptake sensitivity rU (isolates rU).
The annual variants call fair's own calculate_alpha and step_concentration functions.
Initial state (end of 2020): the DICE-2023 reservoirs, cumulative emissions and temperature boxes.
Writes results/fair_benchmark.json.
"""
import numpy as np
import jax, jax.numpy as jnp
from fair.gas_cycle import calculate_alpha
from fair.gas_cycle.forward import step_concentration
from common import dm, save_json, gams_session, gams_controls
import dice_jax as dj

N = 81
C2 = 3.667                       # GtCO2 per GtC, as in the DICE inflow E/3.667
YEARS = (2025, 2050, 2100, 2150)

# FaIR v2.2.5 'ar6' defaults (fair/defaults/data/ar6/species_configs_properties.csv)
PF = np.array([0.2173, 0.224, 0.2824, 0.2763])
LT = np.array([1e9, 394.4, 36.54, 4.304])
G0, G1 = 0.010178, 11.412622
AR6 = dict(r0=29.0, rU=0.00846, rT=4.0, rA=0.000819)          # per GtCO2


def setup():
    p, b = gams_session(9)
    MIU, S, g = gams_controls(9, p)
    alpha, _ = dm.solve_alpha(p, MIU, S)
    q = dj.pack_params(p)
    MIUj, Sj, aj = jnp.asarray(MIU), jnp.asarray(S), jnp.asarray(alpha)
    base = dj.trajectory(q, MIUj, Sj, aj)
    Efix = base['ECO2']   # emissions held at baseline, so the weights do not repeat the output-emissions loop
    w = np.asarray(jax.grad(lambda ts: dj.trajectory(q, MIUj, Sj, aj, T_shift=ts, E_fix=Efix)['UTILITY'])(jnp.zeros(N + 1)))
    lamC = np.zeros(N + 1); lamC[1:] = np.asarray(dj.marginal_utility_C(q, base))
    E = np.asarray(base['ECO2']); T = np.asarray(base['TATM']); MAT = np.asarray(base['MAT'])
    Fco2 = np.zeros(N + 1); Fco2[2:] = p['fco22x'] * np.log(MAT[2:] / p['mateq']) / np.log(2)
    Fnon = np.asarray(base['FORC']) - Fco2          # F_Misc + F_GHGabate, fixed with the controls
    return p, q, MIUj, Sj, aj, w, lamC, E, T, Fnon


def dice_split(q, MIUj, Sj, aj, w, lamC, i):
    """DFAIR in JAX: dT/dE(i) with alpha fixed (direct) and through the IFT (feedback)."""
    def T_of(al1, e):
        al = jnp.concatenate([jnp.zeros(1), al1]); ee = jnp.concatenate([jnp.zeros(1), e])
        return dj.trajectory(q, MIUj, Sj, al, eps_E=ee)['TATM']

    def R_of(al1, e):
        al = jnp.concatenate([jnp.zeros(1), al1]); ee = jnp.concatenate([jnp.zeros(1), e])
        return dj.irf_residual(q, dj.trajectory(q, MIUj, Sj, al, eps_E=ee), al)
    a1 = aj[1:]; e0 = jnp.zeros(N)
    dTdE = np.asarray(jax.jacfwd(T_of, argnums=1)(a1, e0))[1:, i - 1]
    dTda = np.asarray(jax.jacfwd(T_of, argnums=0)(a1, e0))[1:]
    Ja = np.asarray(jax.jacfwd(R_of, argnums=0)(a1, e0)); Je = np.asarray(jax.jacfwd(R_of, argnums=1)(a1, e0))
    dadE = -np.linalg.solve(Ja, Je[:, i - 1])
    dT_dir = dTdE; dT_fb = dTda @ dadE
    return dT_dir, dT_fb


def step(E, boxes, a):
    """One 1-year step of fair.gas_cycle.forward.step_concentration (same formula; check_step())."""
    rate = 1.0 / (a * LT); dec = np.exp(-rate)
    return PF * E / rate * (1 - dec) + boxes * dec


def check_step():
    rng = np.random.default_rng(0)
    b = rng.uniform(0, 500, 4); e, a = 40.0, 0.4
    _, bf, _ = step_concentration(np.array(e).reshape(1, 1, 1, 1), b.reshape(1, 1, 1, 1, 4), b.sum(), a, 0.0, 0.0, 1.0,
                                  LT.reshape(1, 1, 1, 1, 4), PF.reshape(1, 1, 1, 1, 4), 1.0)
    return float(np.max(np.abs(bf.ravel() - step(e, b, a))))


class Annual:
    """FaIR v2 CO2 cycle on 1-year steps (fair's own functions), DICE forcing and temperature."""
    def __init__(self, p, Fnon, r0, rU, rT, rA):
        self.p, self.Fnon = p, Fnon
        self.par = (r0, rA, rT, rU)

    def run(self, Eper, alpha_path=None):
        p = self.p; r0, rA, rT, rU = self.par
        boxes = np.array([p['res00'], p['res10'], p['res20'], p['res30']]) * C2   # GtCO2
        airb = boxes.sum(); cum = p['CumEmiss0'] * C2
        B1, B2 = p['tbox10'], p['tbox20']; T = B1 + B2
        Tout = np.zeros(N + 1); Tout[1] = T
        alphas = []
        k = 0
        for i in range(2, N + 1):
            for yr in range(5):
                if alpha_path is None:
                    a = float(calculate_alpha(np.array([airb]), np.array([cum]), G0, G1, r0, rA, rT, rU, np.array([T]), 100.0)[0])
                else:
                    a = alpha_path[k]
                alphas.append(a); k += 1
                boxes = step(Eper[i], boxes, a)
                airb = float(boxes.sum()); cum += Eper[i]
                MAT = p['mateq'] + airb / C2
                F = p['fco22x'] * np.log(MAT / p['mateq']) / np.log(2) + self.Fnon[i]
                B1 = B1 * np.exp(-1 / p['d1']) + p['teq1'] * F * (1 - np.exp(-1 / p['d1']))
                B2 = B2 * np.exp(-1 / p['d2']) + p['teq2'] * F * (1 - np.exp(-1 / p['d2']))
                T = B1 + B2
            Tout[i] = T
        return Tout, np.array(alphas)

    def split(self, E, i, h=1e-3):
        T0, a0 = self.run(E)
        Ep, Em = E.copy(), E.copy(); Ep[i] += h; Em[i] -= h
        full = (self.run(Ep)[0] - self.run(Em)[0]) / (2 * h)
        frozen = (self.run(Ep, a0)[0] - self.run(Em, a0)[0]) / (2 * h)
        return frozen[1:], (full - frozen)[1:], T0, a0


def scc_parts(w, lamC, i, dT_dir, dT_fb):
    d = -1000 * float(w[1:] @ dT_dir) / lamC[i]; f = -1000 * float(w[1:] @ dT_fb) / lamC[i]
    return d, f


if __name__ == '__main__':
    p, q, MIUj, Sj, aj, w, lamC, E, T, Fnon = setup()
    print('step check vs fair', check_step())
    fc = 1 / C2
    variants = {
        'dfair_closure': dict(r0=p['irf0'], rU=p['irC'] * fc, rT=p['irT'], rA=0.0),
        'fair_rU': dict(r0=p['irf0'], rU=AR6['rU'], rT=p['irT'], rA=0.0),
        'fair_noA': dict(AR6, rA=0.0),
        'fair_ar6': AR6,
    }
    out = {'years': list(YEARS), 'variants': {}}
    o = {'scc': [], 'direct': [], 'feedback': [], 'share': []}
    for y in YEARS:
        i = (y - 2020) // 5 + 1
        d, f = scc_parts(w, lamC, i, *dice_split(q, MIUj, Sj, aj, w, lamC, i))
        o['scc'].append(d + f); o['direct'].append(d); o['feedback'].append(f); o['share'].append(f / (d + f))
    out['variants']['dice'] = o
    print('dice', [round(s, 4) for s in o['share']], [round(s, 2) for s in o['scc']], flush=True)
    for name, par in variants.items():
        m = Annual(p, Fnon, **par)
        o = {'params': par, 'scc': [], 'direct': [], 'feedback': [], 'share': []}
        for y in YEARS:
            i = (y - 2020) // 5 + 1
            dT_dir, dT_fb, T0, a0 = m.split(E, i)
            d, f = scc_parts(w, lamC, i, dT_dir, dT_fb)
            o['scc'].append(d + f); o['direct'].append(d); o['feedback'].append(f); o['share'].append(f / (d + f))
            if y == 2025:
                o['T_base'] = T0[1:41].tolist(); o['alpha_2025_2100'] = a0[:81].tolist()
                o['dT_dir_2025'] = dT_dir[:41].tolist(); o['dT_fb_2025'] = dT_fb[:41].tolist()
        out['variants'][name] = o
        print(name, [round(s, 4) for s in o['share']], [round(s, 2) for s in o['scc']], flush=True)
    out['dice_T_base'] = T[1:41].tolist()
    save_json(out, 'fair_benchmark.json')
