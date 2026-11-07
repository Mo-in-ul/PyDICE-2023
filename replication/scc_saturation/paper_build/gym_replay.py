"""Gymnasium environment on the corrected model, and a replay of the optimal controls.

The environment advances one 5-year period per step with the same forward equations as the
optimizer (diceForward_numba). In each step alpha(t) is solved so that the IRF condition holds
for the period-t state, which itself depends on alpha(t) (a scalar root, brentq).

The environment used in the submitted paper (notebook cell 56) instead solved alpha(t) from the
period t-1 state, used the pre-correction timing, and dropped the last period. The replay below
runs both alpha rules on the same controls, so the old 0.26% welfare gap can be attributed.

Replay: GAMS optimal controls (scenario 9) through the environment; compare total welfare with
the optimizer's forward model and with GAMS. Also timing per step.
Writes results/gym_replay.json.
"""
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.optimize import brentq
from common import dm, save_json, scenario_params, gams_controls

FORWARD_ARGS = ['tstep', 'dk', 'gama', 'eco2Param', 'sigma', 'eland', 'cost1tot', 'expcost2', 'miuup',
                'sLBounds', 'sUBounds', 'AlphaLowerBound', 'Fcoef1', 'Fcoef2', 'CO2E_GHGabateB',
                'emshare0', 'emshare1', 'emshare2', 'emshare3', 'tau0', 'tau1', 'tau2', 'tau3', 'mateq',
                'fco22x', 'F_Misc', 'teq1', 'teq2', 'd1', 'd2', 'a1', 'a2base', 'a3']


class DICE2023Env(gym.Env):
    """Observation: [K, TATM, MAT, CCATOT, t/N] scaled; action: [MIU, S]; reward: the period's
    contribution to UTILITY (tstep*scale1*PERIODU*L*RR; scale2 added at the last step).
    alpha_rule='contemporaneous' (IRF holds in every period, as in GAMS) or 'lagged' (alpha(t)
    from the period t-1 state, as in the submitted paper)."""

    def __init__(self, params, alpha_rule='contemporaneous'):
        super().__init__()
        self.p = params; self.N = params['num_periods']; self.rule = alpha_rule
        self.action_space = spaces.Box(low=np.array([0.0, 0.0], np.float32), high=np.array([1.2, 1.0], np.float32))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.fargs = [params[k] for k in FORWARD_ARGS]

    def _reward(self, C, t):
        p = self.p
        cpc = 1000.0 * C / p['L'][t]
        u = ((cpc ** (1 - p['elasmu'])) - 1) / (1 - p['elasmu']) - 1
        return p['tstep'] * p['scale1'] * u * p['L'][t] * p['RR'][t]

    def reset(self, seed=None, options=None, S1=None):
        super().reset(seed=seed)
        p = self.p; N = self.N
        self.MIU = np.zeros(N + 1); self.S = np.zeros(N + 1); self.alpha = np.full(N + 1, p['a0'])
        self.MIU[1] = p['miu1']; self.S[1] = p['optlrsav'] if S1 is None else S1
        self.pulse = np.zeros(N + 1)
        K = p['k0']; T = p['tatm0']
        YG = p['eco2Param'][1] * K ** p['gama']
        DAM = p['a1'] * T + p['a2base'] * T ** p['a3']
        Y = YG * (1 - DAM) - YG * p['cost1tot'][1] * self.MIU[1] ** p['expcost2']
        I = self.S[1] * Y; C = Y - I
        E = (p['sigma'][1] * YG + p['eland'][1]) * (1 - self.MIU[1])
        self.st = dict(K=K, I=I, CCATOT=p['CumEmiss0'], FG=p['F_GHGabate2020'], R=[p['res00'], p['res10'], p['res20'], p['res30']],
                       B1=p['tbox10'], B2=p['tbox20'], Eprev=E, Mprev=self.MIU[1], MAT=p['mat0'], TATM=T)
        self.t = 2
        r1 = self._reward(C, 1)
        return self._obs(), {'reward_period1': r1, 'C': C}

    def _advance(self, a_t):
        s = self.st; t = self.t
        self.alpha[t] = a_t
        return dm.diceForward_numba(t, self.MIU, self.S, self.alpha, s['CCATOT'], s['K'], s['I'], s['FG'],
                                    s['R'][0], s['R'][1], s['R'][2], s['R'][3], s['B1'], s['B2'], *self.fargs,
                                    self.pulse, s['Eprev'], s['Mprev'])

    def _irf_lhs(self, CC, MAT, T):
        p = self.p
        return p['irf0'] + p['irC'] * (CC - (MAT - p['mateq'])) + p['irT'] * T

    def step(self, action):
        p = self.p; t = self.t; s = self.st
        self.MIU[t] = float(action[0]); self.S[t] = float(action[1])
        if self.rule == 'lagged':
            target = self._irf_lhs(s['CCATOT'], s['MAT'], s['TATM'])
            a_t = brentq(lambda a: dm.irf_rhs(p, a) - target, 0.05, 100.0, xtol=1e-14)
        else:
            def g(a):
                r = self._advance(a)
                return dm.irf_rhs(p, a) - self._irf_lhs(r[1], r[11], r[12])
            a_t = brentq(g, 0.05, 100.0, xtol=1e-14)
        r = self._advance(a_t)
        (C, CC, K, I, FG, R0, R1, R2, R3, B1, B2, MAT, T, Y, YN, YG, DF, AB, Eflow, Mi) = r
        self.st = dict(K=K, I=I, CCATOT=CC, FG=FG, R=[R0, R1, R2, R3], B1=B1, B2=B2, Eprev=Eflow, Mprev=Mi, MAT=MAT, TATM=T)
        rew = self._reward(C, t)
        done = t == self.N
        if done:
            rew += p['scale2']
        self.t += 1
        return self._obs(), rew, done, False, {'alpha': a_t, 'TATM': T, 'MAT': MAT, 'C': C,
                                              'irf_residual': dm.irf_rhs(p, a_t) - self._irf_lhs(CC, MAT, T)}

    def _obs(self):
        s = self.st
        return np.array([s['K'] / 1e4, s['TATM'] / 10, s['MAT'] / 2e3, s['CCATOT'] / 5e3, (self.t - 1) / self.N], np.float32)


def replay(sc, rule):
    _, p = scenario_params(sc)
    MIU, S, g = gams_controls(sc, p)
    env = DICE2023Env(p, rule)
    _, info = env.reset(S1=S[1])
    W = info['reward_period1']; Ts = []; irf = []
    t0 = time.perf_counter()
    for t in range(2, p['num_periods'] + 1):
        _, r, done, _, inf = env.step(np.array([MIU[t], S[t]]))
        W += r; Ts.append(inf['TATM']); irf.append(abs(inf['irf_residual']))
    dt = time.perf_counter() - t0
    a, _ = dm.solve_alpha(p, MIU, S)
    W_opt = dm.diceTrajectory(p, MIU, S, a)[0]
    W_g = dm._welfare(p, g['C'][:p['num_periods']])
    Ts = np.array(Ts)
    return {'scenario': sc, 'rule': rule, 'welfare_env': W, 'welfare_forward_model': W_opt, 'welfare_gams': W_g,
            'gap_vs_forward_pct': 100 * (W - W_opt) / abs(W_opt), 'gap_vs_gams_pct': 100 * (W - W_g) / abs(W_g),
            'max_dT_vs_gams': float(np.max(np.abs(Ts - g['TATM'][1:p['num_periods']]))),
            'max_irf_residual': float(max(irf)), 'ms_per_step': 1000 * dt / (p['num_periods'] - 1)}


if __name__ == '__main__':
    out = {}
    for sc in (9, 8):
        for rule in ('contemporaneous', 'lagged'):
            replay(sc, rule)   # warm-up (Numba compile)
            o = replay(sc, rule); out[f'{sc}_{rule}'] = o
            print(o, flush=True)
    save_json(out, 'gym_replay.json')
