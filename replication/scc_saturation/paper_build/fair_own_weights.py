"""FaIR v2 carbon-cycle substitution with each calibration's own baseline (referee check for Section 5.2).

exp_c_fair_range.py weights the temperature response to a pulse with marginal welfare weights w(s) taken
from the DICE-2023 optimal trajectory, so that only the impulse response differs across parameter sets.
Here each parameter set is run as a full model (fair_opt.trajectory: DICE-2023 economy, forcing and
two-box temperature with the FaIR v2 CO2 cycle on 1-year steps) under the published DICE-2023 optimal
controls. Its own temperature, damages, consumption and emissions then set the welfare weights, and the
SCC and the feedback share come from automatic differentiation of welfare (fair_opt.split). The policy is
not re-optimized. Sets: the eleven CMIP6 calibrations, the FaIRv2.0.0 default and the AR6 default, as used
and with r0 re-anchored (fair_audit.py). Writes results/fair_own_weights.json.
"""
import json
import numpy as np
import jax.numpy as jnp
from common import save_json, gams_session, gams_controls
import dice_jax as dj
import fair_opt as FO
import exp_c_fair_range as XC

YEARS = (2025, 2050, 2100)

if __name__ == '__main__':
    p, b = gams_session(9)
    MIU, S, g = gams_controls(9, p)
    q = dj.pack_params(p)
    MIUj, Sj = jnp.asarray(MIU), jnp.asarray(S)
    fixed = json.load(open('results/exp_c_fair_range.json'))['sets']
    audit = json.load(open('results/fair_audit.json'))['sets']
    out = {'years': list(YEARS), 'sets': {}}
    for name, o in fixed.items():
        cp = o['params_per_GtCO2']
        row = {'fixed_weights': {'share_2025': o['share_2025'], 'scc_2025': o['scc_2025']}}
        variants = {'as_used': cp}
        if name in audit and 'r0_reanchored' in audit[name]:
            variants['reanchored'] = dict(cp, r0=audit[name]['r0_reanchored'])
            row['fixed_weights_reanchored'] = {'share_2025': audit[name]['share_2025_reanchored'],
                                               'scc_2025': audit[name]['scc_2025_reanchored']}
        for v, c in variants.items():
            tot, dirc, base = FO.split(q, c, MIUj, Sj)
            T = np.asarray(base['TATM'])
            row['own_' + v] = {'scc': {y: float(tot[(y - 2020) // 5]) for y in YEARS},
                               'share': {y: float(1 - dirc[(y - 2020) // 5] / tot[(y - 2020) // 5]) for y in YEARS},
                               'peak_T_2020_2215': float(T[:40].max()), 'T_2100': float(T[16])}
        out['sets'][name] = row
        fw, ow = row['fixed_weights'], row['own_as_used']
        print(f"{name:24s} fixed {100 * fw['share_2025']:6.2f}% ${fw['scc_2025']:6.2f} | own {100 * ow['share'][2025]:6.2f}% "
              f"${ow['scc'][2025]:6.2f} | peak T {ow['peak_T_2020_2215']:.2f}", flush=True)
    cm = [k for k in fixed if k in XC.LEACH and k != 'FaIRv2.0.0 default']
    def rng(key, sub):
        v = np.array([out['sets'][k][key]['share'][2025] if sub else out['sets'][k][key]['share_2025'] for k in cm])
        return {'min': float(v.min()), 'median': float(np.median(v)), 'max': float(v.max())}
    d = np.array([out['sets'][k]['own_as_used']['share'][2025] - out['sets'][k]['fixed_weights']['share_2025'] for k in out['sets']])
    dr = np.array([out['sets'][k]['own_reanchored']['share'][2025] - out['sets'][k]['fixed_weights_reanchored']['share_2025']
                   for k in out['sets'] if 'own_reanchored' in out['sets'][k]])
    ds = np.array([out['sets'][k]['own_as_used']['scc'][2025] / out['sets'][k]['fixed_weights']['scc_2025'] - 1 for k in out['sets']])
    out['summary'] = {'cmip6_fixed': rng('fixed_weights', False), 'cmip6_own': rng('own_as_used', True),
                      'cmip6_own_reanchored': rng('own_reanchored', True),
                      'max_abs_share_diff_pp': float(100 * np.abs(d).max()), 'max_abs_share_diff_reanchored_pp': float(100 * np.abs(dr).max()),
                      'scc_ratio_minus_1_range': [float(ds.min()), float(ds.max())]}
    save_json(out, 'fair_own_weights.json')
    print(out['summary'])
