"""v8 robustness: endogenous policy under alternative carbon cycles (referee comment 6).

C2. For each FaIR v2 parameter set (the eleven CMIP6 calibrations of Leach et al. 2021, the FaIRv2.0.0 default and
    the AR6 default), DICE-2023 with that carbon cycle is re-optimized (fair_opt.solve) and the SCC is split at the
    new optimum. Also reported: the split at the DICE-2023 optimal policy, run through the same model, so the
    policy effect of each carbon cycle can be read off directly.
C3. Climate-sensitivity sweep under three carbon cycles (AR6 default and two CMIP6 sets). For each ECS the share is
    computed (a) with the policy held at that carbon cycle's own optimum at the default ECS and (b) re-optimized.
    This repeats the fixed-policy versus re-optimized comparison of Section 6.1 outside DFAIR.
Writes results/c2_fair_reopt.json.
"""
import sys
import numpy as np
import jax.numpy as jnp
from common import save_json, gams_session, gams_controls
import fair_opt as FO
from exp_c_fair_range import LEACH

C2 = FO.C2
YEARS = list(range(2025, 2201, 5))
ECS0 = 3.93 * (0.324 + 0.44)


def shares(q, cp, MIU, S):
    tot, dirc, base = FO.split(q, cp, MIU, S)
    sh = 1 - dirc / tot
    idx = [(y - 2020) // 5 for y in YEARS]
    s = [float(sh[i]) for i in idx]
    neg = [y for y, v in zip(YEARS, s) if v < 0]
    return {'share_2025': s[0], 'scc_2025': float(tot[idx[0]]), 'feedback_2025': float(tot[idx[0]] - dirc[idx[0]]),
            'share_2100': s[YEARS.index(2100)], 'first_negative_pulse_year': neg[0] if neg else None,
            'peak_T': float(np.asarray(base['TATM'])[:40].max())}


def setp(ecs=None):
    p, b = gams_session(9)
    b['Tlo'] = 0.01
    if ecs is not None:
        sc = ecs / ECS0
        p['teq1'] = 0.324 * sc; p['teq2'] = 0.44 * sc
    return p, b


def run_c2(out):
    p, b = setp()
    MIUg, Sg, _ = gams_controls(9, p)
    x_g = np.concatenate([MIUg[2:], Sg[1:]])
    sets = {k: dict(r0=v[0], rU=v[1] / C2, rT=v[2], rA=v[3] / C2) for k, v in LEACH.items()}
    sets['FaIR v2.2 AR6 default'] = FO.AR6
    out.setdefault('c2', {})
    for name, cp in sets.items():
        q, prob, x, info = FO.solve(cp, p, b, x_g)
        MIU, S = prob.unpack(jnp.asarray(x))
        ro = shares(q, cp, MIU, S)
        fx = shares(q, cp, jnp.asarray(MIUg), jnp.asarray(Sg))
        out['c2'][name] = {'status': int(info['status']), 'reopt': ro, 'dice_policy': fx,
                           'cprice_2030': float(p['PBACKTIME'][3] * float(MIU[3]) ** (p['expcost2'] - 1))}
        print(f"{name:24s} st {info['status']}  share DICE policy {100 * fx['share_2025']:6.2f}%  re-opt {100 * ro['share_2025']:6.2f}%"
              f"  SCC {fx['scc_2025']:6.2f} -> {ro['scc_2025']:6.2f}  neg {fx['first_negative_pulse_year']} -> {ro['first_negative_pulse_year']}", flush=True)
        save_json(out, 'c2_fair_reopt.json')
    v = [o['reopt']['share_2025'] for k, o in out['c2'].items() if k in LEACH and k != 'FaIRv2.0.0 default']
    w = [o['dice_policy']['share_2025'] for k, o in out['c2'].items() if k in LEACH and k != 'FaIRv2.0.0 default']
    out['c2_summary'] = {'reopt': [min(v), float(np.median(v)), max(v)], 'dice_policy': [min(w), float(np.median(w)), max(w)]}
    save_json(out, 'c2_fair_reopt.json'); print(out['c2_summary'])


def run_c3(out, names):
    sets = {k: dict(r0=v[0], rU=v[1] / C2, rT=v[2], rA=v[3] / C2) for k, v in LEACH.items()}
    sets['FaIR v2.2 AR6 default'] = FO.AR6
    out.setdefault('c3', {})
    for name in names:
        cp = sets[name]
        p0, b0 = setp()
        MIUg, Sg, _ = gams_controls(9, p0)
        x_g = np.concatenate([MIUg[2:], Sg[1:]])
        q0, prob0, x0, _ = FO.solve(cp, p0, b0, x_g)
        MIU0, S0 = prob0.unpack(jnp.asarray(x0))
        base = shares(q0, cp, MIU0, S0)
        rows = []
        for ecs in (2.0, 2.5, 3.5, 4.0, 5.0):
            p, b = setp(ecs)
            q, prob, x, info = FO.solve(cp, p, b, np.asarray(x0))
            MIU, S = prob.unpack(jnp.asarray(x))
            fx = shares(q, cp, MIU0, S0)
            ro = shares(q, cp, MIU, S)
            rows.append({'ecs': ecs, 'status': int(info['status']), 'fixed': fx, 'reopt': ro})
            print(f"{name:24s} ECS {ecs}: fixed {100 * fx['share_2025']:6.2f}%  re-opt {100 * ro['share_2025']:6.2f}%"
                  f"  feedback $ {fx['feedback_2025']:6.2f} / {ro['feedback_2025']:6.2f}", flush=True)
        out['c3'][name] = {'baseline': base, 'rows': rows}
        save_json(out, 'c2_fair_reopt.json')


if __name__ == '__main__':
    import os
    from common import RES, load_json
    out = load_json('c2_fair_reopt.json') if os.path.exists(os.path.join(RES, 'c2_fair_reopt.json')) else {}
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('all', 'c2'):
        run_c2(out)
    if what in ('all', 'c3'):
        run_c3(out, ['FaIR v2.2 AR6 default', 'IPSL-CM6A-LR', 'MPI-ESM1-2-LR'])
