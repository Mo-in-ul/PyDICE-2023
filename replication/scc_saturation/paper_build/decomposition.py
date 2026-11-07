"""Gap decomposition: what each correction contributes to matching GAMS.

Starting from the submitted PyDICE model, switch the corrections on one at a time
(cumulative), and separately switch each one off from the full set (leave-one-out).
Every configuration is solved with IPOPT (ipopt_model.py, exact JAX derivatives, warm start
from the GAMS solution), so differences come from the model, not from the optimizer.

Corrections, in the order they are switched on:
  land     control rate applies to land-use CO2            (equation, patch 0001)
  ccatot   cumulative emissions use previous-period ECO2   (equation, patch 0001)
  fghg     other-GHG forcing uses previous-period values   (equation, patch 0001)
  savings  S >= 0 and fixed at 0.28 from period 38          (equation; old: S >= 0.2, optlrsav)
  matfloor atmospheric carbon floor 10 GtC, not 588         (equation, 30 Oct)
  tlo      TATM >= 0.5                                      (GAMS session setting)
  carry    miu.lo, carbon-price caps, K(1) = 409            (GAMS session setting)
  miu1     base run: MIU(2020) set by the carbon-price rule (GAMS session setting)

Metrics per scenario and configuration: welfare gap vs the GAMS consumption path, max |dT|
to 2100 and 2215, and the multiplier SCC vs GAMS in 2025 and 2100.
Usage: python decomposition.py <tag> [--cum-only] <scenarios...>  -> results/decomp_<tag>.json
"""
import sys
import numpy as np
from common import dm, save_json
from ipopt_model import solve

STEPS = ['land', 'ccatot', 'fghg', 'savings', 'matfloor', 'tlo', 'carry', 'miu1']


def config(on):
    on = set(on)
    switches = {'old_land': 'land' not in on, 'old_ccatot': 'ccatot' not in on,
                'old_fghg': 'fghg' not in on, 'mat_floor': 10.0 if 'matfloor' in on else 588.0 + 1e-6}
    flags = []
    if 'tlo' in on: flags.append('tlo')
    if 'carry' in on: flags += ['miulo', 'cpcap', 'k0']
    if 'miu1' in on: flags.append('miu1')
    return dict(switches=switches, flags=tuple(flags), old_savings='savings' not in on)


def metrics(sc, r, g):
    pr = r['problem']; N = pr.N
    T = r['traj']['TATM'][1:]
    ccm = -r['dVdC']; eco = -r['dVdE']
    sccg = -1000 * eco / (1e-5 + ccm)
    Wg = dm._welfare(pr.p, g['C'][:N])
    out = {'status': r['status'], 'U': r['U'], 'welfare_gap_pct': 100 * (r['U'] - Wg) / abs(Wg),
           'dT2100': float(np.max(np.abs(T[:17] - g['TATM'][:17]))),
           'dT2215': float(np.max(np.abs(T[:40] - g['TATM'][:40])))}
    for y, k in ((2025, 'A'), (2100, 'C')):
        i = (y - 2020) // 5 + 1
        out[f'scc{k}'] = float(sccg[i]); out[f'scc{k}_gap_pct'] = 100 * (sccg[i] / g['SCC'][i - 1] - 1)
    return out


def run(sc, loo=True):
    g = dm.load_gams(sc)
    res = {'scenario': sc, 'cumulative': [], 'leave_one_out': []}
    for k in range(len(STEPS) + 1):
        on = STEPS[:k]
        try:
            m = metrics(sc, solve(sc, **config(on)), g)
        except Exception as e:
            m = {'error': repr(e)}
        m['added'] = STEPS[k - 1] if k else 'submitted model'
        res['cumulative'].append(m)
        print(sc, 'cum', m['added'], {kk: (round(v, 6) if isinstance(v, float) else v) for kk, v in m.items() if kk in ('status', 'welfare_gap_pct', 'dT2100', 'dT2215', 'sccA_gap_pct')}, flush=True)
    for s in (STEPS if loo else []):
        try:
            m = metrics(sc, solve(sc, **config([x for x in STEPS if x != s])), g)
        except Exception as e:
            m = {'error': repr(e)}
        m['removed'] = s
        res['leave_one_out'].append(m)
        print(sc, 'loo', s, {kk: (round(v, 6) if isinstance(v, float) else v) for kk, v in m.items() if kk in ('status', 'welfare_gap_pct', 'dT2100', 'dT2215', 'sccA_gap_pct')}, flush=True)
    return res


if __name__ == '__main__':
    tag = sys.argv[1]
    cum_only = '--cum-only' in sys.argv
    scs = [int(s) for s in sys.argv[2:] if s != '--cum-only']
    allr = {}
    for sc in scs:
        allr[str(sc)] = run(sc, loo=not cum_only)
        save_json(allr, f'decomp_{tag}.json')
