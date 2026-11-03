"""Shared setup for the elasticity-letter scripts. Imports the reference model (dice2023_corrected.py) from this folder and reads the GAMS reference solutions from dice/validation/gams_reference."""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
NB = HERE
os.environ.setdefault('PYDICE_GAMS_DIR', os.path.join(REPO, 'dice', 'validation', 'gams_reference'))
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, NB)
import dice2023_corrected as dm  # noqa: E402

dm.RESULTS_DIR = os.path.join(HERE, 'results', 'model_dumps')
RES = os.path.join(HERE, 'results')
os.makedirs(RES, exist_ok=True)

YEARS = 2020 + 5 * np.arange(81)
def idx(year):
    return int(np.searchsorted(YEARS, year))

def save_json(obj, name):
    def conv(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.bool_,)): return bool(o)
        raise TypeError(type(o))
    with open(os.path.join(RES, name), 'w') as f:
        json.dump(obj, f, indent=1, default=conv)

def load_json(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)

def scenario_params(scenario):
    """Params with the scenario tweaks applied, without optimizing (mirrors Dice2023Model.run_model)."""
    m = dm.Dice2023Model(81, scenario)
    p = m.params
    if scenario in (1, 2, 3, 4, 5):
        k0 = {1: 420, 2: 409, 3: 370, 4: 326, 5: 290}[scenario]
        p['k0'] = k0
        kw = {4: dict(cap_after_t=81, cap_exp_value=400), 5: dict(cap_after_t=51, cap_exp_value=255)}.get(scenario, {})
        dm.apply_disc_prstp(p, prstp_value=0.01 * scenario, **kw)
    elif scenario == 8:
        for i in range(1, 82):
            p['miuup'][i] = min(0.05 + 0.04 * (i - 1) - 0.01 * max(0, i - 5), p['limmiu2070'])
    elif scenario == 10:
        p['miuup'][:] = 1.0
    return m, p


# ── GAMS session settings (single source of truth) ─────────────────────────────────────
# The published GAMS outputs come from one session (Putlong-4-3-10.gms) that solves, in order:
# opt, T2, T1.5, altdam, paris, base, DISC5, DISC4, DISC3, DISC2, DISC1. Bounds persist.
GAMS_TLO = 0.5               # FAIR-beta-4-3-1.gms l.98: TATM.lo = 0.5 ("for stability")
GAMS_ALPHA = (0.1, 100.0)    # FAIR-beta-4-3-1.gms l.99-100


def gams_session(scenario, flags=('tlo', 'miulo', 'cpcap', 'k0', 'miu1')):
    """Params and bounds that reproduce the published GAMS run for `scenario`.
    flags (any subset) switch on the session settings one at a time:
      tlo   : TATM >= 0.5 (PyDICE default 0.01)
      miulo : miu.lo = 0.01, set in the Paris run and kept in base, DISC5..DISC1
      cpcap : cprice.up = 500 after base (DISC5), 1000 from DISC4 on (DISC4..DISC1)
      k0    : DISC1 sets k0 = 420 but never re-fixes K(1), which stays at 409 from DISC2
      miu1  : base only: MIU(1) is set by the binding cprice <= cpricebase rule (0.0513), not 0.05
    Returns (params, b) where b holds 1-based bound arrays miu_lo, miu_up, Tlo, Tup."""
    flags = set(flags)
    _, p = scenario_params(scenario)
    N = p['num_periods']
    if scenario not in (1, 2, 3, 4, 5):
        p['rartp'] = np.exp(p['prstp'] + p['betaclim'] * p['pi']) - 1
    if 'k0' in flags and scenario == 1:
        p['k0'] = 409.0
    miu_up = p['miuup'].copy()
    cap = {5: 500.0, 4: 1000.0, 3: 1000.0, 2: 1000.0, 1: 1000.0}.get(scenario) if 'cpcap' in flags else None
    if cap is not None:
        miu_up[1:] = np.minimum(miu_up[1:], (cap / p['PBACKTIME'][1:]) ** (1 / (p['expcost2'] - 1)))
    if scenario == 10:   # def-base: cprice <= cpricebase for t < 47, miu fixed at 1 for t > 57
        t = np.arange(1, 47)
        miu_up[t] = np.minimum(miu_up[t], (p['cpricebase'][t] / p['PBACKTIME'][t]) ** (1 / (p['expcost2'] - 1)))
    miu_lo = np.zeros(N + 1)
    if 'miulo' in flags and scenario in (10, 5, 4, 3, 2, 1):
        miu_lo[:] = 0.01
    if scenario == 10:
        miu_lo[58:] = 1.0; miu_up[58:] = 1.0
    free_miu1 = scenario == 10 and 'miu1' in flags
    if free_miu1:
        # the cap binds: MIU(1) = (cpricebase(1)/PBACKTIME(1))^(1/(expcost2-1)) = 0.0513. The Numba
        # forward model forces MIU[1] = p['miu1'], so put the binding value there (sigma(1) is
        # already computed from the historical miu1 = 0.05, as in GAMS).
        p['miu1'] = float(miu_up[1])
        miu_lo[1] = min(miu_lo[1], miu_up[1])
    else:
        miu_lo[1] = miu_up[1] = p['miu1']
    # GAMS never fixes MIU(1); it is capped by miuup(1) = 0.05, which binds in every run except
    # base, where def-base resets the cap to 1 and the cprice <= cpricebase rule sets MIU(1) = 0.0513.
    p['miuup'] = miu_up
    b = {'miu_lo': np.minimum(miu_lo, miu_up), 'miu_up': miu_up,
         'Tlo': GAMS_TLO if 'tlo' in flags else 0.01,
         'Tup': {6: 1.5, 7: 2.0, 10: 15.0}.get(scenario, 20.0), 'free_miu1': free_miu1}
    return p, b


def gams_controls(scenario, p):
    """GAMS control paths (1-based MIU, S) for `scenario`. Sets p['miu1'] to the GAMS MIU(1),
    because the forward model forces MIU[1] = p['miu1']: GAMS caps MIU(1) at 0.05 and the cap
    binds in every run except base, where MIU(1) = 0.0513."""
    g = dm.load_gams(scenario)
    N = p['num_periods']
    MIU = np.zeros(N + 1); S = np.zeros(N + 1)
    MIU[1:] = g['MIU'][:N]; S[1:] = g['S'][:N]
    p['miu1'] = float(MIU[1])
    return MIU, S, g
