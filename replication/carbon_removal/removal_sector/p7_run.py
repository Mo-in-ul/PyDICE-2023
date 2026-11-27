"""Revision runs of 12 Nov 2026 (more defensible discounting; sensitivity to the near-term ceiling). Results: results/p7_runs.json.
Nothing in removal3.py, removal3_stoch.py or the p3 to p5 scripts is changed; the plans and the regret are computed as in p5_regret.py.
Case names:  disc:RHO:ETA  published GAMS session with utility discount rate RHO and elasticity ETA (as p5_tests.py);
             d2            the 2 percent calibration of Barrage and Nordhaus (2024).
Item syntax: CASE@F@T[@noceil]   F = factor applied to the growth of the near-term control-rate ceiling after 2030 (1 = DICE-2023),
                                 T = learning year (2050 or 2070); p = 0.5.
The learning-2050 problem is solved jointly; if that fails or beats no benchmark plan, and for learning in 2070, in reduced space
(as p3_reduced.py), with a checkpoint so that a run can be resumed after the 150 s time budget of one call (env BUDGET).
Usage: python p7_run.py disc:0.001:1.1@1@2050 d2@2@2070 ..."""
import os, sys, json, time
import numpy as np
from scipy.optimize import minimize
import removal3 as R3, removal3_stoch as ST
import dice2023_corrected as dm
from p4_common import *

OUT = 'p7_runs.json'; STATE = os.path.join(RES, 'p7_state' + os.environ.get('SLOT', '') + '.json'); BUDGET = float(os.environ.get('BUDGET', '160')); NCTRL = 5
_orig = R3.mu_ceiling; START = time.time()


def set_factor(f):
    def mc(b, p, sc, ramp=True, cpcap=True):
        up = _orig(b, p, sc, ramp, cpcap)
        if f != 1.0 and ramp:
            piv = up[1 + iy(2030)]
            for i in range(1, p['num_periods'] + 1):
                if 2030 < R3.YR[i - 1] < 2070:
                    up[i] = min(1.0, piv + f * (up[i] - piv))
        return up
    R3.mu_ceiling = mc


def mkses(case):
    if case.startswith('disc'):
        _, rho, eta = case.split(':')
        def mk():
            p, b = gams_session(9); dm.apply_disc_prstp(p, prstp_value=float(rho), elasmu_value=float(eta)); return p, b
        return 9, mk
    return SC[case], (lambda: None)


def fixed(sc, mk, cal, allow, xc, k):
    pr = R3.Problem(sc, cal=cal, allow=allow, session=mk()); pr.fixed = (np.asarray(xc), k)
    x, info, r, d, _ = pr.solve(x0=np.asarray(xc)); return pr.summary(x, info, r, d)


def cum(s, k): return float(5 * np.array(s['paths']['ABT'])[:k].sum())


class State:
    def __init__(self, sc, mk, cal, allow, k):
        self.pr = R3.Problem(sc, cal=cal, allow=allow, session=mk()); self.pr.fixed_blocks = range(NCTRL); self.k = k; self.xref = None

    def solve(self, xfix):
        pr = self.pr; pr.fixed = (xfix, self.k)
        x0 = xfix if self.xref is None else np.where(self.mask, xfix, self.xref)
        x, info, r, d, dt = pr.solve(x0=x0)
        g = pr.gradient(x) + pr.jacobian(x).reshape(-1, pr.nv).T @ np.asarray(info['mult_g'])
        return float(r['UTILITY']), -g, int(info['status']), (x, info, r, d)


def reduced(tag, p, sc, mk, calS, calF, k, lam):
    S = State(sc, mk, calS, 'NGM', k); F = State(sc, mk, calF, '', k)
    xS, *_ = S.pr.solve(); xF, *_ = F.pr.solve(); S.xref = np.array(xS); F.xref = np.array(xF)
    N = S.pr.N; S.pr.fixed = None; lo, hi, _, _ = F.pr.bounds(); loS, hiS, _, _ = S.pr.bounds()
    lo = np.maximum(lo, loS); hi = np.minimum(hi, hiS)
    idx = np.concatenate([np.arange(b * N, b * N + k) for b in range(NCTRL)]); free = idx[hi[idx] - lo[idx] > 1e-12]
    mask = np.zeros(S.pr.nv, bool); mask[idx] = True; S.mask = F.mask = mask
    base = np.array(xF, float); base[idx] = np.clip(p * np.asarray(xS)[idx] + (1 - p) * np.asarray(xF)[idx], lo[idx], hi[idx])
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    z0 = np.array(st[tag]['z']) if tag in st else base[free]
    z0 = np.clip(z0, lo[free], hi[free]); base[free] = z0
    t0 = time.time(); ref = [None]; last = {'z': z0.copy(), 'n': 0}

    def fg(z):
        if time.time() - START > BUDGET:
            raise TimeoutError
        xf = base.copy(); xf[free] = z
        uS, gS, stS, _ = S.solve(xf); uF, gF, stF, _ = F.solve(xf)
        EU = p * uS + (1 - p) * uF; grad = (p * gS + (1 - p) * gF)[free]
        if stS not in (0, 1) or stF not in (0, 1) or not np.all(np.isfinite(grad)):
            return 1e3, np.zeros(len(free))
        if ref[0] is None: ref[0] = EU
        last['z'] = z.copy(); last['n'] += 1
        return -(EU - ref[0]) / lam, -grad / lam
    try:
        for rnd in range(5):
            res = minimize(fg, z0, jac=True, method='L-BFGS-B', bounds=list(zip(lo[free], hi[free])), options={'maxiter': 80, 'ftol': 1e-15, 'gtol': 1e-6, 'maxcor': 30})
            z0 = res.x; gz = -res.jac
            pgz = np.where((z0 <= lo[free] + 1e-7) & (gz < 0), 0, np.where((z0 >= hi[free] - 1e-7) & (gz > 0), 0, gz))
            if np.max(np.abs(pgz)) < 1e-2: break
    except TimeoutError:
        st = json.load(open(STATE)) if os.path.exists(STATE) else {}; st[tag] = {'z': last['z'].tolist()}; json.dump(st, open(STATE, 'w'))
        print(f"RESUME {tag}: time budget reached after {last['n']} evaluations", flush=True); sys.exit(3)
    xf = base.copy(); xf[free] = res.x
    uS, gS, stS, oS = S.solve(xf); uF, gF, stF, oF = F.solve(xf)
    sS = S.pr.summary(*oS); sF = F.pr.summary(*oF); gfin = (p * gS + (1 - p) * gF)[free] / lam; z = res.x
    pg = np.where((z <= lo[free] + 1e-7) & (gfin < 0), 0, np.where((z >= hi[free] - 1e-7) & (gfin > 0), 0, gfin))
    return {'EU': p * uS + (1 - p) * uF, 'abt': cum(sS, k), 'peakF': sF['peakT'], 'scc': p * sS['scc2025'] + (1 - p) * sF['scc2025'],
            'status': f'reduced {stS}/{stF}', 'pg': float(np.max(np.abs(pg)))}


def run(item):
    case, f, t, *rest = item.split('@'); f = float(f); t = int(t); noceil = 'noceil' in rest; p = 0.5
    set_factor(f); sc, mk = mkses(case); k = iy(t); base = dict(ramp_set=False) if noceil else {}
    out = load(OUT).get(item, {})
    if 'regret_expected' in out: return
    calS, calF = capcal(1.0, base), dict(base)
    xS, sS, _ = det(sc, calS, 'NGM', mk()); xF, sF, _ = det(sc, calF, '', mk()); xE, sE, _ = det(sc, capcal(p, base), 'NGM', mk())
    lam = sF['lamC2025'] / 5
    plans = {}
    for lab, xc in (('optimist', xS), ('pessimist', xF), ('expected', xE)):
        a = fixed(sc, mk, calS, 'NGM', xc, k); b = fixed(sc, mk, calF, '', xc, k)
        plans[lab] = {'US': a['U'], 'UF': b['U'], 'abt': cum(a, k), 'peakF': b['peakT'], 'st': f"{a['status']}/{b['status']}"}
        plans[lab]['EU'] = p * a['U'] + (1 - p) * b['U']
    plans['expected']['scc'] = sE['scc2025']
    st = None
    if t == 2050:
        m = ST.TwoState(sc, calS, 'NGM', calF, '', p, t, session=mk())
        x, info, scc, dt = m.solve(x0=m.start(xS, xF, common_from='S'))
        s2 = m.summary(x, info, scc)
        st = {'EU': s2['EU'], 'abt': s2['S']['cumABT_to_res'], 'peakF': s2['F']['peakT'], 'scc': s2['scc2025'], 'status': f"joint {s2['status']}"}
        if s2['status'] not in (0, 1) or st['EU'] < max(plans[l]['EU'] for l in plans) - 1e-9: st = None
    if st is None:
        st = reduced(item, p, sc, mk, calS, calF, k, lam)
        if st['EU'] < max(plans[l]['EU'] for l in plans) - 1e-7: print('WARNING: stochastic optimum below a benchmark plan', flush=True)
    o = {'lam': lam, 'stochastic': st, 'plans': plans, 'perfect_EU': p * sS['U'] + (1 - p) * sF['U'], 'sccK': sS['scc2025'], 'scc0': sF['scc2025'], 'sccE': sE['scc2025'],
         'V': (plans['optimist']['US'] - plans['pessimist']['UF']) / lam}
    for lab in plans: o[f'regret_{lab}'] = (st['EU'] - plans[lab]['EU']) / lam
    o['perfect_gain'] = (o['perfect_EU'] - st['EU']) / lam; o['premium_pct'] = 100 * (st['scc'] - sE['scc2025']) / sE['scc2025']
    o['regret_pct_pV'] = 100 * o['regret_expected'] / (0.5 * o['V'])
    save(OUT, item, o)
    print(f"{item}: premium {o['premium_pct']:.2f}% regret exp {o['regret_expected']:.4f} opt {o['regret_optimist']:.4f} pes {o['regret_pessimist']:.4f} "
          f"({o['regret_pct_pV']:.3f}% of pV) abt stoch {st['abt']:.1f} exp {plans['expected']['abt']:.1f} peakF {st['peakF']:.3f}/{plans['expected']['peakF']:.3f} {st['status']}", flush=True)


if __name__ == '__main__':
    for it in sys.argv[1:]:
        if time.time() - START > float(os.environ.get('PAUSE', '45')) and 'regret_expected' not in load(OUT).get(it, {}): print('PAUSE before', it); sys.exit(4)
        run(it)
    print('done')
