"""Gradual learning (12 Nov 2026): a signal in 2050 moves the probability of success from the prior 0.5 to PG (good signal, probability 0.5)
or PB (bad signal, probability 0.5); the outcome is learned in 2070. Four terminal states (good/bad signal, success/failure);
decisions are common before 2050, common within each signal branch from 2050 to 2070, and state-specific after 2070.
Solved in reduced space as in p3_reduced.py and p7_run.py: the outer variables are the decisions before 2070 (common block before 2050,
one block per signal branch from 2050 to 2070); each terminal state is solved with them fixed; envelope gradients.
Benchmarks: the expected-capacity planner who updates (plans on the current expected capacity before 2050, on PG*K or PB*K after the
signal, re-optimises after 2070) and the one who never updates before 2070 (taken from p7_runs.json, case@1@2070).
Item syntax: CASE@PG@PB, e.g. d2@0.75@0.25. Results: results/p7_tree.json. Usage: python p7_tree.py d2@0.75@0.25"""
import os, sys, json, time
import numpy as np
from scipy.optimize import minimize
from p7_run import *
import p7_run

TOUT = 'p7_tree.json'


def tree(item):
    case, pg, pb = item.split('@'); pg = float(pg); pb = float(pb); q = 0.5; prior = q * pg + (1 - q) * pb
    sc, mk = mkses(case); k1 = iy(2050); k2 = iy(2070); out = load(TOUT).get(item, {})
    if 'regret_updating' in out: return
    set_factor(1.0)
    calS, calF = capcal(1.0, {}), {}
    xS, sS, _ = det(sc, calS, 'NGM', mk()); xF, sF, _ = det(sc, calF, '', mk()); xE, sE, _ = det(sc, capcal(prior, {}), 'NGM', mk())
    lam = sF['lamC2025'] / 5
    S = State(sc, mk, calS, 'NGM', k2); F = State(sc, mk, calF, '', k2); S.xref = np.array(xS); F.xref = np.array(xF)
    N = S.pr.N; S.pr.fixed = None; lo, hi, _, _ = F.pr.bounds(); loS, hiS, _, _ = S.pr.bounds(); lo = np.maximum(lo, loS); hi = np.minimum(hi, hiS)
    idx1 = np.concatenate([np.arange(b * N, b * N + k1) for b in range(NCTRL)]); idx2 = np.concatenate([np.arange(b * N + k1, b * N + k2) for b in range(NCTRL)])
    f1 = idx1[hi[idx1] - lo[idx1] > 1e-12]; f2 = idx2[hi[idx2] - lo[idx2] > 1e-12]; n1, n2 = len(f1), len(f2)
    mask = np.zeros(S.pr.nv, bool); mask[idx1] = True; mask[idx2] = True; S.mask = F.mask = mask
    base = np.array(xF, float)
    for idx, w in ((idx1, prior), (idx2, None)):
        base[idx] = np.clip(w * np.asarray(xS)[idx] + (1 - w) * np.asarray(xF)[idx], lo[idx], hi[idx]) if w is not None else base[idx]
    stt = json.load(open(STATE)) if os.path.exists(STATE) else {}; tag = 'tree:' + item
    if tag in stt: z0 = np.array(stt[tag]['z'])
    else:
        z0 = np.concatenate([base[f1], np.clip(pg * np.asarray(xS)[f2] + (1 - pg) * np.asarray(xF)[f2], lo[f2], hi[f2]), np.clip(pb * np.asarray(xS)[f2] + (1 - pb) * np.asarray(xF)[f2], lo[f2], hi[f2])])
    zl = np.concatenate([lo[f1], lo[f2], lo[f2]]); zu = np.concatenate([hi[f1], hi[f2], hi[f2]]); z0 = np.clip(z0, zl, zu)
    t0 = time.time(); ref = [None]; last = {'z': z0.copy(), 'n': 0}

    def vec(z, which):
        x = base.copy(); x[f1] = z[:n1]; x[f2] = z[n1:n1 + n2] if which == 'G' else z[n1 + n2:]; return x

    def evaluate(z):
        r = {}
        for br, w, pr_ in (('G', q, pg), ('B', 1 - q, pb)):
            x = vec(z, br); r[br] = (w, pr_, S.solve(x), F.solve(x))
        return r

    def fg(z):
        if time.time() - p7_run.START > p7_run.BUDGET: raise TimeoutError
        r = evaluate(z); EU = 0.0; grad = np.zeros(len(z)); ok = True
        for br, (w, pr_, (uS, gS, stS, _), (uF, gF, stF, _)) in r.items():
            ok &= stS in (0, 1) and stF in (0, 1); EU += w * (pr_ * uS + (1 - pr_) * uF); g = w * (pr_ * gS + (1 - pr_) * gF)
            grad[:n1] += g[f1]
            if br == 'G': grad[n1:n1 + n2] += g[f2]
            else: grad[n1 + n2:] += g[f2]
        if not ok or not np.all(np.isfinite(grad)): return 1e3, np.zeros(len(z))
        if ref[0] is None: ref[0] = EU
        last['z'] = z.copy(); last['n'] += 1
        return -(EU - ref[0]) / lam, -grad / lam
    try:
        for rnd in range(5):
            res = minimize(fg, z0, jac=True, method='L-BFGS-B', bounds=list(zip(zl, zu)), options={'maxiter': 80, 'ftol': 1e-15, 'gtol': 1e-6, 'maxcor': 30})
            z0 = res.x; gz = -res.jac
            pgz = np.where((z0 <= zl + 1e-7) & (gz < 0), 0, np.where((z0 >= zu - 1e-7) & (gz > 0), 0, gz))
            if np.max(np.abs(pgz)) < 1e-2: break
    except TimeoutError:
        s_ = json.load(open(STATE)) if os.path.exists(STATE) else {}; s_[tag] = {'z': last['z'].tolist()}; json.dump(s_, open(STATE, 'w'))
        print(f"RESUME {tag}: time budget after {last['n']} evaluations", flush=True); sys.exit(3)
    z = res.x; r = evaluate(z); EU = 0.0; scc = 0.0; sts = []; gtot = np.zeros(len(z))
    for br, (w, pr_, (uS, gS, stS, oS), (uF, gF, stF, oF)) in r.items():
        EU += w * (pr_ * uS + (1 - pr_) * uF); sts.append(f'{stS}{stF}')
        scc += w * (pr_ * S.pr.summary(*oS)['scc2025'] + (1 - pr_) * F.pr.summary(*oF)['scc2025'])
        g = w * (pr_ * gS + (1 - pr_) * gF) / lam; gtot[:n1] += g[f1]
        if br == 'G': gtot[n1:n1 + n2] += g[f2]
        else: gtot[n1 + n2:] += g[f2]
    pg_ = np.where((z <= zl + 1e-7) & (gtot < 0), 0, np.where((z >= zu - 1e-7) & (gtot > 0), 0, gtot))
    # benchmark: expected-capacity planner who updates after the signal
    plan = {}
    for br, pr_ in (('G', pg), ('B', pb)):
        xb = fixed_start(sc, mk, capcal(pr_, {}), xE, k1)
        a = fixed(sc, mk, calS, 'NGM', xb, k2); b = fixed(sc, mk, calF, '', xb, k2); plan[br] = (a['U'], b['U'])
    EUplan = q * (pg * plan['G'][0] + (1 - pg) * plan['G'][1]) + (1 - q) * (pb * plan['B'][0] + (1 - pb) * plan['B'][1])
    base70 = load('p7_runs.json').get(item.split('@')[0] + '@1@2070', {})
    o = {'lam': lam, 'EU_tree': EU, 'EU_updating': EUplan, 'regret_updating': (EU - EUplan) / lam, 'scc_tree': scc, 'sccE': sE['scc2025'], 'premium_pct': 100 * (scc - sE['scc2025']) / sE['scc2025'],
         'status': sts, 'pg_norm': float(np.max(np.abs(pg_)))}
    if base70:
        o['EU_2070only'] = base70['stochastic']['EU']; o['value_of_signal'] = (EU - o['EU_2070only']) / lam
        o['regret_noupdate'] = (EU - base70['plans']['expected']['EU']) / lam; o['regret_2070only_expected'] = base70['regret_expected']
    save(TOUT, item, o)
    print(item, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()}, flush=True)


def fixed_start(sc, mk, cal, xE, k1):
    """Deterministic solve at capacity cal with the first k1 periods fixed to the plan xE; returns the full solution."""
    pr = R3.Problem(sc, cal=cal, allow='NGM', session=mk()); pr.fixed = (np.asarray(xE), k1)
    x, info, r, d, _ = pr.solve(x0=np.asarray(xE)); return x


if __name__ == '__main__':
    for it in sys.argv[1:]: tree(it)
    print('done')
