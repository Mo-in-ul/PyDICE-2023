"""Reduced-space solution of the two-state problem (paper 2, v0.6).

The decisions before the resolution date (control rate, removal, savings in periods before t_res) are the only outer
variables. For given outer variables each state is solved as a deterministic problem with those periods held fixed and the
carbon-cycle variable free. By the envelope theorem the derivative of each state's optimal welfare with respect to the fixed
variables is the derivative of the Lagrangian, grad f + J' lambda, at the inner solution. The outer problem
max p W_S + (1-p) W_F is solved with L-BFGS-B over the box bounds. Usage: python p3_reduced.py KEY p [start] ...
KEY e.g. U1_pub_2070; start: segment (default), optimist, pessimist. Results: results/p3_reduced.json (resumable)."""
import os, sys, json, time
import numpy as np
from scipy.optimize import minimize
import removal3 as R3
from p3_stoch import SC, CASES, iy

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'results', 'p3_reduced.json')
RUNS = json.load(open(os.path.join(HERE, 'results', 'p3_runs.json')))
HL = json.load(open(os.path.join(HERE, 'results', 'p3_hedge_line.json')))
NCTRL = 5


class State:
    def __init__(self, sc, cal, allow, k):
        self.pr = R3.Problem(sc, cal=cal, allow=allow); self.pr.fixed_blocks = range(NCTRL); self.k = k
        self.x = None; self.xref = None

    def solve(self, xfix):
        pr = self.pr; pr.fixed = (xfix, self.k)
        x0 = xfix if self.xref is None else np.where(self.mask, xfix, self.xref)   # always start from the deterministic optimum
        x, info, r, d, dt = pr.solve(x0=x0)
        self.x = x; self.info = info
        g = pr.gradient(x) + pr.jacobian(x).reshape(-1, pr.nv).T @ np.asarray(info['mult_g'])
        return float(r['UTILITY']), -g, int(info['status']), (x, info, r, d)


def run(key, p, start):
    case, c, t = key.rsplit('_', 2); t = int(t)
    if case not in CASES:
        import p3_prec
        CASES.update(p3_prec.CASES)
    calS, allowS, calF, allowF = CASES[case]; sc = SC[c]; k = iy(t)
    lam = RUNS[f'{c}_none']['lamC2025'] / 5
    S = State(sc, calS, allowS, k); F = State(sc, calF, allowF, k)
    xS, *_ = S.pr.solve(); xF, *_ = F.pr.solve(); S.xref = np.array(xS); F.xref = np.array(xF)
    N = S.pr.N
    S.pr.fixed = None; lo, hi, _, _ = F.pr.bounds(); loS, hiS, _, _ = S.pr.bounds()
    lo = np.maximum(lo, loS); hi = np.minimum(hi, hiS)
    idx = np.concatenate([np.arange(b * N, b * N + k) for b in range(NCTRL)])
    free = idx[hi[idx] - lo[idx] > 1e-12]
    mask = np.zeros(S.pr.nv, bool); mask[idx] = True; S.mask = F.mask = mask
    if start in ('optimist', 'pessimist'):
        l = 1.0 if start == 'optimist' else 0.0
    else:
        l = HL[key]['p'][str(p)]['l'] if key in HL else 0.5
    base = np.array(xF, float)
    base[idx] = l * np.asarray(xS)[idx] + (1 - l) * np.asarray(xF)[idx]
    base[idx] = np.clip(base[idx], lo[idx], hi[idx])
    hist = []; last = []; t0 = time.time()

    def fg(z):
        xf = base.copy(); xf[free] = z
        uS, gS, stS, _ = S.solve(xf); uF, gF, stF, _ = F.solve(xf)
        EU = p * uS + (1 - p) * uF; grad = (p * gS + (1 - p) * gF)[free]
        if stS not in (0, 1) or stF not in (0, 1) or not np.all(np.isfinite(grad)):   # failed inner solve: reject the trial point
            print(f"  inner failure st {stS}{stF}; rejecting trial point", flush=True)
            return 1e3 + (hist[-1]['f'] if hist else 0.0), (last[0] if last else np.zeros(len(free)))
        hist.append({'EU': EU, 'stS': stS, 'stF': stF, 'gnorm': float(np.max(np.abs(grad / lam))), 't': time.time() - t0})
        print(f"  {key} p={p} {start} eval {len(hist)}: EU {EU:.6f} st {stS}{stF} |g| {hist[-1]['gnorm']:.2e} T$/unit {hist[-1]['t']:.0f}s", flush=True)
        hist[-1]['f'] = -(EU - EUref[0]) / lam; last[:] = [-grad / lam]
        return -(EU - EUref[0]) / lam, -grad / lam
    z0 = base[free]; z0 = np.where(hi[free] - z0 < 1e-6, hi[free], np.where(z0 - lo[free] < 1e-6, lo[free], z0)); base[free] = z0
    EUref = [0.0]
    EUref[0] = fg(z0)[0] * 0 + hist[-1]['EU']; hist[-1]['f'] = 0.0
    bnds = list(zip(lo[free], hi[free]))
    for rnd in range(4):                                   # restart L-BFGS-B until the projected gradient is small
        res = minimize(fg, z0, jac=True, method='L-BFGS-B', bounds=bnds, options={'maxiter': 80, 'ftol': 1e-15, 'gtol': 1e-6, 'maxcor': 30})
        z0 = res.x; gz = -res.jac
        pgz = np.where((z0 <= lo[free] + 1e-7) & (gz < 0), 0, np.where((z0 >= hi[free] - 1e-7) & (gz > 0), 0, gz))
        print(f"  round {rnd}: {res.message} |pg| {np.max(np.abs(pgz)):.2e}", flush=True)
        if np.max(np.abs(pgz)) < 1e-2:
            break
    xf = base.copy(); xf[free] = res.x
    uS, gS, stS, outS = S.solve(xf); uF, gF, stF, outF = F.solve(xf)
    sS = S.pr.summary(*outS); sF = F.pr.summary(*outF)
    gfin = (p * gS + (1 - p) * gF)[free] / lam
    # projected gradient (stationarity on the box)
    z = res.x; pg = np.where((z <= lo[free] + 1e-7) & (gfin < 0), 0, np.where((z >= hi[free] - 1e-7) & (gfin > 0), 0, gfin))
    EU = p * uS + (1 - p) * uF
    out = {'key': key, 'p': p, 'start': start, 'l_start': l, 'EU': EU, 'EU_start': hist[0]['EU'], 'gain_over_start_T': (EU - hist[0]['EU']) / lam,
           'proj_grad_max_T': float(np.max(np.abs(pg))), 'n_eval': len(hist), 'lbfgs_msg': str(res.message), 'status_S': stS, 'status_F': stF,
           'scc2025': p * sS['scc2025'] + (1 - p) * sF['scc2025'], 'scc2025_S': sS['scc2025'], 'scc2025_F': sF['scc2025'],
           'peakS': sS['peakT'], 'peakF': sF['peakT'], 'abt_to_res': float(5 * np.array(sS['paths']['ABT'])[:k].sum()),
           'US': uS, 'UF': uF, 'hist': hist, 'xcommon': xf[idx].tolist()}
    print(f"{key} p={p} {start}: EU {EU:.6f} start {hist[0]['EU']:.6f} gain {out['gain_over_start_T']:.5f} T$ |pg| {out['proj_grad_max_T']:.1e} "
          f"SCC {out['scc2025']:.2f} peakF {out['peakF']:.3f} abt {out['abt_to_res']:.1f} {res.message}", flush=True)
    return out


if __name__ == '__main__':
    args = sys.argv[1:]; key = args[0]; jobs = []
    rest = args[1:]
    for a in rest:
        p, _, st = a.partition(':'); jobs.append((float(p), st or 'segment'))
    for p, st in jobs:
        tag = f'{key}|{p}|{st}'
        res = json.load(open(OUT)) if os.path.exists(OUT) else {}
        if tag in res:
            continue
        o = run(key, p, st)
        res = json.load(open(OUT)) if os.path.exists(OUT) else {}
        res[tag] = o; json.dump(res, open(OUT, 'w'))
    print('done')
