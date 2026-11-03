"""Hold the near-term consumption discount rate R(h) fixed at its phi = 0.95 value by adjusting rho_e.
Usage: python s2_rho.py h   (h = 50 | 80 | 100). GAMS 0.5 C bound, sigma = 1%, cap 235 y, N = 81."""
import sys, numpy as np, s2_core as S
h = int(sys.argv[1])
name = f's2_rho_h{h}.json'
out = S.load(name) or []
pr = S.problem(81, 0.95); o0, x0 = S.solve(pr, pr.start_from_gams())
RHO0 = float(pr.p['rartp'])
target = S.R_h(o0['lamC'], h)
print('target R', h, target, 'rho_e0', RHO0, flush=True)

def F(eta, rho_e, x):
    p = S.problem(81, eta, rho_e=rho_e); o, xn = S.solve(p, x)
    return S.R_h(o['lamC'], h) - target, o, xn

done = {o['eta'] for o in out}
for direction in (np.round(np.arange(0.95, 1.7501, 0.05), 2), np.round(np.arange(0.95, 0.4999, -0.05), 2)):
    x, r0 = x0, RHO0
    prev = []           # (eta, rho) for extrapolating the starting guess
    for e in direction:
        e = float(e)
        guess = r0 if len(prev) < 2 else prev[-1][1] + (prev[-1][1] - prev[-2][1])
        a, b_ = guess, guess - 0.002
        fa, oa, xa = F(e, a, x); fb, ob, xb = F(e, b_, x)
        n = 2
        while abs(fb) > 1e-7 and n < 12:
            c = b_ - fb * (b_ - a) / (fb - fa)
            a, fa, oa, xa = b_, fb, ob, xb
            b_ = c; fb, ob, xb = F(e, b_, xa); n += 1
        prev.append((e, b_)); r0 = b_; x = xb
        if e in done:
            continue
        ob.update({'eta': e, 'h': h, 'rho_e': b_, 'resid': fb, 'solves': n, 'target': target,
                   'rho_implied_beta_fixed': float(np.log(1 + b_) - 0.5 * 0.05),
                   'beta_implied_rho_fixed': float((np.log(1 + b_) - 0.001) / 0.05)})
        ob['R50'], ob['R80'], ob['R100'], ob['R200'] = [S.R_h(ob['lamC'], k) for k in (50, 80, 100, 200)]
        out.append(ob); done.add(e); S.save(out, name)
        print(h, e, ob['status'], round(b_, 5), '%.1e' % fb, n, round(ob['scc_2025'], 3), round(ob['peakT'], 3),
              round(ob['rho_implied_beta_fixed'], 5), round(ob['beta_implied_rho_fixed'], 3), flush=True)
