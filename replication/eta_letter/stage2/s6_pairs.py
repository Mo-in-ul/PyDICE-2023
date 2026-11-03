"""Published (rho, phi) pairs inserted into DICE-2023 (sigma = 1%, cap 235 y, GAMS 0.5 C bound).
Each pair is run with the climate beta switched off (beta = 0, as in the source calibrations, which target
consumption / risk-free rates) and kept at DICE's 0.5; rho_e = exp(rho + beta*pi) - 1. For comparison each is
also run without the precautionary term (sigma = 0). Horizon check: N = 121 for the sigma = 1% runs."""
import numpy as np, s2_core as S
PAIRS = [('DICE-2023 default', 0.001, 0.95), ('Stern Review', 0.001, 1.0), ('Drupp et al. median', 0.005, 1.0),
         ('Drupp et al. mean', 0.011, 1.35), ('NPP 2% rate', 0.001, 1.25), ('NPP 2.5% rate', 0.004, 1.4),
         ('NPP 3% rate', 0.008, 1.53), ('DICE-2016R', 0.015, 1.45)]
PI = 0.05
out = S.load('s6_pairs.json') or []; done = {(o['label'], o['beta'], o['sigma'], o['N']) for o in out}
for N, sig in ((81, 0.01), (81, 0.0), (121, 0.01)):
    for beta in (0.0, 0.5):
        x = None
        for lab, rho, phi in PAIRS:
            re = float(np.exp(rho + beta * PI) - 1)
            pr = S.problem(N, phi, sigma=sig, rho_e=re)
            if x is None: x = pr.start_from_gams() if N == 81 else pr.start_default()
            if (lab, beta, sig, N) in done: continue
            o, xn = S.solve(pr, x)
            if o['status'] in (0, 1): x = xn
            kappa = phi ** 2 * sig ** 2 * 235 * (1 + re) / (2 * re)
            o.update({'label': lab, 'rho': rho, 'phi': phi, 'beta': beta, 'sigma': sig, 'rho_e': re, 'kappa': kappa,
                      'R80': S.R_h(o['lamC'], 80)}); o.pop('lamC')
            out.append(o); S.save(out, 's6_pairs.json')
            print('pair', N, sig, beta, lab, o['status'], round(o['scc_2025'], 2), 'kappa %.2f' % kappa, 'R80 %.4f' % o['R80'],
                  'peakT %.2f' % o['peakT'], 'floor' if o['Tlo_binds'] else '', flush=True)
