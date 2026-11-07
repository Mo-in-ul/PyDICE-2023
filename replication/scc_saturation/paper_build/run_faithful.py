"""Optimize one scenario with the GAMS session settings reproduced.

The published GAMS outputs come from one GAMS session that solves the scenarios in the order
opt, T2, T1.5, altdam, paris, base, DISC5, DISC4, DISC3, DISC2, DISC1 (Putlong-4-3-10.gms).
Bounds set in one scenario persist into later ones. Settings reproduced here:
  * TATM >= 0.5 in every scenario (FAIR-beta-4-3-1.gms, line 98); PyDICE default is 0.01
  * miu.lo = 0.01 from the Paris run on (def-paris line 24): affects base and DISC5..DISC1
  * cprice.up = 500 after the base run (def-base line 25): DISC5
  * cprice.up = 1000 from DISC4 on (def_DISC4 line 12): DISC4..DISC1
  * K(1) stays at 409 in DISC1 (def_DISC1 sets k0 = 420 but never re-fixes K(tfirst))
  * base: MIU(1) is not fixed in GAMS; the cprice <= cpricebase rule sets it to 0.0513
A carbon-price cap is a control bound: MIU <= (cap / PBACKTIME)^(1/(expcost2-1)).
The settings themselves live in common.gams_session, shared with ipopt_model.py.
Usage: python run_faithful.py <scenario> [flags]   flags: all (default) or comma list of
  tlo,miulo,cpcap,k0,miu1   -> results/faith_scen<N>[_flags].json/.npz
"""
import sys, time, os
import numpy as np
from scipy.optimize import minimize
from common import dm, save_json, RES, gams_session

sc = int(sys.argv[1])
flags = sys.argv[2] if len(sys.argv) > 2 else 'all'
F = {'tlo', 'miulo', 'cpcap', 'k0', 'miu1'} if flags == 'all' else set(f for f in flags.split(',') if f and f != 'none')
FTOLS = (1e-9,) if '--tight' in sys.argv else (1e-5, 1e-7, 1e-9)
warm = len(sys.argv) > 3 and sys.argv[3] == 'gams'
tag = ('_tight' if '--tight' in sys.argv else '') + ('' if flags == 'all' else '_' + (flags.replace(',', '-') or 'none')) + ('_warm' if warm else '')

# All settings come from common.gams_session, shared with the IPOPT runs (ipopt_model.py).
p, b = gams_session(sc, F)
N = p['num_periods']
Tup, Tlo = b['Tup'], b['Tlo']
miu_lo = float(b['miu_lo'][2]); cap = {5: 500.0, 4: 1000.0, 3: 1000.0, 2: 1000.0, 1: 1000.0}.get(sc) if 'cpcap' in F else None

prob = dm.DiceFunc(N, p, TempUpperConstraint=Tup, TempLowerConstraint=Tlo)
x0 = np.concatenate([prob.MIU[1:], prob.S[1:], prob.Alpha[1:]])
x0[:N] = np.clip(x0[:N], b['miu_lo'][1:], b['miu_up'][1:])
if warm:  # start from the GAMS solution (diagnostic: is it a KKT point of the PyDICE problem?)
    g = dm.load_gams(sc)
    x0 = np.concatenate([np.clip(g['MIU'][:N], b['miu_lo'][1:], b['miu_up'][1:]), g['S'][:N],
                         np.maximum(g['ALPHA'][:N], p['AlphaLowerBound'])])
# MIU(1) is fixed at p['miu1'] (the forward model forces it); gams_session sets it to the binding cap in base
mb = [(p['miu1'], p['miu1'])] + [(b['miu_lo'][t], b['miu_up'][t]) for t in range(2, N + 1)]
x0[0] = p['miu1']
sb = [(p['sLBounds'][i], p['sUBounds'][i]) for i in range(1, N + 1)]
ab = [(p['AlphaLowerBound'], p['AlphaUpperBound'])] * N
# base: the cprice <= cpricebase rule is in the MIU upper bounds (gams_session)
cons = [{'type': 'eq', 'fun': prob.irf_residual},
        {'type': 'ineq', 'fun': prob.temp_up},
        {'type': 'ineq', 'fun': prob.temp_lo}]

t0 = time.time()
attempts = []
x = x0
for ftol in FTOLS:
    res = minimize(prob.objective, x, method='SLSQP', bounds=mb + sb + ab, constraints=cons,
                   options={'maxiter': 25000, 'ftol': ftol, 'eps': 1e-6})
    attempts.append({'ftol': ftol, 'status': int(res.status), 'message': str(res.message),
                     'nit': int(res.nit), 'W': float(-res.fun)})
    x = res.x
    if res.success:
        break
elapsed = time.time() - t0
irf = float(np.max(np.abs(prob.irf_residual(x))))
output = dm.recoverAllVars(x, p)


class M:  # minimal stand-in for compare_with_gams
    pass
mm = M(); mm.params = p; mm.scenario = sc
meta = {'irf_max_residual': irf, 'infeasible': False}
summary, _ = dm.compare_with_gams(mm, output, meta, show=False, save=(flags == 'all' and not warm),
                                  results_dir=os.path.join(RES, 'compare_faithful'))
summary.update({'flags': sorted(F), 'attempts': attempts, 'success': bool(res.success),
                'final_status': int(res.status), 'elapsed_s': elapsed,
                'Tlo': Tlo, 'miu_lo': miu_lo, 'cprice_cap': cap, 'k0': p['k0'],
                'min_T': float(output[:, 3].min())})
np.savez(os.path.join(RES, f'faith_scen{sc}{tag}.npz'), x_opt=x, output=output)
save_json(summary, f'faith_scen{sc}{tag}.json')
print('DONE', sc, flags, 'success', res.success, 'gap', summary['welfare_gap_pct'], 'dT', summary['max_dT_to_2215'], elapsed)
