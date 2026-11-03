import json, os, numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
def _c(o):
    if isinstance(o, dict): return {k: _c(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_c(v) for v in o]
    if isinstance(o, np.ndarray): return _c(o.tolist())
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    return o
def save(obj, name): json.dump(_c(obj), open(os.path.join(R, name), 'w'), indent=1)
def load(name):
    p = os.path.join(R, name); return json.load(open(p)) if os.path.exists(p) else None
def row(N, eta, variant, tlo, start, d, n, dt):
    return {'N': N, 'eta': float(eta), 'variant': variant, 'tlo': tlo, 'start': start, 'status': d['status'],
            'iters': n, 'time_s': dt, 'U': d['U'], 'scc_2025': d['scc'][2], 'scc_2050': d['scc'][7],
            'scc_2100': d['scc'][17], 'peakT': d['peakT'], 'minT': d['minT'], 'argminT_year': d['argminT_year'],
            'Tlo_binding_years': d['Tlo_binding_years'], 'Tlo_mult_sum': d['Tlo_mult_sum'],
            'kkt': d['kkt_stationarity_inf'], 'irf_max': d['irf_max'], 'minMAT': d['minMAT'],
            'miu_at_lo_hi': d['miu_at_lo_hi'], 'alpha_at_lo_hi': d['alpha_at_lo_hi'],
            'cumE_2100': float(5 * np.sum(d['E'][1:17]))}
