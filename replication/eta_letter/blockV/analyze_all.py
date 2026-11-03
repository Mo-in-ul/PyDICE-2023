import json, numpy as np
from bv_io import load
h = load('v5_horizon.json'); H = {(o['variant'], o['eta'], o['N']): o for o in h}
print('HORIZON (SCC 2025; end year 2020+5(N-1))')
for v in ('dice', 'fixed'):
    for e in sorted({o['eta'] for o in h if o['variant'] == v}):
        row = [H.get((v, e, N)) for N in (61, 81, 101, 121, 141)]
        s = [r['scc_2025'] if r else np.nan for r in row]
        steps = [100 * (s[i + 1] / s[i] - 1) for i in range(4)]
        print(v, e, ' '.join(f'{x:8.2f}' for x in s), '| step %', ' '.join(f'{x:6.2f}' for x in steps),
              '| minT', ' '.join(f"{r['minT']:.3f}" if r else '-' for r in row), '| status', [r['status'] if r else None for r in row])
m = load('v3_multistart.json')
print('MULTISTART')
for e in sorted({o['eta'] for o in m}):
    rs = [o for o in m if o['eta'] == e]; U = np.array([o['U'] for o in rs]); S = np.array([o['scc_2025'] for o in rs])
    print(e, len(rs), 'status', {o['status'] for o in rs}, 'iters', [o['iters'] for o in rs], 'U relspread %.1e' % ((U.max() - U.min()) / abs(U).max()), 'SCC spread %% %.1e' % (100 * (S.max() - S.min()) / S.mean()), 'kkt max %.1e' % max(o['kkt'] for o in rs))
print('BRUTE FORCE')
for tl in ('none', '0.01', '0.5'):
    for r in load(f'v2_bf_tlo{tl}.json'):
        for y in ('2025', '2050'):
            q = r[y]
            print(tl, r['eta'], y, 'mult %.4f reopt %.4f fixed %.4f | gap reopt %.1e%% fixed %.2f%% | h-cons %.1e%% | minT %.3f | st_reopt %s st_fixed %s' % (
                q['multiplier'], q['reopt']['scc_rich'], q['fixed_policy']['scc_rich'], q['gap_reopt_pct'], q['gap_fixed_pct'], q['h_consistency_reopt_pct'], r['minT'],
                sorted(set(q['reopt']['parts']['E']['status'] + q['reopt']['parts']['C']['status'])), sorted(set(q['fixed_policy']['parts']['E']['status'] + q['fixed_policy']['parts']['C']['status']))))
