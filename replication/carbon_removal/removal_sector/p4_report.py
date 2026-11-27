"""Collect the p4 results into tables: results/p4_tables.json (for the sheet) and results/p4_tables.md (readable).
Reads results/p4_*.json and, for comparison, p3_mech.json, p3_stoch.json, p3_reduced.json, p3_runs.json. No model runs."""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); R = lambda n: json.load(open(os.path.join(HERE, 'results', n)))
M = R('p4_mech.json'); YR = 2020 + 5 * np.arange(81)
T = {}                                                   # table name -> (header, rows)
CAL = {'pub': 'Published', 'd1': '1%', 'd2': '2%', 'd3': '3%', 'd4': '4%', 'd5': '5%'}
VARN = {'A': 'Hard caps', 'D': 'No storage limit', 'H': 'No storage limit, no temperature bound'}
r2 = lambda v, n=2: None if v is None else round(float(v), n)


def gap(c, p=0.5):
    return p * c['1.0']['scc2025'] + (1 - p) * c['0.0']['scc2025'] - c[str(p)]['scc2025']


def floor_year(r):
    f = [int(y) for y, t in zip(YR, r['T']) if t <= 0.5005]
    return f[0] if f else None


# 1. Curves on eighths
rows = []
for key in ('curve:A_pub', 'curve:D_pub', 'curve:A_d3', 'curve:D_d3', 'curve:A_d2', 'curve:D_d2', 'curve:H_d2'):
    c = M[key]; v, cal = key.split(':')[1].split('_'); prev = None
    for s in sorted(c, key=float):
        r = c[s]
        rows.append([CAL[cal], VARN[v], float(s), r2(r['scc2025'], 3), None if prev is None else r2(r['scc2025'] - prev, 3), r2(r['cumR_2200'], 0),
                     r2(c['0.0']['cumABT_2100'] - r['cumABT_2100'], 0), r2(r['T'][36], 3), r2(min(r['T']), 3), floor_year(r), r['first_at_max'], r['last_on_ramp'], r['status']])
        prev = r['scc2025']
T['A2_curves'] = (['Calibration', 'Storage', 'Capacity (share)', 'SCC 2025 ($/t)', 'Change per step ($/t)', 'Removal to 2200 (Gt)', 'Abatement given up to 2100 (Gt)',
                   'T in 2200 (C)', 'Lowest T (C)', 'First year at 0.5 C bound', 'Control rate first at one', 'Last year on near-term ceiling', 'IPOPT status'], rows)

# 2. Free against fixed abatement
rows = []
for cal in ('pub', 'd3', 'd2'):
    for v in ('A', 'D') + (('H',) if cal == 'd2' else ()):
        cf, cx = M[f'curve:{v}_{cal}'], M.get(f'fixmu:{v}_{cal}')
        if cx is None:
            continue
        for p in (0.25, 0.5, 0.75):
            gf, gx = gap(cf, p), gap(cx, p)
            rows.append([CAL[cal], VARN[v], p, r2(gf, 3), r2(100 * gf / cf[str(p)]['scc2025']), r2(gx, 3), r2(100 * gx / cx[str(p)]['scc2025']),
                         r2(cf['1.0']['cumR_2200'], 0), r2(cx['1.0']['cumR_2200'], 0), floor_year(cf['1.0']), floor_year(cx['1.0'])])
T['A1_fixed_abatement'] = (['Calibration', 'Storage', 'p', 'Jensen gap, abatement re-optimised ($/t)', 'same (%)', 'Jensen gap, control rate held ($/t)', 'same (%)',
                            'Removal to 2200 at full capacity, re-optimised (Gt)', 'same, held (Gt)', 'First year at or below 0.5 C at full capacity, re-optimised', 'same, held'], rows)

# 3. Variants: ceiling, abatement cost, discount rates
rows = []
for key, lab in [(f'{k}:{v}_{c}', (k, v, c)) for k in ('curve', 'ceil') for c in ('pub', 'd2') for v in ('A', 'D')] + \
                [(f'abc:{v}_{c}:{m}', ('abc' + m, v, c)) for c in ('pub', 'd2') for v in ('A', 'D') for m in ('0.5', '2.0')] + \
                [(f'disc:{v}_{c}', ('curve', v, c)) for c in ('d1', 'd4', 'd5') for v in ('A', 'D')] + [(f'curve:{v}_d3', ('curve', v, 'd3')) for v in ('A', 'D')]:
    if key not in M:
        continue
    c = M[key]; k, v, cal = lab
    name = {'curve': 'Central', 'ceil': 'DICE ceiling on control rate (1.10 from 2120)', 'abc0.5': 'Abatement cost x0.5', 'abc2.0': 'Abatement cost x2'}[k]
    g = gap(c)
    two = M.get(f"two:{v}_{cal}:{'ceil' if k == 'ceil' else 'abc:' + k[3:]}", {}).get('p0.5') if k != 'curve' else None
    rows.append([CAL[cal], VARN[v], name, r2(c['0.0']['scc2025'], 2), r2(c['0.5']['scc2025'], 2), r2(c['1.0']['scc2025'], 2), r2(g, 3), r2(100 * g / c['0.5']['scc2025']),
                 None if two is None else r2(100 * two['premium'] / two['sccE']), floor_year(c['0.5']), floor_year(c['1.0']), c['0.0']['first_at_max'], c['1.0']['first_at_max']])
T['A3_C3_B8_variants'] = (['Calibration', 'Storage', 'Variant', 'SCC, no removal', 'SCC, half capacity', 'SCC, full capacity', 'Jensen gap p=0.5 ($/t)', 'Jensen gap (%)',
                           'Two-state premium (%)', 'First year at 0.5 C bound, half capacity', 'same, full capacity', 'Control rate at max from, no removal', 'same, full capacity'], rows)

# 4. Other lotteries
if os.path.exists(os.path.join(HERE, 'results', 'p4_nstate.json')):
    NSj = R('p4_nstate.json'); rows = []
    LN = {'check': 'two states {0, 1}', 'three': 'three states {0, 0.5, 1}', 'narrow': 'two states {0.25, 0.75}', 'skew': 'three states {0, 0.5, 1}', 'wide4': 'four states {0, 0.25, 0.75, 1}'}
    for key, o in NSj.items():
        lot, vc = key.split(':'); v, cal = vc.split('_'); sc, pr = np.array(o['scales']), np.array(o['probs'])
        var = float(pr @ (sc - o['mean']) ** 2)
        rows.append([CAL[cal], VARN[v], LN[lot], ', '.join(f'{x:g}' for x in pr), r2(o['mean'], 3), r2(var, 4), r2(o['scc_ns'], 3), r2(o['sccE'], 3), r2(o['premium'], 3),
                     r2(100 * o['premium'] / o['sccE']), r2(o['jensen'], 3), r2(o['premium'] - o['jensen'], 4), o['status']])
    T['B1_B2_lotteries'] = (['Calibration', 'Storage', 'Capacity lottery (share of central)', 'Probabilities', 'Mean', 'Variance', 'SCC under uncertainty', 'SCC at mean capacity',
                             'Premium ($/t)', 'Premium (%)', 'Jensen gap from deterministic curve ($/t)', 'Difference ($/t)', 'IPOPT status'], rows)

# 5. Local curvature formula against the exact gap (deterministic curves only)
rows = []
for key in ('curve:A_pub', 'curve:A_d2', 'curve:D_d2'):
    c = M[key]; v, cal = key.split(':')[1].split('_'); f = lambda s: c[str(s)]['scc2025']
    for lo, hi in ((0.375, 0.625), (0.25, 0.75), (0.0, 1.0), (0.0, 0.5), (0.5, 1.0)):
        m = (lo + hi) / 2; h = 0.125; curv = (f(m + h) - 2 * f(m) + f(m - h)) / h ** 2; var = ((hi - lo) / 2) ** 2
        exact = 0.5 * (f(lo) + f(hi)) - f(m)
        rows.append([CAL[cal], VARN[v], f'{lo:g} or {hi:g}', m, r2(var, 4), r2(curv, 2), r2(0.5 * curv * var, 3), r2(exact, 3), r2(100 * exact / f(m))])
T['B1_local_formula'] = (['Calibration', 'Storage', 'Capacity is (equal chances)', 'Mean', 'Variance', "SCC'' at mean (h = 1/8)", 'Half curvature x variance ($/t)', 'Exact Jensen gap ($/t)', 'Exact gap (%)'], rows)

# 6. Engineered-only failure
if os.path.exists(os.path.join(HERE, 'results', 'p4_partial.json')):
    P = R('p4_partial.json'); m3 = R('p3_mech.json'); rows = []
    CN = {'E70': 'All removal from 2070; biological survives failure', 'E25': 'Central schedule; biological certain from 2025'}
    for cal in ('pub', 'd3', 'd2'):
        for p in ('0.25', '0.5', '0.75'):
            a = m3[f'A_{cal}']; s = a['stoch'][p]
            rows.append([CAL[cal], 'All removal from 2070; nothing survives failure (central case)', float(p), r2(s['scc2025'], 3), r2(s['ce'], 3), r2(s['scc2025'] - s['ce'], 3),
                         r2(100 * (s['scc2025'] - s['ce']) / s['ce']), None, None, None, None, None, s['status']])
    for key, o in P.items():
        case, cal = key.split('_')
        for p in ('0.25', '0.5', '0.75'):
            if p not in o:
                continue
            q = o[p]; ok = case == 'E70' or cal == 'pub'
            rows.append([CAL[cal], CN[case], float(p), r2(q['scc2s'], 3), r2(q['sccE'], 3), r2(q['premium'], 3), r2(100 * q['premium'] / q['sccE']), r2(q['voh_opt'], 4) if ok else None,
                         r2(q['voh_pes'], 4), r2(q['weight_opt'], 3), r2(q['value_S'], 1), r2(q['value_F'], 1), q['status']])
    T['B3_engineered_failure'] = (['Calibration', 'Case', 'p', 'SCC under uncertainty', 'SCC at expected capacity', 'Premium ($/t)', 'Premium (%)', 'Value of hedging vs optimist (T$)',
                                   'vs pessimist (T$)', 'Weight on optimist plan', 'Value of removal if it succeeds (T$)', 'if it fails (T$)', 'IPOPT status'], rows)

# 7. Learning dates and uncertain arrival
if os.path.exists(os.path.join(HERE, 'results', 'p4_learn.json')):
    L = R('p4_learn.json'); red = R('p3_reduced.json'); runs = R('p3_runs.json'); rows = []
    for cal in ('pub', 'd2'):
        lam = runs[f'{cal}_none']['lamC2025'] / 5
        for p in ('0.5', '0.25', '0.75'):
            k70 = f'U1_{cal}_2070|{float(p)}|segment'
            if k70 not in red:
                continue
            eu70 = red[k70]['EU']
            for t in (2030, 2040, 2050, 2060):
                o = L.get(f'learn:{cal}:{t}', {}).get(p)
                if o:
                    rows.append([CAL[cal], float(p), t, r2((o['EU'] - eu70) / lam, 4), r2(o['voh_opt'], 4), r2(o['voh_pes'], 4), r2(o['loss_2s_vs_pi'], 4), r2(o['scc2s'], 3), r2(o['weight_opt'], 3), r2(o['peakF'], 3), o['status']])
            rows.append([CAL[cal], float(p), 2070, 0.0, None, None, None, r2(red[k70]['scc2025'], 3), None, r2(red[k70]['peakF'], 3), f"reduced space, {red[k70]['status_S']}/{red[k70]['status_F']}"])
    T['B5_learning_date'] = (['Calibration', 'p', 'Outcome learned in', 'Value of learning then rather than in 2070 (T$)', 'Value of hedging vs optimist (T$)', 'vs pessimist (T$)',
                              'Loss against perfect information (T$)', 'SCC 2025', 'Weight on optimist', 'Peak if removal fails (C)', 'Solver status'], rows)
    rows = []
    for cal in ('pub', 'd2'):
        for p in ('0.25', '0.5', '0.75'):
            o = L.get(f'timing:{cal}', {}).get(p)
            if o:
                lin = float(p) * o['sccS'] + (1 - float(p)) * o['sccF']
                rows.append([CAL[cal], float(p), r2(o['scc2s'], 3), r2(o['sccS'], 3), r2(o['sccF'], 3), r2(lin, 3), r2(o['scc2s'] - lin, 3), r2(o['scc2090'], 3) if float(p) == 0.5 else None,
                             r2(100 * (o['scc2s'] - o['scc2090']) / o['scc2090']) if float(p) == 0.5 else None, r2(o['voh_opt'], 4), r2(o['voh_pes'], 4), r2(o['weight_opt'], 3),
                             r2(o['peakS'], 3), r2(o['peakF'], 3), r2(o['peak_none'], 3), r2(o['peakF_opt'], 3), o['status']])
    T['B7_arrival_date'] = (['Calibration', 'p (arrival 2070)', 'SCC under uncertainty', 'SCC, known 2070', 'SCC, known 2110', 'Probability-weighted SCC', 'Difference ($/t)',
                             'SCC, known 2090', 'Premium over known 2090 (%)', 'Value of hedging vs optimist (T$)', 'vs pessimist (T$)', 'Weight on optimist', 'Peak if 2070 (C)',
                             'Peak if 2110 (C)', 'Peak, no removal (C)', 'Peak if 2110 under optimist plan (C)', 'IPOPT status'], rows)

# 8. Carbon cycle and elasticity
pk = lambda r: max(r['T'][:37])                          # peak warming to 2200
rows = []
runs = R('p3_runs.json'); an = R('p3_anticip.json') if os.path.exists(os.path.join(HERE, 'results', 'p3_anticip.json')) else {}
if os.path.exists(os.path.join(HERE, 'results', 'p4_carbon.json')):
    Cc = R('p4_carbon.json')
    for cal, o in Cc.items():
        if 'a2110' not in o:
            continue
        ref = M[f'curve:A_{cal}']; refD = M[f'curve:D_{cal}']
        g = 0.5 * (o['a2070']['scc2025'] + o['none']['scc2025']) - o['a2070_half']['scc2025']
        gD = 0.5 * (o['D_a2070']['scc2025'] + o['none']['scc2025']) - o['D_a2070_half']['scc2025'] if 'D_a2070_half' in o else None
        rows.append([CAL[cal], 'FaIR v2 (AR6) feedback parameters', r2(o['none']['scc2025'], 2), r2(pk(o['none']), 3), r2(pk(o['a2070']) - pk(o['none']), 3),
                     r2(pk(o['a2090']) - pk(o['none']), 3), r2(pk(o['a2110']) - pk(o['none']), 3), r2(100 * g / o['a2070_half']['scc2025']),
                     None if gD is None else r2(100 * gD / o['D_a2070_half']['scc2025']), max(o[k]['status'] for k in o)])
        rows.append([CAL[cal], 'DICE-2023 (central)', r2(ref['0.0']['scc2025'], 2), r2(pk(ref['0.0']), 3), r2(pk(ref['1.0']) - pk(ref['0.0']), 3), None, None,
                     r2(100 * gap(ref) / ref['0.5']['scc2025']), r2(100 * gap(refD) / refD['0.5']['scc2025']), 0])
if os.path.exists(os.path.join(HERE, 'results', 'p4_eta.json')):
    Ee = R('p4_eta.json')
    for eta, o in Ee.items():
        for lab, nm in (('matched', f'Elasticity {eta}, discount factors re-matched (kappa {o.get("kappa", 0):.3f})'), ('asis', f'Elasticity {eta}, discount factors as published')):
            q = o.get(lab, {})
            if 'D_1.0' not in q:
                continue
            g = 0.5 * (q['a2070']['scc2025'] + q['none']['scc2025']) - q['A_0.5']['scc2025']; gD = 0.5 * (q['D_1.0']['scc2025'] + q['none']['scc2025']) - q['D_0.5']['scc2025']
            rows.append(['Published', nm, r2(q['none']['scc2025'], 2), r2(pk(q['none']), 3), r2(pk(q['a2070']) - pk(q['none']), 3), None,
                         r2(pk(q['a2110']) - pk(q['none']), 3), r2(100 * g / q['A_0.5']['scc2025']), r2(100 * gD / q['D_0.5']['scc2025']), max(q[k]['status'] for k in q)])
T['C1_C2_carbon_elasticity'] = (['Calibration', 'Variant', 'SCC 2025, no removal', 'Peak to 2200, no removal (C)', 'Change in peak, removal from 2070 (C)', 'from 2090 (C)', 'from 2110 (C)',
                                 'Jensen gap p=0.5, hard caps (%)', 'no storage limit (%)', 'Worst IPOPT status'], rows)

json.dump(T, open(os.path.join(HERE, 'results', 'p4_tables.json'), 'w'), indent=1)
with open(os.path.join(HERE, 'results', 'p4_tables.md'), 'w') as f:
    for name, (hd, rows) in T.items():
        f.write(f'## {name}\n\n| ' + ' | '.join(hd) + ' |\n|' + '---|' * len(hd) + '\n')
        for r in rows:
            f.write('| ' + ' | '.join('' if v is None else str(v) for v in r) + ' |\n')
        f.write('\n')
print({k: len(v[1]) for k, v in T.items()})
