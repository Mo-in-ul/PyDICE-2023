"""Welfare by removal-access date (paper 2, v0.6). Results: results/p3_welfare.json (resumable).

For each calibration and first access year: welfare gain over no removal (trillion 2025 dollars, exact, from the objective),
and an accounting of the change in present-value damages before and after the new temperature peak, abatement cost and
removal cost against the no-removal run. Flows are discounted with the no-removal run's consumption discount factors
(dV/dC_t / dV/dC_2025), a common measuring rod; the accounting is first-order and is not an exact decomposition of welfare.
Peak warming is reported over the full horizon and to 2200."""
import os, json
import numpy as np
import removal3 as R3

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'results', 'p3_welfare.json')
SC = {'pub': 9, 'd3': 3, 'd2': 2}
YEARS = (2030, 2050, 2070, 2080, 2090, 2100, 2110, 2130, 2150)
res = json.load(open(OUT)) if os.path.exists(OUT) else {}


def solve(sc, cal, allow):
    pr = R3.Problem(sc, cal=cal, allow=allow); x, info, r, d, dt = pr.solve()
    s = pr.summary(x, info, r, d)
    g = lambda k: np.asarray(r[k])[1:]
    YG, T = g('YGROSS'), g('TATM')
    DAM = float(pr.q['a1']) * T + float(pr.qd['a2base']) * T ** float(pr.qd['a3'])
    return {'U': s['U'], 'status': s['status'], 'scc2025': s['scc2025'], 'T': T.tolist(), 'damages': (YG * DAM).tolist(),
            'abate': g('ABATECOST').tolist(), 'remcost': g('REMCOST').tolist(), 'dVdC': np.asarray(d['dVdC'])[1:].tolist(),
            'lamC2025': s['lamC2025']}


for c, sc in SC.items():
    if f'{c}_none' not in res:
        res[f'{c}_none'] = solve(sc, {}, ''); json.dump(res, open(OUT, 'w'))
    for y in YEARS:
        key = f'{c}_{y}'
        if key in res:
            continue
        res[key] = solve(sc, dict(startN=float(y), startG=float(y), startM=float(y)), 'NGM'); json.dump(res, open(OUT, 'w'))
        print(key, res[key]['status'], round(res[key]['scc2025'], 2), flush=True)
# summary
YR = 2020 + 5 * np.arange(len(res['pub_none']['T']))
summ = {}
for c in SC:
    b = res[f'{c}_none']; dv = np.array(b['dVdC']); w = dv / dv[1] * 5        # 5-year flows to 2025 dollars
    lam = b['lamC2025'] / 5; T0 = np.array(b['T'])
    for y in YEARS:
        r = res[f'{c}_{y}']; T1 = np.array(r['T']); ip = int(np.argmax(T1)); pre = np.arange(len(T1)) <= ip
        dD = (np.array(r['damages']) - np.array(b['damages'])) * w
        summ[f'{c}_{y}'] = {'gain_T': (r['U'] - b['U']) / lam, 'dpeak': float(T1.max() - T0.max()),
                            'dpeak_2200': float(T1[YR <= 2200].max() - T0[YR <= 2200].max()),
                            'dDam_pre_T': float(dD[pre].sum()), 'dDam_post_T': float(dD[~pre].sum()),
                            'dAbate_T': float(((np.array(r['abate']) - np.array(b['abate'])) * w).sum()),
                            'remcost_T': float(((np.array(r['remcost']) - np.array(b['remcost'])) * w).sum()),
                            'peak_year': int(YR[ip]), 'status': r['status']}
        print(c, y, {k: round(v, 3) if isinstance(v, float) else v for k, v in summ[f'{c}_{y}'].items()})
res['summary'] = summ; json.dump(res, open(OUT, 'w'))
print('done')
