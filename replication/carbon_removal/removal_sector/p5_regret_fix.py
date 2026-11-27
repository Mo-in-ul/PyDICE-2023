"""Stochastic optimum for the no-ceiling regret cases where the joint solve stopped below a feasible plan
(2%, learned 2050, no near-term ceiling). Solved in reduced space (p3_reduced.run) with a case U1nc added to the case
table: removal from 2070 or never, near-term ceiling off in both states. Updates results/p5_regret.json in place
(the joint-solve values are kept under 'stochastic_joint'). Usage: python p5_regret_fix.py d2 0.25 0.5 0.75"""
import sys
import p3_stoch, p3_reduced
from p4_common import *

p3_stoch.CASES['U1nc'] = (dict(A70, ramp_set=False), 'NGM', dict(ramp_set=False), '')
p3_reduced.CASES['U1nc'] = p3_stoch.CASES['U1nc']

if __name__ == '__main__':
    c = sys.argv[1]; item = f'{c}:2050:noceil'
    for p in [float(a) for a in sys.argv[2:]]:
        o = p3_reduced.run(f'U1nc_{c}_2050', p, 'segment')
        cur = load('p5_regret.json').get(item, {})
        r = cur.get(str(p))
        if r is None:
            print('no regret record for', p); continue
        lam = r['lam']
        r['stochastic_joint'] = r['stochastic']
        r['stochastic'] = {'EU': o['EU'], 'abt': o['abt_to_res'], 'peakF': o['peakF'], 'scc': o['scc2025'],
                           'status': f"reduced {o['status_S']}/{o['status_F']}", 'proj_grad': o['proj_grad_max_T']}
        for lab in ('expected', 'optimist', 'pessimist'):
            r[lab]['regret'] = (o['EU'] - r[lab]['EU']) / lam
        r['perfect']['gain'] = (r['perfect']['EU'] - o['EU']) / lam
        cur[str(p)] = r; save('p5_regret.json', item, cur)
        print(f"{item} p={p} reduced: EU {o['EU']:.6f} (joint {r['stochastic_joint']['EU']:.6f}) regret expected {r['expected']['regret']:+.4f} "
              f"optimist {r['optimist']['regret']:+.4f} pessimist {r['pessimist']['regret']:+.4f} abt {o['abt_to_res']:.1f} exp {r['expected']['abt']:.1f} |pg| {o['proj_grad_max_T']:.1e}", flush=True)
    print('done')
