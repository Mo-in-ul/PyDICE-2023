"""The 1.5 C pathway: what the temperature cap costs with and without capital disposal.

GAMS writes capital accumulation as K(t+1) =L= (1-dk)^5 K(t) + 5 I(t), so capital can be
scrapped. With equality accumulation no control path keeps temperature below 1.632 C (2050),
so the 1.5 C run is infeasible (Dice2023Model.check_temp_feasibility). Here the scenario is
solved with IPOPT for a range of caps, with equality accumulation and with a disposal control
D >= 0, to show where disposal starts to be used and what it does to output and welfare.
Warm starts: the 2 C GAMS solution for caps >= 1.65 and the 1.5 C GAMS solution below.
Writes results/onepointfive.json and figs/fig_1p5c_cap.pdf.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import dm, save_json, HERE
from ipopt_model import solve, Problem
from house_style import BLUE, ORANGE, DOUBLE, MM, panel_label   # after common: the model module resets matplotlib defaults

CAPS = [1.5, 1.55, 1.6, 1.632, 1.65, 1.7, 1.8, 1.9, 2.0, 2.2, 2.6]


def start(cap, disposal):
    sc = 6 if cap < 1.65 else 7
    pr = Problem(sc, disposal=True)
    x = pr.start_from_gams()
    N = pr.N
    return x if disposal else x[:3 * N]


def summarize(r, cap, disposal):
    tr = r['traj']; N = r['problem'].N; p = r['problem'].p
    T = tr['TATM'][1:]; K = tr['K'][1:]; C = tr['C'][1:]
    o = {'cap': cap, 'disposal': disposal, 'status': r['status'], 'U': r['U'], 'peak_T': float(T[:40].max()),
         'peak_year': int(2020 + 5 * np.argmax(T[:40])), 'K2025': float(K[1]),
         'cpc2025': float(1000 * C[1] / p['L'][2]), 'cpc2050': float(1000 * C[6] / p['L'][7]),
         'scc2025': float(r['scc'][2])}
    if disposal:
        D = r['x'][3 * N:]
        o['disposal_total'] = float(D.sum()); o['disposal_max'] = float(D.max())
        o['disposal_first_year'] = int(2020 + 5 * np.argmax(D > 1e-3)) if np.any(D > 1e-3) else None
    return o


def plot(out):
    ok = [o for o in out if 'error' not in o and o['status'] in (0, 1)]
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE, 62 * MM))
    for disp, col, lab in ((True, BLUE, 'Capital can be scrapped (GAMS)'), (False, ORANGE, 'Equality accumulation')):
        rs = sorted([o for o in ok if o['disposal'] == disp], key=lambda o: o['cap'])
        ax[0].plot([o['cap'] for o in rs], [o['U'] for o in rs], 'o-', color=col, label=lab, ms=2.5)
        ax[1].plot([o['cap'] for o in rs], [o['cpc2025'] for o in rs], 'o-', color=col, ms=2.5)
    for a in ax:
        a.axvline(1.632, color='0.6', ls=':', lw=0.8); a.set_xlabel(r'Temperature cap ($^\circ$C)')
    panel_label(ax[0], '(a)'); panel_label(ax[1], '(b)')
    ax[0].set_ylabel('Welfare (DICE utility units)'); ax[0].legend(frameon=False, loc='lower right')
    ax[1].set_ylabel('Consumption per person, 2025 (thousand \\$)')
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'figs', 'fig_1p5c_cap.pdf')); plt.close(fig)


if __name__ == '__main__':
    import sys
    if '--plot-only' in sys.argv:
        from common import load_json
        plot(load_json('onepointfive.json')['runs']); sys.exit()
    out = []
    g9 = dm.load_gams(9); opt_cpc2025 = float(g9['CPC'][1])
    for disposal in (True, False):
        for cap in CAPS:
            try:
                r = solve(6, disposal=disposal, tup=cap, x0=start(cap, disposal), warm_start=disposal,
                          tol=1e-10, max_iter=5000)
                o = summarize(r, cap, disposal)
            except Exception as e:
                o = {'cap': cap, 'disposal': disposal, 'error': repr(e)}
            out.append(o); print(o, flush=True)
            save_json({'runs': out, 'optimal_cpc2025': opt_cpc2025}, 'onepointfive.json')
    plot(out)
