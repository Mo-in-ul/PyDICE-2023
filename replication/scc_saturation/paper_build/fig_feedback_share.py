"""Figure: saturation-feedback share of the SCC by pulse year (from check3_ift.json, exact
IFT values), for the five clean scenarios, with the first year of net-negative emissions
marked. Writes figs/fig_feedback_share.pdf and results/feedback_share_summary.json."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import dm, load_json, save_json, HERE
from house_style import BLUE, ORANGE, DOUBLE, MM

NAMES = {4: '4% discounting', 5: '5% discounting', 8: 'Paris', 9: 'Cost-benefit optimal', 10: 'Baseline'}
ift = load_json('check3_ift.json')
fig, ax = plt.subplots(figsize=(DOUBLE, 65 * MM))
summ = {}
for sc, col in ((9, BLUE), (8, ORANGE), (10, '0.3'), (4, '0.5'), (5, '0.7')):
    o = ift[str(sc)]
    yrs = np.array(o['years_all']); sh = 100 * np.array(o['share_all'])
    k = (yrs >= 2025) & (yrs <= 2200)
    ax.plot(yrs[k], sh[k], color=col, lw=1.4 if sc == 9 else 1.0, ls='--' if sc in (4, 5) else '-', label=NAMES[sc])
    E = dm.load_gams(sc)['ECO2'][:81]
    neg = yrs[E < 0]
    fy = int(neg[0]) if len(neg) else None
    cross = yrs[k][np.where(sh[k] < 0)[0][0]] if np.any(sh[k] < 0) else None
    summ[sc] = {'share_2025': float(sh[1]), 'share_2050': float(sh[6]), 'share_2100': float(sh[16]),
                'first_net_negative_year': fy, 'first_negative_share_year': int(cross) if cross is not None else None}
    if fy and fy <= 2200:
        ax.axvline(fy, color=col, lw=0.8, ls=':')
ax.axhline(0, color='0.5', lw=0.7)
ax.set_ylabel('Feedback share of SCC (%)'); ax.set_xlabel('Year of emissions pulse')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'figs', 'fig_feedback_share.pdf')); plt.close(fig)
save_json(summ, 'feedback_share_summary.json')
print(summ)
