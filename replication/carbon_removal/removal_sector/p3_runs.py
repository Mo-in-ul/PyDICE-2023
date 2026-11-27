"""Paper 2, model v3 experiment set (after the referee review of v0.3). Results: results/p3_runs.json (resumable).
Central: three families (biological, geological injection, mineral), biological leakage 0.5%/yr."""
import os, sys, json
import removal3 as R3
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'results', 'p3_runs.json')
runs = json.load(open(OUT)) if os.path.exists(OUT) else {}
SC = {'pub': 9, 'd3': 3, 'd2': 2}
SOFT = dict(capN=None, capG=None, sN=200.0, sG=200.0, nu=4.0)
J = []
def add(name, sc, allow='NGM', **cal):
    J.append((name, sc, allow, cal))
for c, sc in SC.items():
    add(f'{c}_none', sc, '')
    for a in ('N', 'G', 'M', 'NGM'):
        add(f'{c}_{a}', sc, a)
    add(f'{c}_NGM_soft', sc, **SOFT)
    add(f'{c}_NG_v2', sc, 'NG', delta0=0.0, RG_inf=20.0)          # version 0.3 central (EW inside geological)
# storage: soft-limit grid (published)
for sv in (100.0, 200.0, 400.0):
    for nu in (2.0, 4.0, 8.0):
        if (sv, nu) != (200.0, 4.0):
            add(f'pub_NGM_soft_s{sv:g}_nu{nu:g}', 9, **dict(SOFT, sN=sv, sG=sv, nu=nu))
# leakage
for d0 in (0.0, 0.002, 0.01):
    add(f'pub_NGM_delta{d0:g}', 9, delta0=d0)
for gam in (0.5, 1.0):
    add(f'pub_NGM_gamma{gam:g}', 9, gamma=gam)
    add(f'd2_NGM_gamma{gam:g}', 2, gamma=gam)
# effectiveness by family
add('pub_NGM_eff_moderate', 9, effN=0.9, effG=0.9, effM=0.8)
add('pub_NGM_eff_severe', 9, effN=0.7, effG=0.8, effM=0.6)
for c in ('d3', 'd2'):
    add(f'{c}_NGM_eff_severe', SC[c], effN=0.7, effG=0.8, effM=0.6)
# mineral calibration
for k, v in [('RM_inf', 2.0), ('RM_inf', 10.0), ('aMinf', 50.0), ('aMinf', 150.0)]:
    add(f'pub_NGM_{k}={v:g}', 9, **{k: v})
# capacity investment (geological)
for c, sc in SC.items():
    add(f'{c}_NGM_inv', sc, invest=True)
    add(f'{c}_NGM_inv_learn', sc, invest=True, learn=1.0)
    add(f'{c}_NGM_inv_fdg', sc, invest=True, g=0.25)              # value of a faster growth limit
    add(f'{c}_NGM_inv_learn_fdg', sc, invest=True, learn=1.0, g=0.25)
for gv in (0.1, 0.3):
    add(f'pub_NGM_inv_g{gv:g}', 9, invest=True, g=gv)
for ph in (0.3, 0.7):
    add(f'pub_NGM_inv_phi{ph:g}', 9, invest=True, phi=ph)
add('pub_NGM_learn', 9, learn=1.0)
add('pub_NGM_inv_soft', 9, invest=True, **SOFT)
# Direction B: point-source CCS in the shared geological store; land competition
for c, sc in SC.items():
    for cum in (200.0, 400.0, 600.0):
        add(f'{c}_NGM_ccs{cum:g}', sc, ccs_cum2100=cum)
    add(f'{c}_NGM_soft_ccs400', sc, ccs_cum2100=400.0, **SOFT)
for lc in (0.25, 0.5):
    add(f'pub_NGM_land{lc:g}', 9, land_c=lc)
    add(f'd2_NGM_land{lc:g}', 2, land_c=lc)
# Direction A: late access to removal (all families from year y), and with investment, soft limits, learning
for c, sc in SC.items():
    for y in (2050.0, 2070.0, 2090.0, 2110.0, 2130.0, 2150.0):
        add(f'{c}_NGM_from{y:g}', sc, startN=y, startG=y, startM=y)
for y in (2070.0, 2110.0, 2150.0):
    add(f'pub_NGM_inv_from{y:g}', 9, invest=True, startN=y, startG=y, startM=y)
    add(f'pub_NGM_soft_from{y:g}', 9, startN=y, startG=y, startM=y, **SOFT)
    add(f'pub_G_from{y:g}', 9, 'G', startG=y)
    add(f'pub_N_from{y:g}', 9, 'N', startN=y)

only = sys.argv[1:]
for name, sc, allow, cal in J:
    if name in runs or (only and not any(o in name for o in only)):
        continue
    pr = R3.Problem(sc, cal=cal, allow=allow)
    x, info, r, d, dt = pr.solve()
    s = pr.summary(x, info, r, d); s['name'] = name; s['cal'] = dict(pr.cal)
    runs[name] = s
    json.dump(runs, open(OUT, 'w'))
    print(f"{name:30s} st{s['status']} SCC25 {s['scc2025']:8.2f} peakT {s['peakT']:.3f}@{s['peakT_year']} T2100 {s['T2100']:.2f} NZ {s['netzero_year']} "
          f"N {s['cumN']:.0f} G {s['cumG']:.0f} M {s['cumM_2200']:.0f} vN {s['vN2025']:.1f} vG {s['vG2025']:.1f} miu50 {s['miu2050']:.3f} {dt:.0f}s", flush=True)
print('done', len(runs))
