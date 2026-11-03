import numpy as np, bv_core as B, jax.numpy as jnp, json
from bv_io import load
for eta in (1.75, 1.9, 2.0):
    pr = B.problem(81, eta, 'dice', None)
    x, info, n, _ = B.solve(pr, pr.start_from_gams(), tol=1e-12, max_iter=20000)
    N = pr.N; al = x[2*N:3*N]; zl = info['mult_x_L'][2*N:3*N]
    r,_ = pr.run(jnp.asarray(x), pr.z, pr.z)
    E = np.asarray(r['ECO2'])[1:]; MAT=np.asarray(r['MAT'])[1:]; T=np.asarray(r['TATM'])[1:]
    idx = np.where(al < 0.1 + 1e-7)[0]
    yrs = 2020 + 5*idx
    print('eta', eta, 'alpha at lower bound in', yrs.tolist(), 'bound mult', np.round(zl[idx], 6).tolist())
    for k in idx[:6]:
        print('   ', 2020+5*k, 'E', round(E[k],2), 'MAT', round(MAT[k],1), 'T', round(T[k],3), 'lamIRF', info['mult_g'][k])
v = load('v2_bf_tlonone.json')
for rec in v:
    if rec['eta'] == 2.0:
        for y in ('2025', '2050'):
            print(y, 'fixed-policy statuses', rec[y]['fixed_policy']['parts']['E']['status'], rec[y]['fixed_policy']['parts']['C']['status'])
