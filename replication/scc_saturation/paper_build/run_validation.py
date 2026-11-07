"""Optimize one scenario with the corrected model and compare with GAMS.
Usage: python run_validation.py <scenario 1..10>
Writes results/val_scen<N>.json and results/val_scen<N>.npz (x_opt, output)."""
import sys, time
import numpy as np
from common import dm, save_json, RES
import os

sc = int(sys.argv[1])
t0 = time.time()
model = dm.Dice2023Model(num_times=81, scenario=sc)
x_opt, output, meta = model.run_model()
elapsed = time.time() - t0
summary, _ = dm.compare_with_gams(model, output, meta, show=False, save=True,
                                  results_dir=os.path.join(RES, 'compare'))
summary['elapsed_s'] = elapsed
summary['meta'] = {k: v for k, v in meta.items()}
np.savez(os.path.join(RES, f'val_scen{sc}.npz'), x_opt=x_opt, output=output)
save_json(summary, f'val_scen{sc}.json')
print('DONE', sc, elapsed)
