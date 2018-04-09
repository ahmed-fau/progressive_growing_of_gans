import cPickle as pickle
import os
import sys

import numpy as np

ckpt_fp, out_dir = sys.argv[1:]

step = int(ckpt_fp.split('-')[-1].split('.')[0])

with open(ckpt_fp, 'rb') as f:
  G, _, Gs = pickle.load(f)

for net, netname in zip([G, Gs], ['G', 'Gs']):
  net_vars = []
  for param in net.trainable_params():
    param_name = param.name
    param_gpuarray = param.eval()
    param_numpy = np.array(param.eval())
    net_vars.append((param_name, param_numpy))

  with open(os.path.join(out_dir, '{}_{}.pkl'.format(str(step).zfill(5), netname)), 'wb') as f:
    pickle.dump(net_vars, f)
